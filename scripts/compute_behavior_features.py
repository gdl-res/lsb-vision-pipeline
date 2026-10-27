#!/usr/bin/env python3
"""Aggregate framewise data into behavioral features (SPECIFICATION.md 11, 18.4).

Produces two tables under --out:
    features_per_window.csv   one row per video x dish x time window (11.1)
    features_per_video.csv    one row per video x dish, whole video (11.2)

Inputs:
    --tracks      larva_tracks.csv     (confidence, morphology, quality flags)
    --kinematics  larva_kinematics.csv (speed, movement_state, turns, area change)
                  defaults to the sibling of --tracks; kinematic features become
                  NaN if it is absent.
    --metadata    experiment_metadata.csv (optional; fills blank experiment_id)

Divergence-from-control features are written as NaN (deferred to a post-process,
SPECIFICATION.md 26.4). All measurement language stays behavioral-proxy only (23).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import crop_petris_yolo_seg as cp
import generate_heatmaps as gh

# --------------------------------------------------------------------------- #
# Column contracts
# --------------------------------------------------------------------------- #
FEATURE_KEYS = [
    "n_frames", "valid_fraction",
    # activity
    "total_distance_mm", "mean_speed_mm_s", "median_speed_mm_s",
    "max_speed_mm_s", "speed_std_mm_s", "immobility_fraction",
    "moving_fraction", "movement_bout_count", "mean_bout_duration_s",
    "median_bout_duration_s", "max_bout_duration_s",
    # path geometry
    "net_displacement_mm", "path_efficiency", "tortuosity",
    "mean_abs_turn_angle_rad", "turn_angle_std_rad", "sharp_turn_count",
    # spatial occupancy
    "center_fraction", "middle_fraction", "edge_fraction", "mean_r_norm",
    "median_r_norm", "std_r_norm", "min_r_norm", "max_r_norm",
    "occupancy_entropy", "occupied_cells", "revisit_rate",
    "rim_occlusion_fraction", "rim_occlusion_event_count",
    # area & morphology
    "mean_area_px", "median_area_px", "std_area_px", "area_cv", "min_area_px",
    "max_area_px", "area_range_px", "mean_area_norm_petri",
    "std_area_norm_petri", "mean_abs_area_change_px", "std_area_change_px",
    "area_oscillation_count", "mean_aspect_ratio", "std_aspect_ratio",
    "mean_eccentricity", "mean_compactness",
    # detection quality
    "mean_larva_conf", "median_larva_conf", "min_larva_conf",
    "missing_fraction", "interpolated_fraction", "track_switch_count",
    "outlier_fraction", "fraction_outside_petri", "quality_score",
]
HEATMAP_KEYS = ["heatmap_entropy", "heatmap_peak_density",
                "heatmap_peak_x_norm", "heatmap_peak_y_norm",
                "heatmap_spread_x", "heatmap_spread_y"]
DIVERGENCE_KEYS = ["js_divergence_from_control",
                   "wasserstein_distance_from_control"]

IDENT = ["dataset_id", "experiment_id", "video_id", "dish_id"]
WINDOW_COLUMNS = IDENT + ["window_id", "window_start_s",
                          "window_end_s"] + FEATURE_KEYS
VIDEO_COLUMNS = IDENT + FEATURE_KEYS + HEATMAP_KEYS + DIVERGENCE_KEYS

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Small pure helpers (testable)
# --------------------------------------------------------------------------- #
def agg(arr: np.ndarray, fn: Callable) -> float:
    """Apply a numpy reducer over finite values; NaN if none."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else NAN


def movement_bouts(states: list[str], median_dt: float) -> list[float]:
    """Durations of contiguous `moving` runs (SPECIFICATION.md 11.1)."""
    durations: list[float] = []
    n = len(states)
    i = 0
    while i < n:
        if states[i] != "moving":
            i += 1
            continue
        j = i
        while j < n and states[j] == "moving":
            j += 1
        durations.append((j - i) * median_dt)
        i = j
    return durations


def band_fractions(r: np.ndarray) -> tuple[float, float, float]:
    """Fraction of points in center (<1/3), middle (1/3..2/3), edge (>2/3)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return NAN, NAN, NAN
    center = float((r < 1 / 3).mean())
    middle = float(((r >= 1 / 3) & (r < 2 / 3)).mean())
    edge = float((r >= 2 / 3).mean())
    return center, middle, edge


def zero_crossings(a: np.ndarray) -> int:
    """Number of sign changes in a sequence (ignoring NaN and zeros)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    signs = np.sign(a)
    signs = signs[signs != 0]
    return int((np.diff(signs) != 0).sum()) if signs.size > 1 else 0


def occupancy_features(x: np.ndarray, y: np.ndarray, bins: int) -> dict:
    """Occupancy entropy / occupied cells / revisit rate from a 2D histogram."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return {"occupancy_entropy": NAN, "occupied_cells": 0,
                "revisit_rate": NAN}
    counts, xe, ye = gh.histogram_norm(x, y, bins)
    counts = counts * gh.circular_mask(xe, ye)
    total = counts.sum()
    occupied = int((counts > 0).sum())
    prob = gh.to_probability(counts)
    entropy = gh.heatmap_stats(prob, xe, ye)["heatmap_entropy"]
    revisit = float(1.0 - occupied / total) if total > 0 else NAN
    return {"occupancy_entropy": entropy, "occupied_cells": occupied,
            "revisit_rate": revisit}


# --------------------------------------------------------------------------- #
# Column accessors (tolerant to missing kinematics columns)
# --------------------------------------------------------------------------- #
def _num(g, col: str) -> np.ndarray:
    import pandas as pd
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
    return np.full(len(g), NAN)


def _bool(g, col: str) -> np.ndarray:
    if col in g.columns:
        return g[col].astype(str).str.lower().eq("true").to_numpy()
    return np.zeros(len(g), dtype=bool)


def _states(g) -> list[str]:
    if "movement_state" in g.columns:
        return g["movement_state"].astype(str).tolist()
    return [""] * len(g)


def _norm_xy(g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized coords, preferring smoothed; returns (x, y, r).

    Falls back to raw coords when the smoothed columns are absent OR present
    but empty (the tracker writes empty smoothed placeholders; smoothed values
    only arrive when a kinematics file is merged)."""
    x, y = _num(g, "x_norm_smooth"), _num(g, "y_norm_smooth")
    if not (np.isfinite(x).any() and np.isfinite(y).any()):
        x, y = _num(g, "x_norm_raw"), _num(g, "y_norm_raw")
    r = _num(g, "r_norm_smooth")
    if not np.isfinite(r).any():
        r = _num(g, "r_norm_raw")
    if not np.isfinite(r).any():
        r = np.hypot(x, y)
    return x, y, r


# --------------------------------------------------------------------------- #
# Core feature computation
# --------------------------------------------------------------------------- #
def compute_features(g, bins: int, sharp_turn_thr: float,
                     rim_r: float = 0.85) -> dict:
    """Compute the shared FEATURE_KEYS for one frame-group (window or video)."""
    n = len(g)
    f = {k: NAN for k in FEATURE_KEYS}
    f["n_frames"] = n
    if n == 0:
        return f

    times = _num(g, "time_s")
    dt = np.diff(times[np.isfinite(times)])
    median_dt = float(np.median(dt)) if dt.size else NAN

    valid = _bool(g, "valid_detection")
    n_valid = int(valid.sum())
    f["valid_fraction"] = n_valid / n

    # activity
    step_mm = _num(g, "step_distance_mm")
    step_norm = _num(g, "step_distance_norm")
    speed = _num(g, "speed_mm_s")
    f["total_distance_mm"] = float(np.nansum(step_mm)) \
        if np.isfinite(step_mm).any() else NAN
    f["mean_speed_mm_s"] = agg(speed, np.mean)
    f["median_speed_mm_s"] = agg(speed, np.median)
    f["max_speed_mm_s"] = agg(speed, np.max)
    f["speed_std_mm_s"] = agg(speed, np.std)

    states = _states(g)
    valid_states = [s for s in states if s in ("moving", "immobile")]
    if valid_states:
        f["moving_fraction"] = valid_states.count("moving") / len(valid_states)
        f["immobility_fraction"] = \
            valid_states.count("immobile") / len(valid_states)
    bouts = movement_bouts(states, median_dt) if np.isfinite(median_dt) else []
    f["movement_bout_count"] = len(bouts)
    if bouts:
        f["mean_bout_duration_s"] = float(np.mean(bouts))
        f["median_bout_duration_s"] = float(np.median(bouts))
        f["max_bout_duration_s"] = float(np.max(bouts))

    # path geometry (ratios use scale-free normalized distances)
    x, y, r = _norm_xy(g)
    fin = np.where(np.isfinite(x) & np.isfinite(y))[0]
    total_norm = float(np.nansum(step_norm)) if np.isfinite(step_norm).any() \
        else NAN
    if fin.size >= 2:
        net_norm = float(np.hypot(x[fin[-1]] - x[fin[0]],
                                  y[fin[-1]] - y[fin[0]]))
    else:
        net_norm = NAN
    # mm-per-normalized-unit (== petri radius in mm) from the kinematics ratio
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = step_mm / step_norm
    mm_per_norm = agg(ratio, np.median)
    f["net_displacement_mm"] = net_norm * mm_per_norm \
        if np.isfinite(net_norm) and np.isfinite(mm_per_norm) else NAN
    if np.isfinite(net_norm) and np.isfinite(total_norm) and total_norm > 0:
        f["path_efficiency"] = net_norm / total_norm
        f["tortuosity"] = total_norm / net_norm if net_norm > 0 else NAN

    abs_turn = _num(g, "abs_turn_angle_rad")
    f["mean_abs_turn_angle_rad"] = agg(abs_turn, np.mean)
    f["turn_angle_std_rad"] = agg(_num(g, "turn_angle_rad"), np.std)
    at = abs_turn[np.isfinite(abs_turn)]
    f["sharp_turn_count"] = int((at > sharp_turn_thr).sum())

    # spatial occupancy
    f["center_fraction"], f["middle_fraction"], f["edge_fraction"] = \
        band_fractions(r)
    f["mean_r_norm"] = agg(r, np.mean)
    f["median_r_norm"] = agg(r, np.median)
    f["std_r_norm"] = agg(r, np.std)
    f["min_r_norm"] = agg(r, np.min)
    f["max_r_norm"] = agg(r, np.max)
    f.update(occupancy_features(x, y, bins))
    # rim-seeking: time spent / events where the larva tucks under the dish
    # edge (occlusion gaps that begin at the rim). Behavioral signal, not just
    # a QC artifact -- see [[dish5-rim-occlusion-finding]].
    missing_g = _bool(g, "missing")
    f["rim_occlusion_fraction"] = cp.rim_occlusion_fraction(missing_g, r, rim_r)
    f["rim_occlusion_event_count"] = cp.rim_occlusion_events(
        missing_g, r, rim_r)

    # area & morphology
    area = _num(g, "mask_area_px")
    f["mean_area_px"] = agg(area, np.mean)
    f["median_area_px"] = agg(area, np.median)
    f["std_area_px"] = agg(area, np.std)
    f["min_area_px"] = agg(area, np.min)
    f["max_area_px"] = agg(area, np.max)
    if np.isfinite(f["max_area_px"]) and np.isfinite(f["min_area_px"]):
        f["area_range_px"] = f["max_area_px"] - f["min_area_px"]
    if np.isfinite(f["mean_area_px"]) and f["mean_area_px"] != 0:
        f["area_cv"] = f["std_area_px"] / f["mean_area_px"]
    f["mean_area_norm_petri"] = agg(_num(g, "mask_area_norm_petri"), np.mean)
    f["std_area_norm_petri"] = agg(_num(g, "mask_area_norm_petri"), np.std)
    area_change = _num(g, "area_change_px")
    f["mean_abs_area_change_px"] = agg(np.abs(area_change), np.mean)
    f["std_area_change_px"] = agg(area_change, np.std)
    f["area_oscillation_count"] = zero_crossings(area_change)
    f["mean_aspect_ratio"] = agg(_num(g, "mask_aspect_ratio"), np.mean)
    f["std_aspect_ratio"] = agg(_num(g, "mask_aspect_ratio"), np.std)
    f["mean_eccentricity"] = agg(_num(g, "mask_eccentricity"), np.mean)
    f["mean_compactness"] = agg(_num(g, "mask_compactness"), np.mean)

    # detection quality
    conf = _num(g, "larva_conf")
    f["mean_larva_conf"] = agg(conf, np.mean)
    f["median_larva_conf"] = agg(conf, np.median)
    f["min_larva_conf"] = agg(conf, np.min)
    missing = _bool(g, "missing")
    interp = _bool(g, "interpolated")
    f["missing_fraction"] = float(missing.mean()) if n else NAN
    f["interpolated_fraction"] = float(interp.mean()) if n else NAN
    f["track_switch_count"] = int(_bool(g, "track_switch").sum())
    outlier = _bool(g, "outlier_position") | _bool(g, "outlier_area")
    f["outlier_fraction"] = float(outlier.mean()) if n else NAN
    inside = _bool(g, "inside_petri")
    f["fraction_outside_petri"] = \
        float((~inside[valid]).mean()) if n_valid else NAN
    vf, fop = f["valid_fraction"], f["fraction_outside_petri"]
    f["quality_score"] = round(vf * (1 - (fop if np.isfinite(fop) else 0)), 6)
    return f


def heatmap_features(g, bins: int) -> dict:
    x, y, _ = _norm_xy(g)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return {k: NAN for k in HEATMAP_KEYS}
    counts, xe, ye = gh.histogram_norm(x, y, bins)
    counts = counts * gh.circular_mask(xe, ye)
    prob = gh.to_probability(counts)
    return gh.heatmap_stats(prob, xe, ye)


def _round_features(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and np.isfinite(v):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# IO / orchestration
# --------------------------------------------------------------------------- #
def load_merged(tracks_path: Path, kinematics_path: Optional[Path]):
    import pandas as pd

    tracks = pd.read_csv(tracks_path)
    if kinematics_path and kinematics_path.is_file():
        kin = pd.read_csv(kinematics_path)
        keys = ["video_id", "frame_idx", "dish_id"]
        # The kinematics file is derived from tracks and re-emits the smoothed /
        # derivative columns. larva_tracks.csv carries some of those (e.g.
        # x_norm_smooth) as EMPTY placeholders, so kinematics must win for any
        # overlapping non-key column -- otherwise the populated smoothed coords
        # are dropped and path_efficiency/tortuosity/occupancy go all-NaN.
        overlap = [c for c in kin.columns
                   if c in tracks.columns and c not in keys]
        merged = tracks.merge(kin, on=keys, how="left", suffixes=("_trk", ""))
        for c in overlap:
            trk = f"{c}_trk"
            if trk in merged.columns:
                # prefer kinematics; fall back to tracks only where kin is NaN
                merged[c] = merged[c].where(merged[c].notna(), merged[trk])
                merged = merged.drop(columns=[trk])
        return merged
    print("WARNING: kinematics file not found; speed/movement/turn/area-change "
          "features will be NaN.", file=sys.stderr)
    return tracks


def fill_identifiers(df, metadata_path: Optional[Path]):
    import pandas as pd
    for col in IDENT:
        if col not in df.columns:
            df[col] = ""
    if metadata_path and metadata_path.is_file():
        meta = pd.read_csv(metadata_path)
        keys = [c for c in ("video_id", "dish_id") if c in meta.columns]
        extra = [c for c in ("dataset_id", "experiment_id")
                 if c in meta.columns]
        if keys and extra:
            df = df.merge(meta[keys + extra], on=keys, how="left",
                          suffixes=("", "_meta"))
            for c in extra:
                mc = f"{c}_meta"
                if mc in df.columns:
                    # an all-empty id column reads back as float NaN; cast to
                    # object so filling it with metadata strings doesn't raise
                    # LossySetitemError, and catch NaN via isna() (astype(str)
                    # on NaN is not reliably "nan" across pandas versions).
                    df[c] = df[c].astype(object)
                    blank = df[c].isna() | df[c].astype(str).str.strip().isin(
                        ["", "nan", "None", "<NA>"])
                    df.loc[blank, c] = df.loc[blank, mc]
                    df = df.drop(columns=[mc])
    return df


def write_provenance(out_dir: Path, args, n_windows: int, n_videos: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_features"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/compute_behavior_features.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.tracks), str(args.kinematics)],
        "output_path": str(args.out),
        "n_window_rows": n_windows, "n_video_rows": n_videos,
        "parameters": {
            "window_size_s": args.window_size_s,
            "heatmap_bins": args.heatmap_bins,
            "sharp_turn_threshold_rad": args.sharp_turn_threshold_rad,
            "rim_occlusion_r": args.rim_occlusion_r,
        },
    }
    with open(out_dir / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", required=True, type=Path)
    p.add_argument("--kinematics", type=Path, default=None,
                   help="defaults to sibling larva_kinematics.csv")
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--window-size-s", type=float, default=60.0)
    p.add_argument("--heatmap-bins", type=int, default=50)
    p.add_argument("--sharp-turn-threshold-rad", type=float, default=1.0)
    p.add_argument("--rim-occlusion-r", type=float, default=0.85,
                   help="r_norm at/above which a larva is 'at the rim'; gaps "
                        "beginning there count as rim-seeking occlusion")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    if args.kinematics is None:
        sib = args.tracks.parent / "larva_kinematics.csv"
        args.kinematics = sib if sib.is_file() else sib  # keep path for provenance
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_merged(args.tracks, args.kinematics)
    df = fill_identifiers(df, args.metadata)
    df["window_id"] = (pd.to_numeric(df["time_s"], errors="coerce")
                       // args.window_size_s).astype("Int64")

    window_rows, video_rows = [], []
    for (vid, dish), vg in df.groupby(["video_id", "dish_id"], sort=True):
        ids = {"dataset_id": vg["dataset_id"].iloc[0],
               "experiment_id": vg["experiment_id"].iloc[0],
               "video_id": vid, "dish_id": dish}
        # per-window
        for wid, wg in vg.groupby("window_id", sort=True):
            if pd.isna(wid):
                continue
            feats = _round_features(
                compute_features(wg, args.heatmap_bins,
                                 args.sharp_turn_threshold_rad,
                                 args.rim_occlusion_r))
            window_rows.append({
                **ids, "window_id": int(wid),
                "window_start_s": round(int(wid) * args.window_size_s, 3),
                "window_end_s": round((int(wid) + 1) * args.window_size_s, 3),
                **feats})
        # per-video
        vfeats = _round_features(
            compute_features(vg, args.heatmap_bins,
                             args.sharp_turn_threshold_rad,
                             args.rim_occlusion_r))
        vfeats.update(_round_features(heatmap_features(vg, args.heatmap_bins)))
        vfeats.update({k: NAN for k in DIVERGENCE_KEYS})  # deferred (26.4)
        video_rows.append({**ids, **vfeats})
        print(f"  {vid} dish {dish}: {len(vg)} frames")

    win_df = pd.DataFrame(window_rows).reindex(columns=WINDOW_COLUMNS)
    vid_df = pd.DataFrame(video_rows).reindex(columns=VIDEO_COLUMNS)
    win_df.to_csv(args.out / "features_per_window.csv", index=False)
    vid_df.to_csv(args.out / "features_per_video.csv", index=False)
    write_provenance(args.out, args, len(win_df), len(vid_df))
    print(f"\nWrote {len(win_df)} window rows and {len(vid_df)} video rows "
          f"to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
