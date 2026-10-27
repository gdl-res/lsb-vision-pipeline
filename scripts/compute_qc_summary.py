#!/usr/bin/env python3
"""Per video x dish quality-control summary (SPECIFICATION.md 13).

Reads the framewise tables and writes qc_summary.csv with detection-quality
metrics and exclusion *recommendations*. Per SPECIFICATION.md 13 the default behavior
never deletes data: it flags. Exclusion thresholds are all configurable.

Inputs:
    --tracks      larva_tracks.csv     (required)
    --petri-tracks petri_tracks.csv    (default: sibling; for petri_valid/conf)
    --kinematics  larva_kinematics.csv (default: sibling; interpolated fraction)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp
from crop_petris_yolo_seg import rim_occlusion_fraction  # shared QC + features
from generate_fair_metadata import QC_COLUMNS

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Pure helpers (testable)
# --------------------------------------------------------------------------- #
def longest_run(mask: np.ndarray) -> int:
    """Length (in elements) of the longest run of True in a boolean array."""
    best = run = 0
    for v in np.asarray(mask, dtype=bool):
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def median_dt(times: np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    d = np.diff(np.sort(t))
    return float(np.median(d)) if d.size else NAN


def duration_seconds(times: np.ndarray, n: int, dt: float) -> float:
    """Observation span in seconds: prefer the time_s range, fall back to
    (n-1) * median_dt when timestamps are missing/degenerate."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if t.size >= 2:
        span = float(t.max() - t.min())
        if span > 0:
            return span
    if np.isfinite(dt) and n > 1:
        return (n - 1) * dt
    return NAN


def mad_outlier_fraction(values: np.ndarray, k: float) -> float:
    """Fraction of finite values whose modified z-score exceeds k (MAD-based)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return 0.0
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return 0.0
    modz = 0.6745 * (v - med) / mad
    return float((np.abs(modz) > k).mean())


def step_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean steps between consecutive finite (x, y) points."""
    pts = np.column_stack([x, y])
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return np.array([])
    return np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))


def semantic_switch_count(x: np.ndarray, y: np.ndarray,
                          frame_idx: np.ndarray, max_step: float) -> int:
    """Count physically-implausible single-frame position jumps.

    A "semantic switch" is a step between two *frame-adjacent* finite points
    (frame_idx difference of exactly 1) whose magnitude exceeds max_step. In
    petri-radius units a step this large in one inter-frame interval is far
    faster than a larva can crawl, so it marks the selected primary jumping
    to a *different physical object* (an artifact or a mask fragment) --
    genuine trajectory corruption -- rather than the nominal ByteTrack ID
    churn that `track_switch_count` conflates it with.

    Computed on *raw* coordinates on purpose: smoothing (savgol) damps a
    one-frame spike and interpolation bridges gaps, both of which would hide
    the teleport this is meant to catch. Restricting to frame-adjacent pairs
    avoids penalizing a legitimately large reappearance jump after a genuine
    multi-frame occlusion gap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fi = np.asarray(frame_idx, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(fi)
    x, y, fi = x[finite], y[finite], fi[finite]
    if x.size < 2 or not np.isfinite(max_step):
        return 0
    adjacent = np.diff(fi) == 1
    step = np.hypot(np.diff(x), np.diff(y))
    return int(np.sum(adjacent & (step > max_step)))


def position_spread(x: np.ndarray, y: np.ndarray) -> float:
    """Spatial dispersion of finite (x, y) points: hypot of the per-axis
    standard deviations. A near-zero value means the point barely moved."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return NAN
    return float(np.hypot(np.std(x[finite]), np.std(y[finite])))


def suspected_empty_dish(mean_area_norm: float, mean_conf: float,
                         pos_spread: float, thr: dict) -> bool:
    """Flag a dish whose 'detections' look like a static artifact, not a larva.

    A persistent dish-surface mark / reflection can be detected frame after
    frame at low confidence and small size in a fixed spot, producing a fake
    'track' that the jump-hysteresis gate cannot catch (a stationary artifact
    never jumps). This is distinct from a poorly-tracked *real* larva.

    Requires ALL THREE signals together, which is what separates an empty
    dish from a genuinely immobile larva (immobility is itself a stress
    phenotype we must not discard): sub-larva mask area AND low confidence AND
    near-zero positional spread. On real footage a true larva always clears at
    least one of these -- e.g. a motionless larva still has larva-sized area
    and normal confidence. Conservative by design: only fires on positive
    artifact-like evidence, never on missing area/conf/spread.
    """
    a, c, s = mean_area_norm, mean_conf, pos_spread
    return bool(
        np.isfinite(a) and a < thr["empty_max_area_norm"] and
        np.isfinite(c) and c < thr["empty_max_conf"] and
        np.isfinite(s) and s < thr["empty_max_spread"])


def morphology_reliability(valid: np.ndarray, solidity: np.ndarray,
                           min_solidity: float) -> float:
    """Fraction of valid-detection frames that carry a usable larva mask.

    A per-dish confidence signal for the *morphology* features (area,
    eccentricity, compactness, ...), reported (never used for exclusion):
    those features are only as trustworthy as the masks they are computed
    from. A frame counts as usable when a mask was actually produced (mask
    solidity is finite -- box-only detections leave it NaN) and is not
    grossly fragmented (solidity >= min_solidity; a split mask spans a big
    convex-hull gap and scores low). Low values mean a dish's morphology is
    built on comparatively few clean masks.

    This is a *relative* per-dish signal; it does NOT capture the absolute
    boundary-noise ceiling of the segmenter (mask mAP@50-95 ~0.70 on the
    validation split), which limits ALL morphology values globally -- see the
    data dictionary / report caveats.
    """
    valid = np.asarray(valid, dtype=bool)
    solidity = np.asarray(solidity, dtype=float)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return NAN
    usable = valid & np.isfinite(solidity) & (solidity >= min_solidity)
    return float(usable.sum()) / n_valid


def evaluate_exclusion(metrics: dict, thr: dict) -> tuple[bool, str]:
    """Apply configurable exclusion rules (SPECIFICATION.md 13). Returns (flag, reason)."""
    reasons = []
    vdf = metrics["valid_detection_fraction"]
    if np.isfinite(vdf) and vdf < thr["min_valid_fraction"]:
        reasons.append(f"valid_detection_fraction<{thr['min_valid_fraction']}")
    gap = metrics["longest_missing_gap_s"]
    if np.isfinite(gap) and gap > thr["max_missing_gap_s"]:
        reasons.append(f"longest_missing_gap_s>{thr['max_missing_gap_s']}")
    fop = metrics["fraction_outside_petri"]
    if np.isfinite(fop) and fop > thr["max_fraction_outside"]:
        reasons.append(f"fraction_outside_petri>{thr['max_fraction_outside']}")
    # Trajectory integrity: gate on the *semantic* switch rate (physically
    # implausible single-frame jumps), NOT the nominal track_switch_rate.
    # The latter counts every ByteTrack ID re-acquisition -- which for a
    # <=1-larva-per-dish problem is mostly benign detection-flicker churn on
    # the same animal, and was excluding ~72% of dishes on real footage for
    # no trajectory-level reason. semantic_switch_rate only fires when the
    # selected centroid actually teleports across the dish.
    ssr = metrics.get("semantic_switch_rate_per_min", NAN)
    if np.isfinite(ssr) and ssr > thr["max_semantic_switch_rate_per_min"]:
        reasons.append(
            "semantic_switch_rate_per_min>"
            f"{thr['max_semantic_switch_rate_per_min']}")
    # The nominal track_switch_rate rule is retained but OPT-IN (default
    # disabled): pass a finite --max-track-switch-rate-per-min to re-enable
    # it, e.g. to audit tracker-ID stability specifically.
    tsr_thr = thr.get("max_track_switch_rate_per_min")
    tsr = metrics.get("track_switch_rate_per_min", NAN)
    if tsr_thr is not None and np.isfinite(tsr) and tsr > tsr_thr:
        reasons.append(f"track_switch_rate_per_min>{tsr_thr}")
    # No real subject: exclude, but with a distinct reason so downstream can
    # tell "empty dish" apart from "larva present but poorly tracked".
    if metrics.get("suspected_empty"):
        reasons.append("suspected_empty")
    return (len(reasons) > 0), ";".join(reasons)


# --------------------------------------------------------------------------- #
# Column accessors
# --------------------------------------------------------------------------- #
def _num(g, col):
    import pandas as pd
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
    return np.full(len(g), NAN)


def _bool(g, col):
    if col in g.columns:
        return g[col].astype(str).str.lower().eq("true").to_numpy()
    return np.zeros(len(g), dtype=bool)


def _norm_xy(g):
    """Best available normalized coords, preferring smoothed then raw.

    larva_tracks.csv carries empty x_norm_smooth/y_norm_smooth placeholders
    (the kinematics step is what populates smoothing), so a naive "column
    exists -> use it" check silently reads all-NaN and zeroes out anything
    built on it (this is what left position_outlier_fraction dead). The
    kinematics merge brings the populated copy in under a _kin suffix; fall
    through smoothed -> smoothed_kin -> raw, taking the first that is
    actually finite."""
    for xc, yc in (("x_norm_smooth", "y_norm_smooth"),
                   ("x_norm_smooth_kin", "y_norm_smooth_kin"),
                   ("x_norm_raw", "y_norm_raw")):
        xs, ys = _num(g, xc), _num(g, yc)
        if np.isfinite(xs).any() and np.isfinite(ys).any():
            return xs, ys
    return _num(g, "x_norm_raw"), _num(g, "y_norm_raw")


# --------------------------------------------------------------------------- #
# Per-group QC
# --------------------------------------------------------------------------- #
def compute_qc(g, petri_summary, thr, area_mad, pos_mad, rim_r,
               morph_min_solidity=0.5) -> dict:
    import pandas as pd

    g = g.sort_values("frame_idx")
    n = len(g)
    valid = _bool(g, "valid_detection")
    n_valid = int(valid.sum())
    times = _num(g, "time_s")
    dt = median_dt(times)

    missing = _bool(g, "missing") if "missing" in g.columns else ~valid
    missing_count = int(missing.sum())
    longest_gap_frames = longest_run(missing)
    longest_gap_s = longest_gap_frames * dt if np.isfinite(dt) else NAN

    switch_count = int(_bool(g, "track_switch").sum())
    duration_s = duration_seconds(times, n, dt)
    dur_min = duration_s / 60.0 if np.isfinite(duration_s) and duration_s > 0 \
        else NAN
    switch_rate = switch_count / dur_min if np.isfinite(dur_min) else NAN

    # Semantic (physically-implausible) switches on RAW normalized coords --
    # the trajectory-integrity signal the exclusion rule actually gates on.
    frame_idx = _num(g, "frame_idx")
    x_raw, y_raw = _num(g, "x_norm_raw"), _num(g, "y_norm_raw")
    sem_count = semantic_switch_count(x_raw, y_raw, frame_idx, thr["max_step_norm"])
    sem_rate = sem_count / dur_min if np.isfinite(dur_min) else NAN

    conf = _num(g, "larva_conf")
    inside = _bool(g, "inside_petri")
    area = _num(g, "mask_area_px")
    x, y = _norm_xy(g)
    r_norm = _num(g, "r_norm_raw")
    if not np.isfinite(r_norm).any():
        r_norm = np.hypot(x, y)

    # Empty-dish / static-artifact signals: sub-larva size + low conf +
    # near-zero motion together mark a fixed artifact rather than a larva.
    # Uses raw normalized coords/area over valid frames (missing frames are
    # NaN and drop out of the nanmean/spread).
    mean_conf = float(np.nanmean(conf)) if np.isfinite(conf).any() else NAN
    area_norm = _num(g, "mask_area_norm_petri")
    mean_area_norm = float(np.nanmean(area_norm)) \
        if np.isfinite(area_norm).any() else NAN
    pos_spread = position_spread(x_raw, y_raw)

    fop = float((~inside[valid]).mean()) if n_valid else NAN
    metrics = {
        "valid_detection_fraction": n_valid / n if n else NAN,
        "mean_larva_conf": mean_conf,
        "median_larva_conf": float(np.nanmedian(conf)) if np.isfinite(conf).any() else NAN,
        "missing_frame_count": missing_count,
        "longest_missing_gap_s": round(longest_gap_s, 4) if np.isfinite(longest_gap_s) else NAN,
        "track_switch_count": switch_count,
        "track_switch_rate_per_min": round(switch_rate, 4) if np.isfinite(switch_rate) else NAN,
        "semantic_switch_count": sem_count,
        "semantic_switch_rate_per_min": round(sem_rate, 4) if np.isfinite(sem_rate) else NAN,
        "fraction_outside_petri": fop,
        "rim_occlusion_fraction": round(
            rim_occlusion_fraction(missing, r_norm, rim_r), 6),
        "mean_mask_area_px": float(np.nanmean(area)) if np.isfinite(area).any() else NAN,
        "mask_area_outlier_fraction": mad_outlier_fraction(area, area_mad),
        "morphology_reliability": morphology_reliability(
            valid, _num(g, "mask_solidity"), morph_min_solidity),
        "position_outlier_fraction": mad_outlier_fraction(step_distances(x, y), pos_mad),
        "interpolated_fraction": float(_bool(g, "interpolated").mean()) if n else NAN,
        "suspected_empty": suspected_empty_dish(
            mean_area_norm, mean_conf, pos_spread, thr),
    }
    metrics["overall_quality_score"] = round(
        metrics["valid_detection_fraction"] *
        (1 - (fop if np.isfinite(fop) else 0)), 6)
    exclude, reason = evaluate_exclusion(metrics, thr)

    row = {
        "dataset_id": g["dataset_id"].iloc[0] if "dataset_id" in g else "",
        "experiment_id": g["experiment_id"].iloc[0] if "experiment_id" in g else "",
        "video_id": g["video_id"].iloc[0],
        "dish_id": g["dish_id"].iloc[0],
        "petri_valid": petri_summary.get("petri_valid", ""),
        "petri_conf": petri_summary.get("petri_conf", NAN),
        **{k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else v)
           for k, v in metrics.items()},
        "exclude_recommended": exclude,
        "exclude_reason": reason,
    }
    return row


def summarize_petri(petri_df):
    """Per (video_id, dish_id) petri validity/confidence summary."""
    import pandas as pd
    out = {}
    if petri_df is None:
        return out
    for (vid, dish), g in petri_df.groupby(["video_id", "dish_id"]):
        valid = g["petri_valid"].astype(str).str.lower().eq("true") \
            if "petri_valid" in g.columns else pd.Series([True] * len(g))
        conf = pd.to_numeric(g.get("petri_conf"), errors="coerce") \
            if "petri_conf" in g.columns else pd.Series([np.nan])
        out[(vid, dish)] = {
            "petri_valid": bool(valid.mean() >= 0.5),
            "petri_conf": round(float(conf.mean()), 6)
            if conf.notna().any() else NAN,
        }
    return out


# --------------------------------------------------------------------------- #
# IO / orchestration
# --------------------------------------------------------------------------- #
def write_provenance(out_dir: Path, args, n_rows: int, thr: dict) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_qc"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/compute_qc_summary.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.tracks)],
        "output_path": str(args.out),
        "n_rows": n_rows,
        "parameters": {**thr, "area_outlier_mad": args.area_outlier_mad,
                       "position_outlier_mad": args.position_outlier_mad,
                       "rim_occlusion_r": args.rim_occlusion_r,
                       "morphology_min_solidity": args.morphology_min_solidity},
    }
    with open(out_dir / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", required=True, type=Path)
    p.add_argument("--petri-tracks", type=Path, default=None)
    p.add_argument("--kinematics", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="default: sibling qc_summary.csv")
    # exclusion thresholds (SPECIFICATION.md 13; configurable, not hard-coded).
    # Larvae genuinely leave the petri and linger at the rim (thigmotaxis) -
    # both drive real, expected missing/outside-petri readings rather than
    # tracking failures, so these three are tuned looser than the original
    # illustrative SPECIFICATION.md 13 example values to avoid flagging normal
    # behavior.
    p.add_argument("--min-valid-fraction", type=float, default=0.40)
    p.add_argument("--max-missing-gap-s", type=float, default=60.0)
    p.add_argument("--max-fraction-outside", type=float, default=0.30)
    # Trajectory integrity gate. A semantic switch is a single-frame position
    # jump larger than --max-step-norm (petri-radius units); the exclusion
    # rule flags dishes whose per-minute rate of these exceeds
    # --max-semantic-switch-rate-per-min. This replaces the old nominal
    # track-switch gate, which counted benign ByteTrack ID churn and excluded
    # ~72% of real dishes for no trajectory-level reason.
    p.add_argument("--max-step-norm", type=float, default=0.5,
                   help="single-frame position jump (in petri radii) above "
                        "which a step is a physically-implausible 'semantic "
                        "switch'; ~0.5 radius/frame is far faster than a "
                        "larva crawls, so it marks a jump to a different "
                        "object rather than real motion")
    p.add_argument("--max-semantic-switch-rate-per-min", type=float,
                   default=6.0,
                   help="exclude when semantic (implausible-jump) switches "
                        "exceed this per-minute rate; length-independent")
    p.add_argument("--max-track-switch-rate-per-min", type=float, default=None,
                   help="OPT-IN (default disabled): also exclude when the "
                        "nominal ByteTrack ID-switch rate exceeds this. Off "
                        "by default because ID churn is mostly benign for "
                        "<=1 larva/dish; set it only to audit tracker-ID "
                        "stability specifically")
    # suspected-empty / static-artifact detection: a dish is flagged when its
    # detections are ALL of sub-larva size AND low confidence AND near-
    # stationary (an immobile real larva clears at least one, so it is safe).
    p.add_argument("--empty-max-area-norm", type=float, default=0.001,
                   help="mean mask area (fraction of petri area) below which "
                        "detections are too small to be a larva "
                        "(real larvae ~0.004; a fixed artifact ~0.0002)")
    p.add_argument("--empty-max-conf", type=float, default=0.45,
                   help="mean confidence below which detections look "
                        "artifact-like")
    p.add_argument("--empty-max-spread", type=float, default=0.15,
                   help="positional spread (petri radii) below which the "
                        "'track' is effectively stationary")
    # morphology reliability: mask counts as usable at/above this solidity
    p.add_argument("--morphology-min-solidity", type=float, default=0.5,
                   help="mask solidity below which a mask is treated as "
                        "fragmented/degenerate when scoring "
                        "morphology_reliability (reported, not an exclusion "
                        "rule)")
    # outlier sensitivity
    p.add_argument("--area-outlier-mad", type=float, default=3.5)
    p.add_argument("--position-outlier-mad", type=float, default=3.5)
    # rim-occlusion attribution: a missing gap counts as occlusion when the
    # larva was last seen at r_norm >= this (near the dish edge)
    p.add_argument("--rim-occlusion-r", type=float, default=0.85)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    if args.petri_tracks is None:
        sib = args.tracks.parent / "petri_tracks.csv"
        args.petri_tracks = sib if sib.is_file() else None
    if args.kinematics is None:
        sib = args.tracks.parent / "larva_kinematics.csv"
        args.kinematics = sib if sib.is_file() else None
    if args.out is None:
        args.out = args.tracks.parent / "qc_summary.csv"

    tracks = pd.read_csv(args.tracks)
    if args.kinematics and args.kinematics.is_file():
        kin = pd.read_csv(args.kinematics)
        keep = [c for c in ("video_id", "frame_idx", "dish_id",
                            "x_norm_smooth", "y_norm_smooth", "missing",
                            "interpolated") if c in kin.columns]
        tracks = tracks.merge(kin[keep],
                              on=["video_id", "frame_idx", "dish_id"],
                              how="left", suffixes=("", "_kin"))
    petri_df = pd.read_csv(args.petri_tracks) \
        if args.petri_tracks and args.petri_tracks.is_file() else None
    petri_summary = summarize_petri(petri_df)

    thr = {"min_valid_fraction": args.min_valid_fraction,
           "max_missing_gap_s": args.max_missing_gap_s,
           "max_fraction_outside": args.max_fraction_outside,
           "max_step_norm": args.max_step_norm,
           "max_semantic_switch_rate_per_min":
               args.max_semantic_switch_rate_per_min,
           "max_track_switch_rate_per_min":
               args.max_track_switch_rate_per_min,
           "empty_max_area_norm": args.empty_max_area_norm,
           "empty_max_conf": args.empty_max_conf,
           "empty_max_spread": args.empty_max_spread}

    rows = []
    for (vid, dish), g in tracks.groupby(["video_id", "dish_id"], sort=True):
        rows.append(compute_qc(g, petri_summary.get((vid, dish), {}),
                               thr, args.area_outlier_mad,
                               args.position_outlier_mad,
                               args.rim_occlusion_r,
                               args.morphology_min_solidity))
        flag = rows[-1]["exclude_recommended"]
        print(f"  {vid} dish {dish}: valid="
              f"{rows[-1]['valid_detection_fraction']}"
              + ("  EXCLUDE" if flag else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=QC_COLUMNS)
    df.to_csv(args.out, index=False)
    write_provenance(args.out.parent, args, len(df), thr)
    n_excl = int(df["exclude_recommended"].sum()) if len(df) else 0
    print(f"\nWrote {len(df)} QC rows to {args.out} "
          f"({n_excl} flagged for exclusion).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
