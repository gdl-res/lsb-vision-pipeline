#!/usr/bin/env python3
"""Compute framewise smoothed coordinates and kinematics from larva_tracks.csv.

Implements SPECIFICATION.md 10 and the locked v1 processing policy (SPECIFICATION.md 26.3,
26.4). This is the analysis step that turns raw measurements into derived
per-frame kinematics; it does not modify the raw tables.

Per (video_id, dish_id) series, ordered by frame_idx:
  1. Linear-interpolate position gaps <= --max-interp-gap-s; longer gaps stay
     missing (SPECIFICATION.md 26.3).
  2. Smooth the interpolated global coordinates with the configured filter
     (--smoothing-filter: savgol | moving_average | kalman | none; default
     savgol window 7 polyorder 2, the locked v1 policy); re-derive
     Petri-normalized smoothed coordinates.
  3. Compute kinematics from the smoothed coordinates (SPECIFICATION.md 10).

`jerk` is intentionally dropped (SPECIFICATION.md 26.4); `acceleration_mm_s2` is kept.
Output: larva_kinematics.csv (one row per frame x dish).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp

OUTPUT_COLUMNS = [
    "dataset_id", "experiment_id", "video_id", "frame_idx", "time_s",
    "dish_id", "larva_track_id",
    # smoothed coordinates (SPECIFICATION.md 8.2 / 26.3)
    "cx_global_smooth", "cy_global_smooth",
    "x_norm_smooth", "y_norm_smooth", "r_norm_smooth",
    # kinematics (SPECIFICATION.md 10; jerk dropped per 26.4)
    "dx_norm", "dy_norm", "step_distance_norm", "step_distance_mm",
    "speed_norm_s", "speed_mm_s", "acceleration_mm_s2",
    "turn_angle_rad", "abs_turn_angle_rad", "angular_speed_rad_s",
    "radial_speed_norm_s",
    "area_change_px", "area_change_abs_px", "area_change_rate_px_s",
    "is_moving", "is_immobile", "movement_state",
    # provenance of the derivation
    "missing", "interpolated",
]


# --------------------------------------------------------------------------- #
# Pure numeric helpers (testable without pandas/scipy where possible)
# --------------------------------------------------------------------------- #
def interpolate_gaps(values: np.ndarray, times: np.ndarray,
                     max_gap_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolate NaN gaps no longer than max_gap_s (SPECIFICATION.md 26.3).

    Returns (filled, interpolated_mask, still_missing_mask). Gaps longer than
    max_gap_s, and leading/trailing NaNs, are left missing.
    """
    v = np.asarray(values, dtype=float).copy()
    t = np.asarray(times, dtype=float)
    n = len(v)
    interpolated = np.zeros(n, dtype=bool)
    valid_idx = np.where(~np.isnan(v))[0]
    if len(valid_idx) < 2:
        return v, interpolated, np.isnan(v)
    for a, b in zip(valid_idx[:-1], valid_idx[1:]):
        if b == a + 1:
            continue
        span = t[b] - t[a]
        if span <= max_gap_s and span > 0:
            for k in range(a + 1, b):
                frac = (t[k] - t[a]) / span
                v[k] = v[a] + frac * (v[b] - v[a])
                interpolated[k] = True
    return v, interpolated, np.isnan(v)


SMOOTHING_FILTERS = ("savgol", "moving_average", "kalman", "none")


def _apply_per_run(values, fn, times=None) -> np.ndarray:
    """Apply ``fn`` to each contiguous non-NaN run, preserving NaN gaps.

    ``fn`` receives the run's value slice (and, if ``times`` is given, the
    matching time slice) and returns a smoothed array of the same length.
    """
    v = np.asarray(values, dtype=float)
    out = v.copy()
    isn = np.isnan(v)
    n = len(v)
    i = 0
    while i < n:
        if isn[i]:
            i += 1
            continue
        j = i
        while j < n and not isn[j]:
            j += 1
        if times is None:
            out[i:j] = fn(v[i:j])
        else:
            out[i:j] = fn(v[i:j], np.asarray(times, dtype=float)[i:j])
        i = j
    return out


def savgol_smooth(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smooth, applied per contiguous non-NaN run.

    Runs shorter than the filter need are left as their raw values; the window
    is shrunk to the largest valid odd length for short runs. NaNs are preserved.
    """
    from scipy.signal import savgol_filter

    def _run(v):
        length = len(v)
        if length < polyorder + 1:
            return v
        w = min(window, length)
        if w % 2 == 0:
            w -= 1
        if w <= polyorder:
            return v
        return savgol_filter(v, w, polyorder)

    return _apply_per_run(values, _run)


def moving_average_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smooth, applied per contiguous non-NaN run.

    The window is forced odd and shrunk to the run length; edges use the
    largest symmetric window that fits, so the run length is preserved. NaNs
    are preserved.
    """
    w = window if window % 2 == 1 else window - 1
    w = max(w, 1)

    def _run(v):
        length = len(v)
        if length < 2 or w < 3:
            return v
        half = w // 2
        out = np.empty(length, dtype=float)
        for k in range(length):
            lo = max(0, k - half)
            hi = min(length, k + half + 1)
            # keep the window symmetric about k at the edges
            reach = min(k - lo, hi - 1 - k)
            out[k] = v[k - reach:k + reach + 1].mean()
        return out

    return _apply_per_run(values, _run)


def kalman_smooth(values: np.ndarray, times: np.ndarray,
                  process_var: float, measurement_var: float) -> np.ndarray:
    """Constant-velocity Kalman (RTS) smoother, per contiguous non-NaN run.

    1-D position state with a latent velocity, white-noise-acceleration process
    model that accounts for variable dt. ``process_var`` is the acceleration
    spectral density; ``measurement_var`` is the position observation variance.
    NaNs are preserved.
    """
    def _run(v, t):
        n = len(v)
        if n < 2:
            return v
        # forward filter
        xf = np.zeros((n, 2))           # filtered means [pos, vel]
        Pf = np.zeros((n, 2, 2))        # filtered covariances
        xp = np.zeros((n, 2))           # predicted means
        Pp = np.zeros((n, 2, 2))        # predicted covariances
        H = np.array([[1.0, 0.0]])
        R = np.array([[measurement_var]])
        x = np.array([v[0], 0.0])
        P = np.array([[measurement_var, 0.0], [0.0, 1.0]])
        xf[0], Pf[0] = x, P
        xp[0], Pp[0] = x, P
        for k in range(1, n):
            dt = t[k] - t[k - 1]
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
            F = np.array([[1.0, dt], [0.0, 1.0]])
            q = process_var * np.array([[dt**3 / 3.0, dt**2 / 2.0],
                                        [dt**2 / 2.0, dt]])
            x = F @ x
            P = F @ P @ F.T + q
            xp[k], Pp[k] = x, P
            y = v[k] - (H @ x)[0]
            S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + (K @ np.array([y]))
            P = (np.eye(2) - K @ H) @ P
            xf[k], Pf[k] = x, P
        # RTS backward smoother
        xs = xf.copy()
        Ps = Pf.copy()
        for k in range(n - 2, -1, -1):
            dt = t[k + 1] - t[k]
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
            F = np.array([[1.0, dt], [0.0, 1.0]])
            C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
            xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
            Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
        return xs[:, 0]

    return _apply_per_run(values, _run, times=times)


def smooth_series(values: np.ndarray, times: np.ndarray, args) -> np.ndarray:
    """Dispatch to the configured smoothing filter (--smoothing-filter)."""
    method = args.smoothing_filter
    if method == "none":
        return np.asarray(values, dtype=float).copy()
    if method == "savgol":
        return savgol_smooth(values, args.smoothing_window, args.savgol_polyorder)
    if method == "moving_average":
        return moving_average_smooth(values, args.smoothing_window)
    if method == "kalman":
        return kalman_smooth(values, times,
                             args.kalman_process_var, args.kalman_measurement_var)
    raise ValueError(f"unknown smoothing filter: {method!r}")


def diff_prepend_nan(a: np.ndarray) -> np.ndarray:
    """First differences with a leading NaN (length preserved)."""
    a = np.asarray(a, dtype=float)
    d = np.full(len(a), np.nan)
    if len(a) > 1:
        d[1:] = a[1:] - a[:-1]
    return d


def turn_angles(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Signed turn angle between consecutive displacement vectors (radians)."""
    n = len(dx)
    ta = np.full(n, np.nan)
    for i in range(2, n):
        ax, ay, bx, by = dx[i - 1], dy[i - 1], dx[i], dy[i]
        if any(math.isnan(v) for v in (ax, ay, bx, by)):
            continue
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        ta[i] = math.atan2(cross, dot)
    return ta


def classify_movement(speed_mm_s: float, threshold: float,
                      missing: bool, interpolated: bool) -> str:
    """Per-frame movement_state (controlled vocab, SPECIFICATION.md 10/15.3)."""
    if missing:
        return "missing"
    if interpolated:
        return "interpolated"
    if speed_mm_s is None or (isinstance(speed_mm_s, float)
                              and math.isnan(speed_mm_s)):
        return "uncertain"
    return "moving" if speed_mm_s > threshold else "immobile"


# --------------------------------------------------------------------------- #
# Per-series kinematics
# --------------------------------------------------------------------------- #
def process_series(df, args):
    """Compute kinematics for one (video_id, dish_id) group. Mutates a copy."""
    import pandas as pd

    g = df.sort_values("frame_idx").reset_index(drop=True)
    t = g["time_s"].to_numpy(dtype=float)

    missing_in = ~g["valid_detection"].astype(str).str.lower().eq("true").to_numpy()
    raw_x = g["cx_global_raw"].to_numpy(dtype=float, copy=True)
    raw_y = g["cy_global_raw"].to_numpy(dtype=float, copy=True)
    raw_x[missing_in] = np.nan
    raw_y[missing_in] = np.nan

    # 1. interpolate short gaps
    fx, ix, mx = interpolate_gaps(raw_x, t, args.max_interp_gap_s)
    fy, _, _ = interpolate_gaps(raw_y, t, args.max_interp_gap_s)
    interpolated = ix
    missing = mx  # still-missing after interpolation

    # 2. smooth global coords, re-derive normalized smoothed coords
    sx = smooth_series(fx, t, args)
    sy = smooth_series(fy, t, args)

    pcx = g["petri_cx"].to_numpy(dtype=float)
    pcy = g["petri_cy"].to_numpy(dtype=float)
    prad = g["petri_radius"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        xn = (sx - pcx) / prad
        yn = (sy - pcy) / prad
        rn = np.hypot(xn, yn)
        if args.petri_diameter_mm:
            mm_per_px = np.where((prad > 0) & np.isfinite(prad),
                                 args.petri_diameter_mm / (2 * prad), np.nan)
        else:
            mm_per_px = np.full(len(prad), np.nan)

    # 3. kinematics from smoothed coordinates
    dxn = diff_prepend_nan(xn)
    dyn = diff_prepend_nan(yn)
    dxg = diff_prepend_nan(sx)
    dyg = diff_prepend_nan(sy)
    dt = diff_prepend_nan(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        step_norm = np.hypot(dxn, dyn)
        step_px = np.hypot(dxg, dyg)
        step_mm = step_px * mm_per_px
        speed_norm = step_norm / dt
        speed_mm = step_mm / dt
        accel_mm = diff_prepend_nan(speed_mm) / dt
        ta = turn_angles(dxg, dyg)
        ang_speed = ta / dt
        radial_speed = diff_prepend_nan(rn) / dt
        area = g["mask_area_px"].to_numpy(dtype=float)
        area_change = diff_prepend_nan(area)
        area_rate = area_change / dt

    out = g[["dataset_id", "experiment_id", "video_id", "frame_idx",
             "time_s", "dish_id", "larva_track_id"]].copy()
    out["cx_global_smooth"] = np.round(sx, 3)
    out["cy_global_smooth"] = np.round(sy, 3)
    out["x_norm_smooth"] = np.round(xn, 6)
    out["y_norm_smooth"] = np.round(yn, 6)
    out["r_norm_smooth"] = np.round(rn, 6)
    out["dx_norm"] = np.round(dxn, 6)
    out["dy_norm"] = np.round(dyn, 6)
    out["step_distance_norm"] = np.round(step_norm, 6)
    out["step_distance_mm"] = np.round(step_mm, 6)
    out["speed_norm_s"] = np.round(speed_norm, 6)
    out["speed_mm_s"] = np.round(speed_mm, 6)
    out["acceleration_mm_s2"] = np.round(accel_mm, 6)
    out["turn_angle_rad"] = np.round(ta, 6)
    out["abs_turn_angle_rad"] = np.round(np.abs(ta), 6)
    out["angular_speed_rad_s"] = np.round(ang_speed, 6)
    out["radial_speed_norm_s"] = np.round(radial_speed, 6)
    out["area_change_px"] = np.round(area_change, 3)
    out["area_change_abs_px"] = np.round(np.abs(area_change), 3)
    out["area_change_rate_px_s"] = np.round(area_rate, 3)

    thr = args.immobility_speed_threshold_mm_s
    states, moving, immobile = [], [], []
    for k in range(len(out)):
        st = classify_movement(speed_mm[k], thr, bool(missing[k]),
                               bool(interpolated[k]))
        states.append(st)
        moving.append(True if st == "moving" else
                      (False if st == "immobile" else ""))
        immobile.append(True if st == "immobile" else
                        (False if st == "moving" else ""))
    out["is_moving"] = moving
    out["is_immobile"] = immobile
    out["movement_state"] = states
    out["missing"] = missing
    out["interpolated"] = interpolated
    return out


# --------------------------------------------------------------------------- #
# IO / orchestration
# --------------------------------------------------------------------------- #
def load_inputs(tracks_path: Path, petri_path: Optional[Path]):
    import pandas as pd

    tracks = pd.read_csv(tracks_path)
    if petri_path and petri_path.is_file():
        petri = pd.read_csv(petri_path)[
            ["video_id", "frame_idx", "dish_id",
             "petri_cx", "petri_cy", "petri_radius"]]
        tracks = tracks.merge(petri, on=["video_id", "frame_idx", "dish_id"],
                              how="left")
    else:
        # Fall back to reconstructing geometry from raw normalized coords.
        for col in ("petri_cx", "petri_cy", "petri_radius"):
            if col not in tracks.columns:
                tracks[col] = np.nan
    return tracks


def write_provenance(out_dir: Path, args) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_kinematics"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/compute_kinematics.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.tracks)],
        "output_path": str(args.out),
        "parameters": {
            "max_interp_gap_s": args.max_interp_gap_s,
            "smoothing_filter": args.smoothing_filter,
            "smoothing_window": args.smoothing_window,
            "savgol_polyorder": args.savgol_polyorder,
            "kalman_process_var": args.kalman_process_var,
            "kalman_measurement_var": args.kalman_measurement_var,
            "immobility_speed_threshold_mm_s":
                args.immobility_speed_threshold_mm_s,
            "petri_diameter_mm": args.petri_diameter_mm,
        },
    }
    with open(out_dir / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracks", required=True, type=Path,
                   help="larva_tracks.csv")
    p.add_argument("--petri-tracks", type=Path, default=None,
                   help="petri_tracks.csv (defaults to sibling of --tracks)")
    p.add_argument("--out", type=Path, default=None,
                   help="output larva_kinematics.csv (defaults to sibling)")
    p.add_argument("--max-interp-gap-s", type=float, default=0.5)
    p.add_argument("--smoothing-filter", choices=SMOOTHING_FILTERS,
                   default="savgol",
                   help="position smoothing filter (default: savgol, the "
                        "locked v1 policy in SPECIFICATION.md 26.3)")
    p.add_argument("--smoothing-window", type=int, default=7,
                   help="window length for savgol/moving_average filters")
    p.add_argument("--savgol-polyorder", type=int, default=2,
                   help="polynomial order for the savgol filter")
    p.add_argument("--kalman-process-var", type=float, default=1.0,
                   help="acceleration spectral density for the kalman filter")
    p.add_argument("--kalman-measurement-var", type=float, default=4.0,
                   help="position observation variance (px^2) for the kalman "
                        "filter")
    p.add_argument("--immobility-speed-threshold-mm-s", type=float, default=1.0)
    p.add_argument("--petri-diameter-mm", type=float, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    if args.petri_tracks is None:
        sibling = args.tracks.parent / "petri_tracks.csv"
        args.petri_tracks = sibling if sibling.is_file() else None
    if args.out is None:
        args.out = args.tracks.parent / "larva_kinematics.csv"

    if args.petri_diameter_mm is None:
        print("WARNING: --petri-diameter-mm not set; mm-based speeds and "
              "immobility classification will be NaN/uncertain.",
              file=sys.stderr)

    tracks = load_inputs(args.tracks, args.petri_tracks)
    groups = []
    for (vid, dish), grp in tracks.groupby(["video_id", "dish_id"], sort=True):
        groups.append(process_series(grp, args))
        print(f"  {vid} dish {dish}: {len(grp)} frames")

    result = pd.concat(groups, ignore_index=True) if groups else \
        pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = result.reindex(columns=OUTPUT_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    write_provenance(args.out.parent, args)
    print(f"\nWrote {len(result)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
