#!/usr/bin/env python3
"""Generate Petri-normalized occupancy heatmaps per video x dish (SPECIFICATION.md 12).

Heatmaps live in the canonical Petri coordinate system (center 0, edge radius 1)
on a fixed domain so they are comparable across videos, cameras and resolutions.
For each (video_id, dish_id) the following are written under --out:

    npy/<video>_dish<NN>_counts.npy        raw 2D histogram (scientific product)
    npy/<video>_dish<NN>_probability.npy   normalized, circular-masked
    metadata/<video>_dish<NN>_metadata.json
    png/<video>_dish<NN>_heatmap.png       visualization only
    png/<video>_dish<NN>_trajectory.png    visualization only

The .npy arrays are the reusable data; PNGs are for QC only (SPECIFICATION.md 12).
Heatmap-derived video features (SPECIFICATION.md 11.2) are recorded in the metadata so
the feature step can reuse them without recomputation.
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

DOMAIN = (-1.0, 1.0, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Pure functions (testable without matplotlib)
# --------------------------------------------------------------------------- #
def bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def histogram_norm(x: np.ndarray, y: np.ndarray, bins: int,
                   domain=DOMAIN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of normalized points. counts[i, j] indexes (x_bin, y_bin)."""
    xe = np.linspace(domain[0], domain[1], bins + 1)
    ye = np.linspace(domain[2], domain[3], bins + 1)
    counts, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    return counts, xe, ye


def circular_mask(xe: np.ndarray, ye: np.ndarray) -> np.ndarray:
    """Boolean mask (x_bin, y_bin) of cells whose center lies within radius 1."""
    xc, yc = bin_centers(xe), bin_centers(ye)
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    return np.hypot(gx, gy) <= 1.0


def to_probability(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def heatmap_stats(prob: np.ndarray, xe: np.ndarray, ye: np.ndarray) -> dict:
    """Heatmap-derived features (SPECIFICATION.md 11.2). Entropy is in nats."""
    xc, yc = bin_centers(xe), bin_centers(ye)
    total = prob.sum()
    nan = float("nan")
    if total <= 0:
        return {"heatmap_entropy": nan, "heatmap_peak_density": nan,
                "heatmap_peak_x_norm": nan, "heatmap_peak_y_norm": nan,
                "heatmap_spread_x": nan, "heatmap_spread_y": nan}
    p = prob / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    pi, pj = np.unravel_index(np.argmax(p), p.shape)
    mean_x = float((p * xc[:, None]).sum())
    mean_y = float((p * yc[None, :]).sum())
    spread_x = float(math.sqrt((p * (xc[:, None] - mean_x) ** 2).sum()))
    spread_y = float(math.sqrt((p * (yc[None, :] - mean_y) ** 2).sum()))
    return {
        "heatmap_entropy": entropy,
        "heatmap_peak_density": float(p.max()),
        "heatmap_peak_x_norm": float(xc[pi]),
        "heatmap_peak_y_norm": float(yc[pj]),
        "heatmap_spread_x": spread_x,
        "heatmap_spread_y": spread_y,
    }


def select_xy(df, prefer_smooth: bool):
    """Pick normalized coordinate columns, preferring smoothed when available."""
    cols = df.columns
    if prefer_smooth and "x_norm_smooth" in cols and "y_norm_smooth" in cols:
        return "x_norm_smooth", "y_norm_smooth"
    if "x_norm_raw" in cols and "y_norm_raw" in cols:
        return "x_norm_raw", "y_norm_raw"
    raise KeyError("No normalized coordinate columns (x_norm_smooth/raw) found")


# --------------------------------------------------------------------------- #
# PNG rendering (visualization only; degrades gracefully)
# --------------------------------------------------------------------------- #
def render_pngs(prob: np.ndarray, x: np.ndarray, y: np.ndarray,
                heatmap_path: Path, traj_path: Path, bins: int) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"  (matplotlib unavailable, skipping PNGs: {exc})",
              file=sys.stderr)
        return False

    circle = plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(prob.T, origin="lower", extent=DOMAIN, cmap="inferno")
    ax.add_patch(circle)
    ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title("occupancy probability")
    fig.savefig(heatmap_path, dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="k", lw=1))
    ax.plot(x, y, lw=0.6, color="tab:blue", alpha=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title("trajectory (normalized)")
    fig.savefig(traj_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def process_group(df, video_id, dish_id, args, dirs) -> dict:
    import pandas as pd

    xcol, ycol = select_xy(df, prefer_smooth=not args.use_raw)
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy()
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    counts, xe, ye = histogram_norm(x, y, args.bins)
    counts *= circular_mask(xe, ye)  # zero cells outside the dish
    prob = to_probability(counts)
    stats = heatmap_stats(prob, xe, ye)

    stem = f"{video_id}_dish{int(dish_id):02d}"
    np.save(dirs["npy"] / f"{stem}_counts.npy", counts)
    np.save(dirs["npy"] / f"{stem}_probability.npy", prob)

    inside = int((np.hypot(x, y) <= 1.0).sum())
    metadata = {
        "video_id": video_id, "dish_id": int(dish_id), "bins": args.bins,
        "domain": list(DOMAIN), "valid_points": int(len(x)),
        "points_inside_petri": inside,
        "coordinate_source": xcol,
        "normalization": "petri_radius", "circular_mask_applied": True,
        "entropy_units": "nats",
        **stats,
    }
    (dirs["metadata"] / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2))

    if not args.no_png:
        render_pngs(prob, x, y,
                    dirs["png"] / f"{stem}_heatmap.png",
                    dirs["png"] / f"{stem}_trajectory.png", args.bins)
    return metadata


def write_provenance(out_dir: Path, args, n_groups: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_heatmaps"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/generate_heatmaps.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.input)],
        "output_path": str(args.out),
        "n_heatmaps": n_groups,
        "parameters": {"bins": args.bins, "domain": list(DOMAIN),
                       "use_raw": args.use_raw},
    }
    with open(out_dir / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="larva_kinematics.csv (or larva_tracks.csv with --use-raw)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--use-raw", action="store_true",
                   help="Use x_norm_raw/y_norm_raw instead of smoothed coords.")
    p.add_argument("--no-png", action="store_true",
                   help="Skip PNG rendering (npy arrays are the scientific data).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    dirs = {name: args.out / name for name in ("npy", "png", "metadata")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "video_id" not in df.columns or "dish_id" not in df.columns:
        print("Input must contain video_id and dish_id columns",
              file=sys.stderr)
        return 1

    n = 0
    for (vid, dish), grp in df.groupby(["video_id", "dish_id"], sort=True):
        meta = process_group(grp, vid, dish, args, dirs)
        n += 1
        print(f"  {vid} dish {dish}: {meta['valid_points']} points "
              f"({meta['points_inside_petri']} inside)")

    write_provenance(args.out, args, n)
    print(f"\nWrote {n} heatmap set(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
