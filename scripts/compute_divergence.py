#!/usr/bin/env python3
"""Divergence-from-control post-processor (SPECIFICATION.md 11.2, 26.4).

v1 writes the two divergence columns of features_per_video.csv as NaN because a
control reference is not guaranteed at feature time. This deferred step fills
them in once the whole dataset (heatmaps + experiment metadata) is available.

Reference (SPECIFICATION.md 26.4 / v1.1 default):

    pool all treatment="control" dishes within the same experiment_id into one
    probability heatmap, then compare every per-dish heatmap against it.

For each (video_id, dish_id) two divergences from the experiment's pooled
control reference are computed:

    js_divergence_from_control          Jensen-Shannon divergence of the 2D
                                        occupancy heatmap (natural log -> nats,
                                        bounded [0, ln 2]).
    wasserstein_distance_from_control   Wasserstein distance of the radial
                                        occupancy marginal (petri radii); a
                                        well-defined 1D earth-mover distance that
                                        captures center<->edge occupancy shifts
                                        under the circular Petri geometry.

Outputs:
    features/features_per_video.csv     updated in place (two columns filled)
    features/divergence_from_control.csv audit table (one row per video x dish)

Reusing the saved heatmap *_counts.npy arrays keeps this consistent with the
heatmaps the feature step already produced; nothing is recomputed from raw
coordinates. Interpretation stays behavioral-proxy only (SPECIFICATION.md 23): a larger
divergence is a *trajectory/occupancy deviation from the control distribution*,
not a direct measure of welfare.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp
import generate_heatmaps as gh

DIVERGENCE_KEYS = ["js_divergence_from_control",
                   "wasserstein_distance_from_control"]

# Audit table written alongside features_per_video.csv.
DETAIL_COLUMNS = [
    "dataset_id", "experiment_id", "video_id", "dish_id", "treatment",
    "n_control_dishes_ref", "reference_valid_points",
    "js_divergence_from_control", "wasserstein_distance_from_control",
    "divergence_status",
]

# divergence_status controlled vocabulary.
STATUS_OK = "ok"
STATUS_NO_METADATA = "no_metadata"
STATUS_NO_CONTROL = "no_control_in_experiment"
STATUS_NO_HEATMAP = "no_heatmap"
STATUS_EMPTY = "empty_heatmap"
STATUS_SELF_ONLY = "self_only_control"
STATUS_VOCAB = "|".join([STATUS_OK, STATUS_NO_METADATA, STATUS_NO_CONTROL,
                         STATUS_NO_HEATMAP, STATUS_EMPTY, STATUS_SELF_ONLY])

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Pure functions (testable without pandas)
# --------------------------------------------------------------------------- #
def heatmap_stem(video_id: str, dish_id) -> str:
    """Match the stem written by generate_heatmaps.process_group."""
    return f"{video_id}_dish{int(dish_id):02d}"


def load_counts(heatmaps_dir: Path, video_id: str, dish_id) -> Optional[np.ndarray]:
    """Load a dish's raw count heatmap, or None when it is absent."""
    path = heatmaps_dir / "npy" / f"{heatmap_stem(video_id, dish_id)}_counts.npy"
    if not path.is_file():
        return None
    return np.load(path).astype(float)


def pool_counts(arrays: list[np.ndarray]) -> Optional[np.ndarray]:
    """Sum a list of count grids into a pooled reference (None if empty)."""
    valid = [a for a in arrays if a is not None]
    if not valid:
        return None
    ref = np.zeros_like(valid[0], dtype=float)
    for a in valid:
        if a.shape != ref.shape:
            raise ValueError(
                f"heatmap shape mismatch: {a.shape} vs {ref.shape}; "
                "regenerate heatmaps with a single --bins value")
        ref += a
    return ref


def normalize(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def js_divergence(p: np.ndarray, q: np.ndarray, base: str = "e") -> float:
    """Jensen-Shannon divergence of two grids; nats (base e) or bits (base 2).

    Bounded [0, ln 2] (nats). NaN if either input has no mass.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return NAN
    p, q = p / sp, q / sq
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    d = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if base == "2":
        d /= np.log(2.0)
    return d


def radial_grid(bins: int) -> np.ndarray:
    """Flattened radial distance of every bin center on the Petri domain."""
    xe = np.linspace(gh.DOMAIN[0], gh.DOMAIN[1], bins + 1)
    ye = np.linspace(gh.DOMAIN[2], gh.DOMAIN[3], bins + 1)
    xc, yc = gh.bin_centers(xe), gh.bin_centers(ye)
    gx, gy = np.meshgrid(xc, yc, indexing="ij")
    return np.hypot(gx, gy).ravel()


def wasserstein_radial(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein distance between the radial marginals of two grids.

    Both grids share the same bin geometry, so the bin-center radii are the
    common support and the grid masses are the weights. Units: petri radii.
    """
    from scipy.stats import wasserstein_distance

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return NAN
    r = radial_grid(p.shape[0])
    return float(wasserstein_distance(r, r, p.ravel(), q.ravel()))


def score_dish(dish_counts: Optional[np.ndarray],
               reference_counts: Optional[np.ndarray],
               base: str = "e") -> tuple[float, float, str]:
    """Return (js, wasserstein, status) for one dish vs a pooled reference."""
    if reference_counts is None or reference_counts.sum() <= 0:
        return NAN, NAN, STATUS_NO_CONTROL
    if dish_counts is None:
        return NAN, NAN, STATUS_NO_HEATMAP
    if dish_counts.sum() <= 0:
        return NAN, NAN, STATUS_EMPTY
    p = normalize(dish_counts)
    q = normalize(reference_counts)
    return js_divergence(p, q, base), wasserstein_radial(p, q), STATUS_OK


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_treatment_lookup(metadata_path: Optional[Path]):
    """Map (video_id, str(dish_id)) -> (treatment, experiment_id) from metadata."""
    import pandas as pd

    if not (metadata_path and Path(metadata_path).is_file()):
        return None
    meta = pd.read_csv(metadata_path)
    need = {"video_id", "dish_id", "treatment"}
    if not need.issubset(meta.columns):
        print(f"WARNING: {metadata_path} lacks {need - set(meta.columns)}; "
              "treatments unknown.", file=sys.stderr)
        return None
    lookup = {}
    for _, r in meta.iterrows():
        exp = str(r["experiment_id"]) if "experiment_id" in meta.columns else ""
        lookup[_key(r["video_id"], r["dish_id"])] = (
            str(r["treatment"]).strip().lower(), exp)
    return lookup


def _key(video_id, dish_id) -> tuple[str, str]:
    try:
        return (str(video_id), str(int(dish_id)))
    except (ValueError, TypeError):
        return (str(video_id), str(dish_id))


def compute_references(feat, lookup, heatmaps_dir: Path, control: str):
    """Pool control-dish counts per experiment_id.

    Returns (references, control_keys, by_exp) where references maps
    experiment_id -> pooled count grid (or None), control_keys is the set of
    control dish (video_id, dish_id) keys (for leave-one-out scoring), and
    by_exp keeps the per-experiment (key, counts) lists.
    """
    by_exp: dict[str, list[tuple[tuple[str, str], np.ndarray]]] = {}
    control_keys: set[tuple[str, str]] = set()
    for _, row in feat.iterrows():
        key = _key(row["video_id"], row["dish_id"])
        treatment, exp = (lookup or {}).get(key, ("", ""))
        if not exp:
            exp = str(row.get("experiment_id", "") or "")
        if treatment == control:
            counts = load_counts(heatmaps_dir, row["video_id"], row["dish_id"])
            if counts is not None and counts.sum() > 0:
                by_exp.setdefault(exp, []).append((key, counts))
                control_keys.add(key)
    references = {exp: pool_counts([c for _, c in items])
                  for exp, items in by_exp.items()}
    return references, control_keys, by_exp


def run(feat, lookup, heatmaps_dir: Path, control: str, base: str,
        exclude_self: bool):
    """Compute divergences for every feature row; returns a list of detail dicts."""
    _references, control_keys, by_exp = compute_references(
        feat, lookup, heatmaps_dir, control)

    detail = []
    for _, row in feat.iterrows():
        key = _key(row["video_id"], row["dish_id"])
        treatment, exp = (lookup or {}).get(key, (None, ""))
        if not exp:
            exp = str(row.get("experiment_id", "") or "")
        dish_counts = load_counts(heatmaps_dir, row["video_id"], row["dish_id"])

        if lookup is None or treatment is None:
            js, wass, status = NAN, NAN, STATUS_NO_METADATA
            ref_n, ref_pts = 0, 0
        else:
            ref_items = by_exp.get(exp, [])
            if exclude_self and key in control_keys:
                ref_items = [(k, c) for k, c in ref_items if k != key]
            ref_counts = pool_counts([c for _, c in ref_items])
            ref_n = len(ref_items)
            ref_pts = int(ref_counts.sum()) if ref_counts is not None else 0
            js, wass, status = score_dish(dish_counts, ref_counts, base)
            if status == STATUS_NO_CONTROL and exclude_self \
                    and key in control_keys and not ref_items:
                status = STATUS_SELF_ONLY

        detail.append({
            "dataset_id": str(row.get("dataset_id", "") or ""),
            "experiment_id": exp,
            "video_id": str(row["video_id"]),
            "dish_id": row["dish_id"],
            "treatment": "" if treatment is None else treatment,
            "n_control_dishes_ref": ref_n,
            "reference_valid_points": ref_pts,
            "js_divergence_from_control": js,
            "wasserstein_distance_from_control": wass,
            "divergence_status": status,
        })
    return detail


def write_provenance(out_dir: Path, args, n_rows: int, n_scored: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_divergence"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/compute_divergence.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.features), str(args.heatmaps),
                        str(args.metadata) if args.metadata else None],
        "output_path": str(args.features),
        "n_rows": n_rows, "n_scored": n_scored,
        "parameters": {
            "control_treatment": args.control_treatment,
            "js_base": args.js_base,
            "wasserstein_metric": "radial_marginal",
            "exclude_self": args.exclude_self,
            "reference_policy": "pooled_control_dishes_per_experiment_id",
        },
    }
    with open(out_dir / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path,
                   help="features_per_video.csv (updated in place)")
    p.add_argument("--heatmaps", required=True, type=Path,
                   help="heatmaps/ directory containing npy/<stem>_counts.npy")
    p.add_argument("--metadata", type=Path, default=None,
                   help="experiment_metadata.csv (treatment + experiment_id)")
    p.add_argument("--control-treatment", default="control",
                   help="treatment value pooled into the reference")
    p.add_argument("--js-base", choices=["e", "2"], default="e",
                   help="log base for JS divergence (e=nats, 2=bits)")
    p.add_argument("--exclude-self", action="store_true",
                   help="leave-one-out: drop a control dish from its own "
                        "reference (removes self-similarity bias)")
    p.add_argument("--detail-out", type=Path, default=None,
                   help="audit CSV path (default: <features dir>/"
                        "divergence_from_control.csv)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    if not args.features.is_file():
        print(f"features file not found: {args.features}", file=sys.stderr)
        return 1
    feat = pd.read_csv(args.features)
    for col in ("video_id", "dish_id"):
        if col not in feat.columns:
            print(f"{args.features} missing required column '{col}'",
                  file=sys.stderr)
            return 1

    lookup = build_treatment_lookup(args.metadata)
    if lookup is None:
        print("No usable experiment metadata; divergences will be NaN "
              "(status=no_metadata).", file=sys.stderr)

    detail = run(feat, lookup, args.heatmaps, args.control_treatment,
                 args.js_base, args.exclude_self)

    # Write the two columns back into features_per_video.csv by (video, dish).
    by_key = {_key(d["video_id"], d["dish_id"]): d for d in detail}
    for col in DIVERGENCE_KEYS:
        feat[col] = [round(by_key[_key(v, dd)][col], 6)
                     if np.isfinite(by_key[_key(v, dd)][col]) else NAN
                     for v, dd in zip(feat["video_id"], feat["dish_id"])]
    feat.to_csv(args.features, index=False)

    detail_out = args.detail_out or (args.features.parent /
                                     "divergence_from_control.csv")
    det_df = pd.DataFrame(detail).reindex(columns=DETAIL_COLUMNS)
    for col in DIVERGENCE_KEYS:
        det_df[col] = det_df[col].round(6)
    det_df.to_csv(detail_out, index=False)

    write_provenance(args.features.parent, args, len(feat),
                     int((det_df["divergence_status"] == STATUS_OK).sum()))

    n_ok = int((det_df["divergence_status"] == STATUS_OK).sum())
    statuses = det_df["divergence_status"].value_counts().to_dict()
    print(f"Scored {n_ok}/{len(det_df)} dish(es) against a control reference.")
    print(f"  status breakdown: {statuses}")
    print(f"  updated: {args.features}")
    print(f"  audit:   {detail_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
