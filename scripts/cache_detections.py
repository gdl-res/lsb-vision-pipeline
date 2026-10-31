#!/usr/bin/env python3
"""Cache raw larva detections once, so several trackers can be replayed on them.

This is the shared detector pass for offline replay
(config/replay.yaml). It exists because the production tables cannot
be reused: `track_larvae_video.py` runs the segmenter through `model.track()`,
and ultralytics' `on_predict_postprocess_end` replaces the results object with
only the rows the tracker emitted --

    idx = tracks[:, -1].astype(int)
    predictor.results[i] = result[idx]

-- so `larva_detections_all.csv` is already ByteTrack-filtered. Replaying other
trackers on it would hand ByteTrack a private advantage. This script therefore
re-runs the segmenter in `predict()` mode (no association whatsoever) and writes
every detection it produces, once, for all configurations to share.

Two things make the cache faithful to production rather than merely similar:

  * Arena geometry is NOT re-detected. Under the locked --detect-petri-first-frame
    policy the crop window is frozen at frame 0, and the production run already
    recorded it for every frame in `_work/<video>/petri_tracks.csv`. Reading it
    back makes the crop windows identical to production by construction, and
    means the petri model is not needed at all.
  * Detections are cached down to `detection.cache_conf` (0.10), which is the
    lowest threshold any configured tracker consults (bytetrack's
    track_low_thresh). Caching below every tracker's floor means no configuration
    is starved of detections it would otherwise have seen; the replay then
    applies one common threshold to all of them.

Per frame x dish it also records image statistics (contrast, intensity) that only
exist while the frame is in hand -- these become the tracker-independent
"low contrast" failure stratum downstream.

Outputs, per video, under --out:
    <video_id>.detections.csv.gz   one row per raw detection (+ morphology)
    <video_id>.frames.csv.gz       one row per frame x dish (incl. empty ones)
    <video_id>._cache_done.json    sentinel for --skip-existing

Segmenter swap (config/segmenters.yaml): with --segmenter NAME
the model comes from that config's `segmenters:` entry through
scripts/segmenters.py, the output goes to the entry's cache_dir, and the video
list defaults to the config's pinned `videos:` (all videos when none are
pinned). Detection thresholds are inherited from the config's `base_config`, so
every segmenter is cached under identical settings. --regression-check proves the
segmenter path reproduces an existing reference cache byte-for-byte;
--runtime-guard times a prefix of the first video against the config's
`cache_runtime:` budget before a long pass.
"""
from __future__ import annotations

import argparse
import gzip
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp
import segmenters as sg
import track_larvae_video as tl

# --------------------------------------------------------------------------- #
# Schemas (stable: the replay depends on these column names)
# --------------------------------------------------------------------------- #
MORPH_KEYS = [
    "area_px", "perimeter_px", "compactness", "solidity", "extent",
    "major_axis_px", "minor_axis_px", "aspect_ratio", "orientation_rad",
    "eccentricity", "bbox_w", "bbox_h", "bbox_area_px", "bbox_area_ratio",
]

DETECTION_COLUMNS = [
    "video_id", "frame_idx", "dish_id", "det_id",
    "class_id", "conf",
    "x1_crop", "y1_crop", "x2_crop", "y2_crop",
    # centroid + area follow LarvaDetection: polygon-derived when a mask exists,
    # box-derived otherwise. Cached so the replay never needs the polygon.
    "cx_crop", "cy_crop", "area_px", "has_polygon",
] + [f"mask_{k}" for k in MORPH_KEYS]

FRAME_COLUMNS = [
    "video_id", "frame_idx", "dish_id", "time_s", "n_detections",
    "crop_contrast", "crop_mean_intensity", "crop_std_intensity",
]

GEOMETRY_COLUMNS = [
    "dish_id", "row", "col", "petri_x1", "petri_y1", "petri_x2", "petri_y2",
    "petri_conf", "petri_valid", "crop_x0", "crop_y0", "crop_size",
    "petri_geometry_source", "petri_fit_cx", "petri_fit_cy",
    "petri_fit_a", "petri_fit_b", "petri_fit_valid",
]


# --------------------------------------------------------------------------- #
# Geometry: read back what production froze at frame 0
# --------------------------------------------------------------------------- #
def load_frozen_geometry(petri_tracks: Path) -> list[tl.DishGeom]:
    """Rebuild the production DishGeom list from a video's petri_tracks.csv.

    Only frame 0 is read: under --detect-petri-first-frame the geometry is
    identical on every row, and re-deriving it here would risk drifting from what
    the tracked dataset actually used.
    """
    import pandas as pd

    df = pd.read_csv(petri_tracks, nrows=4096)
    first = df["frame_idx"].min()
    df = df[df["frame_idx"] == first].sort_values("dish_id")
    dishes: list[tl.DishGeom] = []
    for _, r in df.iterrows():
        box = cp.PetriBox(float(r["petri_x1"]), float(r["petri_y1"]),
                          float(r["petri_x2"]), float(r["petri_y2"]),
                          float(r["petri_conf"]))
        geom = cp.CropGeom(int(r["crop_x0"]), int(r["crop_y0"]),
                           int(r["crop_size"]))
        fit = None
        if str(r.get("petri_fit_valid", "")).lower() == "true":
            fit = cp.DishFit(float(r["petri_fit_cx"]), float(r["petri_fit_cy"]),
                             float(r["petri_fit_a"]), float(r["petri_fit_b"]),
                             True)
        dishes.append(tl.DishGeom(int(r["dish_id"]), int(r["row"]),
                                  int(r["col"]), box, geom, fit))
    return dishes


def geometry_rows(dishes: list[tl.DishGeom]) -> list[dict]:
    """Serializable geometry, so the replay does not re-read the dataset."""
    rows = []
    for dg in dishes:
        rows.append({
            "dish_id": dg.dish_id, "row": dg.row, "col": dg.col,
            "petri_x1": dg.box.x1, "petri_y1": dg.box.y1,
            "petri_x2": dg.box.x2, "petri_y2": dg.box.y2,
            "petri_conf": dg.box.conf, "petri_valid": True,
            "crop_x0": dg.geom.x0, "crop_y0": dg.geom.y0,
            "crop_size": dg.geom.size,
            "petri_geometry_source": "circlefit" if (
                dg.fit is not None and dg.fit.valid) else "box",
            "petri_fit_cx": "" if dg.fit is None else dg.fit.cx,
            "petri_fit_cy": "" if dg.fit is None else dg.fit.cy,
            "petri_fit_a": "" if dg.fit is None else dg.fit.a,
            "petri_fit_b": "" if dg.fit is None else dg.fit.b,
            "petri_fit_valid": bool(dg.fit is not None and dg.fit.valid),
        })
    return rows


# --------------------------------------------------------------------------- #
# Image statistics for the low-contrast stratum
# --------------------------------------------------------------------------- #
def dish_mask(size: int, cx: float, cy: float, radius: float,
              step: int) -> np.ndarray:
    """Boolean mask of the dish interior on a `step`-strided crop grid."""
    ys = np.arange(0, size, step, dtype=float)
    xs = np.arange(0, size, step, dtype=float)
    dy = ys[:, None] - cy
    dx = xs[None, :] - cx
    return (dx * dx + dy * dy) <= radius * radius


def crop_statistics(crop: np.ndarray, mask: np.ndarray,
                    step: int) -> tuple[float, float, float]:
    """(contrast, mean, std) of dish-interior intensity.

    Contrast is p95-p05 of the grayscale interior: robust to the specular rim
    highlight and to a few saturated pixels, unlike max-min. Computed on a
    strided view rather than the full crop -- this runs 4.5M times, and the
    stratum only needs to rank frames, not measure them precisely.
    """
    import cv2

    sub = crop[::step, ::step]
    if sub.ndim == 3:
        sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    h = min(sub.shape[0], mask.shape[0])
    w = min(sub.shape[1], mask.shape[1])
    vals = sub[:h, :w][mask[:h, :w]]
    if vals.size < 16:
        nan = float("nan")
        return nan, nan, nan
    vals = vals.astype(np.float32)
    p05, p95 = np.percentile(vals, (5.0, 95.0))
    return float(p95 - p05), float(vals.mean()), float(vals.std())


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detections_from_results(results) -> list[tl.LarvaDetection]:
    """Reuse the production parser so cached boxes/centroids/areas match it."""
    return tl._parse_larva_results(results)


def cache_video(model: "sg.Segmenter", video: Path, dishes: list[tl.DishGeom],
                out_dir: Path, args) -> dict:
    """Run the segmenter over one video and write its cache. Returns a summary."""
    import cv2

    video_id = video.stem
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    masks = {}
    for dg in dishes:
        step = max(1, dg.geom.size // 128)
        ecx, ecy, erad = tl.effective_geometry(dg)
        masks[dg.dish_id] = (
            step,
            dish_mask(dg.geom.size, ecx - dg.geom.x0, ecy - dg.geom.y0,
                      erad, step))

    det_path = out_dir / f"{video_id}.detections.csv.gz"
    frm_path = out_dir / f"{video_id}.frames.csv.gz"
    t0 = time.time()
    n_frames = 0
    n_dets = 0

    with gzip.open(det_path, "wt", newline="") as dfh, \
            gzip.open(frm_path, "wt", newline="") as ffh:
        dw = csv.DictWriter(dfh, fieldnames=DETECTION_COLUMNS)
        fw = csv.DictWriter(ffh, fieldnames=FRAME_COLUMNS)
        dw.writeheader()
        fw.writeheader()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and n_frames >= args.max_frames:
                break
            time_s = n_frames / fps
            crops = [cp.extract_square_crop(frame, dg.geom) for dg in dishes]
            per_dish = model.detect_batch(crops)
            for dg, crop, dets in zip(dishes, crops, per_dish):
                step, mask = masks[dg.dish_id]
                contrast, mean_i, std_i = crop_statistics(crop, mask, step)
                fw.writerow({
                    "video_id": video_id, "frame_idx": n_frames,
                    "dish_id": dg.dish_id, "time_s": round(time_s, 4),
                    "n_detections": len(dets),
                    "crop_contrast": round(contrast, 3),
                    "crop_mean_intensity": round(mean_i, 3),
                    "crop_std_intensity": round(std_i, 3)})
                for j, d in enumerate(dets):
                    morph = (tl.polygon_morphology(d.polygon)
                             if d.polygon is not None else None)
                    row = {
                        "video_id": video_id, "frame_idx": n_frames,
                        "dish_id": dg.dish_id, "det_id": j,
                        "class_id": tl.LARVA_CLASS_ID,
                        "conf": round(d.conf, 6),
                        "x1_crop": round(d.x1, 3), "y1_crop": round(d.y1, 3),
                        "x2_crop": round(d.x2, 3), "y2_crop": round(d.y2, 3),
                        "cx_crop": round(d.cx, 3), "cy_crop": round(d.cy, 3),
                        "area_px": round(d.area_px, 3),
                        "has_polygon": d.polygon is not None}
                    for k in MORPH_KEYS:
                        row[f"mask_{k}"] = ("" if morph is None
                                            else round(morph[k], 6))
                    dw.writerow(row)
                    n_dets += 1
            n_frames += 1
            if n_frames % 500 == 0:
                print(f"  {video_id}: {n_frames} frames, {n_dets} detections",
                      flush=True)
    cap.release()

    summary = {"video_id": video_id, "n_frames": n_frames,
               "n_dishes": len(dishes), "n_detections": n_dets,
               "fps": fps, "wall_s": round(time.time() - t0, 1),
               "cache_conf": args.cache_conf, "imgsz": args.imgsz,
               "geometry": geometry_rows(dishes)}
    if args.segmenter is not None:
        summary.update(segmenter=args.segmenter, backend=model.backend,
                       weights=str(model.weights),
                       weights_sha256=args.weights_sha256)
    (out_dir / f"{video_id}._cache_done.json").write_text(
        json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_provenance(out: Path, args, summaries: list[dict]) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_cache_detections"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/cache_detections.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.videos), str(args.dataset)],
        "output_path": str(out),
        "larva_model": str(args.larva_model),
        "larva_model_sha256": args.weights_sha256,
        "segmenter": args.segmenter,
        "segmenter_backend": (args.seg_spec or {}).get("backend",
                                                       "ultralytics"),
        "n_videos": len(summaries),
        "n_frames_total": sum(s["n_frames"] for s in summaries),
        "n_detections_total": sum(s["n_detections"] for s in summaries),
        "parameters": {
            "cache_conf": args.cache_conf, "iou": args.iou,
            "imgsz": args.imgsz, "device": args.device,
            "geometry_source": "frozen_from_dataset_petri_tracks",
        },
    }
    with open(out / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("config/replay.yaml"))
    p.add_argument("--videos", type=Path, default=None)
    p.add_argument("--dataset", type=Path, default=None,
                   help="production dataset supplying frozen arena geometry")
    p.add_argument("--larva-model", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cache-conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap frames per video (smoke tests)")
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated video ids to cache")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip videos with a _cache_done.json sentinel")
    p.add_argument("--segmenter", type=str, default=None,
                   help="segmenter name from a segmenter config")
    p.add_argument("--regression-check", action="store_true",
                   help="re-cache the first --regression-frames of the first "
                        "pinned video and require byte-identical rows to the "
                        "existing cache, then exit")
    p.add_argument("--regression-frames", type=int, default=300)
    p.add_argument("--runtime-guard", action="store_true",
                   help="time --guard-frames first; if the projected wall "
                        "time exceeds the config budget, cut the video subset "
                        "(written to <out>/video_subset.json for every "
                        "segmenter)")
    p.add_argument("--guard-frames", type=int, default=2000)
    return p


def resolve_args(args) -> argparse.Namespace:
    """CLI wins; anything unset falls back to the experiment config.

    A segmenter-comparison config (one with `segmenters:`) inherits detection
    thresholds from its `base_config`; the model, imgsz and cache_dir come from
    the `--segmenter` entry, and the video list defaults to its pinned subset.
    """
    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    args.seg_spec = None
    args.seg_cfg = None
    if "segmenters" in cfg:
        if args.segmenter is None:
            raise SystemExit("--segmenter is required with a segmenter config")
        seg_cfg = cfg
        spec = seg_cfg["segmenters"][args.segmenter]
        base = yaml.safe_load(Path(seg_cfg["base_config"]).read_text())
        cfg = {"paths": {"videos": seg_cfg["paths"]["videos"],
                         "dataset": seg_cfg["paths"]["production_dataset"],
                         "larva_model": spec["weights"],
                         "cache": spec["cache_dir"]},
               "detection": {**base["detection"], "imgsz": spec["imgsz"]}}
        args.seg_spec, args.seg_cfg = spec, seg_cfg
        if args.only is None:
            subset = active_video_subset(seg_cfg)
            if subset:
                args.only = ",".join(subset)
    paths, det = cfg["paths"], cfg["detection"]
    if args.videos is None:
        args.videos = Path(paths["videos"])
    if args.dataset is None:
        args.dataset = Path(paths["dataset"])
    if args.larva_model is None:
        args.larva_model = Path(paths["larva_model"])
    if args.out is None:
        args.out = Path(paths["cache"])
    if args.cache_conf is None:
        args.cache_conf = float(det["cache_conf"])
    if args.iou is None:
        args.iou = float(det["iou"])
    if args.imgsz is None:
        args.imgsz = int(det["imgsz"])
    args.weights_sha256 = cp.sha256_file(args.larva_model)
    return args


def active_video_subset(seg_cfg: dict) -> list[str]:
    """Pinned videos, or the runtime-guard's cut subset once it exists."""
    f = Path(seg_cfg["paths"]["out"]) / "video_subset.json"
    if f.is_file():
        return list(json.loads(f.read_text())["videos"])
    return list(seg_cfg.get("videos") or [])


def load_detector(args) -> "sg.Segmenter":
    """The configured segmenter, or the legacy single-model ultralytics path."""
    spec = args.seg_spec or {"backend": "ultralytics",
                             "imgsz": args.imgsz}
    return sg.load_segmenter(args.segmenter or "production", spec,
                             conf=args.cache_conf, iou=args.iou,
                             device=args.device, weights=args.larva_model,
                             imgsz=args.imgsz)


def compare_cache_prefix(new_dir: Path, ref_dir: Path, video_id: str,
                         n_frames: int) -> list[str]:
    """Differences between a fresh cache and the reference, over the frames
    the fresh one covers. Empty list == byte-identical rows."""
    problems = []
    for kind in ("detections", "frames"):
        def rows(d):
            with gzip.open(d / f"{video_id}.{kind}.csv.gz", "rt") as fh:
                r = list(csv.reader(fh))
            head, body = r[0], r[1:]
            fi = head.index("frame_idx")
            return head, [x for x in body if int(x[fi]) < n_frames]
        h_new, b_new = rows(new_dir)
        h_ref, b_ref = rows(ref_dir)
        if h_new != h_ref:
            problems.append(f"{kind}: header differs")
            continue
        if len(b_new) != len(b_ref):
            problems.append(f"{kind}: {len(b_new)} rows vs {len(b_ref)} "
                            f"in reference")
        for i, (a, b) in enumerate(zip(b_new, b_ref)):
            if a != b:
                problems.append(f"{kind}: first difference at row {i}: "
                                f"{a} != {b}")
                break
    return problems


def run_regression_check(args) -> int:
    """The segmenter layer reproduces an existing reference cache."""
    import tempfile
    if not args.only:
        print("--regression-check needs --only VIDEO_ID", file=sys.stderr)
        return 2
    vid = args.only.split(",")[0]
    video = next(v for v in discover_videos(args) if v.stem == vid)
    ref_dir = args.out
    if not (ref_dir / f"{vid}._cache_done.json").is_file():
        print(f"no reference cache for {vid} in {ref_dir}", file=sys.stderr)
        return 1
    seg_out = Path(args.seg_cfg["paths"]["out"]) if args.seg_cfg else ref_dir
    tmp = Path(tempfile.mkdtemp(prefix="regression_", dir=seg_out))
    args.max_frames = args.regression_frames
    dishes = load_frozen_geometry(
        args.dataset / "_work" / vid / "petri_tracks.csv")
    cache_video(load_detector(args), video, dishes, tmp, args)
    problems = compare_cache_prefix(tmp, ref_dir, vid, args.regression_frames)
    report = {"video_id": vid, "frames": args.regression_frames,
              "reference": str(ref_dir), "fresh": str(tmp),
              "identical": not problems, "problems": problems}
    (seg_out / "cache_regression.json").write_text(json.dumps(report, indent=2))
    if problems:
        print("REGRESSION: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"regression OK: {vid}, first {args.regression_frames} frames "
          f"byte-identical to {ref_dir}")
    return 0


def apply_runtime_guard(args, model, videos: list[Path]) -> list[Path]:
    """Time a prefix; cut the subset if over the configured hour budget.

    The cut subset is written to <experiment out>/video_subset.json, which
    `active_video_subset` (cache) and the replay/analysis read, so every
    segmenter is compared on the same videos.
    """
    import tempfile
    seg_cfg = args.seg_cfg
    if not seg_cfg or "cache_runtime" not in seg_cfg:
        raise SystemExit("--runtime-guard needs a segmenter config with a "
                         "cache_runtime: section")
    out = Path(seg_cfg["paths"]["out"])
    budget = float(seg_cfg["cache_runtime"]["max_projected_hours_maskrcnn"])
    guard_file = out / "runtime_guard.json"
    if guard_file.is_file():
        decided = json.loads(guard_file.read_text())
        keep = set(decided["videos"])
        return [v for v in videos if v.stem in keep]

    ref_cache = Path(seg_cfg["segmenters"][seg_cfg["reference_segmenter"]]
                     ["cache_dir"])
    n_frames = {v.stem: json.loads((ref_cache / f"{v.stem}._cache_done.json")
                                   .read_text())["n_frames"] for v in videos}
    v0 = videos[0]
    tmp = Path(tempfile.mkdtemp(prefix="timing_", dir=out))
    saved = args.max_frames
    args.max_frames = args.guard_frames
    dishes = load_frozen_geometry(
        args.dataset / "_work" / v0.stem / "petri_tracks.csv")
    s = cache_video(model, v0, dishes, tmp, args)
    args.max_frames = saved
    per_frame = s["wall_s"] / max(1, s["n_frames"])
    projected_h = per_frame * sum(n_frames.values()) / 3600.0
    keep = [v.stem for v in videos]
    cut = projected_h > budget
    if cut:
        keep = cut_subset(seg_cfg, keep)
    decision = {"timed_video": v0.stem, "timed_frames": s["n_frames"],
                "s_per_frame": round(per_frame, 4),
                "projected_hours_all": round(projected_h, 2),
                "budget_hours": budget, "cut": cut, "videos": keep}
    guard_file.write_text(json.dumps(decision, indent=2))
    if cut:
        (out / "video_subset.json").write_text(json.dumps(
            {"reason": "runtime guard", **decision}, indent=2))
    print(f"runtime guard: {per_frame:.3f} s/frame -> {projected_h:.1f} h "
          f"projected (budget {budget} h); "
          f"{'CUT to ' + str(len(keep)) if cut else 'keeping all'} videos")
    return [v for v in videos if v.stem in set(keep)]


def stratum_of(video_id: str) -> Optional[str]:
    """Campaign x condition stratum from a video filename, or None."""
    if video_id.startswith("202605"):
        for cond in ("nostress", "warm", "cold"):
            if cond in video_id:
                return f"may_{cond}"
        return None
    if video_id.startswith("202606"):
        if "nostress" in video_id:
            return "jun_nostress"
        m = re.search(r"38\D*C_(\d)h", video_id)
        if m:
            return f"jun_38c_{m.group(1)}h"
    return None


def cut_subset(seg_cfg: dict, videos: list[str]) -> list[str]:
    """Drop one video from each 2-video stratum (in config order) until the
    fallback size is reached, so every stratum stays represented."""
    target = int(seg_cfg["cache_runtime"]["fallback_n_videos"])
    keep = list(videos)
    by_stratum: dict[str, list[str]] = {}
    for v in keep:
        by_stratum.setdefault(stratum_of(v), []).append(v)
    for members in by_stratum.values():
        if len(keep) <= target:
            break
        if len(members) >= 2:
            keep.remove(members[-1])
    return keep


def discover_videos(args) -> list[Path]:
    vids = sorted([p for p in args.videos.iterdir()
                   if p.suffix.lower() in (".mov", ".mp4", ".avi")])
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        vids = [v for v in vids if v.stem in want]
    if args.limit_videos is not None:
        vids = vids[:args.limit_videos]
    return vids


def main(argv: Optional[list[str]] = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    args.out.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(args)
    if not videos:
        print(f"No videos found under {args.videos}", file=sys.stderr)
        return 1
    print(f"Caching detections for {len(videos)} video(s) "
          f"at conf>={args.cache_conf}, imgsz={args.imgsz} -> {args.out}")

    if args.regression_check:
        return run_regression_check(args)
    model = load_detector(args)
    if args.runtime_guard and args.seg_cfg is not None:
        videos = apply_runtime_guard(args, model, videos)
    summaries, failed = [], []
    for i, video in enumerate(videos, 1):
        sentinel = args.out / f"{video.stem}._cache_done.json"
        if args.skip_existing and sentinel.is_file():
            print(f"[{i}/{len(videos)}] {video.stem}: skipped (cached)")
            summaries.append(json.loads(sentinel.read_text()))
            continue
        petri = args.dataset / "_work" / video.stem / "petri_tracks.csv"
        if not petri.is_file():
            print(f"[{i}/{len(videos)}] {video.stem}: NO GEOMETRY at {petri}",
                  file=sys.stderr)
            failed.append(video.stem)
            continue
        print(f"[{i}/{len(videos)}] {video.stem}")
        try:
            dishes = load_frozen_geometry(petri)
            s = cache_video(model, video, dishes, args.out, args)
        except Exception as exc:  # isolate one bad video, keep the batch going
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(video.stem)
            continue
        summaries.append(s)
        print(f"  {s['n_frames']} frames, {s['n_detections']} detections, "
              f"{s['wall_s']}s")

    write_provenance(args.out, args, summaries)
    print(f"\nCached {len(summaries)} video(s), {len(failed)} failed.")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
