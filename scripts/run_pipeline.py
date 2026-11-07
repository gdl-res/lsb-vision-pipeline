#!/usr/bin/env python3
"""End-to-end driver: raw tray videos -> full FAIR dataset (SPECIFICATION.md 16, 24).

Chains the whole pipeline in-process:

    track_larvae_video    per video  -> per-video framewise tables
    (combine)                        -> unified petri/ and larva/ tables
    compute_kinematics               -> kinematics/larva_kinematics.csv
    generate_heatmaps                -> heatmaps/
    compute_behavior_features        -> features/
    compute_qc_summary               -> qc/qc_summary.csv
    generate_fair_metadata           -> README, data_dictionary, config, ...

The result under --out is the recommended FAIR layout (SPECIFICATION.md 16). Use
--skip-tracking to (re)run only the analysis chain on already-combined tables.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import crop_petris_yolo_seg as cp
import track_larvae_video as tl
import larva_trackers as lt
import compute_kinematics as ck
import generate_heatmaps as gh
import compute_behavior_features as cf
import compute_divergence as dv
import compute_qc_summary as qc
import generate_fair_metadata as fm
import generate_report as gr
import aggregate_report as ag

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
LAYOUT_DIRS = ["petri", "larva", "kinematics", "features", "heatmaps",
               "qc", "metadata", "model_cards", "aggregate", "_work"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def discover_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.suffix.lower() in VIDEO_EXTS)
    return []


def run_step(name: str, fn, argv: list[str]) -> None:
    print(f"\n=== {name} ===")
    rc = fn(argv)
    if rc != 0:
        raise RuntimeError(f"step '{name}' failed (rc={rc})")


def build_track_argv(video: Path, work_dir: Path, args) -> list[str]:
    argv = ["--video", str(video), "--out", str(work_dir / video.stem),
            "--petri-model", str(args.petri_model),
            "--larva-model", str(args.larva_model),
            "--rows", str(args.rows), "--cols", str(args.cols),
            "--margin", str(args.margin), "--petri-geometry", args.petri_geometry,
            "--petri-imgsz", str(args.petri_imgsz),
            "--larva-imgsz", str(args.larva_imgsz),
            "--conf", str(args.conf), "--larva-conf", str(args.larva_conf),
            "--iou", str(args.iou), "--tracker", args.tracker,
            "--low-conf-threshold", str(args.low_conf_threshold),
            "--max-jump-px", str(args.max_jump_px),
            "--max-jump-growth-px", str(args.max_jump_growth_px),
            "--max-jump-cap-px", str(args.max_jump_cap_px),
            "--dataset-id", args.dataset_id,
            "--video-id", video.stem]
    if args.experiment_id:
        argv += ["--experiment-id", args.experiment_id]
    if args.petri_diameter_mm is not None:
        argv += ["--petri-diameter-mm", str(args.petri_diameter_mm)]
    if args.tracker_config is not None:
        argv += ["--tracker-config", str(args.tracker_config)]
    if args.area_min_px is not None:
        argv += ["--area-min-px", str(args.area_min_px)]
    if args.area_max_px is not None:
        argv += ["--area-max-px", str(args.area_max_px)]
    if args.reacquire_after_s is not None:
        argv += ["--reacquire-after-s", str(args.reacquire_after_s)]
    if args.device is not None:
        argv += ["--device", args.device]
    if args.track:
        argv += ["--track"]
    if args.save_masks:
        argv += ["--save-masks"]
    if args.save_overlays:
        argv += ["--save-overlays"]
    if args.max_frames is not None:
        argv += ["--max-frames", str(args.max_frames)]
    return argv


def combine_tables(work_dir: Path, out: Path) -> None:
    """Concatenate per-video framewise tables into the unified FAIR layout."""
    import pandas as pd

    csv_map = {
        "petri_tracks.csv": out / "petri" / "petri_tracks.csv",
        "larva_tracks.csv": out / "larva" / "larva_tracks.csv",
        "larva_detections_all.csv": out / "larva" / "larva_detections_all.csv",
    }
    video_dirs = [d for d in work_dir.iterdir() if d.is_dir()]
    for name, dest in csv_map.items():
        frames = [pd.read_csv(d / name) for d in video_dirs
                  if (d / name).is_file()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(dest, index=False)
            print(f"  combined {len(frames)} -> {dest.name} "
                  f"({sum(len(f) for f in frames)} rows)")

    mask_files = [d / "larva_masks.jsonl" for d in video_dirs
                  if (d / "larva_masks.jsonl").is_file()]
    if mask_files:
        with open(out / "larva" / "larva_masks.jsonl", "w") as dst:
            for mf in mask_files:
                dst.write(mf.read_text())


def copy_experiment_metadata(src: Path, out: Path) -> int:
    """Ship the curated experiment_metadata.csv (SPECIFICATION.md 14) in the release.

    Only rows for videos present in features_per_video.csv are kept, so the
    release documents exactly the recordings it contains. Written before the
    FAIR step, which then leaves it alone instead of dropping a blank template.
    Returns the number of rows written.
    """
    import pandas as pd

    meta = pd.read_csv(src, dtype={"video_id": str})
    feat_path = out / "features" / "features_per_video.csv"
    if feat_path.is_file():
        videos = set(pd.read_csv(feat_path, usecols=["video_id"],
                                 dtype={"video_id": str})["video_id"])
        meta = meta[meta["video_id"].isin(videos)]
    dest = out / "metadata" / "experiment_metadata.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(dest, index=False)
    print(f"  experiment metadata: {len(meta)} rows -> {dest}")
    return len(meta)


def gather_provenance(out: Path) -> None:
    """Merge every processing_provenance.jsonl into metadata/."""
    target = out / "metadata" / "processing_provenance.jsonl"
    lines = []
    for prov in out.rglob("processing_provenance.jsonl"):
        if prov.resolve() == target.resolve():
            continue
        lines.extend(prov.read_text().splitlines())
    if lines:
        with open(target, "a") as fh:
            for line in lines:
                if line.strip():
                    fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# Batch resilience: preflight, per-video isolation, resume, batch index
# --------------------------------------------------------------------------- #
DONE_SENTINEL = "_track_done.json"
FAILURE_FILE = "failure.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def probe_video(path: Path) -> dict:
    """Read basic properties; report readability without raising."""
    import cv2

    info = {"video": path.name, "readable": False, "frames": 0,
            "fps": 0.0, "width": 0, "height": 0, "error": None}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            info["error"] = "cannot open (unreadable or unsupported codec)"
            return info
        info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        if not ok:
            info["error"] = "opened but first frame unreadable"
            return info
        info["readable"] = True
    finally:
        cap.release()
    return info


def _wants_gpu(device: Optional[str]) -> bool:
    return bool(device) and "cpu" not in device.lower()


def preflight(videos: list[Path], args) -> int:
    """Validate inputs before committing the machine to a long batch.

    Returns the number of BLOCKERs (0 == good to go). Preflight contract:
    list hard blockers, warn on soft issues, estimate
    runtime/disk so a multi-hour run is not started blind.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    print("=== preflight ===")

    # models present
    for label, mp in (("petri-model", args.petri_model),
                      ("larva-model", args.larva_model)):
        if mp is None or not Path(mp).is_file():
            blockers.append(f"missing {label}: {mp}")
        else:
            print(f"  ok   {label}: {mp}")

    # GPU availability if a cuda device was requested
    if _wants_gpu(args.device):
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ok   device {args.device}: CUDA available")
            else:
                blockers.append(f"device {args.device} requested but "
                                "torch.cuda.is_available() is False")
        except Exception as exc:  # torch missing/broken
            blockers.append(f"device {args.device} requested but torch "
                            f"unavailable: {exc}")

    # videos readable + frame total for ETA
    total_frames = 0
    for v in videos:
        info = probe_video(v)
        if not info["readable"]:
            blockers.append(f"unreadable video {v.name}: {info['error']}")
            continue
        cap = args.max_frames if args.max_frames is not None else info["frames"]
        cap = min(cap, info["frames"]) if info["frames"] else cap
        total_frames += cap
        print(f"  ok   {v.name}: {info['frames']} frames, "
              f"{info['fps']:.1f} fps, {info['width']}x{info['height']}")

    # rough runtime + disk estimate (coarse, video-only)
    n_dishes = args.rows * args.cols
    # ~ from the 600-frame smoke run: ~0.08 s/frame for petri+6 larva on GPU
    sec_per_frame = 0.08 if _wants_gpu(args.device) else 0.8
    est_runtime_s = total_frames * sec_per_frame
    # tables ~1 KB/dish/frame; overlay mp4 dominates when enabled
    est_bytes = total_frames * n_dishes * 1024
    if args.save_overlays:
        est_bytes += total_frames * 60_000  # ~60 KB/frame encoded overlay
    free_bytes = shutil.disk_usage(args.out.parent if args.out.parent.exists()
                                   else Path.cwd()).free
    print(f"  estimate: ~{total_frames} frames, "
          f"~{est_runtime_s / 60:.0f} min, ~{est_bytes / 1e9:.1f} GB output; "
          f"{free_bytes / 1e9:.1f} GB free")
    if est_bytes > 0.8 * free_bytes:
        warnings.append(f"estimated output ({est_bytes / 1e9:.1f} GB) exceeds "
                        f"80% of free disk ({free_bytes / 1e9:.1f} GB)")
    if args.min_free_gb is not None and free_bytes < args.min_free_gb * 1e9:
        blockers.append(f"free disk {free_bytes / 1e9:.1f} GB below "
                        f"--min-free-gb {args.min_free_gb}")

    for w in warnings:
        print(f"  WARN    {w}")
    for b in blockers:
        print(f"  BLOCKER {b}")
    print(f"  -> {len(blockers)} blocker(s), {len(warnings)} warning(s)")
    return len(blockers)


def track_one(video: Path, work: Path, args) -> dict:
    """Track a single video in isolation.

    On success writes a DONE_SENTINEL marker (enables --skip-existing resume);
    on failure writes failure.json and returns status='failed' instead of
    propagating, so a bad video does not abort the whole batch.
    """
    vdir = work / video.stem
    vdir.mkdir(parents=True, exist_ok=True)
    rec = {"video": video.name, "video_id": video.stem, "status": "running",
           "wall_s": 0.0, "started": _utcnow(), "error": None}
    t0 = time.monotonic()
    try:
        run_step(f"track {video.name}", tl.main,
                 build_track_argv(video, work, args))
        if not (vdir / "larva_tracks.csv").is_file():
            raise RuntimeError("track step produced no larva_tracks.csv")
        rec["status"] = "done"
        rec["wall_s"] = round(time.monotonic() - t0, 2)
        (vdir / FAILURE_FILE).unlink(missing_ok=True)
        (vdir / DONE_SENTINEL).write_text(json.dumps(rec, indent=2))
    except Exception as exc:
        rec["status"] = "failed"
        rec["wall_s"] = round(time.monotonic() - t0, 2)
        rec["error"] = f"{type(exc).__name__}: {exc}"
        (vdir / FAILURE_FILE).write_text(json.dumps(
            {**rec, "traceback": traceback.format_exc()}, indent=2))
        (vdir / DONE_SENTINEL).unlink(missing_ok=True)
        print(f"  FAILED {video.name}: {rec['error']}", file=sys.stderr)
    return rec


def write_batch_index(out: Path, records: list[dict]) -> None:
    """One machine-readable record per (video) run for monitoring/resume."""
    index = {"generated": _utcnow(), "n_videos": len(records),
             "n_done": sum(r["status"] == "done" for r in records),
             "n_failed": sum(r["status"] == "failed" for r in records),
             "n_skipped": sum(r["status"] == "skipped" for r in records),
             "runs": records}
    (out / "metadata" / "batch_index.json").write_text(
        json.dumps(index, indent=2))
    print(f"  batch index: {index['n_done']} done, {index['n_failed']} "
          f"failed, {index['n_skipped']} skipped "
          f"-> metadata/batch_index.json")


def run_tracking_batch(videos: list[Path], work: Path, args) -> list[dict]:
    """Track every video with per-video isolation, resume, and an index."""
    records: list[dict] = []
    for i, v in enumerate(videos, 1):
        vdir = work / v.stem
        if args.skip_existing and (vdir / DONE_SENTINEL).is_file():
            prev = json.loads((vdir / DONE_SENTINEL).read_text())
            print(f"\n[{i}/{len(videos)}] skip (done) {v.name}")
            records.append({**prev, "status": "skipped"})
            continue
        print(f"\n[{i}/{len(videos)}] {v.name}")
        rec = track_one(v, work, args)
        records.append(rec)
        if rec["status"] == "failed" and args.fail_fast:
            print("  --fail-fast set; aborting batch.", file=sys.stderr)
            break
    return records


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_analysis_chain(out: Path, args) -> None:
    petri = out / "petri" / "petri_tracks.csv"
    larva = out / "larva" / "larva_tracks.csv"
    kin = out / "kinematics" / "larva_kinematics.csv"
    if not larva.is_file():
        raise FileNotFoundError(f"missing combined tracks: {larva}")

    kin_argv = ["--tracks", str(larva), "--petri-tracks", str(petri),
                "--out", str(kin),
                "--max-interp-gap-s", str(args.max_interp_gap_s),
                "--smoothing-filter", args.smoothing_filter,
                "--smoothing-window", str(args.smoothing_window),
                "--savgol-polyorder", str(args.savgol_polyorder),
                "--kalman-process-var", str(args.kalman_process_var),
                "--kalman-measurement-var", str(args.kalman_measurement_var),
                "--immobility-speed-threshold-mm-s",
                str(args.immobility_speed_threshold_mm_s)]
    if args.petri_diameter_mm is not None:
        kin_argv += ["--petri-diameter-mm", str(args.petri_diameter_mm)]
    run_step("kinematics", ck.main, kin_argv)

    run_step("heatmaps", gh.main,
             ["--input", str(kin), "--out", str(out / "heatmaps"),
              "--bins", str(args.heatmap_bins)])

    feat_argv = ["--tracks", str(larva), "--kinematics", str(kin),
                 "--out", str(out / "features"),
                 "--window-size-s", str(args.window_size_s),
                 "--heatmap-bins", str(args.heatmap_bins),
                 "--rim-occlusion-r", str(args.rim_occlusion_r)]
    if args.metadata is not None:
        feat_argv += ["--metadata", str(args.metadata)]
    run_step("features", cf.main, feat_argv)

    if not args.no_divergence:
        div_argv = ["--features", str(out / "features" /
                                      "features_per_video.csv"),
                    "--heatmaps", str(out / "heatmaps"),
                    "--control-treatment", args.control_treatment]
        if args.metadata is not None:
            div_argv += ["--metadata", str(args.metadata)]
        if args.divergence_exclude_self:
            div_argv += ["--exclude-self"]
        run_step("divergence", dv.main, div_argv)

    qc_argv = ["--tracks", str(larva), "--petri-tracks", str(petri),
               "--kinematics", str(kin),
               "--out", str(out / "qc" / "qc_summary.csv"),
               "--rim-occlusion-r", str(args.rim_occlusion_r),
               "--min-valid-fraction", str(args.qc_min_valid_fraction),
               "--max-missing-gap-s", str(args.qc_max_missing_gap_s),
               "--max-fraction-outside", str(args.qc_max_fraction_outside),
               "--max-step-norm", str(args.qc_max_step_norm),
               "--max-semantic-switch-rate-per-min",
               str(args.qc_max_semantic_switch_rate_per_min),
               "--empty-max-area-norm", str(args.qc_empty_max_area_norm),
               "--empty-max-conf", str(args.qc_empty_max_conf),
               "--empty-max-spread", str(args.qc_empty_max_spread),
               "--morphology-min-solidity",
               str(args.qc_morphology_min_solidity)]
    # nominal track-switch gate is opt-in (default disabled); only forward it
    # when the user explicitly set a threshold.
    if args.qc_max_track_switch_rate_per_min is not None:
        qc_argv += ["--max-track-switch-rate-per-min",
                    str(args.qc_max_track_switch_rate_per_min)]
    run_step("qc", qc.main, qc_argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", type=Path,
                   help="video file or directory (required unless --skip-tracking)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--petri-model", type=Path)
    p.add_argument("--larva-model", type=Path, default=tl.DEFAULT_LARVA_MODEL,
                   help="larva segmenter weights (default: the crop-trained "
                        "v1.2 model, SPECIFICATION.md 26.6)")
    p.add_argument("--skip-tracking", action="store_true",
                   help="run analysis only on already-combined tables")
    # batch resilience
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: skip videos that already tracked successfully "
                        "(have a _track_done.json marker in _work/)")
    p.add_argument("--fail-fast", action="store_true",
                   help="abort the batch on the first failed video "
                        "(default: isolate the failure and continue)")
    p.add_argument("--preflight", action="store_true",
                   help="validate inputs (videos, models, GPU, disk) and exit "
                        "without processing; non-zero exit on any BLOCKER")
    p.add_argument("--min-free-gb", type=float, default=None,
                   help="preflight BLOCKER if free disk is below this many GB")
    # geometry / models
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--petri-geometry", choices=["box", "circlefit"],
                   default="box",
                   help="normalization geometry passed to the tracker "
                        "(circlefit = ellipse-fit the dish; SPECIFICATION.md 26.2)")
    p.add_argument("--petri-imgsz", type=int, default=640,
                   help="pinned inference imgsz for petri detection")
    p.add_argument("--larva-imgsz", type=int, default=640,
                   help="pinned inference imgsz for larva detection")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--larva-conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--track", action="store_true", default=True)
    p.add_argument("--tracker", type=lt.normalize_tracker_name,
                   choices=lt.TRACKERS, default="bytetrack",
                   help="larva association algorithm passed to the tracker: "
                        "bytetrack (locked v1 default) | botsort | ocsort "
                        "(SPECIFICATION.md 26.1)")
    p.add_argument("--tracker-config", type=Path, default=None,
                   help="override config/trackers/<tracker>.yaml")
    # Primary-larva selection policy (SPECIFICATION.md 6). Defaults mirror
    # track_larvae_video.build_parser; a mismatch would silently re-track a
    # batch under a different policy than a single-video run.
    p.add_argument("--low-conf-threshold", type=float, default=0.40)
    p.add_argument("--area-min-px", type=float, default=None,
                   help="plausible larva-area gate, lower bound")
    p.add_argument("--area-max-px", type=float, default=None,
                   help="plausible larva-area gate, upper bound")
    p.add_argument("--max-jump-px", type=float, default=60.0,
                   help="hysteresis gate: max centroid jump a lone "
                        "new-identity detection may make and still be "
                        "accepted as primary (SPECIFICATION.md 6)")
    p.add_argument("--max-jump-growth-px", type=float, default=5.0,
                   help="extra jump allowance per consecutive missing frame "
                        "(additive)")
    p.add_argument("--max-jump-cap-px", type=float, default=240.0,
                   help="absolute ceiling on the jump gate")
    p.add_argument("--reacquire-after-s", type=float,
                   default=tl.DEFAULT_REACQUIRE_AFTER_S,
                   help="passed to track_larvae_video.py (gate "
                        "re-acquisition; default 5 s, 0 disables)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--petri-diameter-mm", type=float, default=None)
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    # analysis (locked v1 defaults, SPECIFICATION.md 26)
    p.add_argument("--max-interp-gap-s", type=float, default=0.5)
    p.add_argument("--smoothing-filter", choices=ck.SMOOTHING_FILTERS,
                   default="savgol")
    p.add_argument("--smoothing-window", type=int, default=7)
    p.add_argument("--savgol-polyorder", type=int, default=2)
    p.add_argument("--kalman-process-var", type=float, default=1.0)
    p.add_argument("--kalman-measurement-var", type=float, default=4.0)
    p.add_argument("--immobility-speed-threshold-mm-s", type=float, default=1.0)
    p.add_argument("--window-size-s", type=float, default=60.0)
    p.add_argument("--heatmap-bins", type=int, default=50)
    p.add_argument("--rim-occlusion-r", type=float, default=0.85,
                   help="r_norm at/above which a missing gap counts as "
                        "rim-seeking occlusion; shared by the features and "
                        "QC steps")
    # QC exclusion thresholds (SPECIFICATION.md 13), passed through to
    # compute_qc_summary.py; defaults tuned for real rim-seeking/petri-exit
    # behavior rather than the illustrative SPECIFICATION.md 13 example values
    p.add_argument("--qc-min-valid-fraction", type=float, default=0.40)
    p.add_argument("--qc-max-missing-gap-s", type=float, default=60.0)
    p.add_argument("--qc-max-fraction-outside", type=float, default=0.30)
    p.add_argument("--qc-max-step-norm", type=float, default=0.5)
    p.add_argument("--qc-max-semantic-switch-rate-per-min", type=float,
                   default=6.0)
    # suspected-empty / static-artifact detection
    p.add_argument("--qc-empty-max-area-norm", type=float, default=0.001)
    p.add_argument("--qc-empty-max-conf", type=float, default=0.45)
    p.add_argument("--qc-empty-max-spread", type=float, default=0.15)
    # morphology reliability (reported per-dish weight, not an exclusion rule)
    p.add_argument("--qc-morphology-min-solidity", type=float, default=0.5)
    # opt-in (default disabled): nominal ByteTrack ID-switch exclusion gate
    p.add_argument("--qc-max-track-switch-rate-per-min", type=float,
                   default=None)
    p.add_argument("--metadata", type=Path, default=None,
                   help="experiment_metadata.csv for the feature step")
    # divergence-from-control post-processing (SPECIFICATION.md 26.4)
    p.add_argument("--no-divergence", action="store_true",
                   help="skip the divergence-from-control post-processor")
    p.add_argument("--control-treatment", default="control",
                   help="treatment pooled into the divergence reference")
    p.add_argument("--divergence-exclude-self", action="store_true",
                   help="leave-one-out reference for control dishes")
    # identifiers / FAIR
    p.add_argument("--dataset-id", default="lsb_tmolitor_v1")
    p.add_argument("--experiment-id", default="")
    p.add_argument("--title", default="Tenebrio molitor "
                                      "behavioral phenotyping dataset")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--creators", default="")
    p.add_argument("--contact", default="")
    p.add_argument("--license", default="CC-BY-4.0",
                   help="data license (LICENSE-DATA)")
    p.add_argument("--code-license", default="Apache-2.0",
                   help="code license (LICENSE)")
    # human-readable report
    p.add_argument("--no-report", action="store_true",
                   help="skip the human-readable report.html/report.md step")
    p.add_argument("--link-report-images", action="store_true",
                   help="report references PNGs by relative path instead of "
                        "embedding them (smaller, but not standalone)")
    # treatment-grouped aggregate (best-effort, needs --metadata treatment labels)
    p.add_argument("--no-aggregate", action="store_true",
                   help="skip the treatment-grouped aggregate report")
    p.add_argument("--aggregate-factors", nargs="*", default=[],
                   help="extra metadata columns refining conditions, e.g. "
                        "stressor_duration_s light_condition")
    p.add_argument("--aggregate-min-valid-fraction", type=float, default=0.0,
                   help="exclude dishes below this valid_fraction from the "
                        "aggregate's behavioral contrasts")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out
    for d in LAYOUT_DIRS:
        (out / d).mkdir(parents=True, exist_ok=True)

    if not args.skip_tracking:
        if not (args.videos and args.petri_model and args.larva_model):
            print("Tracking needs --videos, --petri-model and --larva-model "
                  "(or pass --skip-tracking).", file=sys.stderr)
            return 1
        videos = discover_videos(args.videos)
        if not videos:
            print(f"No videos found at {args.videos}", file=sys.stderr)
            return 1

        if args.preflight:
            return 1 if preflight(videos, args) else 0

        print(f"Tracking {len(videos)} video(s)...")
        work = out / "_work"
        records = run_tracking_batch(videos, work, args)
        write_batch_index(out, records)
        if not any(r["status"] in ("done", "skipped") for r in records):
            print("All videos failed; nothing to analyse.", file=sys.stderr)
            return 1
        combine_tables(work, out)
    else:
        print("Skipping tracking; using existing combined tables.")

    run_analysis_chain(out, args)

    if args.metadata is not None:
        copy_experiment_metadata(args.metadata, out)

    model_argv = []
    for flag, mp in (("--larva-model", args.larva_model),
                     ("--petri-model", args.petri_model)):
        if mp is not None:
            model_argv += [flag, str(mp)]
    run_step("FAIR metadata", fm.main,
             model_argv +
             ["--out", str(out), "--dataset-id", args.dataset_id,
              "--title", args.title, "--version", args.version,
              "--creators", args.creators, "--contact", args.contact,
              "--license", args.license,
              "--code-license", args.code_license, "--no-pip-freeze",
              "--tracker", args.tracker,
              "--max-jump-px", str(args.max_jump_px),
              "--max-jump-growth-px", str(args.max_jump_growth_px),
              "--max-jump-cap-px", str(args.max_jump_cap_px),
              "--reacquire-after-s", str(args.reacquire_after_s),
              "--smoothing-filter", args.smoothing_filter,
              "--smoothing-window", str(args.smoothing_window),
              "--savgol-polyorder", str(args.savgol_polyorder),
              "--kalman-process-var", str(args.kalman_process_var),
              "--kalman-measurement-var", str(args.kalman_measurement_var)])

    if not args.no_report:
        report_argv = ["--dataset", str(out)]
        if args.link_report_images:
            report_argv += ["--link-images"]
        run_step("report", gr.main, report_argv)

    # Treatment-grouped aggregate: best-effort, only when curated treatment
    # labels are available. A non-zero rc (e.g. no usable treatment) is a
    # warning, not a pipeline failure.
    if args.metadata is not None and not args.no_aggregate:
        print("\n=== aggregate ===")
        agg_argv = ["--features", str(out / "features" /
                                      "features_per_video.csv"),
                    "--metadata", str(args.metadata),
                    "--out", str(out / "aggregate"),
                    "--control-treatment", args.control_treatment,
                    "--min-valid-fraction",
                    str(args.aggregate_min_valid_fraction)]
        if args.aggregate_factors:
            agg_argv += ["--factors", *args.aggregate_factors]
        if ag.main(agg_argv) != 0:
            print("  aggregate report skipped (no usable treatment labels in "
                  "--metadata).", file=sys.stderr)

    gather_provenance(out)
    print(f"\nPipeline complete. FAIR dataset at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
