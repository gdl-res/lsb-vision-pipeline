#!/usr/bin/env python3
"""Replay several association backends over one cached detection set.

Second stage of the offline replay. `cache_detections.py` ran the
segmenter once; this script replays every configuration over that single cache,
so the detector output, thresholds, arena geometry, crops and preprocessing are
identical across configurations by construction rather than by assertion. It runs
on CPU and never opens a video or a model.

Configurations are (method x continuity-gate):

    bytetrack   ultralytics BYTETracker, driven directly
    botsort     ultralytics BOTSORT, driven directly
    ocsort      larva_trackers.OCSort
    nearest     null associator -- emits no track ids at all

`nearest` is the tracker-free baseline. With no ids, select_primary_larva
(SPECIFICATION.md 6) can never take its `same_track_id` branch and reduces to
"plausible-area gate -> closest to the previous accepted centroid", i.e. exactly
a nearest-valid-centroid tracker. It reuses the same selection code as every
other configuration, so the contrast isolates what association contributes.

The gate is the hysteresis jump gate; `gate_off` sets max_jump_px to 0, which
short-circuits plausible_jump() to True.

DRIVING ULTRALYTICS OFFLINE. BYTETracker.update duck-types its `results`
argument (it needs .conf, .cls, .xywh, boolean-mask indexing and __len__), so an
`ultralytics.engine.results.Boxes` built from cached rows is sufficient. Two
details are load-bearing:

  * frame_rate=30 is passed regardless of the true 60 fps, because
    ultralytics/trackers/track.py hardcodes it, making max_time_lost equal to
    track_buffer. Production ran with that value; reproducing production requires
    reproducing it.
  * All six per-dish trackers are constructed UP FRONT. BYTETracker.__init__
    calls reset_id() on the process-global BaseTrack._count, so lazy per-dish
    construction (what the production pipeline does) lets dish 2's tracker zero a
    counter dish 1 is already using. Constructing first makes per-dish id
    numbering deterministic; canonical_track_id's dish_id*1000 namespacing makes
    the difference immaterial downstream.

EMISSION POLICY. Each backend is replayed as it behaves in the pipeline, not
normalized to a common one. ultralytics' on_predict_postprocess_end replaces the
results object with only the rows the tracker emitted, so for bytetrack/botsort
only emitted detections reach the selection step -- suppression is part of what
those methods do (except when the tracker emits nothing at all, where that code
short-circuits and every raw detection survives; replicated above). OCSort and
nearest annotate every detection instead. n_detections_fed vs
n_detections_associated is recorded per recording so the report can separate "the
tracker filtered detections" from "the tracker assigned identities" rather than
conflating them.

ONE DELIBERATE DEVIATION FROM PRODUCTION. ultralytics also overwrites the box
coordinates with the tracker's Kalman-smoothed box
(`results.update(boxes=tracks[:, :-1])`), so x1_crop..y2_crop and the bbox_*
columns in the released dataset are ByteTrack-filtered rather than raw. This
replay writes the raw detector box for every configuration, which keeps those
columns tracker-independent and comparable. Centroids, confidences and mask
morphology are unaffected -- they come from the detection either way -- and no
evaluated metric reads the bbox columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp
import track_larvae_video as tl
import larva_trackers as lt
import cache_detections as cd

RUNTIME_COLUMNS = [
    "config_id", "method", "gate", "video_id", "dish_id",
    "n_frames", "n_detections_fed", "n_detections_associated",
    "assoc_s", "select_s", "row_s",
]


# --------------------------------------------------------------------------- #
# Association backends
# --------------------------------------------------------------------------- #
class Associator:
    """Assigns `track_id` to a frame's detections, in place.

    `update` returns the detections that reach the selection step, which is not
    always the input list -- see EMISSION POLICY in the module docstring.
    """

    def update(self, dets: list[tl.LarvaDetection]) -> list[tl.LarvaDetection]:
        raise NotImplementedError


class NullAssociator(Associator):
    """The tracker-free baseline: no identities, ever."""

    def update(self, dets):
        for d in dets:
            d.track_id = None
        return dets


class OCSortAssociator(Associator):
    def __init__(self, cfg: dict):
        self.tracker = lt.ocsort_from_config(cfg)

    def update(self, dets):
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2, d.conf] for d in dets],
                         dtype=float).reshape(-1, 5)
        for d, tid in zip(dets, self.tracker.update(boxes)):
            d.track_id = tid
        return dets


class UltralyticsAssociator(Associator):
    """BYTETracker / BOTSORT driven directly from cached boxes."""

    def __init__(self, cfg: dict, frame_rate: int, orig_shape: tuple[int, int]):
        from types import SimpleNamespace
        from ultralytics.trackers import BOTSORT, BYTETracker

        klass = {"bytetrack": BYTETracker, "botsort": BOTSORT}[cfg["tracker_type"]]
        self.args = SimpleNamespace(**cfg)
        self.tracker = klass(args=self.args, frame_rate=frame_rate)
        self.orig_shape = orig_shape
        self.high_thresh = float(cfg["track_high_thresh"])

    def update(self, dets):
        from ultralytics.engine.results import Boxes

        for d in dets:
            d.track_id = None
        if not dets:
            # step the tracker anyway so lost tracks age
            self.tracker.update(Boxes(np.zeros((0, 6), dtype=np.float32),
                                      self.orig_shape), img=None)
            return []
        data = np.array([[d.x1, d.y1, d.x2, d.y2, d.conf, tl.LARVA_CLASS_ID]
                         for d in dets], dtype=np.float32)
        tracks = self.tracker.update(Boxes(data, self.orig_shape), img=None)
        if len(tracks) == 0:
            # Faithful to ultralytics: on_predict_postprocess_end short-circuits
            # with `if len(tracks) == 0: continue`, leaving the results object
            # untouched -- so the pipeline still sees every raw detection, with
            # no ids. Returning [] here instead would invent extra missing
            # frames that production never had.
            return dets
        # The trailing column of `tracks` indexes into the score-filtered subset
        # init_track was called on, not into `dets`. It coincides with `dets` only
        # when every detection cleared track_high_thresh -- which the experiment
        # guarantees (det_conf == track_high_thresh) but this cannot assume.
        direct = bool(np.all(data[:, 4] >= self.high_thresh))
        out = []
        for t in tracks:
            i = int(t[-1])
            if not direct or i >= len(dets):
                i = int(np.argmin(np.abs(data[:, :4] - t[:4]).sum(axis=1)))
            dets[i].track_id = int(t[4])
            out.append(dets[i])
        return out


def make_associator(method: str, tracker_cfg: dict, frame_rate: int,
                    orig_shape: tuple[int, int]) -> Associator:
    if method == "nearest":
        return NullAssociator()
    if method == "ocsort":
        return OCSortAssociator(tracker_cfg)
    return UltralyticsAssociator(tracker_cfg, frame_rate, orig_shape)


# --------------------------------------------------------------------------- #
# Cache loading
# --------------------------------------------------------------------------- #
def load_cache(cache_dir: Path, video_id: str):
    """(frames_df, detections_df, dishes, summary) for one cached video."""
    import pandas as pd

    summary = json.loads(
        (cache_dir / f"{video_id}._cache_done.json").read_text())
    frames = pd.read_csv(cache_dir / f"{video_id}.frames.csv.gz")
    dets = pd.read_csv(cache_dir / f"{video_id}.detections.csv.gz")
    dishes = []
    for g in summary["geometry"]:
        box = cp.PetriBox(g["petri_x1"], g["petri_y1"], g["petri_x2"],
                          g["petri_y2"], g["petri_conf"])
        geom = cp.CropGeom(g["crop_x0"], g["crop_y0"], g["crop_size"])
        fit = (cp.DishFit(g["petri_fit_cx"], g["petri_fit_cy"], g["petri_fit_a"],
                          g["petri_fit_b"], True) if g["petri_fit_valid"]
               else None)
        dishes.append(tl.DishGeom(g["dish_id"], g["row"], g["col"], box, geom,
                                  fit))
    return frames, dets, dishes, summary


def detections_by_frame(dets_df, dish_id: int, det_conf: float) -> dict:
    """{frame_idx: [(LarvaDetection, morph_dict|None), ...]} for one dish."""
    d = dets_df[(dets_df["dish_id"] == dish_id)
                & (dets_df["conf"] >= det_conf)]
    out: dict[int, list] = {}
    morph_cols = [f"mask_{k}" for k in cd.MORPH_KEYS]
    for rec in d.sort_values(["frame_idx", "det_id"]).to_dict("records"):
        has_poly = str(rec["has_polygon"]).lower() == "true"
        det = tl.LarvaDetection(
            None, float(rec["conf"]),
            float(rec["x1_crop"]), float(rec["y1_crop"]),
            float(rec["x2_crop"]), float(rec["y2_crop"]),
            float(rec["cx_crop"]), float(rec["cy_crop"]),
            float(rec["area_px"]),
            # a sentinel, never used for geometry: build_larva_row only tests it
            # for None to decide whether to write the mask_* columns, and the
            # morphology itself is handed over precomputed from the cache
            np.empty((0, 2)) if has_poly else None)
        morph = ({k: float(rec[c]) for k, c in zip(cd.MORPH_KEYS, morph_cols)}
                 if has_poly else None)
        out.setdefault(int(rec["frame_idx"]), []).append((det, morph))
    return out


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def replay_video(cache_dir: Path, video_id: str, method: str, gate: dict,
                 tracker_cfg: dict, opts, writer) -> list[dict]:
    """Replay one video under one configuration; append rows via `writer`."""
    frames, dets_df, dishes, summary = load_cache(cache_dir, video_id)
    fps = float(summary["fps"])
    ids = {"dataset_id": opts.dataset_id, "experiment_id": opts.experiment_id,
           "video_id": video_id}
    orig_shape = (dishes[0].geom.size, dishes[0].geom.size)
    frame_ids = sorted(frames["frame_idx"].unique().tolist())

    # Construct every dish's tracker BEFORE any frame: see module docstring.
    assoc = {dg.dish_id: make_associator(method, tracker_cfg,
                                         opts.tracker_frame_rate, orig_shape)
             for dg in dishes}
    per_dish = {dg.dish_id: detections_by_frame(dets_df, dg.dish_id,
                                                opts.det_conf)
                for dg in dishes}
    stats = {dg.dish_id: {"n_fed": 0, "n_assoc": 0, "assoc_s": 0.0,
                          "select_s": 0.0, "row_s": 0.0} for dg in dishes}
    prev = {dg.dish_id: {"track_id": None, "centroid": None,
                         "missing_streak": 0} for dg in dishes}

    for frame_idx in frame_ids:
        time_s = frame_idx / fps
        for dg in dishes:
            st = stats[dg.dish_id]
            pairs = per_dish[dg.dish_id].get(frame_idx, [])
            dets = [p[0] for p in pairs]
            morphs = {id(p[0]): p[1] for p in pairs}
            st["n_fed"] += len(dets)

            t = time.perf_counter()
            emitted = assoc[dg.dish_id].update(dets)
            st["assoc_s"] += time.perf_counter() - t
            st["n_assoc"] += len(emitted)

            t = time.perf_counter()
            idx, reason, ambiguous = tl.select_primary_larva(
                emitted, prev[dg.dish_id]["track_id"],
                prev[dg.dish_id]["centroid"],
                opts.area_min_px, opts.area_max_px,
                max_jump_px=gate["max_jump_px"],
                missing_streak=prev[dg.dish_id]["missing_streak"],
                jump_growth_px=gate["max_jump_growth_px"],
                jump_cap_px=gate["max_jump_cap_px"])
            st["select_s"] += time.perf_counter() - t

            _, _, erad = tl.effective_geometry(dg)
            mm = (opts.petri_diameter_mm / (2 * erad)
                  if opts.petri_diameter_mm and erad > 0 else None)
            chosen = emitted[idx] if idx is not None else None
            new_tid = tl.canonical_track_id(
                dg.dish_id, chosen.track_id) if chosen is not None else None
            track_switch = (chosen is not None and new_tid is not None
                            and prev[dg.dish_id]["track_id"] is not None
                            and new_tid != prev[dg.dish_id]["track_id"])

            t = time.perf_counter()
            row, _ = tl.build_larva_row(
                ids, frame_idx, time_s, dg, chosen, reason, ambiguous,
                track_switch, mm, opts,
                morph=None if chosen is None else morphs.get(id(chosen)))
            writer.writerow(row)
            st["row_s"] += time.perf_counter() - t

            if chosen is not None:
                prev[dg.dish_id] = {"track_id": new_tid,
                                    "centroid": (chosen.cx, chosen.cy),
                                    "missing_streak": 0}
            else:
                prev[dg.dish_id]["missing_streak"] += 1

    return [{"video_id": video_id, "dish_id": dg.dish_id,
             "n_frames": len(frame_ids),
             "n_detections_fed": stats[dg.dish_id]["n_fed"],
             "n_detections_associated": stats[dg.dish_id]["n_assoc"],
             "assoc_s": round(stats[dg.dish_id]["assoc_s"], 4),
             "select_s": round(stats[dg.dish_id]["select_s"], 4),
             "row_s": round(stats[dg.dish_id]["row_s"], 4)}
            for dg in dishes]


def run_downstream(cfg_dir: Path, petri_tracks: Path, dcfg: dict,
                   petri_diameter_mm: float) -> None:
    """Kinematics -> features -> QC, pinned identical across configurations."""
    import compute_kinematics as ck
    import compute_behavior_features as cf
    import compute_qc_summary as qcs

    tracks = cfg_dir / "larva_tracks.csv"
    kin = cfg_dir / "larva_kinematics.csv"
    rc = ck.main(["--tracks", str(tracks), "--petri-tracks", str(petri_tracks),
                  "--out", str(kin),
                  "--max-interp-gap-s", str(dcfg["max_interp_gap_s"]),
                  "--smoothing-filter", dcfg["smoothing_filter"],
                  "--smoothing-window", str(dcfg["smoothing_window"]),
                  "--savgol-polyorder", str(dcfg["savgol_polyorder"]),
                  "--immobility-speed-threshold-mm-s",
                  str(dcfg["immobility_speed_threshold_mm_s"]),
                  "--petri-diameter-mm", str(petri_diameter_mm)])
    if rc != 0:
        raise RuntimeError("kinematics failed")
    rc = cf.main(["--tracks", str(tracks), "--kinematics", str(kin),
                  "--out", str(cfg_dir / "features"),
                  "--window-size-s", str(dcfg["window_size_s"]),
                  "--heatmap-bins", str(dcfg["heatmap_bins"]),
                  "--rim-occlusion-r", str(dcfg["rim_occlusion_r"])])
    if rc != 0:
        raise RuntimeError("features failed")
    q = dcfg["qc"]
    argv = ["--tracks", str(tracks), "--petri-tracks", str(petri_tracks),
            "--kinematics", str(kin), "--out", str(cfg_dir / "qc_summary.csv"),
            "--min-valid-fraction", str(q["min_valid_fraction"]),
            "--max-missing-gap-s", str(q["max_missing_gap_s"]),
            "--max-fraction-outside", str(q["max_fraction_outside"]),
            "--max-step-norm", str(q["max_step_norm"]),
            "--max-semantic-switch-rate-per-min",
            str(q["max_semantic_switch_rate_per_min"]),
            "--rim-occlusion-r", str(dcfg["rim_occlusion_r"])]
    if q.get("max_track_switch_rate_per_min") is not None:
        argv += ["--max-track-switch-rate-per-min",
                 str(q["max_track_switch_rate_per_min"])]
    if qcs.main(argv) != 0:
        raise RuntimeError("qc failed")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def config_id(method: str, gate: str, prefix: Optional[str] = None) -> str:
    """`<method>__<gate>`, or `<segmenter>__<method>__<gate>` with a prefix
    (segmenter comparison: every segmenter's configs share one configs/ dir)."""
    cid = f"{method}__{gate}"
    return f"{prefix}__{cid}" if prefix else cid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("config/replay.yaml"))
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--petri-tracks", type=Path, default=None,
                   help="petri_tracks.csv shared by every configuration "
                        "(default: the production dataset's, which is the same "
                        "frozen geometry the cache was built from)")
    p.add_argument("--methods", type=str, default=None,
                   help="comma-separated subset of methods")
    p.add_argument("--gates", type=str, default=None)
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated video ids")
    p.add_argument("--dataset-id", type=str, default="wormstress_tmolitor_v1")
    p.add_argument("--experiment-id", type=str, default="replay")
    p.add_argument("--no-downstream", action="store_true",
                   help="write larva_tracks.csv only (harness debugging)")
    p.add_argument("--config-prefix", type=str, default=None,
                   help="prefix config ids with a segmenter name")
    p.add_argument("--all-segmenters", action="store_true",
                   help="segmenter-comparison config: replay every "
                        "segmenter's cache, each under its own prefix")
    p.add_argument("--rerun", action="store_true",
                   help="ignore per-config done-markers")
    return p


def cached_video_ids(cache: Path, only: Optional[list[str]]) -> list[str]:
    ids = sorted(p.name.replace("._cache_done.json", "")
                 for p in cache.glob("*._cache_done.json"))
    if only is not None:
        want = set(only)
        ids = [v for v in ids if v in want]
    return ids


def replay_cache(cache: Path, video_ids: list[str], out: Path, methods,
                 gates, rep: dict, dcfg: dict, opts, petri_tracks: Path,
                 prefix: Optional[str], downstream: bool,
                 rerun: bool) -> None:
    """Replay every (method x gate) over one cache into out/configs/<cid>/.

    Per-config runtime rows go to <cid>/runtime.csv and a `_replay_done.json`
    marker is written last, so an interrupted multi-config replay resumes at
    the first unfinished configuration.
    """
    fields = (["segmenter"] if prefix else []) + RUNTIME_COLUMNS
    for method in methods:
        tracker_cfg = ({} if method == "nearest"
                       else lt.load_tracker_config(method))
        for gate_name in gates:
            gate = dict(rep[gate_name])
            cid = config_id(method, gate_name, prefix)
            cfg_dir = out / "configs" / cid
            done = cfg_dir / "_replay_done.json"
            if done.is_file() and not rerun:
                print(f"  {cid}: done (skipped)")
                continue
            cfg_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            runtime_rows = []
            with open(cfg_dir / "larva_tracks.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=tl.LARVA_COLUMNS)
                w.writeheader()
                for vid in video_ids:
                    for r in replay_video(cache, vid, method, gate,
                                          tracker_cfg, opts, w):
                        row = {"config_id": cid, "method": method,
                               "gate": gate_name, **r}
                        if prefix:
                            row["segmenter"] = prefix
                        runtime_rows.append(row)
            with open(cfg_dir / "runtime.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(runtime_rows)
            print(f"  {cid}: tracks written in {time.time() - t0:.1f}s")
            if downstream:
                run_downstream(cfg_dir, petri_tracks, dcfg,
                               opts.petri_diameter_mm)
                print(f"  {cid}: downstream done "
                      f"({time.time() - t0:.1f}s total)")
            done.write_text(json.dumps({
                "config_id": cid, "cache": str(cache),
                "n_videos": len(video_ids), "videos": video_ids,
                "wall_s": round(time.time() - t0, 1)}))


def merge_runtime(out: Path) -> None:
    """out/runtime.csv = every configuration's runtime rows."""
    rows, fields = [], []
    for f in sorted((out / "configs").glob("*/runtime.csv")):
        with open(f, newline="") as fh:
            r = csv.DictReader(fh)
            for k in r.fieldnames or []:
                if k not in fields:
                    fields.append(k)
            rows.extend(r)
    with open(out / "runtime.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or RUNTIME_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    import yaml

    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    # Segmenter-comparison config: replay settings come from its base config,
    # one cache per segmenter, videos = the pinned (or runtime-cut) subset.
    jobs: list[tuple[Path, Optional[str]]] = []
    only = ([s.strip() for s in args.only.split(",") if s.strip()]
            if args.only else None)
    if "segmenters" in cfg:
        seg_cfg = cfg
        cfg = yaml.safe_load(Path(seg_cfg["base_config"]).read_text())
        out = args.out or Path(seg_cfg["paths"]["out"])
        petri_tracks = args.petri_tracks or (
            Path(seg_cfg["paths"]["production_dataset"]) / "petri"
            / "petri_tracks.csv")
        if only is None:
            only = cd.active_video_subset(seg_cfg) or None
        names = (list(seg_cfg["segmenters"]) if args.all_segmenters
                 else [args.config_prefix])
        if names == [None]:
            print("--all-segmenters or --config-prefix NAME is required with "
                  "a segmenter config", file=sys.stderr)
            return 2
        for name in names:
            jobs.append((args.cache or
                         Path(seg_cfg["segmenters"][name]["cache_dir"]), name))
        methods_default, gates_default = seg_cfg["methods"], seg_cfg["gates"]
    else:
        out = args.out or Path(cfg["paths"]["out"])
        petri_tracks = args.petri_tracks or (
            Path(cfg["paths"]["dataset"]) / "petri" / "petri_tracks.csv")
        jobs.append((args.cache or Path(cfg["paths"]["cache"]),
                     args.config_prefix))
        methods_default, gates_default = cfg["methods"], cfg["gates"]
    rep, dcfg = cfg["replay"], cfg["downstream"]

    methods = ([m.strip() for m in args.methods.split(",")] if args.methods
               else list(methods_default))
    gates = ([g.strip() for g in args.gates.split(",")] if args.gates
             else list(gates_default))

    # A lightweight stand-in for the tracker's argparse Namespace: the row
    # builder and the selection step only read these fields.
    from types import SimpleNamespace
    opts = SimpleNamespace(
        det_conf=float(rep["det_conf"]),
        low_conf_threshold=float(rep["low_conf_threshold"]),
        area_min_px=rep["area_min_px"], area_max_px=rep["area_max_px"],
        tracker_frame_rate=int(rep["tracker_frame_rate"]),
        petri_diameter_mm=float(cfg["detection"]["petri_diameter_mm"]),
        dataset_id=args.dataset_id, experiment_id=args.experiment_id)

    out.mkdir(parents=True, exist_ok=True)
    all_videos: dict[str, list[str]] = {}
    for cache, prefix in jobs:
        video_ids = cached_video_ids(cache, only)
        if only is not None and prefix is not None \
                and len(video_ids) != len(only):
            missing = sorted(set(only) - set(video_ids))
            print(f"{prefix}: cache {cache} lacks {len(missing)} of the "
                  f"pinned videos: {missing}", file=sys.stderr)
            return 1
        if not video_ids:
            print(f"No cached videos under {cache}", file=sys.stderr)
            return 1
        all_videos[str(prefix)] = video_ids
        print(f"Replaying {len(methods) * len(gates)} configuration(s) over "
              f"{len(video_ids)} cached video(s)"
              + (f" [{prefix}]" if prefix else ""))
        replay_cache(cache, video_ids, out, methods, gates, rep, dcfg, opts,
                     petri_tracks, prefix, not args.no_downstream, args.rerun)
    merge_runtime(out)

    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_replay_trackers"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/replay_trackers.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(c) for c, _ in jobs] + [str(petri_tracks)],
        "output_path": str(out),
        "n_videos": max(len(v) for v in all_videos.values()),
        "videos_by_prefix": all_videos,
        "parameters": {"methods": methods, "gates": gates,
                       "det_conf": opts.det_conf,
                       "tracker_frame_rate": opts.tracker_frame_rate,
                       "gate_definitions": {g: rep[g] for g in gates},
                       "downstream": dcfg},
    }
    with open(out / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"\nDone -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
