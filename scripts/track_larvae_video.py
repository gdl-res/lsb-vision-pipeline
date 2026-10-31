#!/usr/bin/env python3
"""Track larvae in a tray video and emit framewise measurement tables.

Implements SPECIFICATION.md 18.3 and the measurement layer of sections 5, 6, 7, 8, 9.
Per SPECIFICATION.md 22/25 this script produces RAW observations only:

    petri_tracks.csv            (8.1)  one row per frame x dish
    larva_tracks.csv            (8.2)  selected larva per frame x dish (+ 9 morphology)
    larva_detections_all.csv    (8.3)  every raw YOLO larva detection
    larva_masks.jsonl           (8.4)  selected larva mask polygons (--save-masks)

Smoothing, interpolation and kinematics (SPECIFICATION.md 10, 26.3) are a separate
post-processing step; the corresponding `*_smooth` columns are written empty here
so the on-disk schema is stable. Behavioral features and heatmaps (11-13) are
likewise downstream.

Locked v1 policy (SPECIFICATION.md 26.1): the larva segmenter runs in stateful tracking
mode with one tracker instance per dish-crop-stream; Petri geometry is detected
once on the first valid frame (SPECIFICATION.md 5) and reused, keeping the crop window
stable. The association algorithm is selectable with --tracker
{bytetrack,botsort,ocsort} and defaults to ByteTrack, which is what every v1
dataset was tracked with; see scripts/larva_trackers.py and config/trackers/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp
import larva_trackers as lt

LARVA_CLASS_ID = 0  # SPECIFICATION.md 3.2

# --------------------------------------------------------------------------- #
# CSV schemas (column order is part of the FAIR contract)
# --------------------------------------------------------------------------- #
PETRI_COLUMNS = [
    "dataset_id", "experiment_id", "video_id", "frame_idx", "time_s",
    "dish_id", "row", "col", "petri_track_id", "petri_conf",
    "petri_x1", "petri_y1", "petri_x2", "petri_y2",
    "petri_cx", "petri_cy", "petri_w", "petri_h", "petri_radius",
    "petri_valid", "crop_x0", "crop_y0", "crop_size",
    "petri_aspect_ratio", "petri_distortion_flag", "petri_offset_norm",
    "petri_detect_policy",
    "petri_geometry_source", "petri_fit_cx", "petri_fit_cy",
    "petri_fit_a", "petri_fit_b", "petri_fit_valid",
]

LARVA_COLUMNS = [
    # identifiers
    "dataset_id", "experiment_id", "video_id", "frame_idx", "time_s",
    "dish_id", "larva_track_id", "larva_class_id", "larva_conf",
    "valid_detection", "selection_reason",
    # crop-space coordinates
    "crop_x0", "crop_y0", "crop_size", "cx_crop_raw", "cy_crop_raw",
    "x1_crop", "y1_crop", "x2_crop", "y2_crop", "bbox_w_crop", "bbox_h_crop",
    # full-frame coordinates
    "cx_global_raw", "cy_global_raw", "x1_global", "y1_global",
    "x2_global", "y2_global", "bbox_w_global", "bbox_h_global",
    # petri-normalized coordinates
    "x_norm_raw", "y_norm_raw", "r_norm_raw", "theta_rad_raw", "inside_petri",
    # smoothed coordinates (reserved; filled by the kinematics step)
    "x_norm_smooth", "y_norm_smooth", "r_norm_smooth",
    "cx_global_smooth", "cy_global_smooth",
    # detection / tracking quality
    "missing", "interpolated", "track_switch", "outlier_position",
    "outlier_area", "quality_flag",
    # segmentation morphology (SPECIFICATION.md 9)
    "mask_area_px", "mask_area_norm_petri", "mask_area_mm2",
    "mask_perimeter_px", "mask_compactness", "mask_solidity", "mask_extent",
    "mask_major_axis_px", "mask_minor_axis_px", "mask_aspect_ratio",
    "mask_orientation_rad", "mask_eccentricity",
    "bbox_area_px", "mask_bbox_area_ratio",
]

DETECTIONS_COLUMNS = [
    "dataset_id", "experiment_id", "video_id", "frame_idx", "time_s",
    "dish_id", "raw_detection_id", "yolo_track_id", "class_id", "conf",
    "x1_crop", "y1_crop", "x2_crop", "y2_crop", "cx_crop", "cy_crop",
    "mask_area_px", "selected_primary", "selection_reason",
]


# --------------------------------------------------------------------------- #
# Pure functions (testable without YOLO / OpenCV)
# --------------------------------------------------------------------------- #
def petri_normalize(cx_global: float, cy_global: float,
                    petri_cx: float, petri_cy: float,
                    petri_radius: float) -> dict:
    """Map a full-frame point into Petri-normalized coordinates (SPECIFICATION.md 7.3)."""
    if petri_radius <= 0:
        return {"x_norm": float("nan"), "y_norm": float("nan"),
                "r_norm": float("nan"), "theta_rad": float("nan"),
                "inside_petri": False}
    x = (cx_global - petri_cx) / petri_radius
    y = (cy_global - petri_cy) / petri_radius
    r = math.hypot(x, y)
    return {"x_norm": x, "y_norm": y, "r_norm": r,
            "theta_rad": math.atan2(y, x), "inside_petri": r <= 1.0}


@dataclass
class LarvaDetection:
    """One raw YOLO larva detection in crop-pixel coordinates."""
    track_id: Optional[int]
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    area_px: float
    polygon: Optional[np.ndarray] = None  # Nx2 crop-pixel coords, if segmented


def select_primary_larva(
    dets: list[LarvaDetection],
    prev_track_id: Optional[int],
    prev_centroid: Optional[tuple[float, float]],
    area_min: Optional[float],
    area_max: Optional[float],
    max_jump_px: Optional[float] = None,
    missing_streak: int = 0,
    jump_growth_px: float = 5.0,
    jump_cap_px: Optional[float] = 240.0,
) -> tuple[Optional[int], str, bool]:
    """Choose the primary larva for a dish (SPECIFICATION.md 6).

    Returns (index_into_dets | None, selection_reason, ambiguous).

    Policy (continuity-first, documented deviation from the literal list order
    so track identity wins over a marginally higher-confidence distractor):
      no detection            -> (None, "no_detection")
      single detection        -> "single_detection" (subject to the
                                  hysteresis jump gate below)
      track id matches prev    -> "same_track_id"
      gate to plausible area + plausible jump; if that leaves exactly one
                               -> "area_filter"
      among remaining: closest to previous centroid -> "closest_to_previous"
      else highest confidence  -> "highest_confidence"
      final fallback           -> "largest_area" (flagged ambiguous)
      implausible lone jump    -> "rejected_implausible" (treated as missing)

    Hysteresis (max_jump_px): batch analysis of real footage found ~85% of
    track-ID switches happen with no preceding missing-detection gap -- the
    tracker was flipping between an established larva track and a spurious
    second detection (a dish-surface artifact or a mask fragment) that
    happens to be the *only* detection in a given frame. Without a jump
    check, the old "single detection -> accept" shortcut let that lone
    distractor instantly hijack the track. With max_jump_px set, a lone
    detection that doesn't match prev_track_id must also land within
    `min(max_jump_px + missing_streak * jump_growth_px, jump_cap_px)` of
    prev_centroid. The small additive (not multiplicative) growth term
    allows a bit more slack the longer the real larva has gone unconfirmed
    (a larger jump becomes plausible after a genuine multi-frame occlusion),
    while jump_cap_px stops that allowance from ever exploding: an early
    multiplicative version of this gate let a ~200px jump through after only
    3-4 missing frames, which is far faster than a larva can plausibly crawl
    and reopened exactly the artifact-vs-larva oscillation this gate exists
    to stop. Failing the check does not discard the detection -- it's still
    recorded in larva_detections_all.csv -- it just isn't promoted to
    primary, so the established track survives the transient frame instead
    of being overwritten by a one-frame artifact.
    """
    if not dets:
        return None, "no_detection", False

    def in_range(d: LarvaDetection) -> bool:
        if area_min is not None and d.area_px < area_min:
            return False
        if area_max is not None and d.area_px > area_max:
            return False
        return True

    def plausible_jump(d: LarvaDetection) -> bool:
        if not max_jump_px or prev_centroid is None:
            return True
        px, py = prev_centroid
        limit = max_jump_px + max(missing_streak, 0) * jump_growth_px
        if jump_cap_px is not None:
            limit = min(limit, jump_cap_px)
        return math.hypot(d.cx - px, d.cy - py) <= limit

    if len(dets) == 1:
        d0 = dets[0]
        if prev_track_id is not None and d0.track_id == prev_track_id:
            return 0, "single_detection", False
        if prev_centroid is not None and not (in_range(d0) and
                                               plausible_jump(d0)):
            return None, "rejected_implausible", False
        return 0, "single_detection", False

    if prev_track_id is not None:
        for i, d in enumerate(dets):
            if d.track_id is not None and d.track_id == prev_track_id:
                return i, "same_track_id", False

    # Plausible-area + plausible-jump gate (falls back in stages so a
    # genuinely ambiguous frame still yields a choice rather than nothing).
    cand = [i for i, d in enumerate(dets)
            if in_range(d) and plausible_jump(d)]
    if not cand:
        cand = [i for i, d in enumerate(dets) if in_range(d)]
    if not cand:
        cand = list(range(len(dets)))
    elif len(cand) == 1:
        return cand[0], "area_filter", False

    if prev_centroid is not None:
        px, py = prev_centroid
        best = min(cand, key=lambda i: math.hypot(dets[i].cx - px,
                                                   dets[i].cy - py))
        return best, "closest_to_previous", False

    confs = [dets[i].conf for i in cand]
    if max(confs) - sorted(confs)[-2] > 1e-6:  # a clear confidence winner
        best = max(cand, key=lambda i: dets[i].conf)
        return best, "highest_confidence", False

    best = max(cand, key=lambda i: dets[i].area_px)
    return best, "largest_area", True  # ambiguous: no track/centroid/conf signal


def polygon_morphology(poly_px: np.ndarray) -> dict:
    """Shape descriptors for a mask polygon in pixel coordinates (SPECIFICATION.md 9).

    Robust metrics (area, perimeter, compactness, solidity, extent, bbox,
    centroid) come from Shapely. Orientation / axis lengths / eccentricity are
    PCA estimates on the boundary vertices: a proxy, not validated posture
    (SPECIFICATION.md 9 caution). Returns NaN-filled dict for degenerate polygons.
    """
    from shapely.geometry import Polygon

    nan = float("nan")
    empty = {k: nan for k in (
        "cx", "cy", "area_px", "perimeter_px", "compactness", "solidity",
        "extent", "major_axis_px", "minor_axis_px", "aspect_ratio",
        "orientation_rad", "eccentricity", "bbox_w", "bbox_h",
        "bbox_area_px", "bbox_area_ratio")}
    pts = np.asarray(poly_px, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return empty
    poly = Polygon(pts)
    if not poly.is_valid or poly.area <= 0:
        return empty

    area = poly.area
    perim = poly.length
    cx, cy = poly.centroid.x, poly.centroid.y
    hull_area = poly.convex_hull.area
    minx, miny, maxx, maxy = poly.bounds
    bbox_w, bbox_h = maxx - minx, maxy - miny
    bbox_area = bbox_w * bbox_h

    # PCA on centered vertices for orientation / axes / eccentricity.
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    minor_var, major_var = float(evals[0]), float(evals[1])
    major_vec = evecs[:, 1]
    orientation = math.atan2(major_vec[1], major_vec[0])
    major_axis = 4.0 * math.sqrt(major_var) if major_var > 0 else 0.0
    minor_axis = 4.0 * math.sqrt(minor_var) if minor_var > 0 else 0.0
    ecc = math.sqrt(1.0 - minor_var / major_var) if major_var > 0 else 0.0

    return {
        "cx": cx, "cy": cy, "area_px": area, "perimeter_px": perim,
        "compactness": 4 * math.pi * area / perim ** 2 if perim > 0 else nan,
        "solidity": area / hull_area if hull_area > 0 else nan,
        "extent": area / bbox_area if bbox_area > 0 else nan,
        "major_axis_px": major_axis, "minor_axis_px": minor_axis,
        "aspect_ratio": major_axis / minor_axis if minor_axis > 0 else nan,
        "orientation_rad": orientation, "eccentricity": ecc,
        "bbox_w": bbox_w, "bbox_h": bbox_h, "bbox_area_px": bbox_area,
        "bbox_area_ratio": area / bbox_area if bbox_area > 0 else nan,
    }


def canonical_track_id(dish_id: int, raw_id: Optional[int]) -> Optional[int]:
    """dish-namespaced track id (SPECIFICATION.md 26.1)."""
    return None if raw_id is None else dish_id * 1000 + int(raw_id)


def should_detect_petri(frame_idx: int, interval: Optional[int],
                        have_geometry: bool) -> bool:
    """Petri (re)detection schedule (SPECIFICATION.md 5)."""
    if not have_geometry:
        return True
    if interval is None:  # first-frame-only (locked default)
        return False
    return frame_idx % interval == 0


def quality_flag(missing: bool, inside_petri: bool, ambiguous: bool,
                 conf: float, low_conf_threshold: float) -> str:
    """Controlled-vocabulary quality flag (SPECIFICATION.md 15.3)."""
    if missing:
        return "missing_detection"
    if not inside_petri:
        return "outside_petri"
    if ambiguous:
        return "ambiguous_detection"
    if conf < low_conf_threshold:
        return "low_confidence"
    return "ok"


# --------------------------------------------------------------------------- #
# YOLO model + detection IO
# --------------------------------------------------------------------------- #
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def run_larva_inference(model, crop: np.ndarray, args,
                        ocsort=None) -> list[LarvaDetection]:
    """Run the larva segmenter (track or predict) on one dish crop.

    ByteTrack and BoT-SORT are driven by ultralytics itself through
    ``model.track``. OC-SORT is not available there (SPECIFICATION.md 26.1), so it is
    passed in as `ocsort`, an `lt.OCSort` stepped over ``model.predict`` output.
    The caller owns that object -- one per dish -- so its state persists across
    frames the way ultralytics' own per-model tracker state does.
    """
    use_ultralytics_tracker = (args.track
                               and args.tracker in lt.ULTRALYTICS_TRACKERS)
    if use_ultralytics_tracker:
        results = model.track(crop, persist=True,
                              tracker=str(args.tracker_config),
                              conf=args.larva_conf, iou=args.iou,
                              imgsz=args.larva_imgsz,
                              device=args.device, verbose=False)
    else:
        results = model.predict(crop, conf=args.larva_conf, iou=args.iou,
                                imgsz=args.larva_imgsz,
                                device=args.device, verbose=False)
    dets = _parse_larva_results(results)
    # OC-SORT must be stepped on EVERY frame, including ones with no
    # detections: that is where lost tracks age out and where the gap length
    # its observation-centric recovery depends on is counted.
    if ocsort is not None:
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2, d.conf] for d in dets],
                         dtype=float).reshape(-1, 5)
        for d, tid in zip(dets, ocsort.update(boxes)):
            d.track_id = tid
    return dets


def _parse_larva_results(results) -> list[LarvaDetection]:
    """Convert one ultralytics Results object into LarvaDetection records."""
    dets: list[LarvaDetection] = []
    if not results or results[0].boxes is None:
        return dets
    r = results[0]
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    ids = (r.boxes.id.cpu().numpy().astype(int)
           if r.boxes.id is not None else [None] * len(xyxy))
    polys = list(r.masks.xy) if r.masks is not None else [None] * len(xyxy)
    for i, ((x1, y1, x2, y2), c, cl, tid) in enumerate(
            zip(xyxy, confs, clss, ids)):
        if cl != LARVA_CLASS_ID:
            continue
        poly = polys[i] if i < len(polys) else None
        if poly is not None and len(poly) >= 3:
            from shapely.geometry import Polygon
            area = Polygon(poly).area
            cxc, cyc = Polygon(poly).centroid.x, Polygon(poly).centroid.y
        else:
            poly = None
            area = float((x2 - x1) * (y2 - y1))
            cxc, cyc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dets.append(LarvaDetection(
            None if tid is None else int(tid), float(c),
            float(x1), float(y1), float(x2), float(y2),
            float(cxc), float(cyc), float(area),
            None if poly is None else np.asarray(poly, dtype=float)))
    return dets


# --------------------------------------------------------------------------- #
# Per-frame geometry detection
# --------------------------------------------------------------------------- #
@dataclass
class DishGeom:
    dish_id: int
    row: int
    col: int
    box: "cp.PetriBox"
    geom: "cp.CropGeom"
    fit: "Optional[cp.DishFit]" = None


def effective_geometry(dg: "DishGeom") -> tuple[float, float, float]:
    """(cx, cy, radius) used for normalization: the circle-fit when valid, else
    the YOLO box. The crop window always stays box-based (detection is
    unaffected); only the normalization geometry is refined."""
    if dg.fit is not None and dg.fit.valid:
        return dg.fit.cx, dg.fit.cy, dg.fit.radius
    return dg.box.cx, dg.box.cy, dg.box.radius


def detect_dish_geometry(model, frame: np.ndarray, args
                         ) -> tuple[list[DishGeom], bool]:
    """Detect petris on a frame and build per-dish crop geometry."""
    # Ultralytics predict on an in-memory array works directly.
    results = model.predict(frame, conf=args.conf, iou=args.iou,
                            imgsz=args.petri_imgsz,
                            device=args.device, verbose=False)
    boxes: list[cp.PetriBox] = []
    if results and results[0].boxes is not None:
        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
            if cl == cp.PETRI_CLASS_ID:
                boxes.append(cp.PetriBox(float(x1), float(y1),
                                         float(x2), float(y2), float(c)))
    selected, complete = cp.select_best_boxes(boxes, args.rows, args.cols)
    dishes_raw, _ = cp.assign_dish_ids(selected, args.rows, args.cols)
    dishes = [DishGeom(did, row, col, box,
                       cp.compute_crop_geom(box, args.margin))
              for (did, row, col, box) in dishes_raw]
    if getattr(args, "petri_geometry", "box") == "circlefit" and dishes:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for dg in dishes:
            dg.fit = cp.fit_dish_geometry(gray, dg.box)
    return dishes, complete


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
def build_petri_row(ids: dict, frame_idx: int, time_s: float, dg: DishGeom,
                    valid: bool, img_w: int, img_h: int, args, policy: str
                    ) -> dict:
    dist = cp.distortion_proxy(dg.box, img_w, img_h, args.distortion_aspect_tol)
    ecx, ecy, erad = effective_geometry(dg)
    use_fit = dg.fit is not None and dg.fit.valid
    fit = dg.fit
    return {
        **ids, "frame_idx": frame_idx, "time_s": round(time_s, 4),
        "dish_id": dg.dish_id, "row": dg.row, "col": dg.col,
        "petri_track_id": dg.box.track_id, "petri_conf": round(dg.box.conf, 6),
        "petri_x1": round(dg.box.x1, 3), "petri_y1": round(dg.box.y1, 3),
        "petri_x2": round(dg.box.x2, 3), "petri_y2": round(dg.box.y2, 3),
        # petri_cx/cy/radius are the geometry USED for normalization (the
        # circle-fit when valid, else the box); petri_x1..h stay the raw box.
        "petri_cx": round(ecx, 3), "petri_cy": round(ecy, 3),
        "petri_w": round(dg.box.w, 3), "petri_h": round(dg.box.h, 3),
        "petri_radius": round(erad, 3), "petri_valid": valid,
        "crop_x0": dg.geom.x0, "crop_y0": dg.geom.y0, "crop_size": dg.geom.size,
        "petri_detect_policy": policy, **dist,
        "petri_geometry_source": "circlefit" if use_fit else "box",
        "petri_fit_cx": round(fit.cx, 3) if fit else "",
        "petri_fit_cy": round(fit.cy, 3) if fit else "",
        "petri_fit_a": round(fit.a, 3) if fit else "",
        "petri_fit_b": round(fit.b, 3) if fit else "",
        "petri_fit_valid": bool(fit.valid) if fit else False,
    }


def build_larva_row(ids: dict, frame_idx: int, time_s: float, dg: DishGeom,
                    det: Optional[LarvaDetection], reason: str, ambiguous: bool,
                    track_switch: bool, mm_per_px: Optional[float],
                    args, morph: Optional[dict] = None
                    ) -> tuple[dict, Optional[np.ndarray]]:
    """Build one larva_tracks row. Returns (row, mask_polygon_crop|None).

    `morph` is an optional precomputed polygon_morphology() result. It exists so
    an offline replay (scripts/replay_trackers.py) can reuse this row builder --
    and therefore the exact LARVA_COLUMNS contract -- from a detection cache that
    stores morphology scalars instead of the polygons themselves. Left at None
    (every in-pipeline caller) the behaviour is unchanged: morphology is computed
    from det.polygon as before.
    """
    base = {k: "" for k in LARVA_COLUMNS}
    base.update({**ids, "frame_idx": frame_idx, "time_s": round(time_s, 4),
                 "dish_id": dg.dish_id, "crop_x0": dg.geom.x0,
                 "crop_y0": dg.geom.y0, "crop_size": dg.geom.size})

    if det is None:  # missing detection
        base.update({
            "larva_track_id": "", "larva_class_id": "", "larva_conf": "",
            "valid_detection": False, "selection_reason": reason,
            "inside_petri": False, "missing": True, "interpolated": False,
            "track_switch": track_switch, "outlier_position": False,
            "outlier_area": False,
            "quality_flag": quality_flag(True, False, False, 0.0,
                                         args.low_conf_threshold)})
        return base, None

    cx_g = dg.geom.x0 + det.cx
    cy_g = dg.geom.y0 + det.cy
    ecx, ecy, erad = effective_geometry(dg)
    norm = petri_normalize(cx_g, cy_g, ecx, ecy, erad)

    if morph is None and det.polygon is not None:
        morph = polygon_morphology(det.polygon)
    petri_area = math.pi * erad ** 2

    base.update({
        "larva_track_id": canonical_track_id(dg.dish_id, det.track_id),
        "larva_class_id": LARVA_CLASS_ID, "larva_conf": round(det.conf, 6),
        "valid_detection": True, "selection_reason": reason,
        "cx_crop_raw": round(det.cx, 3), "cy_crop_raw": round(det.cy, 3),
        "x1_crop": round(det.x1, 3), "y1_crop": round(det.y1, 3),
        "x2_crop": round(det.x2, 3), "y2_crop": round(det.y2, 3),
        "bbox_w_crop": round(det.x2 - det.x1, 3),
        "bbox_h_crop": round(det.y2 - det.y1, 3),
        "cx_global_raw": round(cx_g, 3), "cy_global_raw": round(cy_g, 3),
        "x1_global": round(dg.geom.x0 + det.x1, 3),
        "y1_global": round(dg.geom.y0 + det.y1, 3),
        "x2_global": round(dg.geom.x0 + det.x2, 3),
        "y2_global": round(dg.geom.y0 + det.y2, 3),
        "bbox_w_global": round(det.x2 - det.x1, 3),
        "bbox_h_global": round(det.y2 - det.y1, 3),
        "x_norm_raw": round(norm["x_norm"], 6),
        "y_norm_raw": round(norm["y_norm"], 6),
        "r_norm_raw": round(norm["r_norm"], 6),
        "theta_rad_raw": round(norm["theta_rad"], 6),
        "inside_petri": norm["inside_petri"],
        "missing": False, "interpolated": False, "track_switch": track_switch,
        "outlier_position": False, "outlier_area": False,
        "quality_flag": quality_flag(False, norm["inside_petri"], ambiguous,
                                     det.conf, args.low_conf_threshold),
    })

    if morph is not None:
        area = morph["area_px"]
        base.update({
            "mask_area_px": round(area, 3),
            "mask_area_norm_petri": round(area / petri_area, 6)
            if petri_area > 0 else "",
            "mask_area_mm2": round(area * mm_per_px ** 2, 6)
            if mm_per_px else "",
            "mask_perimeter_px": round(morph["perimeter_px"], 3),
            "mask_compactness": round(morph["compactness"], 6),
            "mask_solidity": round(morph["solidity"], 6),
            "mask_extent": round(morph["extent"], 6),
            "mask_major_axis_px": round(morph["major_axis_px"], 3),
            "mask_minor_axis_px": round(morph["minor_axis_px"], 3),
            "mask_aspect_ratio": round(morph["aspect_ratio"], 6),
            "mask_orientation_rad": round(morph["orientation_rad"], 6),
            "mask_eccentricity": round(morph["eccentricity"], 6),
            "bbox_area_px": round(morph["bbox_area_px"], 3),
            "mask_bbox_area_ratio": round(morph["bbox_area_ratio"], 6),
        })
    return base, det.polygon


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def write_provenance(out: Path, args, n_frames: int) -> None:
    # idempotent: main() has already resolved this, but resolving again keeps
    # write_provenance callable on a raw parsed Namespace.
    tracker_cfg_path = lt.resolve_tracker_config(args.tracker,
                                                 args.tracker_config)
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_track"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/track_larvae_video.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.video)],
        "output_path": str(args.out),
        "petri_model": str(args.petri_model),
        "petri_model_sha256": cp.sha256_file(args.petri_model),
        "larva_model": str(args.larva_model),
        "larva_model_sha256": cp.sha256_file(args.larva_model),
        "n_frames_processed": n_frames,
        "parameters": {
            "rows": args.rows, "cols": args.cols, "margin": args.margin,
            "conf": args.conf, "larva_conf": args.larva_conf, "iou": args.iou,
            "track": args.track, "tracker": args.tracker,
            "tracker_config": str(tracker_cfg_path),
            "tracker_config_sha256": cp.sha256_file(tracker_cfg_path),
            "tracker_params": lt.load_tracker_config(
                args.tracker, tracker_cfg_path),
            # primary-larva selection policy (SPECIFICATION.md 6)
            "low_conf_threshold": args.low_conf_threshold,
            "area_min_px": args.area_min_px, "area_max_px": args.area_max_px,
            "max_jump_px": args.max_jump_px,
            "max_jump_growth_px": args.max_jump_growth_px,
            "max_jump_cap_px": args.max_jump_cap_px,
            "petri_diameter_mm": args.petri_diameter_mm,
            "distortion_aspect_tol": args.distortion_aspect_tol,
            "petri_geometry": args.petri_geometry,
            "petri_imgsz": args.petri_imgsz, "larva_imgsz": args.larva_imgsz,
            "detect_petri_interval": _detect_interval(args),
            # reserved for the kinematics step (SPECIFICATION.md 26.3):
            "smoothing_window": args.smoothing_window,
            "immobility_speed_threshold_mm_s":
                args.immobility_speed_threshold_mm_s,
        },
    }
    with open(out / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def _detect_interval(args) -> Optional[int]:
    if args.detect_petri_every_frame:
        return 1
    if args.detect_petri_every_n_frames is not None:
        return args.detect_petri_every_n_frames
    return None  # first-frame-only (locked default, SPECIFICATION.md 26.1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--petri-model", required=True, type=Path)
    p.add_argument("--larva-model", required=True, type=Path)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--petri-diameter-mm", type=float, default=None)
    p.add_argument("--track", action="store_true")
    p.add_argument("--tracker", type=lt.normalize_tracker_name,
                   choices=lt.TRACKERS, default="bytetrack",
                   help="association algorithm (SPECIFICATION.md 26.1): 'bytetrack' "
                        "(locked v1 default, every released dataset was "
                        "tracked with it), 'botsort' (ultralytics BoT-SORT "
                        "with global motion compensation off) or 'ocsort' "
                        "(OC-SORT; ultralytics does not ship it, so it runs "
                        "on predict() output via scripts/larva_trackers.py). "
                        "A legacy '<name>.yaml' value is normalized to <name>")
    p.add_argument("--tracker-config", type=Path, default=None,
                   help="override the shipped config/trackers/<tracker>.yaml; "
                        "the resolved path and its SHA-256 are recorded in "
                        "provenance either way")
    p.add_argument("--conf", type=float, default=0.30, help="Petri conf")
    # Pinned explicitly: this model's box size is imgsz-sensitive (boxes match
    # the true dish at 640 but shrink ~24% at 1280), so the inference imgsz must
    # not be left to a library default that could change.
    p.add_argument("--petri-imgsz", type=int, default=640,
                   help="inference imgsz for petri detection (pinned)")
    p.add_argument("--larva-imgsz", type=int, default=640,
                   help="inference imgsz for larva detection/tracking (pinned)")
    p.add_argument("--larva-conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--low-conf-threshold", type=float, default=0.40)
    p.add_argument("--distortion-aspect-tol", type=float, default=0.15)
    p.add_argument("--petri-geometry", choices=["box", "circlefit"],
                   default="box",
                   help="normalization geometry: 'box' (YOLO box, locked v1) or "
                        "'circlefit' (ellipse-fit the dish; accurate center/"
                        "radius, falls back to box when the fit fails)")
    p.add_argument("--area-min-px", type=float, default=None)
    p.add_argument("--area-max-px", type=float, default=None)
    p.add_argument("--max-jump-px", type=float, default=60.0,
                   help="hysteresis gate: a lone new-identity detection must "
                        "land within max-jump-px + missing_streak * "
                        "max-jump-growth-px (capped at max-jump-cap-px) of "
                        "the previous centroid to be accepted as primary; "
                        "otherwise the frame is treated as missing instead "
                        "of letting a transient artifact/mask fragment "
                        "hijack the track. 0/None disables the gate")
    p.add_argument("--max-jump-growth-px", type=float, default=5.0,
                   help="extra jump allowance per consecutive missing frame "
                        "(additive, not multiplicative -- see --max-jump-px)")
    p.add_argument("--max-jump-cap-px", type=float, default=240.0,
                   help="absolute ceiling on the jump gate regardless of how "
                        "long the track has been unconfirmed")
    # Petri detection policy (SPECIFICATION.md 5)
    p.add_argument("--detect-petri-first-frame", action="store_true",
                   default=True)
    p.add_argument("--detect-petri-every-frame", action="store_true")
    p.add_argument("--detect-petri-every-n-frames", type=int, default=None)
    # outputs
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    # reserved for the kinematics step (SPECIFICATION.md 26.3), recorded in provenance
    p.add_argument("--smoothing-window", type=int, default=7)
    p.add_argument("--immobility-speed-threshold-mm-s", type=float, default=1.0)
    # identifiers / limits
    p.add_argument("--dataset-id", type=str, default="wormstress_tmolitor_v1")
    p.add_argument("--experiment-id", type=str, default="")
    p.add_argument("--video-id", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    import cv2

    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    video_id = args.video_id or args.video.stem
    ids = {"dataset_id": args.dataset_id, "experiment_id": args.experiment_id,
           "video_id": video_id}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = _detect_interval(args)
    policy = ("every_frame" if interval == 1 else
              f"every_{interval}_frames" if interval else "first_frame")

    args.tracker_config = lt.resolve_tracker_config(args.tracker,
                                                    args.tracker_config)
    tracker_cfg = lt.load_tracker_config(args.tracker, args.tracker_config)
    print(f"Video {video_id}: fps={fps:.3f}, petri policy={policy}, "
          f"track={args.track}, tracker={args.tracker} "
          f"({args.tracker_config})")
    petri_model = load_model(str(args.petri_model))
    # One larva tracker instance per dish keeps tracker state independent
    # across dishes (SPECIFICATION.md 26.1).
    n_dishes = args.rows * args.cols
    larva_models = {d: load_model(str(args.larva_model))
                    for d in range(1, n_dishes + 1)}
    # ByteTrack/BoT-SORT state lives inside each ultralytics model above;
    # OC-SORT state is ours, so it needs the same per-dish treatment.
    ocsorts = ({d: lt.ocsort_from_config(tracker_cfg)
                for d in range(1, n_dishes + 1)}
               if args.track and args.tracker == "ocsort" else {})

    petri_fh = open(args.out / "petri_tracks.csv", "w", newline="")
    larva_fh = open(args.out / "larva_tracks.csv", "w", newline="")
    det_fh = open(args.out / "larva_detections_all.csv", "w", newline="")
    masks_fh = (open(args.out / "larva_masks.jsonl", "w")
                if args.save_masks else None)
    petri_w = csv.DictWriter(petri_fh, fieldnames=PETRI_COLUMNS)
    larva_w = csv.DictWriter(larva_fh, fieldnames=LARVA_COLUMNS)
    det_w = csv.DictWriter(det_fh, fieldnames=DETECTIONS_COLUMNS)
    petri_w.writeheader(); larva_w.writeheader(); det_w.writeheader()

    overlay_writer = None
    dishes: list[DishGeom] = []
    geom_valid = False
    prev: dict[int, dict] = {d: {"track_id": None, "centroid": None,
                                  "missing_streak": 0}
                             for d in range(1, n_dishes + 1)}
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            img_h, img_w = frame.shape[:2]
            time_s = frame_idx / fps

            if should_detect_petri(frame_idx, interval, bool(dishes)):
                new_dishes, complete = detect_dish_geometry(
                    petri_model, frame, args)
                if complete or not dishes:
                    dishes = new_dishes
                    geom_valid = complete

            if not dishes:  # no geometry yet; cannot crop
                frame_idx += 1
                continue

            for dg in dishes:
                petri_w.writerow(build_petri_row(
                    ids, frame_idx, time_s, dg, geom_valid,
                    img_w, img_h, args, policy))

            for dg in dishes:
                crop = cp.extract_square_crop(frame, dg.geom)
                dets = run_larva_inference(larva_models[dg.dish_id], crop,
                                           args, ocsorts.get(dg.dish_id))
                pstate = prev[dg.dish_id]
                idx, reason, ambiguous = select_primary_larva(
                    dets, pstate["track_id"], pstate["centroid"],
                    args.area_min_px, args.area_max_px,
                    max_jump_px=args.max_jump_px,
                    missing_streak=pstate["missing_streak"],
                    jump_growth_px=args.max_jump_growth_px,
                    jump_cap_px=args.max_jump_cap_px)

                _, _, erad = effective_geometry(dg)
                mm = (args.petri_diameter_mm / (2 * erad)
                      if args.petri_diameter_mm and erad > 0 else None)
                chosen = dets[idx] if idx is not None else None
                new_tid = canonical_track_id(
                    dg.dish_id, chosen.track_id) if chosen else None
                track_switch = (chosen is not None and new_tid is not None
                                and pstate["track_id"] is not None
                                and new_tid != pstate["track_id"])

                row, poly = build_larva_row(
                    ids, frame_idx, time_s, dg, chosen, reason, ambiguous,
                    track_switch, mm, args)
                larva_w.writerow(row)

                # all raw detections (SPECIFICATION.md 8.3)
                for j, d in enumerate(dets):
                    det_w.writerow({
                        **ids, "frame_idx": frame_idx,
                        "time_s": round(time_s, 4), "dish_id": dg.dish_id,
                        "raw_detection_id": j,
                        "yolo_track_id": d.track_id, "class_id": LARVA_CLASS_ID,
                        "conf": round(d.conf, 6),
                        "x1_crop": round(d.x1, 3), "y1_crop": round(d.y1, 3),
                        "x2_crop": round(d.x2, 3), "y2_crop": round(d.y2, 3),
                        "cx_crop": round(d.cx, 3), "cy_crop": round(d.cy, 3),
                        "mask_area_px": round(d.area_px, 3),
                        "selected_primary": (j == idx),
                        "selection_reason": reason if j == idx else "",
                    })

                if masks_fh is not None and poly is not None:
                    masks_fh.write(json.dumps({
                        **ids, "frame_idx": frame_idx,
                        "time_s": round(time_s, 4), "dish_id": dg.dish_id,
                        "larva_track_id": new_tid,
                        "mask_format": "polygon_crop",
                        "polygon_crop": np.asarray(poly).round(2).tolist(),
                        "mask_area_px": round(float(chosen.area_px), 3),
                    }) + "\n")

                # update tracker state
                if chosen is not None:
                    prev[dg.dish_id] = {
                        "track_id": new_tid,
                        "centroid": (chosen.cx, chosen.cy),
                        "missing_streak": 0}
                else:
                    pstate["missing_streak"] = pstate["missing_streak"] + 1

                if args.save_overlays:
                    bx = dg.geom
                    cv2.rectangle(frame, (int(dg.box.x1), int(dg.box.y1)),
                                  (int(dg.box.x2), int(dg.box.y2)),
                                  (0, 255, 0), 1)
                    if chosen is not None:
                        cv2.circle(frame, (int(bx.x0 + chosen.cx),
                                           int(bx.y0 + chosen.cy)),
                                   4, (0, 0, 255), -1)

            if args.save_overlays:
                if overlay_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    overlay_writer = cv2.VideoWriter(
                        str(args.out / f"{video_id}_overlay.mp4"),
                        fourcc, fps, (img_w, img_h))
                overlay_writer.write(frame)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  processed {frame_idx} frames")
    finally:
        cap.release()
        petri_fh.close(); larva_fh.close(); det_fh.close()
        if masks_fh is not None:
            masks_fh.close()
        if overlay_writer is not None:
            overlay_writer.release()

    write_provenance(args.out, args, frame_idx)
    print(f"\nDone. Processed {frame_idx} frames across {n_dishes} dishes.")
    print(f"Outputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
