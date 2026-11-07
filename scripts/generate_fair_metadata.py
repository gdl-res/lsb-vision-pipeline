#!/usr/bin/env python3
"""Generate FAIR dataset-release metadata (SPECIFICATION.md 14, 15, 16, 17).

Emits the reusability scaffold for a processed dataset release:

    dataset_description.json   (15.1 Findable)
    data_dictionary.csv        (15.4 every column documented)
    processing_config.yaml      (locked v1 defaults, 26)
    README.md
    LICENSE                    (placeholder pointer)
    requirements.txt
    software_environment.txt   (pip freeze)
    metadata/experiment_metadata.csv  (template header, 14)
    model_cards/petri_detector_model_card.md   (17)
    model_cards/larva_segmenter_model_card.md  (17)

The data dictionary is generated from the real output-column lists imported from
the pipeline scripts, so it cannot silently drift from the actual schemas.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import crop_petris_yolo_seg as cp
import track_larvae_video as tl
import larva_trackers as lt
import compute_kinematics as ck
import compute_behavior_features as cf
import compute_divergence as dv

# --------------------------------------------------------------------------- #
# Schemas to document. QC + experiment_metadata are defined here (their
# producers are documented by SPECIFICATION.md 13 / 14); the rest are imported.
# --------------------------------------------------------------------------- #
QC_COLUMNS = [
    "dataset_id", "experiment_id", "video_id", "dish_id", "petri_valid",
    "petri_conf", "valid_detection_fraction", "mean_larva_conf",
    "median_larva_conf", "missing_frame_count", "longest_missing_gap_s",
    "track_switch_count", "track_switch_rate_per_min",
    "semantic_switch_count", "semantic_switch_rate_per_min",
    "fraction_outside_petri",
    "rim_occlusion_fraction", "mean_mask_area_px", "mask_area_outlier_fraction",
    "morphology_reliability", "position_outlier_fraction",
    "interpolated_fraction", "suspected_empty",
    "overall_quality_score", "exclude_recommended", "exclude_reason",
]
EXPERIMENT_METADATA_COLUMNS = [
    "dataset_id", "experiment_id", "video_id", "dish_id", "larva_id",
    "batch_id", "date", "camera_id", "phone_model", "fps", "frame_width",
    "frame_height", "petri_diameter_mm", "treatment", "stressor",
    "stressor_intensity", "stressor_duration_s", "recovery_time_s",
    "temperature_c", "humidity_percent", "light_condition",
    "handling_protocol", "recording_start_time", "recording_duration_s",
    "operator", "notes",
]
MASKS_JSONL_KEYS = [
    "dataset_id", "experiment_id", "video_id", "frame_idx", "time_s",
    "dish_id", "larva_track_id", "mask_format", "polygon_crop", "mask_area_px",
]

FILES = [
    ("petri/petri_tracks.csv", tl.PETRI_COLUMNS),
    ("larva/larva_tracks.csv", tl.LARVA_COLUMNS),
    ("larva/larva_detections_all.csv", tl.DETECTIONS_COLUMNS),
    ("larva/larva_masks.jsonl", MASKS_JSONL_KEYS),
    ("kinematics/larva_kinematics.csv", ck.OUTPUT_COLUMNS),
    ("features/features_per_window.csv", cf.WINDOW_COLUMNS),
    ("features/features_per_video.csv", cf.VIDEO_COLUMNS),
    ("features/divergence_from_control.csv", dv.DETAIL_COLUMNS),
    ("qc/qc_summary.csv", QC_COLUMNS),
    ("metadata/experiment_metadata.csv", EXPERIMENT_METADATA_COLUMNS),
]

# --------------------------------------------------------------------------- #
# Controlled vocabularies (SPECIFICATION.md 15.3)
# --------------------------------------------------------------------------- #
VOCAB = {
    "movement_state": "moving|immobile|missing|interpolated|uncertain",
    "quality_flag": ("ok|low_confidence|missing_detection|track_switch|"
                     "outside_petri|area_outlier|ambiguous_detection"),
    "selection_reason": ("single_detection|same_track_id|highest_confidence|"
                         "closest_to_previous|area_filter|largest_area|"
                         "no_detection|ambiguous|rejected_implausible|"
                         "reacquired"),
    "treatment": ("control|sham|heat|cold|starvation|dehydration|"
                  "mechanical_disturbance|light_exposure|crowding|unknown"),
    "petri_detect_policy": "first_frame|every_frame|every_N_frames",
    "mask_format": "polygon_crop",
    "exclude_reason": "free text or empty",
    "divergence_status": dv.STATUS_VOCAB,
}

# Columns whose description/unit/type are worth stating explicitly.
EXPLICIT: dict[str, tuple[str, str, str]] = {
    # column: (data_type, unit, description)
    "dataset_id": ("string", "", "Stable dataset identifier."),
    "experiment_id": ("string", "", "Experiment identifier (exp_YYYYMMDD_condition_batch)."),
    "video_id": ("string", "", "Stable source video/image identifier (filename stem or UUID)."),
    "larva_id": ("string", "", "Stable larva identifier within an experiment (curated)."),
    "dish_id": ("integer", "", "Petri dish 1..N in row-major order."),
    "row": ("integer", "", "Grid row index (0-based)."),
    "col": ("integer", "", "Grid column index (0-based)."),
    "frame_idx": ("integer", "", "0-based frame index within the video."),
    "time_s": ("float", "seconds", "Frame timestamp = frame_idx / fps."),
    "petri_track_id": ("integer", "", "YOLO Petri track id, if Petri tracking was used."),
    "larva_track_id": ("integer", "", "Canonical larva track id = dish_id*1000 + raw YOLO id; stable within a dish within a video."),
    "yolo_track_id": ("integer", "", "Raw per-crop YOLO track id (not dish-namespaced)."),
    "petri_radius": ("float", "pixels", "Normalization radius. For petri_geometry_source=box: 0.5*min(petri_w, petri_h); for circlefit: the fitted dish mean semi-axis."),
    "petri_geometry_source": ("string", "", "Geometry used for petri_cx/cy/radius and larva normalization: 'box' (YOLO box) or 'circlefit' (ellipse fitted to the dish)."),
    "petri_fit_cx": ("float", "pixels", "Circle-fit dish center x (full-frame); empty when not fitted."),
    "petri_fit_cy": ("float", "pixels", "Circle-fit dish center y (full-frame); empty when not fitted."),
    "petri_fit_a": ("float", "pixels", "Circle-fit major semi-axis; empty when not fitted."),
    "petri_fit_b": ("float", "pixels", "Circle-fit minor semi-axis; empty when not fitted. petri_fit_a/petri_fit_b is the measured perspective-distortion ratio (SPECIFICATION.md 26.2)."),
    "petri_fit_valid": ("boolean", "", "True when the dish ellipse-fit succeeded; petri geometry uses the fit only when True."),
    "petri_aspect_ratio": ("float", "unitless", "petri_w / petri_h; perspective-distortion proxy."),
    "petri_distortion_flag": ("boolean", "", "True when |1 - aspect_ratio| exceeds the tolerance."),
    "petri_offset_norm": ("float", "unitless", "Distance of dish center from frame center, / half frame diagonal."),
    "x_norm_raw": ("float", "petri radii", "Petri-normalized x (center 0, edge 1)."),
    "y_norm_raw": ("float", "petri radii", "Petri-normalized y (center 0, edge 1)."),
    "r_norm_raw": ("float", "petri radii", "Radial distance from dish center in Petri radii."),
    "theta_rad_raw": ("float", "radians", "Angular position atan2(y_norm, x_norm)."),
    "inside_petri": ("boolean", "", "True when r_norm <= 1."),
    "mask_area_norm_petri": ("float", "unitless", "mask_area_px / (pi * petri_radius_px^2)."),
    "acceleration_mm_s2": ("float", "mm/s^2", "Derived (2nd derivative); flagged low-confidence."),
    "heatmap_entropy": ("float", "nats", "Shannon entropy of the occupancy probability (natural log)."),
    "occupancy_entropy": ("float", "nats", "Shannon entropy of windowed occupancy (natural log)."),
    "js_divergence_from_control": ("float", "nats", "Jensen-Shannon divergence (natural log, in [0, ln 2]) of the dish occupancy heatmap from the pooled control reference for its experiment_id. NaN when no control reference exists. Deferred post-process (SPECIFICATION.md 26.4)."),
    "wasserstein_distance_from_control": ("float", "petri radii", "Wasserstein distance of the radial occupancy marginal from the pooled control reference. NaN when no control reference exists. Deferred post-process (SPECIFICATION.md 26.4)."),
    "n_control_dishes_ref": ("integer", "", "Number of control dishes pooled into the reference heatmap for this row."),
    "reference_valid_points": ("integer", "", "Total occupancy counts in the pooled control reference heatmap."),
    "divergence_status": ("string", "", "Outcome of divergence scoring (controlled vocabulary)."),
    "polygon_crop": ("array", "pixels", "List of [x, y] mask vertices in crop coordinates."),
    "mask_format": ("string", "", "Mask encoding."),
    "petri_diameter_mm": ("float", "millimeters", "Real Petri dish diameter for mm calibration."),
    "treatment": ("string", "", "Experimental treatment (controlled vocabulary)."),
    "exclude_recommended": ("boolean", "", "QC recommendation to exclude this video x dish."),
    "suspected_empty": ("boolean", "", "True when the dish's detections look like a fixed dish-surface artifact rather than a larva: sub-larva mean mask area AND low mean confidence AND near-zero positional spread, all together (thresholds --empty-max-area-norm/-conf/-spread). Distinguishes 'no subject / larva not loaded' from 'larva present but poorly tracked'; contributes the 'suspected_empty' exclude_reason. Conservative -- a motionless real larva still has larva-sized area and normal confidence, so it is not flagged."),
    "morphology_reliability": ("float", "unitless [0,1]", "Fraction of this dish's valid-detection frames that carry a usable larva mask (mask produced and not grossly fragmented: solidity >= --morphology-min-solidity). A per-dish confidence weight for the morphology features (mask_area/eccentricity/compactness/...): they are only as good as the masks they come from, so low values mean sparse/noisier morphology. REPORTED ONLY -- never used for exclusion (a dish with weak masks can still have a clean trajectory). Note: this is a RELATIVE per-dish signal; it does not capture the segmenter's absolute boundary-noise ceiling (mask mAP@50-95 ~0.70 on validation), which caps ALL morphology values globally."),
    "track_switch_rate_per_min": ("float", "1/min", "track_switch_count divided by the dish observation duration in minutes. Nominal ByteTrack ID-switch rate: counts every re-acquisition of the same animal after a detection-flicker frame, so for <=1 larva/dish it is mostly benign churn. DIAGNOSTIC ONLY -- the exclusion rule gates on semantic_switch_rate_per_min instead; the nominal rate is opt-in via --max-track-switch-rate-per-min."),
    "semantic_switch_count": ("integer", "", "Number of physically-implausible single-frame position jumps: steps between frame-adjacent raw detections exceeding max_step_norm (default 0.5 petri radii/frame, far faster than a larva crawls). Marks the selected primary teleporting to a different physical object (artifact/mask fragment) -- genuine trajectory corruption, unlike the nominal track_switch_count."),
    "semantic_switch_rate_per_min": ("float", "1/min", "semantic_switch_count divided by the dish observation duration in minutes; the length-independent trajectory-integrity signal the exclusion rule gates on (threshold --max-semantic-switch-rate-per-min)."),
    "rim_occlusion_fraction": ("float", "unitless", "Fraction of all frames that are missing immediately after the larva was last seen at/near the rim (last-seen r_norm >= rim_occlusion_r, default 0.85); attributes missingness to the larva tucking under the dish edge (real behavior) rather than mid-dish detector failure. Always <= missing_fraction. Present in both qc_summary and the behavioral features."),
    "rim_occlusion_event_count": ("integer", "", "Number of distinct rim-occlusion episodes (contiguous attributed gaps) -- how many times the larva went under the dish edge in the window/video. A rim-seeking/thigmotaxis behavioral signal."),
}


# --------------------------------------------------------------------------- #
# Per-column metadata inference
# --------------------------------------------------------------------------- #
_BOOL_HINTS = ("is_", "valid", "inside_", "missing", "interpolated",
               "track_switch", "outlier_", "exclude_", "selected_primary",
               "petri_valid", "_flag")
_INT_SUFFIX = ("_count", "_idx", "_cells")
_INT_NAMES = {"dish_id", "row", "col", "bins", "window_id", "n_frames",
              "raw_detection_id", "frame_width", "frame_height"}


def _data_type(col: str) -> str:
    if col in EXPLICIT:
        return EXPLICIT[col][0]
    low = col.lower()
    if any(h in low for h in _BOOL_HINTS):
        return "boolean"
    if col in _INT_NAMES or any(low.endswith(s) for s in _INT_SUFFIX):
        return "integer"
    if low.endswith(("_id", "_reason", "_state", "_flag_reason", "policy",
                     "protocol", "_model", "condition", "operator", "notes")):
        return "string"
    return "float"


def _unit(col: str) -> str:
    if col in EXPLICIT and EXPLICIT[col][1]:
        return EXPLICIT[col][1]
    low = col.lower()
    # *_smooth / *_raw share the unit of their base coordinate; strip the
    # variant suffix so it does not mask the coordinate-system suffix.
    for variant in ("_smooth", "_raw"):
        if low.endswith(variant):
            low = low[: -len(variant)]
            break
    rules = [
        ("_mm2", "mm^2"), ("_mm_s2", "mm/s^2"), ("_mm_s", "mm/s"),
        ("_mm", "millimeters"), ("_px_s", "pixels/s"), ("_px", "pixels"),
        ("_rad_s", "rad/s"), ("_rad", "radians"),
        ("_norm_s", "petri radii/s"), ("_norm", "petri radii"),
        ("_global", "pixels"),
        ("temperature", "Celsius"), ("humidity", "percent"),
        ("fps", "frames/s"),
    ]
    for suffix, unit in rules:
        if low.endswith(suffix) or low.startswith(suffix.strip("_")):
            return unit
    if low.endswith(("_fraction", "_efficiency", "_cv")) or "conf" in low:
        return "unitless [0,1]"
    if low.endswith(("_s", "duration_s", "gap_s")) or low == "time_s":
        return "seconds"
    if "entropy" in low:
        return "nats"
    return "unitless"


def _missing_policy(dtype: str) -> str:
    return {"boolean": "true/false", "string": "empty string",
            "array": "empty list"}.get(dtype, "NaN")


def _description(col: str) -> str:
    if col in EXPLICIT:
        return EXPLICIT[col][2]
    return col.replace("_", " ").strip().capitalize() + "."


# Columns produced by the configurable position-smoothing filter
# (compute_kinematics.py --smoothing-filter).
SMOOTH_COLUMNS = ("cx_global_smooth", "cy_global_smooth",
                  "x_norm_smooth", "y_norm_smooth", "r_norm_smooth")


def smoothing_meta(args) -> tuple[dict, str]:
    """Return (processing_config block, human description) for the chosen filter.

    Mirrors compute_kinematics.smooth_series so the FAIR metadata documents the
    filter that actually produced the *_smooth columns.
    """
    method = getattr(args, "smoothing_filter", "savgol")
    if method == "savgol":
        cfg = {"method": "savitzky_golay",
               "window": args.smoothing_window,
               "polyorder": args.savgol_polyorder}
        desc = (f"Savitzky-Golay (window {args.smoothing_window}, "
                f"polyorder {args.savgol_polyorder})")
    elif method == "moving_average":
        cfg = {"method": "moving_average", "window": args.smoothing_window}
        desc = f"centered moving average (window {args.smoothing_window})"
    elif method == "kalman":
        cfg = {"method": "kalman",
               "process_var": args.kalman_process_var,
               "measurement_var": args.kalman_measurement_var}
        desc = (f"constant-velocity Kalman/RTS (process_var "
                f"{args.kalman_process_var}, measurement_var "
                f"{args.kalman_measurement_var})")
    elif method == "none":
        cfg = {"method": "none"}
        desc = "no smoothing (raw interpolated coordinates)"
    else:
        raise ValueError(f"unknown smoothing filter: {method!r}")
    return cfg, desc


def _smooth_descriptions(filter_desc: str) -> dict:
    """Per-column descriptions for the *_smooth columns naming the filter."""
    base = {
        "cx_global_smooth": "Smoothed global x",
        "cy_global_smooth": "Smoothed global y",
        "x_norm_smooth": "Smoothed Petri-normalized x (center 0, edge 1)",
        "y_norm_smooth": "Smoothed Petri-normalized y (center 0, edge 1)",
        "r_norm_smooth": "Smoothed radial distance in Petri radii",
    }
    return {c: f"{t}; filter: {filter_desc}." for c, t in base.items()}


def build_data_dictionary(filter_desc: Optional[str] = None) -> list[dict]:
    smooth_desc = _smooth_descriptions(filter_desc) if filter_desc else {}
    rows = []
    for file_path, columns in FILES:
        for col in columns:
            dtype = _data_type(col)
            desc = smooth_desc.get(col) or _description(col)
            rows.append({
                "file": file_path,
                "column_name": col,
                "data_type": dtype,
                "unit": _unit(col),
                "description": desc,
                "allowed_values": VOCAB.get(col, ""),
                "missing_value_policy": _missing_policy(dtype),
            })
    return rows


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_data_dictionary(path: Path, filter_desc: Optional[str] = None) -> int:
    rows = build_data_dictionary(filter_desc)
    fields = ["file", "column_name", "data_type", "unit", "description",
              "allowed_values", "missing_value_policy"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_dataset_description(path: Path, args) -> None:
    desc = {
        "dataset_id": args.dataset_id,
        "dataset_title": args.title,
        "dataset_version": args.version,
        "description": ("Automated behavioral-phenotyping dataset for "
                        "Tenebrio molitor larvae monitored in Petri dishes. "
                        "Contains framewise trajectory and area-based "
                        "behavioral metrics; not a measure of welfare or "
                        "subjective state."),
        "creators": [c.strip() for c in args.creators.split(";") if c.strip()],
        "affiliations": [a.strip() for a in args.affiliations.split(";")
                         if a.strip()],
        "contact_email": args.contact,
        "creation_date": date.today().isoformat(),
        "license": args.license,
        "data_license": args.license,
        "code_license": args.code_license,
        "keywords": ["automated behavioral phenotyping",
                     "Tenebrio molitor", "stress-associated behavioral deviation",
                     "non-invasive behavioral monitoring", "FAIR"],
        "species": args.species,
        "experimental_context": ("Tray of Petri dishes imaged top-down; one "
                                  "larva per dish; YOLO Petri detection + YOLO "
                                  "larva segmentation/tracking."),
        "identifiers": {
            "dataset_id": args.dataset_id,
            "experiment_id": "exp_YYYYMMDD_condition_batch",
            "video_id": "stable filename stem or UUID",
            "larva_id": "stable within experiment",
            "dish_id": "1-6 row-major",
        },
    }
    path.write_text(json.dumps(desc, indent=2))


def tracking_meta(args) -> dict:
    """The larva_tracking block of processing_config.yaml, from the run's args.

    Records the association algorithm actually used plus the resolved parameter
    values, so a released dataset states its tracker rather than assuming the
    v1 default (SPECIFICATION.md 26.1).
    """
    tracker = getattr(args, "tracker", "bytetrack")
    override = getattr(args, "tracker_config", None)
    cfg_path = lt.resolve_tracker_config(tracker, override).resolve()
    repo_root = lt.CONFIG_DIR.parents[1]
    try:  # keep shipped configs repo-relative so the record is portable
        cfg_str = str(cfg_path.relative_to(repo_root))
    except ValueError:  # an override from outside the repo
        cfg_str = str(cfg_path)
    larva_model = getattr(args, "larva_model", None)
    reacq = getattr(args, "reacquire_after_s", tl.DEFAULT_REACQUIRE_AFTER_S)
    return {"larva_model": None if larva_model is None else Path(larva_model).name,
            "tracker": tracker,
            "tracker_config": cfg_str,
            "tracker_params": lt.load_tracker_config(tracker, cfg_path),
            "persist": True, "per_dish_instance": True, "larva_conf": 0.25,
            "track_id_rule": "dish_id*1000 + raw_yolo_id",
            # primary-larva selection gate (SPECIFICATION.md 6)
            "max_jump_px": getattr(args, "max_jump_px", 60.0),
            "max_jump_growth_px": getattr(args, "max_jump_growth_px", 5.0),
            "max_jump_cap_px": getattr(args, "max_jump_cap_px", 240.0),
            "reacquire_after_s": reacq or None}


def write_processing_config(path: Path, smoothing_cfg: Optional[dict] = None,
                            tracking_cfg: Optional[dict] = None) -> None:
    import yaml
    if smoothing_cfg is None:
        smoothing_cfg = {"method": "savitzky_golay", "window": 7,
                         "polyorder": 2}
    if tracking_cfg is None:
        tracking_cfg = {"tracker": "bytetrack", "persist": True,
                        "per_dish_instance": True, "larva_conf": 0.25,
                        "track_id_rule": "dish_id*1000 + raw_yolo_id"}
    config = {
        "pipeline_version": tl.PIPELINE_VERSION,
        "locked_decisions_ref": "SPECIFICATION.md section 26",
        "petri_detection": {"rows": 3, "cols": 2, "margin": 0.05,
                            "conf": 0.30, "iou": 0.50,
                            "distortion_aspect_tol": 0.15,
                            "detect_policy": "first_frame"},
        "larva_tracking": tracking_cfg,
        "interpolation": {"max_interp_gap_s": 0.5, "method": "linear"},
        "smoothing": smoothing_cfg,
        "kinematics": {"jerk": "dropped",
                       "acceleration": "kept_low_confidence",
                       "immobility_speed_threshold_mm_s": 1.0},
        "heatmaps": {"bins": 50, "domain": [-1, 1, -1, 1],
                     "circular_mask": True, "entropy_units": "nats"},
        "features": {"window_size_s": 60.0, "sharp_turn_threshold_rad": 1.0,
                     "divergence_from_control": "deferred_nan_v1"},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def write_requirements(path: Path) -> None:
    path.write_text("\n".join([
        "numpy", "pandas", "opencv-python", "ultralytics", "shapely",
        "scipy", "scikit-learn", "tqdm", "pyyaml", "matplotlib", "pytest", "",
    ]))


def write_software_environment(path: Path) -> None:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60)
        content = out.stdout or "# pip freeze produced no output\n"
    except Exception as exc:
        content = f"# pip freeze unavailable: {exc}\n"
    header = (f"# Captured {datetime.now(timezone.utc).astimezone().isoformat()}\n"
              f"# python {sys.version.split()[0]}\n")
    path.write_text(header + content)


def write_experiment_metadata_template(path: Path) -> None:
    # Never clobber curated metadata: only drop the blank template if absent.
    if path.exists():
        return
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerow(EXPERIMENT_METADATA_COLUMNS)


APACHE_2_0_LICENSE = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or Derivative
          Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


CC_BY_4_0_LICENSE = """Creative Commons Attribution 4.0 International (CC BY 4.0)

This is a human-readable summary of (and not a substitute for) the license.
The full legal code is at: https://creativecommons.org/licenses/by/4.0/legalcode

You are free to:

  - Share — copy and redistribute the material in any medium or format.
  - Adapt — remix, transform, and build upon the material for any purpose,
    even commercially.

  The licensor cannot revoke these freedoms as long as you follow the
  license terms.

Under the following terms:

  - Attribution — You must give appropriate credit, provide a link to the
    license, and indicate if changes were made. You may do so in any
    reasonable manner, but not in any way that suggests the licensor
    endorses you or your use.

  - No additional restrictions — You may not apply legal terms or
    technological measures that legally restrict others from doing anything
    the license permits.

Notices:

  - You do not have to comply with the license for elements of the material
    in the public domain or where your use is permitted by an applicable
    exception or limitation.

  - No warranties are given. The license may not give you all of the
    permissions necessary for your intended use. For example, other rights
    such as publicity, privacy, or moral rights may limit how you use the
    material.

How to attribute this dataset:

  "{dataset_title}" by {creators} is licensed under CC BY 4.0. Source:
  {dataset_id} (version {version}). To view a copy of this license, visit
  https://creativecommons.org/licenses/by/4.0/
"""


def write_license(path: Path, license_name: str, **fields) -> None:
    key = license_name.strip().lower().replace(" ", "-").replace("_", "-")
    if key in ("apache-2.0", "apache2.0", "apache-2", "apache"):
        path.write_text(APACHE_2_0_LICENSE)
        return
    if key in ("cc-by-4.0", "cc-by-4", "ccby4.0", "cc-by"):
        path.write_text(CC_BY_4_0_LICENSE.format(
            dataset_title=fields.get("dataset_title", "[title]"),
            creators=fields.get("creators", "[creators]"),
            dataset_id=fields.get("dataset_id", "[dataset_id]"),
            version=fields.get("version", "[version]")))
        return
    path.write_text(
        f"{license_name}\n\n"
        "Replace this file with the full text of the chosen license before "
        "release. The dataset_description.json records the license identifier.\n")


MODEL_CARDS_DIR = tl.REPO_ROOT / "model_cards"


def find_model_card(model: Optional[Path],
                    cards_dir: Path = MODEL_CARDS_DIR) -> Optional[Path]:
    """The repository model card whose `- model name:` line names this file."""
    if model is None or not cards_dir.is_dir():
        return None
    tag = f"- model name: `{Path(model).name}`"
    for card in sorted(cards_dir.glob("*.md")):
        if tag in card.read_text():
            return card
    return None


def write_model_cards(card_dir: Path, larva_model: Optional[Path] = None,
                      petri_model: Optional[Path] = None,
                      cards_dir: Path = MODEL_CARDS_DIR) -> None:
    """Ship the real card of each model used (SPECIFICATION.md 17).

    A model with a card in model_cards/ gets that card copied verbatim under
    the release's fixed file name; anything else gets the blank template, to
    be filled in before release.
    """
    common = (
        "## Model\n- model name:\n- model type:\n- YOLO version:\n\n"
        "## Training\n- training dataset:\n- classes:\n- annotation protocol:\n"
        "- number of images:\n- number of boxes/masks:\n- image resolution:\n"
        "- training command:\n\n## Validation\n- validation metrics:\n\n"
        "## Use\n- intended use:\n- not intended use:\n- known failure modes:\n")
    templates = {
        "petri_detector_model_card.md": (
            petri_model,
            "# Petri Detector Model Card\n\n" + common +
            "\n## Notes\nSingle class `petri`; trained on full tray images, "
            "six dishes per image (SPECIFICATION.md 3.1).\n"),
        "larva_segmenter_model_card.md": (
            larva_model,
            "# Larva Segmenter Model Card\n\n" + common +
            "\n## Segmentation specifics\n- mask annotation protocol:\n"
            "- expected number of larvae per Petri: 0 or 1\n"
            "- selection policy when multiple detections occur: "
            "continuity-first (same track id > area gate > closest to "
            "previous > highest conf > largest area), SPECIFICATION.md 6\n"
            "- area measurement limitations:\n"),
    }
    for name, (model, template) in templates.items():
        card = find_model_card(model, cards_dir)
        (card_dir / name).write_text(card.read_text() if card else template)


def write_readme(path: Path, args, n_dict_rows: int,
                 filter_desc: str = "Savitzky-Golay (window 7, polyorder 2)") -> None:
    creators = [c.strip() for c in args.creators.split(";") if c.strip()]
    affiliations = [a.strip() for a in args.affiliations.split(";") if a.strip()]
    year = date.today().year

    creators_line = "; ".join(creators) if creators else "_unspecified_"
    affil_line = "; ".join(affiliations) if affiliations else "_unspecified_"
    contact_line = args.contact or "_unspecified_"
    authors_cite = ", ".join(creators) if creators else "[creators]"
    citation = (f"{authors_cite} ({year}). *{args.title}* "
                f"(version {args.version}) [Data set]. {args.license}.")
    code_license = args.code_license

    path.write_text(f"""# {args.title}

Dataset ID: `{args.dataset_id}`  ·  Version: `{args.version}`  ·  Data: {args.license}  ·  Code: {code_license}

Automated behavioral-phenotyping dataset for *{args.species}* larvae monitored
in Petri dishes. The data products are objective behavioral proxies
(trajectory, occupancy and area dynamics); they do **not** measure welfare,
suffering or subjective state (see *Scientific scope* below; SPECIFICATION.md 23).

## Creators & contact (FAIR: Findable)

- **Creators:** {creators_line}
- **Affiliations:** {affil_line}
- **Contact:** {contact_line}
- **Keywords:** automated behavioral phenotyping; *{args.species}*;
  stress-associated behavioral deviation; non-invasive behavioral monitoring; FAIR

Full machine-readable metadata is in `dataset_description.json`.

## How to cite

> {citation}

## Licensing

This release is dual-licensed to separate software from data:

- **Code** (the pipeline scripts): **{code_license}** — see `LICENSE`.
- **Data** (CSV tables, heatmap `.npy`, JSONL, metadata): **{args.license}** —
  see `LICENSE-DATA`.

Reusing the data requires attribution under {args.license} (cite as above).
Reusing or modifying the code is governed by {code_license}. Both identifiers
are recorded in `dataset_description.json` (`code_license`, `data_license`).

## Layout (SPECIFICATION.md 16)

```
README.md  LICENSE  LICENSE-DATA  dataset_description.json  data_dictionary.csv
processing_config.yaml  requirements.txt  software_environment.txt
metadata/   experiment_metadata.csv  model_metadata.json  processing_provenance.jsonl
petri/      petri_tracks.csv
larva/      larva_tracks.csv  larva_detections_all.csv  larva_masks.jsonl
kinematics/ larva_kinematics.csv
features/   features_per_window.csv  features_per_video.csv
heatmaps/   npy/  png/  metadata/
qc/         qc_summary.csv
model_cards/
```

## Documentation (FAIR: Reusable)

`data_dictionary.csv` documents every column of every table
({n_dict_rows} columns across {len(FILES)} files): file, column name, data type,
unit, description, allowed values and missing-value policy.
`processing_config.yaml` records the processing parameters (locked v1 defaults,
SPECIFICATION.md 26). Provenance for each run — git commit, command, model paths and
SHA-256 hashes, thresholds — is appended to
`metadata/processing_provenance.jsonl`. Model cards are in `model_cards/`.

## Formats (FAIR: Accessible)

Open formats only: CSV (tables), JSON/JSON-LD (metadata), JSONL (framewise
masks/provenance), NumPy `.npy`/`.npz` (heatmaps), YAML (config). PNG/MP4 are
for visualization and QC only — the `.npy` arrays and CSV tables are the
reusable scientific data.

## Units & conventions (FAIR: Interoperable)

| quantity | unit |
|---|---|
| time | seconds |
| distance | millimeters (calibrated) / pixels (otherwise) |
| speed | mm/s (calibrated) |
| angle | radians |
| area | pixels^2 (and mm^2 if calibrated) |
| normalized position | Petri-radius units (center 0, edge 1) |
| temperature / humidity | Celsius / percent |

Column names are `snake_case`. Missing data is explicit: `NaN` (numeric),
empty string/null (string), `false` only when genuinely false — numeric gaps
are **never** encoded as `0`.

### Controlled vocabularies

- `treatment`: control, sham, heat, cold, starvation, dehydration,
  mechanical_disturbance, light_exposure, crowding, unknown
- `movement_state`: moving, immobile, missing, interpolated, uncertain
- `quality_flag`: ok, low_confidence, missing_detection, track_switch,
  outside_petri, area_outlier, ambiguous_detection
- `selection_reason`: single_detection, same_track_id, highest_confidence,
  closest_to_previous, area_filter, largest_area, no_detection, ambiguous,
  rejected_implausible, reacquired

## Coordinate systems

Every larva observation is preserved in three systems: crop, full-frame
(pixels), and Petri-normalized (center `(0,0)`, edge radius `1`). The
Petri-normalized system makes heatmaps comparable across videos, cameras,
resolutions and dishes.

## Processing notes

- Petri geometry is detected on the first valid frame and reused
  (`--detect-petri-first-frame`), keeping the crop window stable.
- Position gaps <= 0.5 s are linearly interpolated; longer gaps stay missing.
- Position smoothing filter for this release: **{filter_desc}**. The filter is
  configurable (`--smoothing-filter savgol|moving_average|kalman|none`) and the
  choice is recorded in `processing_config.yaml`, the `*_smooth` column
  descriptions in `data_dictionary.csv`, and the provenance records.
- `jerk` is dropped; `acceleration_mm_s2` is kept but flagged low-confidence.
  Divergence-from-control features are deferred (NaN in v1). See SPECIFICATION.md 26.

## Scientific scope (SPECIFICATION.md 23)

These data are objective behavioral proxies: altered locomotor activity,
altered spatial occupancy, increased immobility, reduced exploration,
trajectory divergence from a control distribution, and area dynamics. They do
**not** measure wellness, suffering, consciousness, or a definitive welfare
score. Interpretation requires experimental validation.

## Reproducibility

The dataset is reproducible from raw videos + `metadata/experiment_metadata.csv`
+ the pipeline scripts, using the parameters in `processing_config.yaml`, the
environment in `software_environment.txt` / `requirements.txt`, and the model
hashes recorded in the provenance records.
""")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dataset-id", default="lsb_tmolitor_v1")
    p.add_argument("--title", default="Tenebrio molitor "
                                      "behavioral phenotyping dataset")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--creators", default="",
                   help="semicolon-separated creator names")
    p.add_argument("--affiliations", default="",
                   help="semicolon-separated affiliations")
    p.add_argument("--contact", default="")
    p.add_argument("--license", default="CC-BY-4.0",
                   help="data license (tables, heatmaps, metadata); "
                        "written to LICENSE-DATA")
    p.add_argument("--code-license", default="Apache-2.0",
                   help="license for the pipeline code; written to LICENSE")
    p.add_argument("--species", default="Tenebrio molitor")
    p.add_argument("--no-pip-freeze", action="store_true",
                   help="skip capturing software_environment.txt")
    # Smoothing filter actually used by compute_kinematics, so the data
    # dictionary and processing_config document the *_smooth columns correctly.
    # Tracker actually used by track_larvae_video, so processing_config.yaml
    # states the association algorithm the dataset was produced with.
    p.add_argument("--tracker", type=lt.normalize_tracker_name,
                   choices=lt.TRACKERS, default="bytetrack")
    p.add_argument("--tracker-config", type=Path, default=None)
    # Models and selection gate actually used (SPECIFICATION.md 6, 17, 26.6), so the
    # release records them and ships the matching model cards.
    p.add_argument("--larva-model", type=Path, default=tl.DEFAULT_LARVA_MODEL)
    p.add_argument("--petri-model", type=Path, default=None)
    p.add_argument("--max-jump-px", type=float, default=60.0)
    p.add_argument("--max-jump-growth-px", type=float, default=5.0)
    p.add_argument("--max-jump-cap-px", type=float, default=240.0)
    p.add_argument("--reacquire-after-s", type=float,
                   default=tl.DEFAULT_REACQUIRE_AFTER_S)
    p.add_argument("--smoothing-filter",
                   choices=("savgol", "moving_average", "kalman", "none"),
                   default="savgol")
    p.add_argument("--smoothing-window", type=int, default=7)
    p.add_argument("--savgol-polyorder", type=int, default=2)
    p.add_argument("--kalman-process-var", type=float, default=1.0)
    p.add_argument("--kalman-measurement-var", type=float, default=4.0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    (out / "model_cards").mkdir(parents=True, exist_ok=True)

    smoothing_cfg, filter_desc = smoothing_meta(args)
    n_rows = write_data_dictionary(out / "data_dictionary.csv", filter_desc)
    write_dataset_description(out / "dataset_description.json", args)
    write_processing_config(out / "processing_config.yaml", smoothing_cfg,
                            tracking_meta(args))
    write_requirements(out / "requirements.txt")
    if not args.no_pip_freeze:
        write_software_environment(out / "software_environment.txt")
    write_experiment_metadata_template(out / "metadata" /
                                       "experiment_metadata.csv")
    # Dual license: code under --code-license (LICENSE), data under --license
    # (LICENSE-DATA). See the README "Licensing" section.
    creators_str = "; ".join(c.strip() for c in args.creators.split(";")
                             if c.strip()) or "[creators]"
    write_license(out / "LICENSE", args.code_license)
    write_license(out / "LICENSE-DATA", args.license,
                  dataset_title=args.title, creators=creators_str,
                  dataset_id=args.dataset_id, version=args.version)
    write_model_cards(out / "model_cards", args.larva_model, args.petri_model)
    write_readme(out / "README.md", args, n_rows, filter_desc)

    print(f"FAIR scaffold written to {out}")
    print(f"  data_dictionary.csv: {n_rows} documented columns "
          f"across {len(FILES)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
