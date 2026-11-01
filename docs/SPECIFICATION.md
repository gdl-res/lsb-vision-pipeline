# wormstress-vision — pipeline specification

Design specification and data contract of the pipeline. Section numbers are referenced from
the source code as `SPECIFICATION.md <section>`.

## Purpose

This repository implements a computer-vision and behavioral-phenotyping pipeline for *Tenebrio molitor* larvae monitored in Petri dishes.

The experimental setup consists of full-frame images or videos containing a tray with six Petri dishes. Each Petri dish may contain zero or one larva. Two YOLO models are expected:

1. A YOLO object-detection model for Petri dishes.
2. A YOLO segmentation/tracking model for larvae.

The final goal is to generate rich, reusable, FAIR-compliant behavioral data for stress-response and welfare-oriented research in *Tenebrio molitor* larvae.

The core scientific output is not a direct “wellness score.” The correct terminology is:

```text
automated behavioral phenotyping
stress-associated behavioral deviation
trajectory and area-based behavioral metrics
non-invasive behavioral monitoring
```

Avoid claims that the system directly measures subjective suffering, welfare state, or insect consciousness.

---

# 1. High-Level Pipeline

The desired pipeline is:

```text
full tray image/video frame
→ YOLO Petri detector
→ select exactly six Petri detections
→ assign dish IDs in row-major order
→ crop one square region around each Petri dish
→ remap full-frame YOLO larva segmentation labels to Petri crops
→ run YOLO larva segmentation/tracking on each crop
→ extract centroid, area, mask, bounding box, and shape descriptors
→ map larva coordinates back to full-frame coordinates
→ normalize larva coordinates by Petri geometry
→ save framewise raw observations
→ save derived kinematics
→ save heatmaps
→ save behavioral features per time window and per video
→ save metadata and provenance for FAIR compliance
```

The pipeline must separate:

```text
measurement
analysis
interpretation
```

Measurement should be as complete and raw as practical. Analysis can be recomputed later from saved measurements.

---

# 2. Dataset Geometry

The physical setup can appear as a portrait-oriented tray:

```text
1 2
3 4
5 6
```

Therefore the default layout should be:

```text
rows = 3
cols = 2
```

The code must also support other layouts, especially:

```text
rows = 2
cols = 3
```

Dish IDs must always be assigned in row-major order:

```text
row 0, left to right
row 1, left to right
row 2, left to right
```

For `rows=3, cols=2`:

```text
1 2
3 4
5 6
```

For `rows=2, cols=3`:

```text
1 2 3
4 5 6
```

Never hard-code fixed pixel coordinates or fixed image resolutions.

---

# 3. YOLO Models

## 3.1 Petri Detector

Use a YOLO object-detection model for Petri dishes.

Class list:

```yaml
0: petri
```

The Petri detector should be trained on full tray images. Each full image should contain six Petri boxes.

Annotation rule:

```text
Draw each box around the whole Petri dish outer rim.
Include the transparent rim and the white interior.
Avoid excessive black tray margin.
Do not annotate only the white interior.
```

Expected training command example:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=petri_dataset.yaml \
  imgsz=1280 \
  epochs=100 \
  batch=8
```

Use `yolo11n.pt` for quick validation and `yolo11s.pt` if higher accuracy is needed.

## 3.2 Larva Detector/Segmenter

Use a YOLO segmentation model for larvae.

The larva model should output:

```text
segmentation mask or polygon
bounding box
confidence
class ID
track ID if tracking is enabled
```

The larva class list should be explicit in metadata.

Example:

```yaml
0: larva
```

The larva model may be run on Petri crops rather than full tray frames.

---

# 4. Petri Detection Inference Requirements

At inference time:

1. Run the Petri detector on the full image/frame.
2. Keep only detections of class `petri`.
3. Filter by confidence.
4. Apply NMS.
5. Expect exactly six Petri detections.
6. If more than six detections remain, select the best six using confidence and grid consistency.
7. If fewer than six detections remain, mark the frame/image as failed or low confidence.
8. Sort selected Petri boxes into dish IDs using row-major order.
9. Store detection quality and failure flags.

Each selected Petri must store:

```python
{
    "video_id": str,
    "frame_idx": int,
    "dish_id": int,
    "row": int,
    "col": int,
    "petri_track_id": int | None,
    "petri_conf": float,
    "petri_x1": float,
    "petri_y1": float,
    "petri_x2": float,
    "petri_y2": float,
    "petri_cx": float,
    "petri_cy": float,
    "petri_w": float,
    "petri_h": float,
    "petri_radius": float,
    "petri_valid": bool
}
```

Compute:

```python
petri_cx = (x1 + x2) / 2
petri_cy = (y1 + y2) / 2
petri_w = x2 - x1
petri_h = y2 - y1
petri_radius = 0.5 * min(petri_w, petri_h)
```

For cropping:

```python
diameter = max(petri_w, petri_h)
crop_size = diameter * (1.0 + margin)
```

Default:

```python
margin = 0.05
```

The crop must be square and centered on the Petri box center.

---

# 5. Petri Tracking Policy in Videos

If the camera is fixed, Petri detection does not need to run on every frame by default.

Preferred strategy:

```text
detect Petri dishes on the first valid frame
reuse Petri geometry for subsequent frames
optionally re-detect periodically or when quality checks fail
```

CLI options should include:

```bash
--detect-petri-first-frame
--detect-petri-every-frame
--detect-petri-every-n-frames INT
```

Default should be:

```bash
--detect-petri-first-frame
```

If using YOLO tracking for Petri dishes, save the Petri track IDs, but still enforce the expected six-dish layout.

---

# 6. Larva Tracking Policy

For each Petri crop and frame:

1. Run YOLO larva segmentation/tracking.
2. Select zero or one primary larva detection per dish.
3. Store all raw detections if practical.
4. Store the selected larva observation separately.

If multiple larvae are detected in one dish, selection priority is:

```text
same YOLO track ID as previous frame
highest confidence
plausible area range
closest to previous centroid
largest mask area if all else fails
```

Save a field:

```text
selection_reason
```

Allowed values:

```text
single_detection
same_track_id
highest_confidence
closest_to_previous
area_filter
largest_area
no_detection
ambiguous
rejected_implausible
```

**Hysteresis (`--max-jump-px`/`--max-jump-growth-px`/`--max-jump-cap-px`,
`track_larvae_video.py`):** a *single* raw detection in a frame is not
automatically accepted as primary if it doesn't match the previous frame's
track ID. It must also land within `min(max_jump_px + missing_streak *
max_jump_growth_px, max_jump_cap_px)` of the previous centroid; a lone
detection that fails this is recorded as `rejected_implausible` (treated as
missing, not deleted — still present in `larva_detections_all.csv`) instead
of hijacking the established track. The growth term is additive, not
multiplicative — an earlier multiplicative version let a ~200px jump through
after only 3-4 missing frames, far faster than a larva can plausibly crawl,
which reopened the exact artifact-vs-larva oscillation this gate exists to
stop. It keeps the tracker from flipping onto a spurious second detection
(a dish-surface artifact or a segmentation mask fragment) whenever that is the
*only* detection in a frame.

Do not silently discard ambiguous detections. Flag them.

---

# 7. Coordinate Systems

The pipeline must preserve multiple coordinate systems.

## 7.1 Crop Coordinates

Coordinates relative to the square Petri crop.

```text
cx_crop
cy_crop
x1_crop
y1_crop
x2_crop
y2_crop
```

## 7.2 Full-Frame Coordinates

Coordinates relative to the original full frame.

```python
cx_global = crop_x0 + cx_crop
cy_global = crop_y0 + cy_crop
```

Store:

```text
cx_global
cy_global
x1_global
y1_global
x2_global
y2_global
```

## 7.3 Petri-Normalized Coordinates

For heatmap comparability, all behavioral coordinates must be expressible in Petri-normalized coordinates:

```python
x_norm = (cx_global - petri_cx) / petri_radius
y_norm = (cy_global - petri_cy) / petri_radius
r_norm = sqrt(x_norm**2 + y_norm**2)
theta_rad = atan2(y_norm, x_norm)
inside_petri = r_norm <= 1.0
```

The canonical Petri coordinate system is:

```text
Petri center = (0, 0)
Petri edge = radius 1
```

This is essential for comparing heatmaps across videos, cameras, resolutions, crops, and Petri dishes.

---

# 8. Required Raw Outputs

The pipeline must produce raw, reusable outputs. Do not only save final behavioral features.

## 8.1 `petri_tracks.csv`

One row per frame × dish.

Columns:

```text
dataset_id
experiment_id
video_id
frame_idx
time_s
dish_id
row
col
petri_track_id
petri_conf
petri_x1
petri_y1
petri_x2
petri_y2
petri_cx
petri_cy
petri_w
petri_h
petri_radius
petri_valid
crop_x0
crop_y0
crop_size
```

If Petri detection is reused from the first frame, repeat the geometry for every frame or clearly record the reuse policy in metadata.

## 8.2 `larva_tracks.csv`

One row per frame × dish for the selected larva observation.

Columns:

```text
dataset_id
experiment_id
video_id
frame_idx
time_s
dish_id
larva_track_id
larva_class_id
larva_conf
valid_detection
selection_reason
```

Crop-space coordinates:

```text
crop_x0
crop_y0
crop_size
cx_crop_raw
cy_crop_raw
x1_crop
y1_crop
x2_crop
y2_crop
bbox_w_crop
bbox_h_crop
```

Full-frame coordinates:

```text
cx_global_raw
cy_global_raw
x1_global
y1_global
x2_global
y2_global
bbox_w_global
bbox_h_global
```

Petri-normalized coordinates:

```text
x_norm_raw
y_norm_raw
r_norm_raw
theta_rad_raw
inside_petri
```

Smoothed coordinates:

```text
x_norm_smooth
y_norm_smooth
r_norm_smooth
cx_global_smooth
cy_global_smooth
```

Detection and tracking quality:

```text
missing
interpolated
track_switch
outlier_position
outlier_area
quality_flag
```

## 8.3 `larva_detections_all.csv`

Optional but recommended.

One row per raw YOLO larva detection, including detections not selected as the primary larva.

Columns:

```text
dataset_id
experiment_id
video_id
frame_idx
time_s
dish_id
raw_detection_id
yolo_track_id
class_id
conf
x1_crop
y1_crop
x2_crop
y2_crop
cx_crop
cy_crop
mask_area_px
selected_primary
selection_reason
```

This is useful for debugging and model validation.

## 8.4 `larva_masks.jsonl`

Recommended if disk space allows.

One JSON object per selected larva mask.

Each line:

```json
{
  "dataset_id": "wormstress_v1",
  "experiment_id": "exp_001",
  "video_id": "video_001",
  "frame_idx": 123,
  "time_s": 4.92,
  "dish_id": 2,
  "larva_track_id": 17,
  "mask_format": "polygon_crop",
  "polygon_crop": [[123.4, 88.1], [124.0, 89.5], [125.2, 91.0]],
  "mask_area_px": 1042.5
}
```

Do not store huge binary mask images by default unless explicitly requested.

---

# 9. Segmentation-Derived Morphology Outputs

For every selected larva mask, compute and save:

```text
mask_area_px
mask_area_norm_petri
mask_area_mm2
mask_perimeter_px
mask_compactness
mask_solidity
mask_extent
mask_major_axis_px
mask_minor_axis_px
mask_aspect_ratio
mask_orientation_rad
mask_eccentricity
bbox_area_px
mask_bbox_area_ratio
```

Definitions:

```python
mask_area_norm_petri = mask_area_px / (pi * petri_radius_px**2)
compactness = 4 * pi * area / perimeter**2
bbox_area_px = bbox_w * bbox_h
mask_bbox_area_ratio = mask_area_px / bbox_area_px
```

If real Petri diameter is known:

```python
mm_per_px = petri_diameter_mm / (2 * petri_radius_px)
mask_area_mm2 = mask_area_px * mm_per_px**2
```

Do not overinterpret these morphology values as true posture unless validated.

---

# 10. Derived Framewise Kinematics

Create either `larva_kinematics.csv` or append these columns to `larva_tracks.csv`.

Use smoothed coordinates for kinematics, but keep raw coordinates too.

Columns:

```text
dx_norm
dy_norm
step_distance_norm
step_distance_mm
speed_norm_s
speed_mm_s
acceleration_mm_s2
jerk_mm_s3
turn_angle_rad
abs_turn_angle_rad
angular_speed_rad_s
radial_speed_norm_s
area_change_px
area_change_abs_px
area_change_rate_px_s
is_moving
is_immobile
movement_state
```

`movement_state` may be:

```text
moving
immobile
missing
interpolated
uncertain
```

The immobility threshold must be configurable:

```bash
--immobility-speed-threshold-mm-s FLOAT
```

Do not hard-code this threshold.

---

# 11. Behavioral Feature Outputs

Compute aggregated features over time windows and full videos.

## 11.1 `features_per_window.csv`

One row per video × dish × time window.

Required identifiers:

```text
dataset_id
experiment_id
video_id
dish_id
window_id
window_start_s
window_end_s
n_frames
valid_fraction
```

Activity features:

```text
total_distance_mm
mean_speed_mm_s
median_speed_mm_s
max_speed_mm_s
speed_std_mm_s
immobility_fraction
moving_fraction
movement_bout_count
mean_bout_duration_s
median_bout_duration_s
max_bout_duration_s
```

Path geometry:

```text
net_displacement_mm
path_efficiency
tortuosity
mean_abs_turn_angle_rad
turn_angle_std_rad
sharp_turn_count
```

Spatial occupancy:

```text
center_fraction
middle_fraction
edge_fraction
mean_r_norm
median_r_norm
std_r_norm
min_r_norm
max_r_norm
occupancy_entropy
occupied_cells
revisit_rate
```

Area and morphology:

```text
mean_area_px
median_area_px
std_area_px
area_cv
min_area_px
max_area_px
area_range_px
mean_area_norm_petri
std_area_norm_petri
mean_abs_area_change_px
std_area_change_px
area_oscillation_count
mean_aspect_ratio
std_aspect_ratio
mean_eccentricity
mean_compactness
```

Detection quality:

```text
mean_larva_conf
median_larva_conf
min_larva_conf
missing_fraction
interpolated_fraction
track_switch_count
outlier_fraction
fraction_outside_petri
quality_score
```

## 11.2 `features_per_video.csv`

One row per video × dish.

Same features as `features_per_window.csv`, aggregated over the full video.

Also include:

```text
heatmap_entropy
heatmap_peak_density
heatmap_peak_x_norm
heatmap_peak_y_norm
heatmap_spread_x
heatmap_spread_y
js_divergence_from_control
wasserstein_distance_from_control
```

Only compute divergence features when a valid control reference is available.

---

# 12. Heatmap Outputs

Heatmaps must be computed in Petri-normalized coordinates.

Use a fixed normalized domain:

```python
x_edges = np.linspace(-1, 1, bins + 1)
y_edges = np.linspace(-1, 1, bins + 1)
```

Default:

```python
bins = 50
```

Save for every video × dish:

```text
heatmaps/
  video_001_dish01_counts.npy
  video_001_dish01_probability.npy
  video_001_dish01_metadata.json
  video_001_dish01_heatmap.png
  video_001_dish01_trajectory.png
```

Metadata must include:

```json
{
  "video_id": "video_001",
  "dish_id": 1,
  "bins": 50,
  "domain": [-1, 1, -1, 1],
  "valid_points": 12345,
  "normalization": "petri_radius",
  "circular_mask_applied": true
}
```

PNG files are for visualization only. The `.npy` arrays are the reusable scientific data.

---

# 13. Quality Control Outputs

Create `qc_summary.csv`.

One row per video × dish.

Columns:

```text
dataset_id
experiment_id
video_id
dish_id
petri_valid
petri_conf
valid_detection_fraction
mean_larva_conf
median_larva_conf
missing_frame_count
longest_missing_gap_s
track_switch_count
fraction_outside_petri
mean_mask_area_px
mask_area_outlier_fraction
position_outlier_fraction
interpolated_fraction
overall_quality_score
exclude_recommended
exclude_reason
```

Suggested exclusion rules should be configurable, not hard-coded.

Example:

```text
valid_detection_fraction < 0.80
longest_missing_gap_s > 10
fraction_outside_petri > 0.05
track_switch_rate_per_min > threshold   # length-normalized, not a raw count
```

Default behavior should not delete data. It should flag data.

---

# 14. Experimental Metadata

A manually curated `experiment_metadata.csv` is required for meaningful behavioral analysis.

Columns:

```text
dataset_id
experiment_id
video_id
dish_id
larva_id
batch_id
date
camera_id
phone_model
fps
frame_width
frame_height
petri_diameter_mm
treatment
stressor
stressor_intensity
stressor_duration_s
recovery_time_s
temperature_c
humidity_percent
light_condition
handling_protocol
recording_start_time
recording_duration_s
operator
notes
```

This file links behavioral data to experimental conditions.

The pipeline must never infer treatment from filenames unless explicitly
configured. The explicit, opt-in path is `scripts/parse_filenames.py` (§18.5),
which writes a reviewable `*.draft.csv` and never auto-applies it.

---

# 15. FAIR Compliance Requirements

The final data products must be designed to be FAIR:

```text
Findable
Accessible
Interoperable
Reusable
```

## 15.1 Findable

Every dataset release must include:

```text
dataset_id
dataset_title
dataset_version
creator names
affiliations
contact email
creation date
license
keywords
species
experimental context
```

Use stable identifiers where possible.

Recommended identifiers:

```text
dataset_id: wormstress_tmolitor_v1
experiment_id: exp_YYYYMMDD_condition_batch
video_id: stable filename stem or UUID
larva_id: stable within experiment
dish_id: 1-6 row-major
```

Include a top-level `dataset_description.json`.

## 15.2 Accessible

Data should be stored in open, common formats:

```text
CSV for tabular data
JSON or JSON-LD for metadata
JSONL for framewise mask/polygon data
Numpy .npy or .npz for numerical heatmaps
PNG/MP4 only for visualization and quality control
YAML for configuration files
```

Avoid proprietary formats.

Every output file must have a documented path and schema.

## 15.3 Interoperable

Use controlled vocabularies and consistent units.

Required unit conventions:

```text
time: seconds
distance: millimeters when calibrated, pixels otherwise
speed: mm/s when calibrated
angle: radians
temperature: Celsius
humidity: percent
area: pixels^2 and optionally mm^2
normalized position: unitless Petri radius units
```

Column names must be stable and machine-readable:

```text
snake_case
no spaces
no ambiguous abbreviations
```

Use explicit missing values:

```text
NaN for numeric missing data
empty string or null for missing strings
false for boolean flags only when actually false
```

Do not encode missing numeric data as `0`.

Controlled vocabulary examples:

`treatment`:

```text
control
sham
heat
cold
starvation
dehydration
mechanical_disturbance
light_exposure
crowding
unknown
```

`movement_state`:

```text
moving
immobile
missing
interpolated
uncertain
```

`quality_flag`:

```text
ok
low_confidence
missing_detection
track_switch
outside_petri
area_outlier
ambiguous_detection
```

## 15.4 Reusable

Every dataset release must include:

```text
README.md
LICENSE
dataset_description.json
data_dictionary.csv
experiment_metadata.csv
processing_config.yaml
software_environment.txt or requirements.txt
model_cards/
provenance/
```

The data dictionary must document every column:

```text
file
column_name
data_type
unit
description
allowed_values
missing_value_policy
```

Processing provenance must record:

```text
git commit hash
script name
script version
CLI command
processing timestamp
model paths
model hashes
confidence thresholds
NMS thresholds
crop margin
smoothing settings
feature settings
```

The dataset must be reproducible from raw data plus metadata plus code.

---

# 16. Recommended FAIR Dataset Structure

Use this structure for processed outputs:

```text
processed_dataset/
  README.md
  LICENSE
  dataset_description.json
  data_dictionary.csv
  processing_config.yaml
  requirements.txt

  metadata/
    experiment_metadata.csv
    model_metadata.json
    processing_provenance.jsonl

  raw_index/
    source_files.csv

  petri/
    petri_tracks.csv

  larva/
    larva_tracks.csv
    larva_detections_all.csv
    larva_masks.jsonl

  kinematics/
    larva_kinematics.csv

  features/
    features_per_window.csv
    features_per_video.csv

  heatmaps/
    npy/
    png/
    metadata/

  qc/
    qc_summary.csv
    overlays/
    trajectories/
    diagnostic_plots/

  crops/
    images/
    labels/
    crop_metadata.jsonl
```

For large video datasets, use one folder per experiment or video if files become too large.

---

# 17. Model Metadata and Model Cards

For each YOLO model, create a model card.

## `model_cards/petri_detector_model_card.md`

Include:

```text
model name
model type
YOLO version
training dataset
classes
annotation protocol
number of images
number of boxes/masks
image resolution
training command
validation metrics
known failure modes
intended use
not intended use
```

## `model_cards/larva_segmenter_model_card.md`

Include the same information, plus:

```text
mask annotation protocol
expected number of larvae per Petri
selection policy when multiple detections occur
area measurement limitations
```

---

# 18. CLI Requirements

All major scripts must be callable from CLI.

## 18.1 Petri Cropper

Expected command:

```bash
python scripts/crop_petris_yolo_seg.py \
  --images My_First_Project_LSB.v2i.yolov11/train/images \
  --labels My_First_Project_LSB.v2i.yolov11/train/labels \
  --out My_First_Project_LSB_petri_crops/train \
  --petri-model runs/detect/petri_detector/weights/best.pt \
  --rows 3 \
  --cols 2 \
  --margin 0.05 \
  --conf 0.30 \
  --iou 0.50 \
  --debug \
  --max-images 20
```

Required options:

```text
--images
--labels
--out
--petri-model
--rows
--cols
--margin
--conf
--iou
--min-label-area-ratio
--device
--debug
--max-images
--on-failure
```

Failure policy:

```text
--on-failure skip
--on-failure keep-best
--on-failure error
```

Default:

```text
--on-failure skip
```

## 18.3 Video Tracker

Expected command:

```bash
python scripts/track_larvae_video.py \
  --video raw_videos/video_001.mp4 \
  --out outputs/video_001 \
  --petri-model runs/detect/petri_detector/weights/best.pt \
  --larva-model runs/segment/larva_segmenter/weights/best.pt \
  --rows 3 \
  --cols 2 \
  --petri-diameter-mm 90 \
  --track \
  --device cuda:0
```

Important options:

```text
--detect-petri-first-frame
--detect-petri-every-frame
--detect-petri-every-n-frames
--petri-geometry box|circlefit       # §26.2 (default box)
--tracker bytetrack|botsort|ocsort   # §26.1 (default bytetrack)
--tracker-config PATH                # override config/trackers/<tracker>.yaml
--petri-imgsz INT                    # pinned, default 640
--larva-imgsz INT                    # pinned, default 640
--save-masks
--save-overlays
--smoothing-window
--immobility-speed-threshold-mm-s
```

## 18.4 Feature Computation

Expected command:

```bash
python scripts/compute_behavior_features.py \
  --tracks outputs/video_001/larva_tracks.csv \
  --metadata metadata/experiment_metadata.csv \
  --out outputs/video_001/features \
  --window-size-s 60 \
  --heatmap-bins 50
```

## 18.5 Filename Metadata Draft

The **explicit, opt-in** tool referenced in §14: parses condition tokens from
video filenames into a reviewable `experiment_metadata` draft. It never
auto-applies — it writes a `*.draft.csv` with `parse_confidence` /
`parse_warnings` columns for human sign-off.

```bash
python scripts/parse_filenames.py \
  --videos data4evaluation/videos_to_evaluate \
  --out metadata/experiment_metadata.draft.csv \
  --profile auto \
  --rows 3 \
  --cols 2
```

Treatment is tray-level (all dishes share it). `experiment_id` is grouped by
campaign (month + protocol) so each group pools its control + treated videos and
the divergence step (§26.4) has a control reference; the tool warns when a group
has treated videos but no control.

## 18.6 Treatment-Grouped Aggregate Report

Cross-video analysis that contrasts each treated condition against its control
within an `experiment_id` (the feature-level companion to the heatmap-level
divergence of §26.4). Joins `features_per_video.csv` with
`experiment_metadata.csv`, groups by condition (treatment refined by
`--factors`), and reports per feature the mean difference, Hedges' g and a
Mann-Whitney U p-value.

```bash
python scripts/aggregate_report.py \
  --features processed_dataset/features/features_per_video.csv \
  --metadata processed_dataset/metadata/experiment_metadata.csv \
  --out processed_dataset/aggregate \
  --factors stressor_duration_s \
  --min-valid-fraction 0.5
```

Detection-quality features are contrasted separately from behavioral features as
a confound check; `--min-valid-fraction` excludes poorly-tracked dishes from the
behavioral contrasts only. The report embeds the §23 caveats — notably that
dishes within a tray are pseudo-replicates (the tray/video is the experimental
unit) and the comparisons are uncorrected.

---

# 19. Code Style Requirements

Use Python.

Prefer:

```text
pathlib.Path
argparse
pandas
numpy
opencv-python
ultralytics
shapely
scipy
scikit-learn
tqdm
pyyaml
```

Use type hints where practical.

Use small functions with clear responsibilities.

Avoid:

```text
hard-coded absolute paths
fixed image resolution assumptions
fixed Petri coordinates
silent failures
implicit global state
unbounded combinatorial algorithms
large debug output by default
```

All scripts must print useful progress messages and write failure reports.

---

# 20. Testing Requirements

Implement unit tests or at least testable functions for:

```text
YOLO segmentation label parsing
polygon clipping and remapping
crop coordinate conversion
Petri box sorting into row-major order
normalization to Petri coordinates
empty-label crop behavior
feature calculations
heatmap generation
metadata writing
```

Minimum manual validation for any cropper change:

```text
run on 5 images with --debug
inspect Petri boxes
inspect crop squares
inspect remapped larva labels
confirm six crops per source image
confirm dish IDs are correct
confirm empty Petri dishes produce empty .txt files
```

Minimum manual validation for any video tracker change:

```text
run on a short video clip
inspect overlay video
inspect larva_tracks.csv
inspect trajectories
inspect heatmaps
check missing-detection fractions
check normalized coordinates
```

---

# 21. Provenance Requirements

Every processing run must create a provenance record.

Use `processing_provenance.jsonl`, one line per command run.

Example:

```json
{
  "run_id": "2026-05-15T10-30-12_crop_train",
  "timestamp": "2026-05-15T10:30:12+02:00",
  "script": "scripts/crop_petris_yolo_seg.py",
  "git_commit": "abc123",
  "command": "python scripts/crop_petris_yolo_seg.py ...",
  "input_paths": ["train/images", "train/labels"],
  "output_path": "My_First_Project_LSB_petri_crops/train",
  "petri_model": "runs/detect/petri_detector/weights/best.pt",
  "petri_model_sha256": "...",
  "parameters": {
    "rows": 3,
    "cols": 2,
    "margin": 0.05,
    "conf": 0.30,
    "iou": 0.50
  }
}
```

---

# 22. Immediate Development Priorities

Build in this order:

1. YOLO Petri detector inference wrapper.
2. Selection of exactly six Petri boxes.
3. Row-major sorting with configurable `rows` and `cols`.
4. Square Petri crop generation.
5. YOLO segmentation label remapping using Shapely.
6. Crop metadata JSONL.
7. Debug overlays.
8. Split-wise dataset processing.
9. Video larva tracking on Petri crops.
10. Framewise tracking table.
11. Kinematics computation.
12. Heatmap generation.
13. Feature aggregation.
14. FAIR metadata and data dictionary generation.

Do not try to implement the full behavioral-analysis suite before the Petri cropper and framewise tracking tables are robust.

---

# 23. Scientific Language Constraints

Allowed:

```text
stress-associated behavioral deviation
altered locomotor activity
altered spatial occupancy
increased immobility
reduced exploration
trajectory divergence from control distribution
area dynamics
automated behavioral phenotyping
```

Avoid:

```text
this measures wellness
this proves suffering
this measures consciousness
this is a definitive welfare score
```

The system produces objective behavioral proxies. Interpretation requires experimental validation.

---

# 24. Example Final Workflow

## Track Larvae in Video

```bash
python scripts/track_larvae_video.py \
  --video raw_videos/video_001.mp4 \
  --out outputs/video_001 \
  --petri-model runs/detect/petri_detector/weights/best.pt \
  --larva-model runs/segment/larva_segmenter/weights/best.pt \
  --rows 3 \
  --cols 2 \
  --petri-diameter-mm 90 \
  --track \
  --device cuda:0
```

## Compute Behavioral Features

```bash
python scripts/compute_behavior_features.py \
  --tracks outputs/video_001/larva_tracks.csv \
  --metadata metadata/experiment_metadata.csv \
  --out outputs/video_001/features \
  --window-size-s 60 \
  --heatmap-bins 50
```

---

# 25. Core Principle

Save enough information to reconstruct behavior later.

Do not only save final stress metrics.

The most important data product is:

```text
one row per frame per dish containing:
Petri geometry,
larva centroid,
larva mask/area,
YOLO confidence,
YOLO track ID,
global coordinates,
Petri-normalized coordinates,
kinematics,
quality flags,
and provenance.
```

This is the foundation for reproducible, FAIR-compliant behavioral monitoring.

---

# 26. Locked Decisions (v1)

These defaults are locked for the v1 pipeline. They are hard to change once data
is written, so they are part of the spec. All are configurable via CLI/config and
must be recorded in `processing_provenance.jsonl`.

## 26.1 Larva tracking: stateful, per dish-crop-stream

Run the YOLO segmenter in tracking mode on each dish's crop independently,
maintaining one tracker instance per dish across the whole video.

```text
tracker: bytetrack        (locked v1 default; every released v1 dataset)
scope:   larva_track_id is stable within a dish within a video
id rule: canonical larva_track_id = dish_id * 1000 + raw_yolo_id
pairing: always run with --detect-petri-first-frame so the crop window is
         stable frame-to-frame (re-detecting every frame injects crop jitter
         that reads as fake larva motion)
```

**v1.1 selectable association algorithm (`--tracker`).** The association step is
now a choice, so track-identity quality can be compared instead of assumed. The
default is unchanged and a default run reproduces v1 byte-identically.

```text
bytetrack   locked v1 default; ultralytics model.track(persist=True)
botsort     ultralytics BoT-SORT
ocsort      OC-SORT (Cao et al. 2023)
```

Parameters live in `config/trackers/<name>.yaml`, versioned with the code and
recorded in `processing_provenance.jsonl` (name, resolved config path, config
SHA-256 and the full parameter dict) and in the released
`processing_config.yaml`. `--tracker-config PATH` overrides the shipped file.
`config/trackers/bytetrack.yaml` is a verbatim copy of the ultralytics defaults
and a test asserts it stays that way, so an ultralytics upgrade that retunes
ByteTrack fails loudly instead of silently re-tracking future data differently.

Two implementation facts worth knowing:

```text
botsort: gmc_method is set to none, deviating from the ultralytics default of
         sparseOptFlow. Global motion compensation corrects camera egomotion;
         here the camera is fixed AND the crop window is frozen at frame 0, so
         there is none to correct and estimating it from dish texture and the
         larva's own motion can feed apparent motion back into the tracker.
         with_reid stays False (<=1 object per dish gives ReID nothing to do).

ocsort:  ultralytics ships only bytetrack and botsort and raises an
         AssertionError for any other tracker_type, so OC-SORT is implemented
         in scripts/larva_trackers.py and stepped over model.predict() output
         instead of running inside model.track(). It depends only on numpy and
         scipy -- no new dependency, and unit-testable without a model. Its
         observation-centric recovery is what lets a track survive a
         detection gap with the same id, which is the property of interest
         given this pipeline's missing-detection rates. min_hits is 1 so a
         track is confirmed on its second frame, matching ByteTrack's
         is_activated semantics.
```

Whichever tracker is selected, the §6 primary-larva selection and its
hysteresis jump gate run downstream unchanged: the tracker proposes identities,
the selection layer still decides what becomes the primary observation.

## 26.2 Petri distortion: record a proxy, do not correct

Keep `petri_radius = 0.5 * min(petri_w, petri_h)` for normalization. Additionally
record per dish × frame (no correction model committed; v2 may add ellipse-fit):

```text
petri_aspect_ratio    = petri_w / petri_h
petri_distortion_flag = abs(1 - petri_aspect_ratio) > tol     # default tol 0.15
petri_offset_norm     = dist(petri_center, frame_center) / (0.5 * frame_diag)
```

CLI: `--distortion-aspect-tol 0.15`

**v1.1 optional ellipse-fit (`--petri-geometry circlefit`).** The deferred
ellipse-fit is now available opt-in. It Otsu-thresholds the bright disk inside
the YOLO-box ROI and `fitEllipse`s it, giving an accurate dish center/radius and
a *measured* ellipticity (`petri_fit_a / petri_fit_b`) rather than the box-aspect
proxy. When the fit is valid, `petri_cx/cy/radius` and larva normalization use it
(the crop window stays box-based, so detection is unaffected); it falls back to
the box otherwise. The raw box stays in `petri_x1..petri_h`; the fit is recorded
in `petri_geometry_source` + `petri_fit_cx/cy/a/b/valid`. Default is still `box`
(v1 reproducibility). Note: the petri model's box size is imgsz-sensitive
(it matches the dish at 640), so `--petri-imgsz` is pinned
(default 640) — at the pinned imgsz the box geometry already matches the dish, so
circlefit's value is imgsz-robustness + measured ellipticity, not bias fixing.

## 26.3 Interpolation then smoothing

Apply interpolation first, then smoothing, on `cx_global` / `cy_global`.

```text
interpolation: linear, only across gaps <= 0.5 s
               gap > 0.5 s  -> leave NaN, missing=True, movement_state="missing"
               gap <= 0.5 s -> interpolated=True, movement_state="interpolated"
smoothing:     Savitzky-Golay, window 7 (odd), polyorder 2
```

CLI: `--max-interp-gap-s 0.5`, `--smoothing-window 7`, `--savgol-polyorder 2`
Kinematics use smoothed coords; raw coords are always retained.

## 26.4 jerk dropped; divergence deferred

```text
jerk_mm_s3:            DROPPED from v1 schema (3rd derivative ~ noise)
acceleration_mm_s2:    KEPT, flagged low-confidence in data dictionary
js_divergence_*,
wasserstein_*:         DEFERRED to a post-processing step; written as NaN in v1.
                       v1.1 default reference: pool all treatment="control"
                       dishes within the same experiment_id into a probability
                       heatmap, compare per-dish heatmaps against it.
```
