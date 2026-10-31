# wormstress-vision

Computer-vision pipeline for automated behavioral phenotyping of *Tenebrio molitor* larvae in
multi-arena Petri-dish assays. It turns top-down videos of a tray of six Petri dishes, each holding
zero or one larva, into the following outputs:

- framewise larva tracks
- kinematics
- occupancy heatmaps
- behavioral features
- quality-control summaries
- a FAIR-structured dataset with a data dictionary and processing provenance

```
tray video ─► Petri detector (YOLO) ─► 6 dishes, row-major IDs ─► square crop per dish
           ─► larva segmenter + tracker (YOLO-seg, one tracker per dish) ─► primary-larva selection
           ─► crop / full-frame / Petri-normalised coordinates ─► interpolation + smoothing
           ─► kinematics ─► heatmaps ─► features per window and per video ─► QC ─► FAIR metadata
```

The outputs are behavioral proxies (locomotion, immobility, spatial occupancy, area dynamics). They
are not a measure of welfare or subjective state.

## Installation

Python ≥ 3.11. The published data was produced with Python 3.13.

```bash
git clone <this repository>
cd lsb-vision-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: Mask R-CNN backend and COCO-style validation for alternative segmenters
pip install -r requirements-segmenters.txt
```

Install a CUDA build of PyTorch for GPU inference (see pytorch.org). CPU inference works but is slow.

## Models

Two Ultralytics YOLO models are required. Neither is included in this repository:

| Role | Task | Class |
|---|---|---|
| Petri detector (`--petri-model`) | detection or segmentation on full tray frames; only the box is used | `petri` |
| Larva segmenter (`--larva-model`) | instance segmentation on square Petri crops | `larva` |

Both are single-class models and are matched by class index, so the class name itself does not matter.

Validate the larva segmenter on **Petri crops**, the input the pipeline actually feeds it, not only
on full tray images: a model trained on full trays can score well there and still return fragmented
or missing masks on crops. Training the segmenter on crops (`process_dataset_splits.py`, below)
avoids this.

## Quick start

Process a directory of tray videos end to end:

```bash
python scripts/run_pipeline.py \
  --videos path/to/videos \
  --out outputs/my_dataset \
  --petri-model path/to/petri.pt \
  --larva-model path/to/larva.pt \
  --rows 3 --cols 2 \
  --petri-diameter-mm 90 \
  --track --device 0
```

Useful options:

- `--preflight` checks the videos, models, GPU and free disk space, then exits.
- `--skip-existing` resumes an interrupted batch.
- `--skip-tracking` re-runs only the downstream analysis (kinematics → QC) on existing tracking
  tables, without running the models again.
- `--metadata experiment_metadata.csv` attaches experimental conditions.

Run `python scripts/run_pipeline.py --help` for the full list of parameters (tracker,
thresholds, smoothing, QC rules).

The defaults reproduce the published dataset:

| Setting | Value |
|---|---|
| Tracker | ByteTrack |
| Petri / larva inference size | `imgsz` 640 |
| Confidence | Petri 0.30, larva 0.25 |
| Jump gate | 60 px + 5 px per missing frame, capped at 240 px |
| Interpolation | linear, gaps ≤ 0.5 s |
| Smoothing | Savitzky–Golay, window 7, order 2 |
| Immobility threshold | 1 mm/s |
| Feature windows | 60 s |
| Heatmaps | 50 × 50 bins |

## Experimental metadata

Treatment is never inferred silently from filenames. To build the metadata table:

1. Generate a draft from the filenames:

   ```bash
   python scripts/parse_filenames.py --videos path/to/videos \
     --out metadata/experiment_metadata.draft.csv --rows 3 --cols 2
   ```

2. Review the draft by hand.
3. Delete the `parse_*` audit columns.
4. Save it as `experiment_metadata.csv` and pass it with `--metadata`.

The column schema is in `docs/SPECIFICATION.md` §14.

## Output layout

```
<out>/
  README.md  LICENSE  LICENSE-DATA  dataset_description.json  data_dictionary.csv
  processing_config.yaml  requirements.txt
  metadata/    experiment_metadata.csv  processing_provenance.jsonl
  petri/       petri_tracks.csv              one row per frame × dish: dish geometry, crop window
  larva/       larva_tracks.csv              one row per frame × dish: selected larva, all coordinate
                                             systems, morphology, confidence, track id, quality flags
               larva_detections_all.csv      every raw detection, selected or not
  kinematics/  larva_kinematics.csv          speed, acceleration, turning, area change, movement_state
  features/    features_per_window.csv  features_per_video.csv
  heatmaps/    npy/  png/  metadata/         50×50 occupancy in Petri-normalised coordinates
  qc/          qc_summary.csv                quality metrics and exclusion recommendations (flags only)
  report.html                                visual summary
```

Every column is documented in `data_dictionary.csv`, which is generated with the dataset.

Coordinate conventions:

- Dish IDs are row-major: `1 2 / 3 4 / 5 6` for a 3 × 2 portrait tray.
- Petri-normalised coordinates put the dish centre at (0, 0) and the dish rim at radius 1.

## Individual steps

Each stage can also be run on its own:

| Script | Purpose |
|---|---|
| `track_larvae_video.py` | Petri detection, larva segmentation/tracking, framewise tables for one video |
| `compute_kinematics.py` | interpolation, smoothing, kinematics |
| `generate_heatmaps.py` | occupancy heatmaps |
| `compute_behavior_features.py` | per-window and per-video features |
| `compute_qc_summary.py` | quality-control summary and exclusion flags |
| `compute_divergence.py` | heatmap divergence from pooled controls (requires metadata with controls) |
| `aggregate_report.py` | treatment-vs-control feature contrasts within each `experiment_id` |
| `generate_fair_metadata.py` | README, data dictionary, licences, dataset description |
| `generate_report.py` | HTML/Markdown visual report |
| `parse_filenames.py` | draft experiment metadata from filenames (opt-in, needs human review) |

There are also three model-preparation tools:

- `crop_petris_yolo_seg.py` and `process_dataset_splits.py` crop full-tray YOLO-seg datasets into
  per-dish training images, remapping the labels.
- `validate_models.py` computes precision, recall and mAP on a held-out split.

## Using a different segmenter

The larva segmenter can be swapped while every downstream step (primary-larva selection, continuity
gate, interpolation, smoothing, kinematics, features, QC) stays identical. The default pipeline is
unchanged.

1. Run the pipeline once on the videos (`run_pipeline.py`); its output supplies the frozen arena
   geometry. Set the paths in `config/replay.yaml` and `config/segmenters.yaml`.
2. Declare segmenters in `config/segmenters.yaml` (`backend: ultralytics` for any YOLO-seg
   checkpoint, `backend: torchvision_maskrcnn` for Mask R-CNN R50-FPN v2).
3. Optionally train them:

   ```bash
   # Petri-crop training set from a full-tray YOLO-seg dataset
   python scripts/process_dataset_splits.py --dataset data/larva_dataset --out outputs/crop_dataset \
     --petri-model path/to/petri.pt --rows 3 --cols 2 --margin 0.05
   python scripts/train_segmenter.py --config config/segmenters.yaml --segmenter yolo11s_crop
   python scripts/train_maskrcnn.py --config config/segmenters.yaml
   ```

4. Cache raw detections and replay them through the unchanged chain:

   ```bash
   python scripts/cache_detections.py --config config/segmenters.yaml --segmenter yolo11s_crop --skip-existing
   python scripts/replay_trackers.py --config config/segmenters.yaml --config-prefix yolo11s_crop \
     --methods bytetrack --gates gate_on
   ```

   The output folder `<segmenter>__bytetrack__gate_on/` holds `larva_tracks.csv`,
   `larva_kinematics.csv`, `features/` and `qc_summary.csv` in the pipeline's schema. With the
   default segmenter and `bytetrack__gate_on` the replay reproduces the pipeline's own tables.

`replay_trackers.py --config config/replay.yaml` also compares association methods (ByteTrack,
BoT-SORT, OC-SORT, none) and the continuity gate on and off over one detection cache.

## Known limitations

- A larva segmenter trained on full tray images detects larvae less reliably on Petri crops, which
  biases absolute speed, distance and mask area; spatial-occupancy features are the least affected.
- With a crop-trained segmenter, the continuity gate can anchor on the operator's hand or vial
  while the larva is placed in the dish and then reject the real larva. A re-acquisition rule is
  needed before such a model becomes the default.

## Tests

```bash
python -m pytest
```

## Documentation

`docs/SPECIFICATION.md` is the design specification and data contract: coordinate systems,
output schemas, selection policy, QC rules and fixed defaults. Section numbers cited in the source
code refer to this document.

`config/trackers/` holds the tracker parameter files:

- `bytetrack.yaml` is a verbatim copy of the Ultralytics defaults.
- `botsort.yaml` is BoT-SORT with global motion compensation disabled.
- `ocsort.yaml` configures OC-SORT, which is implemented in `scripts/larva_trackers.py`.

## Data

A dataset produced with this pipeline (release v1.0.0) is available on Zenodo:

- Derived data: https://doi.org/10.5281/zenodo.23686764
- Raw videos: https://doi.org/10.5281/zenodo.23686766

## Citation

If you use this software, please cite it as described in `CITATION.cff`.

## License

The code is licensed under Apache-2.0 (see `LICENSE`). Datasets generated by the pipeline are
licensed separately; the default is CC-BY-4.0, written to `LICENSE-DATA` in the output.

Third-party dependencies are not included and keep their own licences: Ultralytics YOLO is
AGPL-3.0 (models trained or run with it are subject to its terms), PyTorch/torchvision are
BSD-3-Clause, pycocotools is BSD-2-Clause. `config/trackers/bytetrack.yaml` is a verbatim copy of
the ByteTrack configuration shipped with Ultralytics 8.4.33.
