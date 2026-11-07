# lsb-vision-pipeline

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
```

Install a CUDA build of PyTorch for GPU inference (see pytorch.org). CPU inference works but is slow.

## Models

Two Ultralytics YOLO models are required. Neither is included in this repository:

| Role | Task | Class |
|---|---|---|
| Petri detector (`--petri-model`) | detection or segmentation on full tray frames; only the box is used | `petri` |
| Larva segmenter (`--larva-model`) | instance segmentation on square Petri crops | `larva` |

Both are single-class models and are matched by class index, so the class name itself does not matter.

Train the larva segmenter on Petri crops, the input the pipeline feeds it. A model trained on full
tray images can score well there and still return fragmented or missing masks on crops. The
segmenter used for the v1.2.0 data (YOLO11s-seg trained on crops) is described, with its SHA-256,
in the model card of that record; the weights are available from the authors on request.

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

The defaults reproduce the published v1.2.0 dataset:

| Setting | Value |
|---|---|
| Tracker | ByteTrack |
| Petri / larva inference size | `imgsz` 640 |
| Confidence | Petri 0.30, larva 0.25 |
| Jump gate | 60 px + 5 px per missing frame, capped at 240 px |
| Gate re-acquisition | after 5 s of a consistent rejected detection (`--reacquire-after-s`, 0 disables) |
| Interpolation | linear, gaps ≤ 0.5 s |
| Smoothing | Savitzky–Golay, window 7, order 2 |
| Immobility threshold | 1 mm/s |
| Feature windows | 60 s |
| Heatmaps | 50 × 50 bins |

Version 1.2 changed two defaults: the crop-trained larva segmenter and gate re-acquisition after
5 s (`docs/SPECIFICATION.md` §6 and §26.6). Values from v1.0/v1.1 and v1.2 runs are not
interchangeable. The v1.0.0 data are reproduced with the v1 full-tray segmenter and
`--reacquire-after-s 0`.

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

Data produced with this pipeline are on Zenodo:

- Raw videos (34 tray videos): https://doi.org/10.5281/zenodo.23686766
- Derived data, v1.2.0 (this release's defaults; includes the evaluation against trajectory ground
  truth): https://doi.org/10.5281/zenodo.24015683
- Derived data, v1.0.0 (pipeline v1.0.0): https://doi.org/10.5281/zenodo.23686764
- Trajectory ground truth (66 hand-checked trajectories): https://doi.org/10.5281/zenodo.24015680
- Segmenter comparison: https://doi.org/10.5281/zenodo.24015682

## Citation

If you use this software, please cite it as described in `CITATION.cff`.

## License

The code is licensed under Apache-2.0 (see `LICENSE`). Datasets generated by the pipeline are
licensed separately; the default is CC-BY-4.0, written to `LICENSE-DATA` in the output.

Third-party dependencies are not included and keep their own licences: Ultralytics YOLO is
AGPL-3.0 (models trained or run with it are subject to its terms), PyTorch/torchvision are
BSD-3-Clause, pycocotools is BSD-2-Clause. `config/trackers/bytetrack.yaml` is a verbatim copy of
the ByteTrack configuration shipped with Ultralytics 8.4.33.
