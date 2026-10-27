"""Unit + smoke tests for generate_fair_metadata."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_fair_metadata as fm  # noqa: E402


# --- data dictionary inference --------------------------------------------- #
def test_data_dictionary_covers_every_column():
    rows = fm.build_data_dictionary()
    by_file: dict[str, set] = {}
    for r in rows:
        by_file.setdefault(r["file"], set()).add(r["column_name"])
    for file_path, columns in fm.FILES:
        assert by_file[file_path] == set(columns), file_path


def test_inferred_units():
    assert fm._unit("step_distance_mm") == "millimeters"
    assert fm._unit("speed_mm_s") == "mm/s"
    assert fm._unit("acceleration_mm_s2") == "mm/s^2"
    assert fm._unit("mask_area_mm2") == "mm^2"
    assert fm._unit("turn_angle_rad") == "radians"
    assert fm._unit("angular_speed_rad_s") == "rad/s"
    assert fm._unit("mask_area_px") == "pixels"
    assert fm._unit("immobility_fraction") == "unitless [0,1]"
    assert fm._unit("x_norm_raw") == "petri radii"
    assert fm._unit("heatmap_entropy") == "nats"
    # smoothed/raw variants inherit their base coordinate-system unit
    assert fm._unit("x_norm_smooth") == "petri radii"
    assert fm._unit("r_norm_smooth") == "petri radii"
    assert fm._unit("cx_global_smooth") == "pixels"
    assert fm._unit("cx_global_raw") == "pixels"
    assert fm._unit("x1_global") == "pixels"


def test_inferred_types():
    assert fm._data_type("movement_bout_count") == "integer"
    assert fm._data_type("inside_petri") == "boolean"
    assert fm._data_type("petri_distortion_flag") == "boolean"
    assert fm._data_type("selection_reason") == "string"
    assert fm._data_type("mean_speed_mm_s") == "float"
    assert fm._data_type("dish_id") == "integer"


def test_missing_value_policy():
    assert fm._missing_policy("float") == "NaN"
    assert fm._missing_policy("boolean") == "true/false"
    assert fm._missing_policy("string") == "empty string"


def test_controlled_vocab_present():
    rows = {(r["file"], r["column_name"]): r for r in fm.build_data_dictionary()}
    mv = rows[("kinematics/larva_kinematics.csv", "movement_state")]
    assert "moving" in mv["allowed_values"] and "uncertain" in mv["allowed_values"]
    sr = rows[("larva/larva_tracks.csv", "selection_reason")]
    assert "same_track_id" in sr["allowed_values"]


# --- end-to-end scaffold --------------------------------------------------- #
def test_end_to_end_scaffold(tmp_path):
    pytest.importorskip("yaml")
    rc = fm.main(["--out", str(tmp_path), "--no-pip-freeze",
                  "--creators", "Jane Doe; John Roe",
                  "--contact", "x@example.org"])
    assert rc == 0

    for rel in ["README.md", "LICENSE", "LICENSE-DATA",
                "dataset_description.json",
                "data_dictionary.csv", "processing_config.yaml",
                "requirements.txt", "metadata/experiment_metadata.csv",
                "model_cards/petri_detector_model_card.md",
                "model_cards/larva_segmenter_model_card.md"]:
        assert (tmp_path / rel).is_file(), rel

    desc = json.loads((tmp_path / "dataset_description.json").read_text())
    assert desc["creators"] == ["Jane Doe", "John Roe"]
    assert desc["species"] == "Tenebrio molitor"
    assert "license" in desc

    # default dual license: code Apache-2.0 (LICENSE), data CC-BY-4.0 (LICENSE-DATA)
    assert desc["code_license"] == "Apache-2.0"
    assert desc["data_license"] == "CC-BY-4.0"
    assert "Apache License" in (tmp_path / "LICENSE").read_text()
    data_lic = (tmp_path / "LICENSE-DATA").read_text()
    assert "Creative Commons Attribution 4.0" in data_lic
    assert "Jane Doe; John Roe" in data_lic  # attribution string filled in

    with open(tmp_path / "data_dictionary.csv") as fh:
        dd = list(csv.DictReader(fh))
    assert len(dd) > 100  # many documented columns
    assert all(r["data_type"] and r["missing_value_policy"] for r in dd)

    import yaml
    cfg = yaml.safe_load((tmp_path / "processing_config.yaml").read_text())
    assert cfg["smoothing"]["window"] == 7
    assert cfg["kinematics"]["jerk"] == "dropped"
    assert cfg["features"]["divergence_from_control"] == "deferred_nan_v1"


def test_default_smoothing_is_savgol_in_dictionary():
    rows = {r["column_name"]: r for r in fm.build_data_dictionary(
        fm.smoothing_meta(fm.build_parser().parse_args(
            ["--out", "x"]))[1])}
    desc = rows["cx_global_smooth"]["description"]
    assert "Savitzky-Golay" in desc and "window 7" in desc


def test_kalman_filter_documented(tmp_path):
    pytest.importorskip("yaml")
    rc = fm.main(["--out", str(tmp_path), "--no-pip-freeze",
                  "--smoothing-filter", "kalman",
                  "--kalman-process-var", "2.0"])
    assert rc == 0
    import yaml
    cfg = yaml.safe_load((tmp_path / "processing_config.yaml").read_text())
    assert cfg["smoothing"] == {"method": "kalman", "process_var": 2.0,
                                "measurement_var": 4.0}
    with open(tmp_path / "data_dictionary.csv") as fh:
        rows = {r["column_name"]: r for r in csv.DictReader(fh)
                if r["file"].endswith("larva_kinematics.csv")}
    assert "Kalman" in rows["r_norm_smooth"]["description"]


def test_experiment_metadata_header(tmp_path):
    fm.write_experiment_metadata_template(tmp_path / "em.csv")
    with open(tmp_path / "em.csv") as fh:
        header = next(csv.reader(fh))
    assert header == fm.EXPERIMENT_METADATA_COLUMNS
    assert "treatment" in header and "petri_diameter_mm" in header


def test_processing_config_records_the_tracker_actually_used(tmp_path):
    """A release must state its association algorithm rather than assuming the
    v1 default (SPECIFICATION.md 26.1)."""
    pytest.importorskip("yaml")
    rc = fm.main(["--out", str(tmp_path), "--no-pip-freeze",
                  "--tracker", "ocsort"])
    assert rc == 0
    import yaml
    cfg = yaml.safe_load((tmp_path / "processing_config.yaml").read_text())
    tracking = cfg["larva_tracking"]
    assert tracking["tracker"] == "ocsort"
    assert tracking["tracker_config"] == "config/trackers/ocsort.yaml"
    assert tracking["tracker_params"]["tracker_type"] == "ocsort"
    assert tracking["per_dish_instance"] is True


def test_processing_config_defaults_to_bytetrack(tmp_path):
    pytest.importorskip("yaml")
    assert fm.main(["--out", str(tmp_path), "--no-pip-freeze"]) == 0
    import yaml
    cfg = yaml.safe_load((tmp_path / "processing_config.yaml").read_text())
    assert cfg["larva_tracking"]["tracker"] == "bytetrack"
    assert cfg["larva_tracking"]["tracker_params"]["track_buffer"] == 30
