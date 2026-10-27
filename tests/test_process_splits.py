"""Unit tests for process_dataset_splits pure functions (no YOLO required)."""
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import process_dataset_splits as pds  # noqa: E402


def _make_dataset(root: Path, splits, with_yaml=True):
    for s in splits:
        (root / s / "images").mkdir(parents=True)
        (root / s / "labels").mkdir(parents=True)
    if with_yaml:
        (root / "data.yaml").write_text("names:\n  0: larva\n")
    return root


def test_discover_splits_subset(tmp_path):
    _make_dataset(tmp_path, ["train", "valid"])
    assert pds.discover_splits(tmp_path) == ["train", "valid"]


def test_discover_splits_ignores_incomplete(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)  # no labels/
    assert pds.discover_splits(tmp_path) == []


def test_build_crop_argv_includes_required_and_optional(tmp_path):
    args = Namespace(
        dataset=tmp_path, out=tmp_path / "out", petri_model=Path("m.pt"),
        rows=3, cols=2, margin=0.05, conf=0.3, iou=0.5,
        min_label_area_ratio=0.0, distortion_aspect_tol=0.15,
        device="cuda:0", debug=True, max_images=20, on_failure="skip",
    )
    argv = pds.build_crop_argv(args, "train")
    assert "--images" in argv and str(tmp_path / "train" / "images") in argv
    assert argv[argv.index("--out") + 1] == str(tmp_path / "out" / "train")
    assert "--debug" in argv
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--max-images") + 1] == "20"


def test_build_crop_argv_omits_unset_optionals(tmp_path):
    args = Namespace(
        dataset=tmp_path, out=tmp_path / "out", petri_model=Path("m.pt"),
        rows=2, cols=3, margin=0.05, conf=0.3, iou=0.5,
        min_label_area_ratio=0.0, distortion_aspect_tol=0.15,
        device=None, debug=False, max_images=None, on_failure="keep-best",
    )
    argv = pds.build_crop_argv(args, "test")
    assert "--device" not in argv
    assert "--max-images" not in argv
    assert "--debug" not in argv
    assert argv[argv.index("--on-failure") + 1] == "keep-best"


def test_load_class_names_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    _make_dataset(tmp_path, ["train"])
    assert pds.load_class_names(tmp_path) == {0: "larva"}


def test_load_class_names_list_form(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "data.yaml").write_text("names: [larva, debris]\n")
    assert pds.load_class_names(tmp_path) == {0: "larva", 1: "debris"}


def test_load_class_names_default_when_missing(tmp_path):
    assert pds.load_class_names(tmp_path) == {0: "larva"}


def test_write_crops_data_yaml(tmp_path):
    pytest.importorskip("yaml")
    import yaml
    out = tmp_path / "out"
    out.mkdir()
    path = pds.write_crops_data_yaml(out, ["train", "valid"], {0: "larva"})
    data = yaml.safe_load(path.read_text())
    assert data["train"] == "train/images"
    assert data["val"] == "valid/images"
    assert "test" not in data
    assert data["names"] == {0: "larva"}
