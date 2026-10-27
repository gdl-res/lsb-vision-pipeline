"""Tests for the model-validation helpers (no ultralytics/weights needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_models as vm  # noqa: E402


def _make_dataset(root: Path, splits=("train", "valid", "test")):
    yaml = pytest.importorskip("yaml")
    for s in splits:
        (root / s / "images").mkdir(parents=True)
        (root / s / "labels").mkdir(parents=True)
    # broken absolute paths, as Roboflow ships them
    (root / "data.yaml").write_text(yaml.safe_dump({
        "train": "/gone/train/images", "val": "/gone/valid/images",
        "test": "/gone/test/images", "nc": 1, "names": ["Tenebrio molitor"]}))
    return root


def test_find_data_yaml(tmp_path):
    pytest.importorskip("yaml")
    _make_dataset(tmp_path)
    assert vm.find_data_yaml(tmp_path).name == "data.yaml"


def test_find_data_yaml_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        vm.find_data_yaml(tmp_path)


def test_resolve_dataset_yaml_rewrites_paths(tmp_path):
    yaml = pytest.importorskip("yaml")
    ds = _make_dataset(tmp_path / "ds")
    out = tmp_path / "out"
    yaml_path, info = vm.resolve_dataset_yaml(ds, out)
    resolved = yaml.safe_load(yaml_path.read_text())
    assert resolved["path"] == str(ds.resolve())
    assert resolved["val"] == "valid/images"          # 'val' -> valid/ dir
    assert resolved["test"] == "test/images"
    assert info["names"] == ["Tenebrio molitor"]
    assert set(info["splits_present"]) == {"train", "val", "test"}


def test_resolve_only_present_splits(tmp_path):
    pytest.importorskip("yaml")
    ds = _make_dataset(tmp_path / "ds", splits=("train", "valid"))  # no test
    _, info = vm.resolve_dataset_yaml(ds, tmp_path / "out")
    assert "test" not in info["splits_present"]
    assert set(info["splits_present"]) == {"train", "val"}


def test_metrics_from_results_dict():
    rd = {"metrics/precision(B)": 0.912345, "metrics/recall(B)": 0.8,
          "metrics/mAP50(B)": 0.95, "metrics/mAP50-95(B)": 0.71,
          "metrics/mAP50(M)": 0.93, "metrics/mAP50-95(M)": 0.66,
          "fitness": 0.69, "irrelevant": 1.0}
    m = vm.metrics_from_results_dict(rd)
    assert m["box_precision"] == 0.91234           # rounded to 5dp
    assert m["box_map50"] == 0.95 and m["mask_map50"] == 0.93
    assert "irrelevant" not in m


def test_count_split_images(tmp_path):
    ds = _make_dataset(tmp_path / "ds")
    for n in ("a.jpg", "b.png", "c.txt"):
        (ds / "test" / "images" / n).write_bytes(b"")
    assert vm.count_split_images(ds, "test") == 2     # .txt ignored


def test_render_report_contains_metrics():
    row = {"model": "m.pt", "split": "test", "n_images": 171, "imgsz": 1024,
           "conf": 0.001, "iou": 0.6, "timestamp": "2026-06-30T00:00:00",
           "box_map50": 0.95, "mask_map50": 0.93, "box_precision": 0.9,
           "fitness": 0.7}
    md = vm.render_report(row, {"names": ["Tenebrio molitor"]})
    assert "mAP@50" in md and "0.95" in md and "0.93" in md
    assert "171 images" in md


def test_main_missing_model(tmp_path):
    rc = vm.main(["--model", str(tmp_path / "nope.pt"),
                  "--dataset", str(tmp_path), "--out", str(tmp_path / "o")])
    assert rc == 1
