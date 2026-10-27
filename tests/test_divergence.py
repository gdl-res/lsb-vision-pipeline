"""Unit + integration tests for compute_divergence (no YOLO)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compute_divergence as dv  # noqa: E402


# --- pure: js_divergence --------------------------------------------------- #
def test_js_divergence_identical_is_zero():
    p = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert dv.js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_disjoint_is_ln2():
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert dv.js_divergence(p, q) == pytest.approx(math.log(2.0), abs=1e-12)


def test_js_divergence_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    assert dv.js_divergence(p, q) == pytest.approx(dv.js_divergence(q, p))


def test_js_divergence_bounded_by_ln2():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.random(16)
        q = rng.random(16)
        d = dv.js_divergence(p, q)
        assert -1e-12 <= d <= math.log(2.0) + 1e-12


def test_js_divergence_base2():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert dv.js_divergence(p, q, base="2") == pytest.approx(1.0, abs=1e-12)


def test_js_divergence_empty_is_nan():
    assert math.isnan(dv.js_divergence(np.zeros(4), np.array([1.0, 0, 0, 0])))


# --- pure: pooling & normalization ----------------------------------------- #
def test_pool_counts_sums():
    a = np.ones((3, 3))
    b = np.full((3, 3), 2.0)
    pooled = dv.pool_counts([a, None, b])
    assert np.all(pooled == 3.0)


def test_pool_counts_none_when_empty():
    assert dv.pool_counts([]) is None
    assert dv.pool_counts([None, None]) is None


def test_pool_counts_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dv.pool_counts([np.ones((3, 3)), np.ones((4, 4))])


def test_normalize_sums_to_one():
    counts = np.array([[1.0, 3.0], [0.0, 0.0]])
    assert dv.normalize(counts).sum() == pytest.approx(1.0)


def test_normalize_all_zero_stays_zero():
    assert dv.normalize(np.zeros((2, 2))).sum() == 0.0


# --- pure: wasserstein on the radial marginal ------------------------------ #
def _center_grid(bins=10):
    g = np.zeros((bins, bins))
    g[bins // 2, bins // 2] = 10.0
    return g


def _edge_grid(bins=10):
    g = np.zeros((bins, bins))
    g[0, bins // 2] = 10.0  # x ~ -0.9 -> large radius
    return g


def test_wasserstein_identical_is_zero():
    g = _center_grid()
    assert dv.wasserstein_radial(g, g) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_center_vs_edge_positive():
    d = dv.wasserstein_radial(_center_grid(), _edge_grid())
    assert d > 0.5  # roughly the radial gap in petri radii


def test_wasserstein_empty_is_nan():
    assert math.isnan(dv.wasserstein_radial(np.zeros((4, 4)), _center_grid(4)))


# --- pure: score_dish status logic ----------------------------------------- #
def test_score_dish_ok():
    js, w, status = dv.score_dish(_center_grid(), _center_grid())
    assert status == dv.STATUS_OK
    assert js == pytest.approx(0.0, abs=1e-9)
    assert w == pytest.approx(0.0, abs=1e-9)


def test_score_dish_no_control():
    js, w, status = dv.score_dish(_center_grid(), None)
    assert status == dv.STATUS_NO_CONTROL and math.isnan(js)


def test_score_dish_no_heatmap():
    _, _, status = dv.score_dish(None, _center_grid())
    assert status == dv.STATUS_NO_HEATMAP


def test_score_dish_empty_heatmap():
    _, _, status = dv.score_dish(np.zeros((10, 10)), _center_grid())
    assert status == dv.STATUS_EMPTY


def test_heatmap_stem_zero_pads():
    assert dv.heatmap_stem("vid", 2) == "vid_dish02"
    assert dv.heatmap_stem("vid", 12) == "vid_dish12"


# --- integration: load_counts + full main ---------------------------------- #
def _write_counts(heatmaps: Path, video: str, dish: int, grid: np.ndarray):
    npy = heatmaps / "npy"
    npy.mkdir(parents=True, exist_ok=True)
    np.save(npy / f"{dv.heatmap_stem(video, dish)}_counts.npy", grid)


def test_load_counts_roundtrip(tmp_path):
    _write_counts(tmp_path, "v1", 1, _center_grid())
    arr = dv.load_counts(tmp_path, "v1", 1)
    assert arr is not None and arr.shape == (10, 10)
    assert dv.load_counts(tmp_path, "v1", 9) is None


def _setup_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    feat = tmp_path / "features"
    heat = tmp_path / "heatmaps"
    feat.mkdir()
    # two control dishes (center) + one treated dish (edge), same experiment
    rows = [
        ("v1", 1, "exp1"), ("v1", 2, "exp1"), ("v1", 3, "exp1"),
    ]
    fdf = pd.DataFrame([
        {"dataset_id": "d", "experiment_id": e, "video_id": v, "dish_id": k,
         "js_divergence_from_control": float("nan"),
         "wasserstein_distance_from_control": float("nan")}
        for (v, k, e) in rows
    ])
    fdf.to_csv(feat / "features_per_video.csv", index=False)
    _write_counts(heat, "v1", 1, _center_grid())
    _write_counts(heat, "v1", 2, _center_grid())
    _write_counts(heat, "v1", 3, _edge_grid())
    meta = pd.DataFrame([
        {"video_id": "v1", "dish_id": 1, "treatment": "control",
         "experiment_id": "exp1"},
        {"video_id": "v1", "dish_id": 2, "treatment": "control",
         "experiment_id": "exp1"},
        {"video_id": "v1", "dish_id": 3, "treatment": "heat",
         "experiment_id": "exp1"},
    ])
    meta_path = tmp_path / "experiment_metadata.csv"
    meta.to_csv(meta_path, index=False)
    return feat, heat, meta_path


def test_main_fills_divergence(tmp_path):
    pd = pytest.importorskip("pandas")
    feat, heat, meta_path = _setup_dataset(tmp_path)
    rc = dv.main(["--features", str(feat / "features_per_video.csv"),
                  "--heatmaps", str(heat), "--metadata", str(meta_path)])
    assert rc == 0

    out = pd.read_csv(feat / "features_per_video.csv")
    by_dish = {r.dish_id: r for r in out.itertuples()}
    # treated dish is far from the control reference -> positive divergence
    assert by_dish[3].js_divergence_from_control > 0
    assert by_dish[3].wasserstein_distance_from_control > 0
    # control dishes coincide with the (center) reference -> ~0
    assert by_dish[1].js_divergence_from_control == pytest.approx(0.0, abs=1e-6)

    audit = pd.read_csv(feat / "divergence_from_control.csv")
    assert set(audit.columns) == set(dv.DETAIL_COLUMNS)
    assert (audit["divergence_status"] == dv.STATUS_OK).all()
    assert (audit["n_control_dishes_ref"] == 2).all()
    # provenance recorded
    prov = (feat / "processing_provenance.jsonl").read_text()
    assert "compute_divergence" in prov


def test_main_exclude_self_leave_one_out(tmp_path):
    pd = pytest.importorskip("pandas")
    feat, heat, meta_path = _setup_dataset(tmp_path)
    rc = dv.main(["--features", str(feat / "features_per_video.csv"),
                  "--heatmaps", str(heat), "--metadata", str(meta_path),
                  "--exclude-self"])
    assert rc == 0
    audit = pd.read_csv(feat / "divergence_from_control.csv").set_index("dish_id")
    # control dishes drop themselves: reference is the one other control dish
    assert audit.loc[1, "n_control_dishes_ref"] == 1
    # treated dish keeps both controls
    assert audit.loc[3, "n_control_dishes_ref"] == 2


def test_main_no_metadata_writes_nan(tmp_path):
    pd = pytest.importorskip("pandas")
    feat, heat, _ = _setup_dataset(tmp_path)
    rc = dv.main(["--features", str(feat / "features_per_video.csv"),
                  "--heatmaps", str(heat)])
    assert rc == 0
    out = pd.read_csv(feat / "features_per_video.csv")
    assert out["js_divergence_from_control"].isna().all()
    audit = pd.read_csv(feat / "divergence_from_control.csv")
    assert (audit["divergence_status"] == dv.STATUS_NO_METADATA).all()


def test_main_no_control_in_experiment(tmp_path):
    pd = pytest.importorskip("pandas")
    feat, heat, _ = _setup_dataset(tmp_path)
    # metadata where nothing is a control
    meta = pd.DataFrame([
        {"video_id": "v1", "dish_id": k, "treatment": "heat",
         "experiment_id": "exp1"} for k in (1, 2, 3)])
    mp = tmp_path / "meta2.csv"
    meta.to_csv(mp, index=False)
    rc = dv.main(["--features", str(feat / "features_per_video.csv"),
                  "--heatmaps", str(heat), "--metadata", str(mp)])
    assert rc == 0
    audit = pd.read_csv(feat / "divergence_from_control.csv")
    assert (audit["divergence_status"] == dv.STATUS_NO_CONTROL).all()
