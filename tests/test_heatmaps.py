"""Unit + smoke tests for generate_heatmaps (no YOLO; matplotlib optional)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_heatmaps as gh  # noqa: E402


# --- histogram & masking (SPECIFICATION.md 12) ------------------------------------ #
def test_histogram_counts_total():
    x = np.array([0.0, 0.5, -0.5, 0.0])
    y = np.array([0.0, 0.5, -0.5, 0.2])
    counts, xe, ye = gh.histogram_norm(x, y, bins=50)
    assert counts.shape == (50, 50)
    assert counts.sum() == 4


def test_circular_mask_zeros_corners():
    _, xe, ye = gh.histogram_norm(np.array([0.0]), np.array([0.0]), bins=10)
    mask = gh.circular_mask(xe, ye)
    assert not mask[0, 0]      # corner (-0.9,-0.9) -> r ~1.27 outside
    assert mask[5, 5]          # near center inside


def test_circular_mask_removes_outside_points():
    # one point at center (inside), one at (0.95, 0.95) -> r ~1.34 outside
    x = np.array([0.0, 0.95])
    y = np.array([0.0, 0.95])
    counts, xe, ye = gh.histogram_norm(x, y, bins=20)
    counts *= gh.circular_mask(xe, ye)
    assert counts.sum() == 1  # outside point masked away


def test_to_probability_sums_to_one():
    counts = np.array([[1.0, 1.0], [2.0, 0.0]])
    prob = gh.to_probability(counts)
    assert math.isclose(prob.sum(), 1.0)


def test_to_probability_empty():
    prob = gh.to_probability(np.zeros((3, 3)))
    assert prob.sum() == 0.0


# --- stats (SPECIFICATION.md 11.2) ------------------------------------------------ #
def test_stats_point_mass_peak_location():
    # all mass near (0.5, -0.5)
    x = np.full(100, 0.5)
    y = np.full(100, -0.5)
    counts, xe, ye = gh.histogram_norm(x, y, bins=50)
    prob = gh.to_probability(counts)
    s = gh.heatmap_stats(prob, xe, ye)
    assert abs(s["heatmap_peak_x_norm"] - 0.5) < 0.05
    assert abs(s["heatmap_peak_y_norm"] - (-0.5)) < 0.05
    assert math.isclose(s["heatmap_entropy"], 0.0, abs_tol=1e-9)  # single cell
    assert math.isclose(s["heatmap_peak_density"], 1.0)
    assert s["heatmap_spread_x"] < 0.05


def test_stats_uniform_has_higher_entropy_than_point():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.7, 0.7, 5000)
    y = rng.uniform(-0.7, 0.7, 5000)
    counts, xe, ye = gh.histogram_norm(x, y, bins=50)
    prob = gh.to_probability(counts)
    spread = gh.heatmap_stats(prob, xe, ye)
    point = gh.heatmap_stats(
        gh.to_probability(gh.histogram_norm(
            np.zeros(100), np.zeros(100), bins=50)[0]), xe, ye)
    assert spread["heatmap_entropy"] > point["heatmap_entropy"]


# --- coordinate column selection ------------------------------------------- #
def test_select_xy_prefers_smooth():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x_norm_smooth": [0.1], "y_norm_smooth": [0.2],
                       "x_norm_raw": [0.3], "y_norm_raw": [0.4]})
    assert gh.select_xy(df, prefer_smooth=True) == ("x_norm_smooth", "y_norm_smooth")
    assert gh.select_xy(df, prefer_smooth=False) == ("x_norm_raw", "y_norm_raw")


def test_select_xy_falls_back_to_raw():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x_norm_raw": [0.3], "y_norm_raw": [0.4]})
    assert gh.select_xy(df, prefer_smooth=True) == ("x_norm_raw", "y_norm_raw")


# --- end-to-end ------------------------------------------------------------ #
def test_end_to_end_writes_outputs(tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        rows.append({"video_id": "v1", "dish_id": 1,
                     "x_norm_smooth": rng.uniform(-0.5, 0.5),
                     "y_norm_smooth": rng.uniform(-0.5, 0.5)})
    inp = tmp_path / "larva_kinematics.csv"
    pd.DataFrame(rows).to_csv(inp, index=False)

    rc = gh.main(["--input", str(inp), "--out", str(tmp_path / "hm"),
                  "--bins", "30", "--no-png"])
    assert rc == 0
    counts = np.load(tmp_path / "hm" / "npy" / "v1_dish01_counts.npy")
    prob = np.load(tmp_path / "hm" / "npy" / "v1_dish01_probability.npy")
    assert counts.shape == (30, 30)
    assert math.isclose(prob.sum(), 1.0)
    meta = json.loads(
        (tmp_path / "hm" / "metadata" / "v1_dish01_metadata.json").read_text())
    assert meta["bins"] == 30
    assert meta["valid_points"] == 200
    assert meta["circular_mask_applied"] is True
    assert "heatmap_entropy" in meta
