"""Unit + smoke tests for compute_kinematics (no YOLO/OpenCV needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compute_kinematics as ck  # noqa: E402


# --- interpolation (SPECIFICATION.md 26.3) ---------------------------------------- #
def test_interpolate_short_gap_filled():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    v = np.array([0.0, np.nan, np.nan, 3.0])
    filled, interp, missing = ck.interpolate_gaps(v, t, max_gap_s=0.5)
    np.testing.assert_allclose(filled, [0.0, 1.0, 2.0, 3.0])
    assert interp.tolist() == [False, True, True, False]
    assert not missing.any()


def test_interpolate_long_gap_left_missing():
    t = np.array([0.0, 0.1, 1.0, 1.1])  # gap from 0.1 -> 1.0 is 0.9 s
    v = np.array([0.0, np.nan, np.nan, 3.0])
    filled, interp, missing = ck.interpolate_gaps(v, t, max_gap_s=0.5)
    assert math.isnan(filled[1]) and math.isnan(filled[2])
    assert not interp.any()
    assert missing.tolist() == [False, True, True, False]


def test_interpolate_leading_trailing_nan_stay_missing():
    t = np.array([0.0, 0.1, 0.2])
    v = np.array([np.nan, 1.0, np.nan])
    filled, interp, missing = ck.interpolate_gaps(v, t, max_gap_s=10.0)
    assert missing.tolist() == [True, False, True]


# --- smoothing ------------------------------------------------------------- #
def test_savgol_preserves_nan_runs():
    pytest.importorskip("scipy")
    v = np.array([1.0, 2.0, 3.0, np.nan, 10.0, 11.0])
    out = ck.savgol_smooth(v, window=7, polyorder=2)
    assert math.isnan(out[3])
    assert len(out) == len(v)


def test_savgol_smooths_linear_unchanged():
    pytest.importorskip("scipy")
    v = np.arange(20, dtype=float)  # a line is invariant under savgol poly>=1
    out = ck.savgol_smooth(v, window=7, polyorder=2)
    np.testing.assert_allclose(out, v, atol=1e-9)


# --- selectable filters ---------------------------------------------------- #
def _args(**kw):
    import argparse
    defaults = dict(smoothing_filter="savgol", smoothing_window=7,
                    savgol_polyorder=2, kalman_process_var=1.0,
                    kalman_measurement_var=4.0)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_moving_average_preserves_nan_and_length():
    v = np.array([1.0, 2.0, 3.0, np.nan, 10.0, 11.0])
    out = ck.moving_average_smooth(v, window=3)
    assert math.isnan(out[3])
    assert len(out) == len(v)


def test_moving_average_constant_unchanged():
    v = np.full(10, 5.0)
    np.testing.assert_allclose(ck.moving_average_smooth(v, window=5), v)


def test_moving_average_reduces_variance():
    rng = np.random.default_rng(0)
    v = 100.0 + rng.normal(0, 1, 200)
    out = ck.moving_average_smooth(v, window=7)
    assert np.std(out) < np.std(v)


def test_kalman_preserves_nan_and_length():
    v = np.array([1.0, 2.0, 3.0, np.nan, 10.0, 11.0])
    t = np.arange(len(v), dtype=float)
    out = ck.kalman_smooth(v, t, process_var=1.0, measurement_var=4.0)
    assert math.isnan(out[3])
    assert len(out) == len(v)


def test_kalman_tracks_linear_motion():
    t = np.arange(30, dtype=float)
    v = 2.0 * t + 5.0  # constant velocity -> CV model fits closely
    out = ck.kalman_smooth(v, t, process_var=1.0, measurement_var=4.0)
    np.testing.assert_allclose(out[5:], v[5:], atol=0.5)


def test_smooth_series_dispatch_none_is_identity():
    v = np.array([1.0, 5.0, 2.0, np.nan, 9.0])
    t = np.arange(len(v), dtype=float)
    out = ck.smooth_series(v, t, _args(smoothing_filter="none"))
    np.testing.assert_array_equal(out[~np.isnan(out)], v[~np.isnan(v)])
    assert out is not v  # returns a copy


def test_smooth_series_dispatch_selects_filter():
    pytest.importorskip("scipy")
    v = 100.0 + np.random.default_rng(1).normal(0, 1, 50)
    t = np.arange(len(v), dtype=float)
    for method in ("savgol", "moving_average", "kalman"):
        out = ck.smooth_series(v, t, _args(smoothing_filter=method))
        assert len(out) == len(v)
        assert np.std(out) < np.std(v)


# --- diffs / turn angles --------------------------------------------------- #
def test_diff_prepend_nan():
    d = ck.diff_prepend_nan(np.array([0.0, 2.0, 5.0]))
    assert math.isnan(d[0])
    np.testing.assert_allclose(d[1:], [2.0, 3.0])


def test_turn_angle_right_turn():
    # move +x then +y -> 90 degree left turn (positive angle)
    dx = np.array([np.nan, 1.0, 0.0])
    dy = np.array([np.nan, 0.0, 1.0])
    ta = ck.turn_angles(dx, dy)
    assert math.isnan(ta[1])
    assert math.isclose(ta[2], math.pi / 2, abs_tol=1e-9)


def test_turn_angle_straight_is_zero():
    dx = np.array([np.nan, 1.0, 1.0])
    dy = np.array([np.nan, 0.0, 0.0])
    ta = ck.turn_angles(dx, dy)
    assert math.isclose(ta[2], 0.0, abs_tol=1e-9)


# --- movement classification (SPECIFICATION.md 10) -------------------------------- #
def test_classify_movement_states():
    assert ck.classify_movement(5.0, 1.0, False, False) == "moving"
    assert ck.classify_movement(0.2, 1.0, False, False) == "immobile"
    assert ck.classify_movement(5.0, 1.0, True, False) == "missing"
    assert ck.classify_movement(5.0, 1.0, False, True) == "interpolated"
    assert ck.classify_movement(float("nan"), 1.0, False, False) == "uncertain"


# --- end-to-end smoke test ------------------------------------------------- #
def test_end_to_end_runs(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    # Larva moving along +x at 1 px/frame, 10 fps, dish radius 100 px @ (200,200)
    n = 12
    rows = []
    for i in range(n):
        cx = 200.0 + i  # moves right
        rows.append({
            "dataset_id": "d", "experiment_id": "e", "video_id": "v",
            "frame_idx": i, "time_s": i / 10.0, "dish_id": 1,
            "larva_track_id": 1001, "valid_detection": True,
            "cx_global_raw": cx, "cy_global_raw": 200.0,
            "mask_area_px": 100.0 + i,
            "petri_cx": 200.0, "petri_cy": 200.0, "petri_radius": 100.0,
        })
    tracks = tmp_path / "larva_tracks.csv"
    pd.DataFrame(rows).to_csv(tracks, index=False)

    rc = ck.main(["--tracks", str(tracks),
                  "--petri-diameter-mm", "90",
                  "--immobility-speed-threshold-mm-s", "1.0"])
    assert rc == 0
    out = pd.read_csv(tmp_path / "larva_kinematics.csv")
    assert len(out) == n
    assert list(out.columns) == ck.OUTPUT_COLUMNS
    # 1 px/frame * (90mm/200px) = 0.45 mm/frame * 10 fps = 4.5 mm/s -> moving
    speeds = out["speed_mm_s"].dropna()
    assert math.isclose(speeds.median(), 4.5, rel_tol=1e-3)
    assert (out["movement_state"] == "moving").sum() >= n - 2
