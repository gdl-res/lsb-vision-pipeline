"""Unit + smoke tests for compute_behavior_features (no YOLO needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compute_behavior_features as cf  # noqa: E402


# --- pure helpers ---------------------------------------------------------- #
def test_agg_ignores_nan_and_handles_empty():
    assert cf.agg(np.array([1.0, np.nan, 3.0]), np.mean) == 2.0
    assert math.isnan(cf.agg(np.array([np.nan, np.nan]), np.mean))


def test_movement_bouts_counts_runs():
    states = ["moving", "moving", "immobile", "moving", "missing", "moving"]
    bouts = cf.movement_bouts(states, median_dt=0.1)
    # runs: [0,1], [3], [5] -> three bouts
    assert len(bouts) == 3
    assert math.isclose(bouts[0], 0.2)
    assert math.isclose(bouts[1], 0.1)


def test_movement_bouts_none():
    assert cf.movement_bouts(["immobile", "missing"], 0.1) == []


def test_band_fractions_partition():
    r = np.array([0.1, 0.2, 0.5, 0.9, 0.95])  # center:2 middle:1 edge:2
    c, m, e = cf.band_fractions(r)
    assert math.isclose(c, 0.4)
    assert math.isclose(m, 0.2)
    assert math.isclose(e, 0.4)
    assert math.isclose(c + m + e, 1.0)


def test_band_fractions_empty():
    c, m, e = cf.band_fractions(np.array([np.nan]))
    assert all(math.isnan(v) for v in (c, m, e))


def test_zero_crossings():
    assert cf.zero_crossings(np.array([1.0, -1.0, 1.0, 2.0])) == 2
    assert cf.zero_crossings(np.array([1.0, 2.0, 3.0])) == 0
    assert cf.zero_crossings(np.array([np.nan, np.nan])) == 0


def test_occupancy_features_single_cell():
    x = np.full(50, 0.0)
    y = np.full(50, 0.0)
    occ = cf.occupancy_features(x, y, bins=20)
    assert occ["occupied_cells"] == 1
    assert math.isclose(occ["occupancy_entropy"], 0.0, abs_tol=1e-9)
    # all 50 points in 1 cell -> 49/50 revisit
    assert math.isclose(occ["revisit_rate"], 49 / 50)


# --- end-to-end ------------------------------------------------------------ #
def _write_inputs(tmp_path):
    pd = pytest.importorskip("pandas")
    n = 120  # 12 s at 10 fps -> 2 windows at 6 s
    track_rows, kin_rows = [], []
    for i in range(n):
        cx = 200.0 + (i % 30)  # oscillating-ish motion, stays in dish
        xn = (cx - 200.0) / 100.0
        track_rows.append({
            "dataset_id": "d", "experiment_id": "", "video_id": "v",
            "frame_idx": i, "time_s": i / 10.0, "dish_id": 1,
            "valid_detection": True, "larva_conf": 0.8,
            "x_norm_raw": xn, "y_norm_raw": 0.0,
            # smoothed columns written as EMPTY placeholders by the tracker
            # (SPECIFICATION.md 8.2); the kinematics file supplies real values.
            "x_norm_smooth": "", "y_norm_smooth": "", "r_norm_smooth": "",
            "inside_petri": True, "mask_area_px": 100.0 + (i % 5),
            "mask_area_norm_petri": 0.003, "mask_aspect_ratio": 2.0,
            "mask_eccentricity": 0.8, "mask_compactness": 0.5,
            "track_switch": False, "outlier_position": False,
            "outlier_area": False,
        })
        kin_rows.append({
            "video_id": "v", "frame_idx": i, "dish_id": 1,
            "x_norm_smooth": xn, "y_norm_smooth": 0.0, "r_norm_smooth": abs(xn),
            "step_distance_norm": 0.01, "step_distance_mm": 0.45,
            "speed_mm_s": 4.5, "abs_turn_angle_rad": 0.1,
            "turn_angle_rad": 0.1, "area_change_px": 1.0 if i % 2 else -1.0,
            "movement_state": "moving", "missing": False, "interpolated": False,
        })
    tracks = tmp_path / "larva_tracks.csv"
    kin = tmp_path / "larva_kinematics.csv"
    pd.DataFrame(track_rows).to_csv(tracks, index=False)
    pd.DataFrame(kin_rows).to_csv(kin, index=False)
    return tracks


def test_end_to_end_features(tmp_path):
    pd = pytest.importorskip("pandas")
    tracks = _write_inputs(tmp_path)
    out = tmp_path / "features"
    rc = cf.main(["--tracks", str(tracks), "--out", str(out),
                  "--window-size-s", "6.0", "--heatmap-bins", "30"])
    assert rc == 0

    win = pd.read_csv(out / "features_per_window.csv")
    vid = pd.read_csv(out / "features_per_video.csv")
    assert list(win.columns) == cf.WINDOW_COLUMNS
    assert list(vid.columns) == cf.VIDEO_COLUMNS
    assert len(win) == 2          # 12 s / 6 s windows
    assert len(vid) == 1
    # everything is "moving" -> moving_fraction 1.0, immobility 0.0
    assert math.isclose(vid["moving_fraction"].iloc[0], 1.0)
    assert math.isclose(vid["immobility_fraction"].iloc[0], 0.0)
    # divergence deferred -> NaN (26.4)
    assert vid["js_divergence_from_control"].isna().all()
    assert vid["wasserstein_distance_from_control"].isna().all()
    # heatmap features present
    assert math.isfinite(vid["heatmap_entropy"].iloc[0])
    # area oscillates every frame -> many oscillations
    assert vid["area_oscillation_count"].iloc[0] > 50
    # smoothed-coord-derived features must NOT be NaN even though the tracks
    # file carries empty x_norm_smooth placeholders (regression: kinematics
    # must win the merge for overlapping columns).
    assert math.isfinite(vid["path_efficiency"].iloc[0])
    assert math.isfinite(vid["tortuosity"].iloc[0])
    assert math.isfinite(vid["occupancy_entropy"].iloc[0])


def test_rim_occlusion_feature_quantifies_rim_seeking(tmp_path):
    pd = pytest.importorskip("pandas")
    # larva sits near the rim then vanishes twice (rim-occlusion), and has one
    # ordinary mid-dish miss that must NOT count.
    rows = []
    rseq = [0.9, 0.95, None, None, 0.9, 0.92, None, 0.3, 0.4, None]
    miss = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    for i, (rv, m) in enumerate(zip(rseq, miss)):
        rows.append({"video_id": "v", "frame_idx": i, "dish_id": 1,
                     "time_s": i / 10.0, "valid_detection": not m,
                     "missing": bool(m),
                     "r_norm_raw": "" if rv is None else rv,
                     "x_norm_raw": "" if rv is None else rv, "y_norm_raw": 0.0})
    g = pd.DataFrame(rows)
    f = cf.compute_features(g, bins=30, sharp_turn_thr=1.0, rim_r=0.85)
    # gaps at idx 2-3 (after r=0.95) and idx 6 (after r=0.92) are rim; idx 9
    # (after r=0.4) is not -> 3 rim frames / 10, 2 events.
    assert math.isclose(f["rim_occlusion_fraction"], 0.3)
    assert f["rim_occlusion_event_count"] == 2


def test_fill_identifiers_empty_id_from_metadata(tmp_path):
    # regression: features with an all-empty experiment_id (reads back as float
    # NaN) + metadata supplying string ids must not raise LossySetitemError.
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"dataset_id": ["", ""], "experiment_id": ["", ""],
                       "video_id": ["v", "v"], "dish_id": [1, 2]})
    df = df.assign(experiment_id=pd.to_numeric(df["experiment_id"],
                                               errors="coerce"))  # -> float NaN
    meta = tmp_path / "experiment_metadata.csv"
    pd.DataFrame({"video_id": ["v", "v"], "dish_id": [1, 2],
                  "dataset_id": ["ds", "ds"],
                  "experiment_id": ["exp1", "exp1"]}).to_csv(meta, index=False)
    out = cf.fill_identifiers(df, meta)
    assert list(out["experiment_id"]) == ["exp1", "exp1"]
    assert list(out["dataset_id"]) == ["ds", "ds"]


def test_load_merged_prefers_populated_kinematics(tmp_path):
    pd = pytest.importorskip("pandas")
    tracks = tmp_path / "larva_tracks.csv"
    kin = tmp_path / "larva_kinematics.csv"
    # tracks has an EMPTY smoothed placeholder; kinematics has the real value
    pd.DataFrame([{"video_id": "v", "frame_idx": 0, "dish_id": 1,
                   "x_norm_smooth": "", "larva_conf": 0.8}]).to_csv(
        tracks, index=False)
    pd.DataFrame([{"video_id": "v", "frame_idx": 0, "dish_id": 1,
                   "x_norm_smooth": 0.42}]).to_csv(kin, index=False)
    merged = cf.load_merged(tracks, kin)
    assert merged["x_norm_smooth"].iloc[0] == 0.42      # kinematics won
    assert merged["larva_conf"].iloc[0] == 0.8          # tracks-only kept
    assert "x_norm_smooth_trk" not in merged.columns    # suffix cleaned up


def test_kinematic_features_nan_without_kinematics(tmp_path):
    pd = pytest.importorskip("pandas")
    tracks = _write_inputs(tmp_path)
    (tmp_path / "larva_kinematics.csv").unlink()  # remove kinematics
    out = tmp_path / "features2"
    rc = cf.main(["--tracks", str(tracks), "--out", str(out),
                  "--window-size-s", "60.0"])
    assert rc == 0
    vid = pd.read_csv(out / "features_per_video.csv")
    # no kinematics -> speed-based features NaN, but occupancy still computed
    assert vid["mean_speed_mm_s"].isna().all()
    assert math.isfinite(vid["mean_r_norm"].iloc[0])
