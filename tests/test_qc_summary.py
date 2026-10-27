"""Unit + smoke tests for compute_qc_summary (no YOLO needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compute_qc_summary as qc  # noqa: E402


# --- pure helpers ---------------------------------------------------------- #
def test_longest_run():
    assert qc.longest_run([False, True, True, False, True]) == 2
    assert qc.longest_run([True, True, True]) == 3
    assert qc.longest_run([False, False]) == 0


def test_median_dt():
    assert math.isclose(qc.median_dt(np.array([0.0, 0.1, 0.2, 0.3])), 0.1)
    assert math.isnan(qc.median_dt(np.array([1.0])))


def test_mad_outlier_fraction_flags_spike():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(10.0, 1.0, 40), [1000.0]])  # one gross outlier
    frac = qc.mad_outlier_fraction(v, k=3.5)
    assert 0 < frac < 0.1


def test_mad_outlier_fraction_degenerate_returns_zero():
    # >50% identical values -> MAD=0 -> cannot flag (documented limitation)
    assert qc.mad_outlier_fraction(np.array([10.0] * 20 + [1000.0]), 3.5) == 0.0


def test_mad_outlier_fraction_no_spread():
    assert qc.mad_outlier_fraction(np.array([5.0, 5.0, 5.0, 5.0]), 3.5) == 0.0


def test_step_distances():
    x = np.array([0.0, 3.0, 3.0])
    y = np.array([0.0, 0.0, 4.0])
    steps = qc.step_distances(x, y)
    np.testing.assert_allclose(steps, [3.0, 4.0])


def _thr(**over):
    """Full threshold dict with the corrected defaults; override per test."""
    base = {"min_valid_fraction": 0.8, "max_missing_gap_s": 10.0,
            "max_fraction_outside": 0.05, "max_step_norm": 0.5,
            "max_semantic_switch_rate_per_min": 6.0,
            "max_track_switch_rate_per_min": None,
            "empty_max_area_norm": 0.001, "empty_max_conf": 0.45,
            "empty_max_spread": 0.15}
    base.update(over)
    return base


def test_evaluate_exclusion_rules():
    thr = _thr()
    good = {"valid_detection_fraction": 0.95, "longest_missing_gap_s": 1.0,
            "fraction_outside_petri": 0.0, "semantic_switch_rate_per_min": 1.0,
            "track_switch_rate_per_min": 1.0}
    flag, reason = qc.evaluate_exclusion(good, thr)
    assert not flag and reason == ""

    bad = {"valid_detection_fraction": 0.5, "longest_missing_gap_s": 20.0,
           "fraction_outside_petri": 0.2, "semantic_switch_rate_per_min": 30.0,
           "track_switch_rate_per_min": 30.0}
    flag, reason = qc.evaluate_exclusion(bad, thr)
    assert flag
    assert "valid_detection_fraction" in reason
    assert "longest_missing_gap_s" in reason
    assert "fraction_outside_petri" in reason
    assert "semantic_switch_rate_per_min" in reason


def test_nominal_track_switch_gate_disabled_by_default():
    """A high nominal track-switch rate must NOT flag while the gate is off."""
    thr = _thr()  # max_track_switch_rate_per_min = None
    m = {"valid_detection_fraction": 0.99, "longest_missing_gap_s": 1.0,
         "fraction_outside_petri": 0.0, "semantic_switch_rate_per_min": 0.0,
         "track_switch_rate_per_min": 999.0}
    flag, reason = qc.evaluate_exclusion(m, thr)
    assert not flag and "track_switch_rate_per_min" not in reason


def test_nominal_track_switch_gate_opt_in():
    """Setting a finite threshold re-enables the nominal-ID exclusion rule."""
    m = {"valid_detection_fraction": 0.99, "longest_missing_gap_s": 1.0,
         "fraction_outside_petri": 0.0, "semantic_switch_rate_per_min": 0.0,
         "track_switch_rate_per_min": 30.0}
    assert qc.evaluate_exclusion(m, _thr())[0] is False
    flag, reason = qc.evaluate_exclusion(
        m, _thr(max_track_switch_rate_per_min=6.0))
    assert flag and "track_switch_rate_per_min" in reason


def test_semantic_switch_gate_is_length_normalized():
    """Same teleport count over a longer clip must not auto-flag."""
    thr = _thr(min_valid_fraction=0.8)
    base = {"valid_detection_fraction": 0.99, "longest_missing_gap_s": 1.0,
            "fraction_outside_petri": 0.0, "track_switch_rate_per_min": 0.0}
    # 9 teleports in 10 s -> 54/min (flag); 9 in 5 min -> 1.8/min (fine)
    short = {**base, "semantic_switch_rate_per_min": 9 / (10 / 60)}
    long = {**base, "semantic_switch_rate_per_min": 9 / 5.0}
    assert qc.evaluate_exclusion(short, thr)[0] is True
    assert qc.evaluate_exclusion(long, thr)[0] is False


def test_semantic_switch_count_flags_single_frame_teleport():
    # one adjacent-frame jump of 1.0 norm-units (>0.5) among tiny real moves
    x = np.array([0.0, 0.01, 1.0, 1.01])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    fi = np.array([0, 1, 2, 3])
    assert qc.semantic_switch_count(x, y, fi, 0.5) == 1


def test_semantic_switch_count_ignores_post_gap_reappearance():
    # a big jump across a frame gap (dframe=2) is a plausible reappearance,
    # not a single-frame teleport -> not counted
    x = np.array([0.0, np.nan, 1.0])
    y = np.array([0.0, np.nan, 0.0])
    fi = np.array([0, 1, 2])
    assert qc.semantic_switch_count(x, y, fi, 0.5) == 0


def test_semantic_switch_count_degenerate():
    assert qc.semantic_switch_count(np.array([0.0]), np.array([0.0]),
                                    np.array([0]), 0.5) == 0
    assert qc.semantic_switch_count(np.array([0.0, 1.0]), np.array([0.0, 0.0]),
                                    np.array([0, 1]), float("nan")) == 0


# --- suspected-empty / static-artifact flag (A2) --------------------------- #
def test_position_spread():
    # a jittery point has spread; a fixed point has ~0
    moving = qc.position_spread(np.array([0.0, 1.0, -1.0, 0.5]),
                                np.array([0.0, -1.0, 1.0, 0.2]))
    still = qc.position_spread(np.array([5.0, 5.0, 5.0]),
                              np.array([2.0, 2.0, 2.0]))
    assert moving > 0.5 and still == 0.0
    assert math.isnan(qc.position_spread(np.array([1.0]), np.array([1.0])))


def test_suspected_empty_flags_static_artifact():
    # small area + low conf + stationary (the dish-6 smudge signature)
    assert qc.suspected_empty_dish(0.0002, 0.29, 0.03, _thr()) is True


def test_suspected_empty_protects_immobile_larva():
    # a motionless REAL larva: tiny spread, but larva-sized area + good conf.
    # Any single non-artifact signal must veto the flag (immobility is a
    # phenotype we must not discard).
    assert qc.suspected_empty_dish(0.0043, 0.82, 0.03, _thr()) is False  # area+conf
    assert qc.suspected_empty_dish(0.0007, 0.68, 0.10, _thr()) is False  # conf
    assert qc.suspected_empty_dish(0.0002, 0.30, 0.60, _thr()) is False  # moving


def test_suspected_empty_missing_signals_do_not_flag():
    nan = float("nan")
    assert qc.suspected_empty_dish(nan, 0.29, 0.03, _thr()) is False
    assert qc.suspected_empty_dish(0.0002, nan, 0.03, _thr()) is False
    assert qc.suspected_empty_dish(0.0002, 0.29, nan, _thr()) is False


def test_suspected_empty_feeds_exclusion_with_distinct_reason():
    m = {"valid_detection_fraction": 0.99, "longest_missing_gap_s": 1.0,
         "fraction_outside_petri": 0.0, "semantic_switch_rate_per_min": 0.0,
         "suspected_empty": True}
    flag, reason = qc.evaluate_exclusion(m, _thr())
    assert flag and reason == "suspected_empty"


# --- morphology reliability (A3) ------------------------------------------- #
def test_morphology_reliability_counts_usable_masks():
    valid = np.array([True, True, True, True])
    # frame1 no mask (NaN), frame2 fragmented (0.3<0.5), frames 3-4 clean
    solidity = np.array([np.nan, 0.3, 0.85, 0.9])
    assert qc.morphology_reliability(valid, solidity, 0.5) == 0.5


def test_morphology_reliability_ignores_masks_on_invalid_frames():
    # a mask on a non-valid (missing) frame must not count toward the ratio
    valid = np.array([True, True, False])
    solidity = np.array([0.9, 0.9, 0.9])
    assert qc.morphology_reliability(valid, solidity, 0.5) == 1.0


def test_morphology_reliability_no_valid_frames_is_nan():
    assert math.isnan(qc.morphology_reliability(
        np.array([False, False]), np.array([0.9, 0.9]), 0.5))


def test_morphology_reliability_not_in_exclusion_rule():
    # even reliability 0 must NOT by itself flag a dish for exclusion
    m = {"valid_detection_fraction": 0.99, "longest_missing_gap_s": 1.0,
         "fraction_outside_petri": 0.0, "semantic_switch_rate_per_min": 0.0,
         "morphology_reliability": 0.0}
    assert qc.evaluate_exclusion(m, _thr())[0] is False


def test_duration_seconds_prefers_time_span():
    import numpy as np
    assert qc.duration_seconds(np.array([0.0, 1.0, 2.0]), 3, 1.0) == 2.0
    # no usable timestamps -> fall back to (n-1)*dt
    assert qc.duration_seconds(np.array([np.nan, np.nan]), 5, 0.5) == 2.0


def test_rim_occlusion_fraction_attributes_only_rim_gaps():
    import numpy as np
    nan = np.nan
    # gap at 2-3 preceded by r=0.95 (rim) -> occlusion; gap at 7 preceded by
    # r=0.40 (mid-dish) -> not occlusion.
    missing = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    r = np.array([0.9, 0.95, nan, nan, 0.2, 0.3, 0.4, nan, 0.5, 0.6])
    assert qc.rim_occlusion_fraction(missing, r, 0.85) == 0.2   # 2/10 frames
    # never <= missing fraction violated, and threshold gates it
    assert qc.rim_occlusion_fraction(missing, r, 0.99) == 0.0   # nothing >=0.99


def test_rim_occlusion_fraction_start_gap_not_attributed():
    import numpy as np
    # gap at the very start has no "last seen" frame -> cannot attribute
    missing = np.array([1, 1, 0], dtype=bool)
    r = np.array([np.nan, np.nan, 0.5])
    assert qc.rim_occlusion_fraction(missing, r, 0.85) == 0.0
    assert np.isnan(qc.rim_occlusion_fraction(np.array([], dtype=bool),
                                              np.array([]), 0.85))


def test_rim_occlusion_events_counts_distinct_episodes():
    import numpy as np
    import crop_petris_yolo_seg as cp
    nan = np.nan
    # two rim gaps (after r=0.95 and r=0.90) + one mid-dish gap (after r=0.50)
    missing = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    r = np.array([0.9, 0.95, nan, nan, 0.9, nan, 0.4, 0.5, 0.92, 0.3])
    assert cp.rim_occlusion_events(missing, r, 0.85) == 2
    assert cp.rim_occlusion_fraction(missing, r, 0.85) == 0.3  # 3/10 frames


# --- end-to-end ------------------------------------------------------------ #
def test_end_to_end_qc(tmp_path):
    pd = pytest.importorskip("pandas")
    rows_track, rows_petri = [], []
    n = 100
    for i in range(n):
        # dish 1: clean; dish 2: 70% missing -> should still be flagged even
        # under the looser rim-seeking/petri-exit-aware defaults
        for dish in (1, 2):
            missing = (dish == 2 and i % 10 < 7)
            rows_track.append({
                "dataset_id": "d", "experiment_id": "e", "video_id": "v",
                "frame_idx": i, "time_s": i / 10.0, "dish_id": dish,
                "valid_detection": not missing,
                "larva_conf": float("nan") if missing else 0.85,
                "inside_petri": not missing,
                "mask_area_px": float("nan") if missing else 100.0,
                "x_norm_raw": float("nan") if missing else 0.1,
                "y_norm_raw": float("nan") if missing else 0.1,
                "track_switch": False, "missing": missing,
                "interpolated": False,
            })
        for dish in (1, 2):
            rows_petri.append({"video_id": "v", "dish_id": dish,
                               "petri_valid": True, "petri_conf": 0.9})
    tracks = tmp_path / "larva_tracks.csv"
    petri = tmp_path / "petri_tracks.csv"
    pd.DataFrame(rows_track).to_csv(tracks, index=False)
    pd.DataFrame(rows_petri).to_csv(petri, index=False)

    rc = qc.main(["--tracks", str(tracks), "--petri-tracks", str(petri)])
    assert rc == 0
    out = pd.read_csv(tmp_path / "qc_summary.csv")
    assert list(out.columns) == qc.QC_COLUMNS
    assert len(out) == 2

    d1 = out[out["dish_id"] == 1].iloc[0]
    d2 = out[out["dish_id"] == 2].iloc[0]
    assert math.isclose(d1["valid_detection_fraction"], 1.0)
    assert not d1["exclude_recommended"]
    assert d2["valid_detection_fraction"] < 0.4
    assert d2["exclude_recommended"]
    assert "valid_detection_fraction" in d2["exclude_reason"]
    assert d1["petri_valid"] in (True, "True", np.True_)
