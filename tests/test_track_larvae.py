"""Unit tests for track_larvae_video pure functions (no YOLO/OpenCV needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import track_larvae_video as tl  # noqa: E402


def det(track_id=None, conf=0.9, cx=50.0, cy=50.0, area=100.0, half=5.0):
    return tl.LarvaDetection(track_id, conf, cx - half, cy - half,
                             cx + half, cy + half, cx, cy, area)


# --- petri normalization (SPECIFICATION.md 7.3) ----------------------------------- #
def test_petri_normalize_center_and_edge():
    c = tl.petri_normalize(100, 100, 100, 100, 50)
    assert c["r_norm"] == 0.0 and c["inside_petri"]
    edge = tl.petri_normalize(150, 100, 100, 100, 50)
    assert math.isclose(edge["x_norm"], 1.0) and edge["inside_petri"]
    out = tl.petri_normalize(160, 100, 100, 100, 50)
    assert out["r_norm"] > 1.0 and not out["inside_petri"]


def test_petri_normalize_theta():
    up = tl.petri_normalize(100, 150, 100, 100, 50)
    assert math.isclose(up["theta_rad"], math.pi / 2, abs_tol=1e-9)


def test_petri_normalize_zero_radius():
    c = tl.petri_normalize(10, 10, 0, 0, 0)
    assert math.isnan(c["r_norm"]) and not c["inside_petri"]


# --- canonical track id (SPECIFICATION.md 26.1) ----------------------------------- #
def test_canonical_track_id_namespaced():
    assert tl.canonical_track_id(3, 7) == 3007
    assert tl.canonical_track_id(1, None) is None


# --- petri detection schedule (SPECIFICATION.md 5) -------------------------------- #
def test_should_detect_first_frame_only():
    assert tl.should_detect_petri(0, None, have_geometry=False) is True
    assert tl.should_detect_petri(5, None, have_geometry=True) is False


def test_should_detect_every_n():
    assert tl.should_detect_petri(10, 5, have_geometry=True) is True
    assert tl.should_detect_petri(11, 5, have_geometry=True) is False


# --- primary larva selection (SPECIFICATION.md 6) --------------------------------- #
def test_select_no_detection():
    idx, reason, amb = tl.select_primary_larva([], None, None, None, None)
    assert idx is None and reason == "no_detection"


def test_select_single():
    idx, reason, _ = tl.select_primary_larva([det()], None, None, None, None)
    assert idx == 0 and reason == "single_detection"


def test_select_same_track_id_wins():
    dets = [det(track_id=5, conf=0.5), det(track_id=9, conf=0.99)]
    idx, reason, _ = tl.select_primary_larva(dets, 9, None, None, None)
    assert idx == 1 and reason == "same_track_id"


def test_select_area_filter_leaves_one():
    dets = [det(area=10.0), det(area=500.0)]
    idx, reason, _ = tl.select_primary_larva(dets, None, None, 100.0, 1000.0)
    assert idx == 1 and reason == "area_filter"


def test_select_closest_to_previous():
    dets = [det(cx=10, cy=10), det(cx=90, cy=90)]
    idx, reason, _ = tl.select_primary_larva(dets, None, (88, 88), None, None)
    assert idx == 1 and reason == "closest_to_previous"


def test_select_highest_confidence_when_no_prior():
    dets = [det(conf=0.4, cx=10), det(conf=0.95, cx=90)]
    idx, reason, _ = tl.select_primary_larva(dets, None, None, None, None)
    assert idx == 1 and reason == "highest_confidence"


def test_select_largest_area_is_ambiguous():
    dets = [det(conf=0.8, area=100), det(conf=0.8, area=300)]
    idx, reason, amb = tl.select_primary_larva(dets, None, None, None, None)
    assert idx == 1 and reason == "largest_area" and amb is True


# --- selection hysteresis (track-switch fix) -------------------------------- #
def test_select_single_rejects_implausible_jump():
    # lone detection, no track-id match, far from the established centroid
    d = det(track_id=99, cx=200, cy=200)
    idx, reason, amb = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0, missing_streak=0)
    assert idx is None and reason == "rejected_implausible" and amb is False


def test_select_single_accepts_plausible_jump():
    d = det(track_id=99, cx=15, cy=10)
    idx, reason, _ = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0, missing_streak=0)
    assert idx == 0 and reason == "single_detection"


def test_select_single_same_track_id_skips_jump_gate():
    # matches prev track id -> accepted regardless of distance
    d = det(track_id=5, cx=500, cy=500)
    idx, reason, _ = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0, missing_streak=0)
    assert idx == 0 and reason == "single_detection"


def test_select_single_no_prior_centroid_skips_jump_gate():
    d = det(track_id=99, cx=500, cy=500)
    idx, reason, _ = tl.select_primary_larva(
        [d], None, None, None, None, max_jump_px=20.0, missing_streak=0)
    assert idx == 0 and reason == "single_detection"


def test_select_single_jump_gate_widens_with_missing_streak():
    # additive growth (not multiplicative): 90px needs many missing frames
    # to become plausible at +5px/frame, unlike a naive x(1+streak) gate
    # that would open this wide again after only 3-4 frames.
    d = det(track_id=99, cx=100, cy=10)  # 90px from prev centroid
    rejected = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0, missing_streak=0,
        jump_growth_px=5.0)
    assert rejected[1] == "rejected_implausible"
    still_rejected = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0, missing_streak=4,
        jump_growth_px=5.0)
    assert still_rejected[1] == "rejected_implausible"
    accepted = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0, missing_streak=15,
        jump_growth_px=5.0)
    assert accepted[1] == "single_detection"


def test_select_single_jump_gate_capped_regardless_of_missing_streak():
    # jump_cap_px stops the growth allowance from ever exploding, even after
    # a very long gap.
    d = det(track_id=99, cx=1000, cy=10)  # 990px from prev centroid
    idx, reason, _ = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=20.0,
        missing_streak=1000, jump_growth_px=5.0, jump_cap_px=240.0)
    assert idx is None and reason == "rejected_implausible"


def test_select_single_jump_gate_disabled_by_default_none():
    d = det(track_id=99, cx=500, cy=500)
    idx, reason, _ = tl.select_primary_larva(
        [d], 5, (10.0, 10.0), None, None, max_jump_px=None, missing_streak=0)
    assert idx == 0 and reason == "single_detection"


def test_select_multi_excludes_jump_implausible_from_tiebreak():
    # 3 candidates: one implausibly far (excluded by the jump gate), then a
    # normal closest-to-previous tie-break among the two plausible ones.
    far = det(track_id=1, cx=500, cy=500)
    near_far = det(track_id=2, cx=25, cy=10)
    near_close = det(track_id=3, cx=15, cy=10)
    idx, reason, _ = tl.select_primary_larva(
        [far, near_far, near_close], None, (10.0, 10.0), None, None,
        max_jump_px=20.0, missing_streak=0)
    assert idx == 2 and reason == "closest_to_previous"


# --- morphology (SPECIFICATION.md 9) ---------------------------------------------- #
def test_polygon_morphology_square():
    pytest.importorskip("shapely")
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    m = tl.polygon_morphology(square)
    assert math.isclose(m["area_px"], 100.0)
    assert math.isclose(m["perimeter_px"], 40.0)
    assert math.isclose(m["extent"], 1.0)       # square fills its bbox
    assert math.isclose(m["solidity"], 1.0)      # convex
    assert math.isclose(m["bbox_area_px"], 100.0)


def test_polygon_morphology_elongated_has_high_eccentricity():
    pytest.importorskip("shapely")
    thin = np.array([[0, 0], [100, 0], [100, 5], [0, 5]], dtype=float)
    m = tl.polygon_morphology(thin)
    assert m["eccentricity"] > 0.9
    assert m["major_axis_px"] > m["minor_axis_px"]
    assert math.isclose(abs(m["orientation_rad"]), 0.0, abs_tol=1e-6)


def test_polygon_morphology_degenerate():
    m = tl.polygon_morphology(np.array([[0, 0], [1, 1]], dtype=float))
    assert math.isnan(m["area_px"])


# --- quality flag (SPECIFICATION.md 15.3) ----------------------------------------- #
def test_quality_flag_vocabulary():
    assert tl.quality_flag(True, False, False, 0.0, 0.4) == "missing_detection"
    assert tl.quality_flag(False, False, False, 0.9, 0.4) == "outside_petri"
    assert tl.quality_flag(False, True, True, 0.9, 0.4) == "ambiguous_detection"
    assert tl.quality_flag(False, True, False, 0.1, 0.4) == "low_confidence"
    assert tl.quality_flag(False, True, False, 0.9, 0.4) == "ok"


# --- tracker selection (SPECIFICATION.md 26.1) ------------------------------------ #
def _parse(*extra):
    return tl.build_parser().parse_args(
        ["--video", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l", *extra])


def test_default_tracker_is_bytetrack():
    """The locked v1 default: every released dataset was tracked with it."""
    assert _parse().tracker == "bytetrack"


@pytest.mark.parametrize("name", ["bytetrack", "botsort", "ocsort"])
def test_every_tracker_name_is_accepted(name):
    assert _parse("--tracker", name).tracker == name


def test_legacy_yaml_tracker_spelling_still_parses():
    """--tracker bytetrack.yaml was the pre-v1.1 default and appears in old
    provenance records and scripts; it must not become a hard error."""
    assert _parse("--tracker", "bytetrack.yaml").tracker == "bytetrack"


def test_unknown_tracker_is_rejected():
    with pytest.raises(SystemExit):
        _parse("--tracker", "deepsort")


# --- gate re-acquisition (lockout fix) -------------------------------------- #
def _run_reacq(frames, reacquire_frames, anchor=(10.0, 10.0), **kw):
    """Drive select_primary_larva + reacquire_step over per-frame detections,
    updating state exactly as the video loop does. Returns the reasons."""
    prev_tid, prev_c, streak, ch = 5, anchor, 0, None
    reasons = []
    for f, dets in enumerate(frames):
        idx, reason, _ = tl.select_primary_larva(
            dets, prev_tid, prev_c, None, None, max_jump_px=20.0,
            missing_streak=streak, jump_growth_px=5.0, jump_cap_px=60.0)
        idx, reason, ch = tl.reacquire_step(
            dets, idx, reason, ch, f, reacquire_frames, None, None,
            20.0, 5.0, 60.0, **kw)
        reasons.append(reason)
        if idx is not None:
            d = dets[idx]
            prev_tid, prev_c, streak = d.track_id, (d.cx, d.cy), 0
        else:
            streak += 1
    return reasons


def test_reacquire_breaks_lockout():
    # anchor at (10,10) (the hand); real larva far away, crawling slowly
    frames = [[det(track_id=7, cx=300 + f, cy=300)] for f in range(20)]
    reasons = _run_reacq(frames, reacquire_frames=10)
    assert reasons[:9] == ["rejected_implausible"] * 9
    assert reasons[9] == "reacquired"
    # once re-anchored, the larva is followed normally
    assert reasons[10:] == ["single_detection"] * 10


def test_reacquire_disabled_passes_through():
    frames = [[det(track_id=7, cx=300, cy=300)] for _ in range(50)]
    assert set(_run_reacq(frames, reacquire_frames=None)) == {
        "rejected_implausible"}


def test_reacquire_reset_by_confirmed_anchor():
    # the artifact case the gate was built for: a far lone artifact that keeps
    # reappearing while the real larva (near the anchor) is still detected
    frames = []
    for f in range(40):
        if f % 4 == 3:
            frames.append([det(track_id=5, cx=12, cy=10)])  # real larva
        else:
            frames.append([det(track_id=8, cx=300, cy=300)])  # artifact
    reasons = _run_reacq(frames, reacquire_frames=10)
    assert "reacquired" not in reasons


def test_reacquire_requires_continuity():
    # rejected detections scattered around the dish never form a challenger
    pts = [(300, 300), (100, 400), (400, 100), (250, 450)]
    frames = [[det(track_id=None, cx=pts[f % 4][0], cy=pts[f % 4][1])]
              for f in range(40)]
    assert "reacquired" not in _run_reacq(frames, reacquire_frames=10)


def test_reacquire_continuity_by_distance_without_track_ids():
    frames = [[det(track_id=None, cx=300 + 2 * f, cy=300)] for f in range(12)]
    assert _run_reacq(frames, reacquire_frames=10)[9] == "reacquired"


def test_reacquire_tolerates_detection_gaps_but_needs_support():
    # detected every other frame -> support 0.5 -> accepted at min_support 0.5
    frames = [[det(track_id=7, cx=300, cy=300)] if f % 2 == 0 else []
              for f in range(30)]
    assert "reacquired" in _run_reacq(frames, reacquire_frames=10)
    # every third frame -> support ~0.33 -> not enough
    frames = [[det(track_id=7, cx=300, cy=300)] if f % 3 == 0 else []
              for f in range(30)]
    assert "reacquired" not in _run_reacq(frames, reacquire_frames=10)


def test_reacquire_respects_area_gate():
    # a lone blob outside the plausible area range is never promoted
    frames = [[det(track_id=7, cx=300, cy=300, area=9000.0)]
              for _ in range(30)]
    prev_c, ch = (10.0, 10.0), None
    for f, dets in enumerate(frames):
        idx, reason, _ = tl.select_primary_larva(
            dets, 5, prev_c, 100.0, 5000.0, max_jump_px=20.0,
            missing_streak=f, jump_growth_px=5.0, jump_cap_px=60.0)
        idx, reason, ch = tl.reacquire_step(
            dets, idx, reason, ch, f, 10, 100.0, 5000.0, 20.0, 5.0, 60.0)
        assert reason == "rejected_implausible"
