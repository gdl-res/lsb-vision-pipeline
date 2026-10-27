"""Tests for the selectable larva association backends (SPECIFICATION.md 26.1).

The registry half checks that every tracker name resolves to a shipped,
parseable config -- including a guard that the vendored ByteTrack config still
matches the installed ultralytics defaults, since every v1 dataset was tracked
with those values.

The OC-SORT half runs the vendored tracker over synthetic boxes. No YOLO is
involved: scripts/larva_trackers.py deliberately imports neither ultralytics nor
torch, which is what makes the tracker testable at all.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import larva_trackers as lt  # noqa: E402


def _box(cx, cy, w=20.0, h=20.0, conf=0.9):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf]


def _ocsort(**overrides):
    cfg = dict(lt.load_tracker_config("ocsort"))
    cfg.update(overrides)
    return lt.ocsort_from_config(cfg)


# --- registry ------------------------------------------------------------ #
@pytest.mark.parametrize("value,expected", [
    ("bytetrack", "bytetrack"),
    ("botsort.yaml", "botsort"),
    ("/opt/cfg/ocsort.yaml", "ocsort"),
    (Path("config/trackers/bytetrack.yaml"), "bytetrack"),
])
def test_normalize_tracker_name(value, expected):
    assert lt.normalize_tracker_name(value) == expected


@pytest.mark.parametrize("name", lt.TRACKERS)
def test_every_tracker_ships_a_config(name):
    path = lt.resolve_tracker_config(name)
    assert path.is_file()
    cfg = lt.load_tracker_config(name)
    # tracker_type must match the filename or ultralytics would load the wrong
    # backend for a name we thought we selected
    assert cfg["tracker_type"] == name == path.stem


def test_resolve_tracker_config_honours_override(tmp_path):
    override = tmp_path / "tuned.yaml"
    override.write_text("tracker_type: bytetrack\n")
    assert lt.resolve_tracker_config("bytetrack", override) == override


def test_resolve_tracker_config_rejects_unknown_and_missing(tmp_path):
    with pytest.raises(ValueError):
        lt.resolve_tracker_config("deepsort")
    with pytest.raises(FileNotFoundError):
        lt.resolve_tracker_config("bytetrack", tmp_path / "nope.yaml")


def test_bytetrack_config_matches_ultralytics_defaults():
    """v1 reproducibility guard.

    config/trackers/bytetrack.yaml is a vendored copy of the ultralytics
    default. If an ultralytics upgrade changes those values, this fails loudly
    instead of silently re-tracking future data with different association
    thresholds than the released v1 datasets.
    """
    pytest.importorskip("ultralytics")
    yaml = pytest.importorskip("yaml")
    import ultralytics
    shipped = (Path(ultralytics.__file__).parent
               / "cfg" / "trackers" / "bytetrack.yaml")
    if not shipped.is_file():
        pytest.skip("ultralytics ships no bytetrack.yaml at the expected path")
    assert (lt.load_tracker_config("bytetrack")
            == yaml.safe_load(shipped.read_text()))


def test_botsort_disables_global_motion_compensation():
    """Fixed camera + crop window frozen at frame 0 -> no egomotion to correct."""
    cfg = lt.load_tracker_config("botsort")
    assert cfg["gmc_method"] == "none"
    assert cfg["with_reid"] is False


# --- OC-SORT ------------------------------------------------------------- #
def test_ocsort_keeps_one_id_through_linear_motion():
    t = _ocsort()
    ids = [t.update(np.array([_box(50 + 2 * i, 50)]))[0] for i in range(30)]
    assert len(set(ids)) == 1 and ids[0] is not None


def test_ocsort_recovers_same_id_after_short_gap():
    """OCR/ORU: a gap shorter than max_age must not mint a new identity.

    This is the property that distinguishes OC-SORT from plain SORT here, and
    the reason it is worth offering given this pipeline's missing-detection
    rates.
    """
    t = _ocsort(max_age=30)
    before = [t.update(np.array([_box(50 + 2 * i, 50)]))[0] for i in range(10)]
    for _ in range(5):
        t.update(np.empty((0, 5)))
    after = t.update(np.array([_box(50 + 2 * 15, 50)]))[0]
    assert after == before[-1] is not None


def test_ocsort_starts_new_id_after_gap_longer_than_max_age():
    t = _ocsort(max_age=10)
    before = [t.update(np.array([_box(50 + 2 * i, 50)]))[0] for i in range(10)]
    for _ in range(20):
        t.update(np.empty((0, 5)))
    t.update(np.array([_box(300, 300)]))          # birth frame: unconfirmed
    after = t.update(np.array([_box(301, 300)]))[0]
    assert after is not None and after != before[-1]


def test_ocsort_tolerates_frames_with_no_detections():
    t = _ocsort()
    assert t.update(np.empty((0, 5))) == []
    assert t.update(None) == []


def test_ocsort_separates_two_objects():
    t = _ocsort()
    out = [t.update(np.array([_box(50 + i, 50), _box(300 + i, 300)]))
           for i in range(15)]
    left, right = zip(*out)
    assert len(set(left)) == len(set(right)) == 1
    assert left[0] != right[0] and left[0] is not None


def test_ocsort_returns_one_id_per_input_row_in_order():
    """Ids are consumed positionally by run_larva_inference, so alignment with
    the input rows -- including rows dropped by det_thresh -- is load-bearing."""
    t = _ocsort(det_thresh=0.25)
    ids = t.update(np.array([_box(50, 50, conf=0.05),
                             _box(200, 200, conf=0.9)]))
    assert len(ids) == 2
    assert ids[0] is None          # below det_thresh: never starts a track
    assert ids[1] is not None


def test_ocsort_state_is_independent_per_instance():
    """One instance per dish is what keeps identities from leaking between
    dishes (SPECIFICATION.md 26.1)."""
    a, b = _ocsort(), _ocsort()
    for i in range(5):
        a.update(np.array([_box(50 + i, 50)]))
    assert len(a.trackers) == 1
    assert b.trackers == []


# --- geometry helpers ---------------------------------------------------- #
def test_iou_batch_basic():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert lt.iou_batch(a, a)[0, 0] == pytest.approx(1.0)
    disjoint = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert lt.iou_batch(a, disjoint)[0, 0] == pytest.approx(0.0)
    assert lt.iou_batch(np.empty((0, 4)), a).shape == (0, 1)


def test_speed_direction_is_a_unit_vector():
    d = lt.speed_direction(np.array([0.0, 0.0, 10.0, 10.0]),
                           np.array([30.0, 0.0, 40.0, 10.0]))
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-4)
    assert d[1] > 0.99          # motion is +x, and speed_direction is (dy, dx)
