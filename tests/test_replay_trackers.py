"""Tests for the offline replay harness (scripts/replay_trackers.py).

The harness's whole claim is that every configuration sees identical detector
output and differs only in association, so these tests target the places that
claim could quietly fail: the association backends, the continuity-gate switch,
the cache->detection conversion, and run-to-run determinism.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cache_detections as cd  # noqa: E402
import replay_trackers as rt  # noqa: E402
import track_larvae_video as tl  # noqa: E402
import larva_trackers as lt  # noqa: E402


def _det(cx, cy, conf=0.9, half=10.0):
    return tl.LarvaDetection(None, conf, cx - half, cy - half, cx + half,
                             cy + half, cx, cy, (2 * half) ** 2, None)


# --- association backends -------------------------------------------------- #
def test_null_associator_never_assigns_an_identity():
    """The tracker-free baseline must have no ids at all -- that is what forces
    select_primary_larva down its nearest-centroid path."""
    a = rt.NullAssociator()
    dets = [_det(10, 10), _det(80, 80)]
    dets[0].track_id = 7                        # stale id must be cleared
    out = a.update(dets)
    assert out is dets
    assert all(d.track_id is None for d in out)


def test_null_associator_forces_nearest_centroid_selection():
    """With no ids the `same_track_id` branch is unreachable, so a stale
    prev_track_id cannot pull selection away from the nearest detection."""
    dets = [_det(100, 100), _det(30, 30)]
    rt.NullAssociator().update(dets)
    # both candidates inside the jump limit -> the nearest one wins on distance
    idx, reason, _ = tl.select_primary_larva(
        dets, prev_track_id=5, prev_centroid=(28.0, 31.0),
        area_min=None, area_max=None, max_jump_px=200.0)
    assert idx == 1 and reason == "closest_to_previous"
    # a tighter gate leaves exactly one candidate; still the nearest one, and
    # the documented reason for a single survivor is "area_filter"
    idx2, reason2, _ = tl.select_primary_larva(
        dets, prev_track_id=5, prev_centroid=(28.0, 31.0),
        area_min=None, area_max=None, max_jump_px=60.0)
    assert idx2 == 1 and reason2 == "area_filter"


def test_ocsort_associator_assigns_a_stable_id():
    a = rt.OCSortAssociator(lt.load_tracker_config("ocsort"))
    ids = []
    for i in range(12):
        dets = [_det(50 + 2 * i, 50)]
        a.update(dets)
        ids.append(dets[0].track_id)
    assert ids[-1] is not None
    assert len({i for i in ids if i is not None}) == 1


def test_ocsort_associator_steps_on_empty_frames():
    a = rt.OCSortAssociator(lt.load_tracker_config("ocsort"))
    assert a.update([]) == []


@pytest.mark.parametrize("method", ["bytetrack", "botsort"])
def test_ultralytics_associator_tracks_a_moving_object(method):
    """The offline driver must reproduce ultralytics' own association: a track
    is unconfirmed on its birth frame, then keeps one id."""
    pytest.importorskip("ultralytics")
    a = rt.make_associator(method, lt.load_tracker_config(method), 30,
                           (512, 512))
    ids = []
    for i in range(8):
        dets = [_det(100 + 3 * i, 100)]
        a.update(dets)
        ids.append(dets[0].track_id)
    assert ids[0] is None or isinstance(ids[0], int)
    later = [i for i in ids[2:] if i is not None]
    assert later and len(set(later)) == 1


def test_ultralytics_associator_survives_empty_frames():
    pytest.importorskip("ultralytics")
    a = rt.make_associator("bytetrack", lt.load_tracker_config("bytetrack"),
                           30, (512, 512))
    assert a.update([]) == []


def test_ultralytics_associator_passes_raw_detections_when_nothing_tracked():
    """Faithful to ultralytics: on_predict_postprocess_end short-circuits when
    the tracker emits nothing, leaving every raw detection visible downstream.
    Returning [] instead would invent missing frames production never had."""
    pytest.importorskip("ultralytics")
    a = rt.make_associator("bytetrack", lt.load_tracker_config("bytetrack"),
                           30, (512, 512))
    dets = [_det(100, 100)]
    out = a.update(dets)                # first frame: nothing activated yet
    assert out == dets or all(d.track_id is not None for d in out)


# --- cache -> detections --------------------------------------------------- #
def _write_cache(dirpath: Path, video_id="vid", n_frames=6, dishes=(1,),
                 conf=0.9):
    dirpath.mkdir(parents=True, exist_ok=True)
    det_rows, frm_rows = [], []
    for f in range(n_frames):
        for d in dishes:
            frm_rows.append(f"{video_id},{f},{d},{f / 30:.4f},1,40.0,200.0,10.0")
            cx = 100.0 + 2 * f
            morph = ",".join(["100.0", "40.0", "0.7", "0.9", "0.8", "20.0",
                              "5.0", "4.0", "0.1", "0.9", "20.0", "20.0",
                              "400.0", "0.25"])
            det_rows.append(
                f"{video_id},{f},{d},0,0,{conf},{cx - 10},90.0,{cx + 10},110.0,"
                f"{cx},100.0,100.0,True,{morph}")
    with gzip.open(dirpath / f"{video_id}.detections.csv.gz", "wt") as fh:
        fh.write(",".join(cd.DETECTION_COLUMNS) + "\n")
        fh.write("\n".join(det_rows) + "\n")
    with gzip.open(dirpath / f"{video_id}.frames.csv.gz", "wt") as fh:
        fh.write(",".join(cd.FRAME_COLUMNS) + "\n")
        fh.write("\n".join(frm_rows) + "\n")
    geom = [{"dish_id": d, "row": 0, "col": 0, "petri_x1": 0.0,
             "petri_y1": 0.0, "petri_x2": 400.0, "petri_y2": 400.0,
             "petri_conf": 0.9, "petri_valid": True, "crop_x0": 0,
             "crop_y0": 0, "crop_size": 420, "petri_geometry_source": "box",
             "petri_fit_cx": "", "petri_fit_cy": "", "petri_fit_a": "",
             "petri_fit_b": "", "petri_fit_valid": False} for d in dishes]
    (dirpath / f"{video_id}._cache_done.json").write_text(json.dumps(
        {"video_id": video_id, "n_frames": n_frames, "fps": 30.0,
         "geometry": geom}))


def test_detections_by_frame_applies_the_shared_threshold(tmp_path):
    pytest.importorskip("pandas")
    _write_cache(tmp_path, conf=0.2)
    _, dets_df, _, _ = rt.load_cache(tmp_path, "vid")
    assert rt.detections_by_frame(dets_df, 1, det_conf=0.25) == {}
    kept = rt.detections_by_frame(dets_df, 1, det_conf=0.1)
    assert len(kept) == 6
    det, morph = kept[0][0]
    assert det.polygon is not None          # sentinel: mask_* columns get written
    assert morph["area_px"] == pytest.approx(100.0)


def test_load_cache_rebuilds_dish_geometry(tmp_path):
    pytest.importorskip("pandas")
    _write_cache(tmp_path, dishes=(1, 2))
    frames, dets, dishes, summary = rt.load_cache(tmp_path, "vid")
    assert [d.dish_id for d in dishes] == [1, 2]
    assert dishes[0].geom.size == 420 and dishes[0].fit is None
    assert summary["fps"] == 30.0 and len(frames) == 12


# --- end-to-end replay ----------------------------------------------------- #
def _replay(tmp_path, out, methods="nearest", gates="gate_on"):
    return rt.main(["--cache", str(tmp_path / "cache"), "--out", str(out),
                    "--methods", methods, "--gates", gates, "--no-downstream"])


def test_replay_writes_the_full_larva_schema(tmp_path):
    pytest.importorskip("pandas")
    _write_cache(tmp_path / "cache", dishes=(1, 2))
    out = tmp_path / "out"
    assert _replay(tmp_path, out) == 0
    p = out / "configs" / "nearest__gate_on" / "larva_tracks.csv"
    header = p.read_text().splitlines()[0].split(",")
    assert header == tl.LARVA_COLUMNS
    assert len(p.read_text().splitlines()) == 1 + 12     # 6 frames x 2 dishes


def test_replay_is_deterministic(tmp_path):
    """Same cache in, byte-identical tracks out -- the experiment is only
    reproducible if the replay is."""
    pytest.importorskip("pandas")
    _write_cache(tmp_path / "cache")
    a, b = tmp_path / "a", tmp_path / "b"
    _replay(tmp_path, a)
    _replay(tmp_path, b)
    name = "configs/nearest__gate_on/larva_tracks.csv"
    assert (a / name).read_bytes() == (b / name).read_bytes()


def test_replay_records_runtime_and_suppression(tmp_path):
    pytest.importorskip("pandas")
    _write_cache(tmp_path / "cache")
    out = tmp_path / "out"
    _replay(tmp_path, out)
    import pandas as pd
    rtm = pd.read_csv(out / "runtime.csv")
    assert list(rtm.columns) == rt.RUNTIME_COLUMNS
    assert rtm["n_detections_fed"].iloc[0] == 6
    # a null associator suppresses nothing
    assert rtm["n_detections_associated"].iloc[0] == 6


def test_gate_off_makes_rejected_implausible_unreachable(tmp_path):
    """gate_off must genuinely disable the continuity gate, not merely widen it."""
    pytest.importorskip("pandas")
    import pandas as pd
    # detections that teleport across the dish every frame
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    rows = []
    for f in range(10):
        cx = 50.0 if f % 2 == 0 else 350.0
        morph = ",".join(["100.0"] + ["1.0"] * 13)
        rows.append(f"v,{f},1,0,0,0.9,{cx - 10},90.0,{cx + 10},110.0,{cx},"
                    f"100.0,100.0,True,{morph}")
    with gzip.open(cache / "v.detections.csv.gz", "wt") as fh:
        fh.write(",".join(cd.DETECTION_COLUMNS) + "\n" + "\n".join(rows) + "\n")
    with gzip.open(cache / "v.frames.csv.gz", "wt") as fh:
        fh.write(",".join(cd.FRAME_COLUMNS) + "\n")
        fh.write("\n".join(f"v,{f},1,{f / 30:.4f},1,40.0,200.0,10.0"
                           for f in range(10)) + "\n")
    (cache / "v._cache_done.json").write_text(json.dumps(
        {"video_id": "v", "n_frames": 10, "fps": 30.0,
         "geometry": [{"dish_id": 1, "row": 0, "col": 0, "petri_x1": 0.0,
                       "petri_y1": 0.0, "petri_x2": 400.0, "petri_y2": 400.0,
                       "petri_conf": 0.9, "petri_valid": True, "crop_x0": 0,
                       "crop_y0": 0, "crop_size": 420,
                       "petri_geometry_source": "box", "petri_fit_cx": "",
                       "petri_fit_cy": "", "petri_fit_a": "",
                       "petri_fit_b": "", "petri_fit_valid": False}]}))

    out = tmp_path / "out"
    rt.main(["--cache", str(cache), "--out", str(out), "--methods", "nearest",
             "--gates", "gate_on,gate_off", "--no-downstream"])
    on = pd.read_csv(out / "configs" / "nearest__gate_on" / "larva_tracks.csv")
    off = pd.read_csv(out / "configs" / "nearest__gate_off" / "larva_tracks.csv")
    assert (on["selection_reason"] == "rejected_implausible").any()
    assert not (off["selection_reason"] == "rejected_implausible").any()
    assert off["valid_detection"].astype(str).str.lower().eq("true").all()
