"""Tests for the shared detector pass (scripts/cache_detections.py).

No YOLO and no video: the tested surface is geometry reconstruction and the
image statistics, which is where a silent error would corrupt every downstream
configuration at once.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cache_detections as cd  # noqa: E402
import track_larvae_video as tl  # noqa: E402

PETRI_HEADER = ("dataset_id,experiment_id,video_id,frame_idx,time_s,dish_id,"
                "row,col,petri_track_id,petri_conf,petri_x1,petri_y1,petri_x2,"
                "petri_y2,petri_cx,petri_cy,petri_w,petri_h,petri_radius,"
                "petri_valid,crop_x0,crop_y0,crop_size,petri_aspect_ratio,"
                "petri_distortion_flag,petri_offset_norm,petri_detect_policy,"
                "petri_geometry_source,petri_fit_cx,petri_fit_cy,petri_fit_a,"
                "petri_fit_b,petri_fit_valid")


def _petri_csv(path: Path, n_frames: int = 3, n_dishes: int = 2):
    """A petri_tracks.csv with geometry replicated on every frame, as the
    first-frame-only policy produces."""
    lines = [PETRI_HEADER]
    for f in range(n_frames):
        for d in range(1, n_dishes + 1):
            x1, y1 = 100.0 * d, 200.0
            x2, y2 = x1 + 400.0, y1 + 400.0
            lines.append(
                f"ds,exp,vid,{f},{f / 30:.4f},{d},0,{d - 1},,0.95,"
                f"{x1},{y1},{x2},{y2},{(x1 + x2) / 2},{(y1 + y2) / 2},"
                f"400.0,400.0,200.0,True,{int(x1) - 10},190,420,"
                f"1.0,False,0.5,first_frame,box,,,,,False")
    path.write_text("\n".join(lines) + "\n")


def test_load_frozen_geometry_reads_only_the_first_frame(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "petri_tracks.csv"
    _petri_csv(p, n_frames=5, n_dishes=3)
    dishes = cd.load_frozen_geometry(p)
    assert [d.dish_id for d in dishes] == [1, 2, 3]
    assert all(isinstance(d, tl.DishGeom) for d in dishes)
    d1 = dishes[0]
    assert (d1.geom.x0, d1.geom.y0, d1.geom.size) == (90, 190, 420)
    assert d1.box.cx == pytest.approx(300.0)
    assert d1.fit is None                      # petri_fit_valid False -> box


def test_frozen_geometry_matches_production_normalization(tmp_path):
    """effective_geometry over the reconstructed dish must equal the box
    geometry the production run normalized with."""
    pytest.importorskip("pandas")
    p = tmp_path / "petri_tracks.csv"
    _petri_csv(p)
    dg = cd.load_frozen_geometry(p)[0]
    cx, cy, r = tl.effective_geometry(dg)
    assert (cx, cy, r) == pytest.approx((300.0, 400.0, 200.0))


def test_geometry_rows_round_trip_through_json(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "petri_tracks.csv"
    _petri_csv(p, n_dishes=2)
    rows = cd.geometry_rows(cd.load_frozen_geometry(p))
    restored = json.loads(json.dumps(rows))
    assert [r["dish_id"] for r in restored] == [1, 2]
    assert set(cd.GEOMETRY_COLUMNS) == set(restored[0])


def test_dish_mask_is_a_circle():
    m = cd.dish_mask(size=100, cx=50, cy=50, radius=25, step=1)
    assert m.shape == (100, 100)
    assert m[50, 50] and not m[0, 0]
    # area of a disc, within the discretization error of a 25 px radius
    assert m.sum() == pytest.approx(np.pi * 25 ** 2, rel=0.05)


def test_crop_statistics_tracks_contrast():
    pytest.importorskip("cv2")
    size, step = 128, 1
    mask = cd.dish_mask(size, size / 2, size / 2, size / 2 - 1, step)
    flat = np.full((size, size, 3), 128, dtype=np.uint8)
    contrast, mean, std = cd.crop_statistics(flat, mask, step)
    assert contrast == pytest.approx(0.0) and mean == pytest.approx(128.0)

    speckled = flat.copy()
    speckled[::2, ::2] = 20
    high, _, _ = cd.crop_statistics(speckled, mask, step)
    assert high > contrast          # a darker larva on a bright dish


def test_crop_statistics_returns_nan_for_a_degenerate_mask():
    pytest.importorskip("cv2")
    m = np.zeros((8, 8), dtype=bool)
    contrast, mean, std = cd.crop_statistics(
        np.zeros((8, 8, 3), np.uint8), m, 1)
    assert np.isnan(contrast) and np.isnan(mean) and np.isnan(std)


def test_schemas_are_stable():
    """The replay reads these by name; renaming a column silently breaks it."""
    for c in ("video_id", "frame_idx", "dish_id", "conf", "cx_crop",
              "area_px", "has_polygon"):
        assert c in cd.DETECTION_COLUMNS
    for c in ("n_detections", "crop_contrast"):
        assert c in cd.FRAME_COLUMNS
    assert all(f"mask_{k}" in cd.DETECTION_COLUMNS for k in cd.MORPH_KEYS)


def test_skip_existing_sentinel_short_circuits(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pandas")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "vid.mp4").write_bytes(b"")
    work = tmp_path / "ds" / "_work" / "vid"
    work.mkdir(parents=True)
    _petri_csv(work / "petri_tracks.csv")
    out = tmp_path / "cache"
    out.mkdir()
    (out / "vid._cache_done.json").write_text(json.dumps(
        {"video_id": "vid", "n_frames": 3, "n_detections": 0}))

    monkeypatch.setattr(cd.tl, "load_model", lambda p: object())
    monkeypatch.setattr(cd.cp, "sha256_file", lambda p: "sha")

    def _boom(*a, **k):
        raise AssertionError("cache_video must not run for a cached video")

    monkeypatch.setattr(cd, "cache_video", _boom)
    rc = cd.main(["--videos", str(videos), "--dataset", str(tmp_path / "ds"),
                  "--larva-model", str(tmp_path / "m.pt"), "--out", str(out),
                  "--skip-existing"])
    assert rc == 0
    assert "skipped (cached)" in capsys.readouterr().out
