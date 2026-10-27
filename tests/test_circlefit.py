"""Tests for the petri circle-fit geometry (SPECIFICATION.md 26.2)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import crop_petris_yolo_seg as cp  # noqa: E402


def _disk_image(cx, cy, r, size=400):
    """Dark frame with one bright filled disk (mimics a dish on the tray)."""
    img = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 255
    return img


def test_dishfit_radius_and_aspect():
    f = cp.DishFit(cx=10, cy=20, a=30, b=20, valid=True)
    assert f.radius == 25.0                       # mean semi-axis
    assert f.aspect_ratio == 1.5


def test_fit_recovers_true_disk():
    pytest.importorskip("cv2")
    img = _disk_image(cx=200, cy=190, r=120)
    # YOLO-style box that is too small AND off-center (the real-world bias)
    box = cp.PetriBox(x1=110, y1=110, x2=250, y2=250, conf=0.9)  # r=70, off
    fit = cp.fit_dish_geometry(img, box)
    assert fit.valid
    assert abs(fit.cx - 200) < 4 and abs(fit.cy - 190) < 4   # true center
    assert abs(fit.radius - 120) < 6                          # true radius
    # box radius was ~70, fit recovers ~120 -> the correction we want
    assert fit.radius > box.radius * 1.4


def test_fit_fails_on_blank_falls_back_to_box():
    pytest.importorskip("cv2")
    blank = np.zeros((300, 300), dtype=np.uint8)
    box = cp.PetriBox(10, 10, 110, 110, 0.9)
    fit = cp.fit_dish_geometry(blank, box)
    assert fit.valid is False
    assert fit.cx == box.cx and fit.radius == box.radius      # carries the box


def test_fit_ellipticity_measured():
    pytest.importorskip("cv2")
    # squashed disk -> ellipse a/b > 1 (the §26.2 distortion signal)
    size = 400
    img = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    img[((xx - 200) / 130.0) ** 2 + ((yy - 200) / 90.0) ** 2 <= 1] = 255
    box = cp.PetriBox(80, 100, 320, 300, 0.9)
    fit = cp.fit_dish_geometry(img, box)
    assert fit.valid and fit.aspect_ratio > 1.2               # ~130/90


# --- effective geometry selection in the tracker --------------------------- #
def test_effective_geometry_prefers_valid_fit():
    import track_larvae_video as tl
    box = cp.PetriBox(0, 0, 100, 100, 0.9)        # cx=50 r=50
    geom = cp.compute_crop_geom(box, 0.05)
    dg = tl.DishGeom(1, 0, 0, box, geom,
                     fit=cp.DishFit(60, 62, 70, 66, True))
    assert tl.effective_geometry(dg) == (60, 62, 68.0)        # fit (mean 68)
    dg.fit = cp.DishFit(60, 62, 70, 66, False)               # invalid -> box
    assert tl.effective_geometry(dg) == (50.0, 50.0, 50.0)
    dg.fit = None                                            # none -> box
    assert tl.effective_geometry(dg) == (50.0, 50.0, 50.0)


def test_petri_columns_include_fit_fields():
    import track_larvae_video as tl
    for c in ("petri_geometry_source", "petri_fit_cx", "petri_fit_cy",
              "petri_fit_a", "petri_fit_b", "petri_fit_valid"):
        assert c in tl.PETRI_COLUMNS
