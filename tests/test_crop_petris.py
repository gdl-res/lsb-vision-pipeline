"""Unit tests for the pure-geometry functions of crop_petris_yolo_seg.

Run: pytest tests/test_crop_petris.py
These cover SPECIFICATION.md 20 requirements: label parsing, row-major sorting,
crop coordinate conversion, polygon remapping, and empty-label behavior.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import crop_petris_yolo_seg as cp  # noqa: E402


def make_box(cx, cy, size=100.0, conf=0.9):
    return cp.PetriBox(cx - size / 2, cy - size / 2,
                       cx + size / 2, cy + size / 2, conf)


# --- label parsing --------------------------------------------------------- #
def test_parse_polygon_line():
    cls, poly = cp.parse_yolo_seg_line("0 0.1 0.1 0.2 0.1 0.2 0.2")
    assert cls == 0
    assert poly.shape == (3, 2)
    np.testing.assert_allclose(poly[1], [0.2, 0.1])


def test_parse_bbox_line_becomes_rectangle():
    cls, poly = cp.parse_yolo_seg_line("0 0.5 0.5 0.2 0.4")
    assert poly.shape == (4, 2)
    np.testing.assert_allclose(poly.min(axis=0), [0.4, 0.3])
    np.testing.assert_allclose(poly.max(axis=0), [0.6, 0.7])


def test_parse_blank_and_malformed():
    assert cp.parse_yolo_seg_line("") is None
    assert cp.parse_yolo_seg_line("0 0.1 0.2 0.3") is None  # odd polygon count


def test_format_roundtrip():
    line = cp.format_yolo_seg_line(0, np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert line.startswith("0 ")
    assert line.split()[1:] == ["0.100000", "0.200000", "0.300000", "0.400000"]


# --- row-major sorting ----------------------------------------------------- #
def test_grid_sort_3x2_row_major():
    # Scrambled 3x2 grid; expect IDs 1..6 left-to-right, top-to-bottom.
    boxes = [make_box(200, 300), make_box(100, 100), make_box(200, 100),
             make_box(100, 300), make_box(100, 200), make_box(200, 200)]
    ordered = cp.grid_sort(boxes, 3, 2)
    centers = [(b.cx, b.cy) for b in ordered]
    assert centers == [(100, 100), (200, 100),
                       (100, 200), (200, 200),
                       (100, 300), (200, 300)]


def test_assign_dish_ids_complete():
    boxes = [make_box(100, 100), make_box(200, 100),
             make_box(100, 200), make_box(200, 200),
             make_box(100, 300), make_box(200, 300)]
    dishes, complete = cp.assign_dish_ids(boxes, 3, 2)
    assert complete
    assert [d[0] for d in dishes] == [1, 2, 3, 4, 5, 6]
    assert dishes[2][1:3] == (1, 0)  # dish 3 -> row 1, col 0


def test_select_best_boxes_picks_top_conf():
    boxes = [make_box(0, 0, conf=c) for c in (0.1, 0.9, 0.5, 0.8, 0.2, 0.7, 0.95)]
    selected, complete = cp.select_best_boxes(boxes, 3, 2)
    assert complete and len(selected) == 6
    assert min(b.conf for b in selected) == 0.2  # 0.1 dropped


def test_select_best_boxes_insufficient():
    boxes = [make_box(0, 0) for _ in range(4)]
    selected, complete = cp.select_best_boxes(boxes, 3, 2)
    assert not complete and len(selected) == 4


# --- crop geometry --------------------------------------------------------- #
def test_compute_crop_geom_square_and_centered():
    box = cp.PetriBox(100, 100, 200, 180, 0.9)  # w=100, h=80
    geom = cp.compute_crop_geom(box, margin=0.0)
    assert geom.size == 100  # max(w,h)
    # centered on (150, 140)
    assert geom.x0 == 100 and geom.y0 == 90


def test_extract_square_crop_pads_edges():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    geom = cp.CropGeom(x0=-10, y0=-10, size=30)
    crop = cp.extract_square_crop(img, geom)
    assert crop.shape == (30, 30, 3)
    assert crop[0, 0].sum() == 0          # padded corner is black
    assert crop[20, 20].sum() == 255 * 3  # interior copied from image


# --- polygon remapping ----------------------------------------------------- #
def test_remap_polygon_into_crop():
    pytest.importorskip("shapely")
    # Full image 100x100; polygon is a 20x20 square at pixels (40..60).
    poly_norm = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
    geom = cp.CropGeom(x0=40, y0=40, size=20)  # crop exactly over the square
    pieces = cp.remap_polygon_to_crop(poly_norm, 100, 100, geom)
    assert len(pieces) == 1
    # square fills the whole crop -> normalized 0..1
    np.testing.assert_allclose(pieces[0].min(axis=0), [0, 0], atol=1e-6)
    np.testing.assert_allclose(pieces[0].max(axis=0), [1, 1], atol=1e-6)


def test_remap_polygon_outside_crop_returns_empty():
    pytest.importorskip("shapely")
    poly_norm = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]])
    geom = cp.CropGeom(x0=500, y0=500, size=20)
    assert cp.remap_polygon_to_crop(poly_norm, 100, 100, geom) == []


# --- distortion proxy ------------------------------------------------------ #
def test_distortion_flag_triggers_on_elongation():
    square = cp.PetriBox(0, 0, 100, 100, 0.9)
    elong = cp.PetriBox(0, 0, 130, 100, 0.9)  # ar=1.3 -> > tol 0.15
    assert cp.distortion_proxy(square, 200, 200, 0.15)["petri_distortion_flag"] is False
    assert cp.distortion_proxy(elong, 200, 200, 0.15)["petri_distortion_flag"] is True
