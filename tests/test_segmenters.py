"""Tests for the segmenter backend layer and its scoring.

The comparison's validity rests on every backend producing detections that are
*measured* identically, so the central tests pin `detection_from_polygon` and
the Mask R-CNN output conversion to the production YOLO parser.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cache_detections as cd  # noqa: E402
import seg_coco  # noqa: E402
import segmenters as sg  # noqa: E402
import track_larvae_video as tl  # noqa: E402

torch = pytest.importorskip("torch")


def _fake_results(xyxy, conf, polys):
    """Minimal ultralytics Results stand-in for tl._parse_larva_results."""
    n = len(xyxy)
    boxes = SimpleNamespace(
        xyxy=torch.tensor(xyxy, dtype=torch.float32),
        conf=torch.tensor(conf, dtype=torch.float32),
        cls=torch.zeros(n), id=None)
    masks = SimpleNamespace(xy=[np.asarray(p, dtype=np.float32)
                                for p in polys])
    return [SimpleNamespace(boxes=boxes, masks=masks)]


POLY = np.array([[10, 10], [40, 12], [42, 20], [12, 18]], dtype=np.float32)


def test_detection_from_polygon_matches_production_parser():
    prod = tl._parse_larva_results(
        _fake_results([[9, 9, 43, 21]], [0.8], [POLY]))[0]
    ours = sg.detection_from_polygon(9, 9, 43, 21, np.float32(0.8), POLY)
    for f in ("conf", "x1", "y1", "x2", "y2", "cx", "cy", "area_px"):
        assert getattr(ours, f) == getattr(prod, f), f
    np.testing.assert_array_equal(ours.polygon, prod.polygon)


def test_degenerate_polygon_falls_back_to_box_like_production():
    bad = POLY[:2]
    prod = tl._parse_larva_results(
        _fake_results([[0, 0, 10, 4]], [0.5], [bad]))[0]
    ours = sg.detection_from_polygon(0, 0, 10, 4, np.float32(0.5), bad)
    assert ours.polygon is None and prod.polygon is None
    assert (ours.cx, ours.cy, ours.area_px) == (prod.cx, prod.cy, prod.area_px)


def test_maskrcnn_output_conversion_filters_and_measures():
    h = w = 64
    m = np.zeros((h, w), np.float32)
    m[20:30, 10:50] = 0.9                           # a 40x10 larva
    out = {"boxes": torch.tensor([[10., 20., 50., 30.], [0., 0., 5., 5.],
                                  [1., 1., 9., 9.]]),
           "scores": torch.tensor([0.9, 0.05, 0.8]),
           "labels": torch.tensor([1, 1, 2]),          # 3rd is not larva
           "masks": torch.from_numpy(np.stack([m, m, m])[:, None])}
    dets = sg.maskrcnn_output_to_detections(out, conf=0.1)
    assert len(dets) == 1
    d = dets[0]
    assert d.polygon is not None
    morph = tl.polygon_morphology(d.polygon)
    assert d.area_px == pytest.approx(morph["area_px"])
    assert (d.cx, d.cy) == pytest.approx((morph["cx"], morph["cy"]))
    assert d.cx == pytest.approx(29.5, abs=0.6)
    assert d.cy == pytest.approx(24.5, abs=0.6)


def test_masks_to_polygons_uses_ultralytics_contours():
    m = np.zeros((1, 32, 32), np.uint8)
    m[0, 5:15, 5:25] = 1
    poly = sg.masks_to_polygons(m)[0]
    assert poly.shape[1] == 2 and len(poly) >= 4
    assert sg.masks_to_polygons(np.zeros((0, 8, 8), np.uint8)) == []


def test_torch_device_strings():
    assert sg.torch_device("0") == "cuda:0"
    assert sg.torch_device(1) == "cuda:1"
    assert sg.torch_device("cpu") == "cpu"


def test_build_maskrcnn_head_has_two_classes():
    m = sg.build_maskrcnn(num_classes=2, coco_init=False)
    assert m.roi_heads.box_predictor.cls_score.out_features == 2
    assert m.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 2
    sg.set_maskrcnn_inference_size(m, 640)
    assert m.transform.min_size == (640,) and m.transform.max_size == 640


# --- COCO scoring ---------------------------------------------------------- #
def _gt_one_square():
    sq = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float)
    return seg_coco.build_coco_gt([{"id": 1, "file_name": "a", "width": 64,
                                    "height": 64, "polygons": [sq]}]), sq


def test_perfect_prediction_scores_one():
    gt, sq = _gt_one_square()
    det = sg.detection_from_polygon(10, 10, 30, 30, 0.9, sq)
    preds = seg_coco.detections_to_coco(1, [det], 64, 64)
    s = seg_coco.score(gt, preds)
    assert s["mask_map50"] == pytest.approx(1.0)
    assert s["box_map50"] == pytest.approx(1.0)
    assert s["mask_precision"] == 1.0 and s["mask_recall"] == 1.0


def test_scoring_does_not_mutate_inputs():
    gt, sq = _gt_one_square()
    det = sg.detection_from_polygon(10, 10, 30, 30, 0.9, sq)
    preds = seg_coco.detections_to_coco(1, [det], 64, 64)
    seg_coco.score(gt, preds)
    assert isinstance(gt["annotations"][0]["segmentation"], list)
    assert preds[0]["bbox"] == [10.0, 10.0, 20.0, 20.0]


def test_false_positive_and_threshold():
    gt, sq = _gt_one_square()
    good = sg.detection_from_polygon(10, 10, 30, 30, 0.9, sq)
    far = sg.detection_from_polygon(40, 40, 60, 60, 0.6, sq + 30)
    low = sg.detection_from_polygon(40, 40, 60, 60, 0.1, sq + 30)
    preds = seg_coco.detections_to_coco(1, [good, far, low], 64, 64)
    p, r = seg_coco.precision_recall_at(gt, preds, "segm", conf=0.25)
    assert (p, r) == (0.5, 1.0)


def test_read_yolo_polygons_scales_and_filters_class(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n1 0.1 0.1 0.2 0.1 0.2 0.2\n")
    polys = seg_coco.read_yolo_polygons(f, 100, 200)
    assert len(polys) == 1
    np.testing.assert_allclose(polys[0][1], [50, 20])


# --- cache helpers --------------------------------------------------------- #
def test_compare_cache_prefix_detects_change(tmp_path):
    import gzip
    for d, val in (("a", "0.5"), ("b", "0.5"), ("c", "0.6")):
        (tmp_path / d).mkdir()
        for kind in ("detections", "frames"):
            with gzip.open(tmp_path / d / f"v.{kind}.csv.gz", "wt") as fh:
                fh.write(f"frame_idx,conf\n0,{val}\n1,{val}\n5,9\n")
    assert cd.compare_cache_prefix(tmp_path / "a", tmp_path / "b", "v", 2) == []
    probs = cd.compare_cache_prefix(tmp_path / "a", tmp_path / "c", "v", 2)
    assert probs and "first difference" in probs[0]


def test_cut_subset_keeps_every_stratum():
    vids = ["20260514_a_video_nostress_nat9",
            "20260514_b_video_warm", "20260514_c_video_warm",
            "20260514_d_video_cold", "20260514_e_video_cold",
            "20260623_nostress_lucecold", "20260624_nostress_lucecold_2",
            "20260624_38°C_1h_lucenat_2", "20260625_38°C_1h_lucewarm_2",
            "20260624_38°C_2h_lucenat_2", "20260625_38°C_2h_lucewarm_2",
            "20260624_38°C_3h_lucenat"]
    cfg = {"cache_runtime": {"fallback_n_videos": 8}}
    keep = cd.cut_subset(cfg, vids)
    assert len(keep) == 8
    assert {cd.stratum_of(v) for v in keep} == {cd.stratum_of(v) for v in vids}


# --- Mask R-CNN training data ---------------------------------------------- #
import train_maskrcnn as tm  # noqa: E402


def test_crop_pad_pads_outside_the_image():
    img = np.full((10, 20, 3), 7, np.uint8)
    out = tm.crop_pad(img, -5, -5, 12)
    assert out.shape == (12, 12, 3)
    assert out[0, 0].tolist() == [0, 0, 0] and out[6, 6].tolist() == [7, 7, 7]
    masks = np.ones((2, 10, 20), np.uint8)
    assert tm.crop_pad(masks, 15, 0, 10).sum() == 2 * 5 * 10


def test_sample_crop_stays_in_bounds_and_is_seeded():
    polys = [np.array([[50., 50.], [60., 50.], [60., 60.]])]
    a = tm.sample_crop(np.random.default_rng(1), 100, 200, polys, (0.2, 1.0),
                       0.5)
    b = tm.sample_crop(np.random.default_rng(1), 100, 200, polys, (0.2, 1.0),
                       0.5)
    assert a == b
    for s in range(50):
        x0, y0, side = tm.sample_crop(np.random.default_rng(s), 100, 200,
                                      polys, (0.2, 1.0), 0.5)
        assert 40 <= side <= 200
        if side <= 200:
            assert 0 <= x0 <= 200 - side
        if side <= 100:
            assert 0 <= y0 <= 100 - side


def test_train_sample_is_deterministic_and_masks_align(tmp_path):
    import cv2
    img = np.full((300, 400, 3), 200, np.uint8)
    cv2.rectangle(img, (100, 100), (160, 120), (20, 20, 20), -1)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    (tmp_path / "a.txt").write_text(
        "0 0.25 0.3333 0.4 0.3333 0.4 0.4 0.25 0.4")
    rec = {"image": tmp_path / "a.jpg", "label": tmp_path / "a.txt"}
    mcfg = {"imgsz": 64, "crop_frac": [1.0, 1.0],
            "instance_centered_fraction": 1.0, "min_instance_visible": 0.3,
            "hflip": 0.0, "color_jitter": {"brightness": 0, "contrast": 0,
                                           "saturation": 0, "hue": 0}}
    i1, m1 = tm.make_train_sample(rec, mcfg, 0, 0, 0)
    i2, m2 = tm.make_train_sample(rec, mcfg, 0, 0, 0)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(m1, m2)
    assert len(m1) == 1
    # the mask covers the dark rectangle
    assert i1[m1[0] > 0].mean() < 80
