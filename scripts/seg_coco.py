#!/usr/bin/env python3
"""COCO-style scoring shared by every segmenter (pycocotools).

One evaluator for all models: ground truth comes from the YOLO
polygon labels, predictions from each model's *polygons* -- the representation
the pipeline actually measures -- so YOLO and Mask R-CNN are scored on equal
terms. Used for Mask R-CNN early stopping (train_maskrcnn.py).

Reported per IoU type ('segm' masks, 'bbox' boxes):
    map50_95, map50       COCO AP (all areas, maxDets 100)
    ar100                 COCO AR@100
    precision, recall     greedy one-to-one matching at IoU >= 0.5 among
                          detections with score >= `pr_conf` (the pipeline's
                          det_conf), i.e. the operating point the tracker uses
"""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
from typing import Optional

import numpy as np

LARVA_CATEGORY = 1


def read_yolo_polygons(label_path: Path, w: int, h: int) -> list[np.ndarray]:
    """YOLO-seg label file -> list of (N, 2) pixel polygons (class 0 only)."""
    polys = []
    if not label_path.is_file():
        return polys
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 7 or int(float(p[0])) != 0:
            continue
        v = np.asarray(p[1:], dtype=float)
        n = len(v) // 2
        xy = v[:2 * n].reshape(n, 2) * np.array([w, h], dtype=float)
        polys.append(xy)
    return polys


def polygon_rle(poly: np.ndarray, h: int, w: int) -> dict:
    from pycocotools import mask as mu
    rles = mu.frPyObjects([np.asarray(poly, dtype=float).ravel().tolist()],
                          h, w)
    return mu.merge(rles)


def build_coco_gt(images: list[dict]) -> dict:
    """images: [{id, file_name, width, height, polygons: [(N,2), ...]}]."""
    from pycocotools import mask as mu
    anns, aid = [], 1
    for im in images:
        for poly in im["polygons"]:
            if len(poly) < 3:
                continue
            rle = polygon_rle(poly, im["height"], im["width"])
            area = float(mu.area(rle))
            if area <= 0:
                continue
            x0, y0 = poly.min(axis=0)
            x1, y1 = poly.max(axis=0)
            anns.append({"id": aid, "image_id": im["id"],
                         "category_id": LARVA_CATEGORY,
                         "segmentation": [poly.ravel().tolist()],
                         "area": area, "iscrowd": 0,
                         "bbox": [float(x0), float(y0),
                                  float(x1 - x0), float(y1 - y0)]})
            aid += 1
    return {"images": [{"id": im["id"], "file_name": im["file_name"],
                        "width": im["width"], "height": im["height"]}
                       for im in images],
            "annotations": anns,
            "categories": [{"id": LARVA_CATEGORY, "name": "larva"}]}


def detections_to_coco(image_id: int, dets, h: int, w: int,
                       sx: float = 1.0, sy: float = 1.0) -> list[dict]:
    """LarvaDetection list -> COCO results; (sx, sy) rescale to GT pixels."""
    out = []
    for d in dets:
        box = [d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sy]
        if d.polygon is not None and len(d.polygon) >= 3:
            poly = np.asarray(d.polygon, dtype=float) * np.array([sx, sy])
            rle = polygon_rle(poly, h, w)
        else:                                   # box-only fallback
            x0, y0, bw, bh = box
            poly = np.array([[x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh],
                             [x0, y0 + bh]])
            rle = polygon_rle(poly, h, w)
        rle = dict(rle)
        rle["counts"] = rle["counts"].decode("ascii") \
            if isinstance(rle["counts"], bytes) else rle["counts"]
        out.append({"image_id": image_id, "category_id": LARVA_CATEGORY,
                    "segmentation": rle, "bbox": [float(v) for v in box],
                    "score": float(d.conf)})
    return out


def _coco_eval(gt: dict, preds: list[dict], iou_type: str):
    """COCOeval on deep copies: pycocotools rewrites both in place (polygons ->
    RLE in the GT, and loadRes replaces segm results' boxes with mask boxes),
    which would corrupt the next iou_type and the P/R pass."""
    from pycocotools.coco import COCO
    gt, preds = copy.deepcopy(gt), copy.deepcopy(preds)
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        cgt = COCO()
        cgt.dataset = gt
        cgt.createIndex()
        if not preds:
            return None
        cdt = cgt.loadRes(preds)
        ev = COCOeval(cgt, cdt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def precision_recall_at(gt: dict, preds: list[dict], iou_type: str,
                        conf: float, iou_thr: float = 0.5) -> tuple[float, float]:
    """Greedy (score-ordered) one-to-one matching at a fixed operating point."""
    from pycocotools import mask as mu
    gts: dict[int, list] = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    sizes = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    tp = fp = 0
    n_gt = len(gt["annotations"])
    by_img: dict[int, list] = {}
    for p in preds:
        if p["score"] >= conf:
            by_img.setdefault(p["image_id"], []).append(p)
    for img_id, ps in by_img.items():
        ps = sorted(ps, key=lambda p: -p["score"])
        g = gts.get(img_id, [])
        if not g:
            fp += len(ps)
            continue
        h, w = sizes[img_id]
        if iou_type == "segm":
            g_obj = [mu.merge(mu.frPyObjects(a["segmentation"], h, w))
                     for a in g]
            d_obj = [p["segmentation"] for p in ps]
        else:
            g_obj = [a["bbox"] for a in g]
            d_obj = [p["bbox"] for p in ps]
        ious = np.asarray(mu.iou(d_obj, g_obj, [0] * len(g_obj)))
        ious = ious.reshape(len(d_obj), len(g_obj))
        used = set()
        for i in range(len(d_obj)):
            cand = [k for k in np.argsort(-ious[i]) if k not in used
                    and ious[i, k] >= iou_thr]
            if cand:
                used.add(int(cand[0]))
                tp += 1
            else:
                fp += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / n_gt if n_gt else float("nan")
    return prec, rec


def score(gt: dict, preds: list[dict], pr_conf: float = 0.25,
          iou_types: tuple[str, ...] = ("segm", "bbox")) -> dict:
    """All metrics for one model on one image set, flat snake_case keys."""
    out: dict[str, Optional[float]] = {"n_images": len(gt["images"]),
                                       "n_gt": len(gt["annotations"]),
                                       "n_pred": len(preds)}
    for t in iou_types:
        prefix = "mask" if t == "segm" else "box"
        ev = _coco_eval(gt, preds, t)
        if ev is None:
            out.update({f"{prefix}_map50_95": 0.0, f"{prefix}_map50": 0.0,
                        f"{prefix}_ar100": 0.0})
        else:
            out.update({f"{prefix}_map50_95": round(float(ev.stats[0]), 5),
                        f"{prefix}_map50": round(float(ev.stats[1]), 5),
                        f"{prefix}_ar100": round(float(ev.stats[8]), 5)})
        p, r = precision_recall_at(gt, preds, t, pr_conf)
        out[f"{prefix}_precision"] = round(p, 5) if np.isfinite(p) else p
        out[f"{prefix}_recall"] = round(r, 5) if np.isfinite(r) else r
    return out
