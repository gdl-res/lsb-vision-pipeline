#!/usr/bin/env python3
"""Segmenter backends behind one interface (segmenter-comparison experiment).

Every backend turns a dish crop into ``list[tl.LarvaDetection]`` -- the exact
contract the production parser `track_larvae_video._parse_larva_results`
produces -- so the cache, replay and analysis downstream cannot tell which
model made a detection. Two rules keep the comparison about the *model*, not
about measurement code:

  * Centroid and area are always computed from the mask polygon with Shapely,
    exactly as the production parser does (`detection_from_polygon` mirrors it
    line for line, and a test pins the two together).
  * A backend that predicts raster masks (Mask R-CNN) converts them to polygons
    with ultralytics' own `ops.masks2segments(strategy="all")`, which is what
    `Results.masks.xy` uses for the YOLO models.

Backends:
    ultralytics            YOLO11 / YOLOv8 / YOLO26 -seg checkpoints
    torchvision_maskrcnn   Mask R-CNN R50-FPN v2 fine-tuned by train_maskrcnn.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import track_larvae_video as tl

MASK_THRESHOLD = 0.5


def torch_device(device) -> str:
    """ultralytics-style device ('0', 0, 'cpu', 'cuda:0') -> torch string."""
    if device is None:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    d = str(device)
    return f"cuda:{d}" if d.isdigit() else d


# --------------------------------------------------------------------------- #
# Shared measurement (mirrors tl._parse_larva_results)
# --------------------------------------------------------------------------- #
def detection_from_polygon(x1: float, y1: float, x2: float, y2: float,
                           conf: float, poly: Optional[np.ndarray],
                           track_id: Optional[int] = None
                           ) -> tl.LarvaDetection:
    """One LarvaDetection, measured exactly as the production parser does."""
    if poly is not None and len(poly) >= 3:
        from shapely.geometry import Polygon
        area = Polygon(poly).area
        cxc, cyc = Polygon(poly).centroid.x, Polygon(poly).centroid.y
    else:
        poly = None
        area = float((x2 - x1) * (y2 - y1))
        cxc, cyc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return tl.LarvaDetection(
        None if track_id is None else int(track_id), float(conf),
        float(x1), float(y1), float(x2), float(y2),
        float(cxc), float(cyc), float(area),
        None if poly is None else np.asarray(poly, dtype=float))


def masks_to_polygons(masks: np.ndarray) -> list[np.ndarray]:
    """(N, H, W) binary masks -> crop-pixel polygons, the YOLO way."""
    from ultralytics.utils import ops
    if len(masks) == 0:
        return []
    return ops.masks2segments(np.asarray(masks, dtype=np.uint8),
                              strategy="all")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Segmenter:
    name: str = ""
    backend: str = ""
    weights: Optional[Path] = None

    def detect(self, crop: np.ndarray) -> list[tl.LarvaDetection]:
        return self.detect_batch([crop])[0]

    def detect_batch(self, crops: list[np.ndarray]
                     ) -> list[list[tl.LarvaDetection]]:
        raise NotImplementedError


class UltralyticsSegmenter(Segmenter):
    """YOLO -seg via ``model.predict`` -- one crop per call, as production.

    Crops are deliberately NOT batched: ultralytics letterboxes a batch to a
    common shape, which would change the input tensor and break bit-identity
    with an existing production cache.
    """
    backend = "ultralytics"

    def __init__(self, name: str, weights, conf: float, iou: float,
                 imgsz: int, device=None):
        self.name, self.weights = name, Path(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.model = tl.load_model(str(weights))

    def detect_batch(self, crops):
        out = []
        for crop in crops:
            results = self.model.predict(
                crop, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                device=self.device, verbose=False)
            out.append(tl._parse_larva_results(results))
        return out


def build_maskrcnn(num_classes: int = 2, coco_init: bool = False):
    """Mask R-CNN R50-FPN v2; COCO weights with a fresh `num_classes` head."""
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if coco_init else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256,
                                                       num_classes)
    return model


def set_maskrcnn_inference_size(model, imgsz: int) -> None:
    """Resize every input so its long side is `imgsz` (YOLO's imgsz rule)."""
    model.transform.min_size = (int(imgsz),)
    model.transform.max_size = int(imgsz)


class MaskRCNNSegmenter(Segmenter):
    """torchvision Mask R-CNN; all crops of a frame in one forward pass.

    NMS and the score floor are set on the ROI heads so they mean the same as
    YOLO's `iou` / `conf`. Inputs are resized so the long side equals `imgsz`
    (square crops -> imgsz x imgsz), matching what YOLO sees.
    """
    backend = "torchvision_maskrcnn"

    def __init__(self, name: str, weights, conf: float, iou: float,
                 imgsz: int, device=None, fp16: bool = True):
        import torch
        self.name, self.weights = name, Path(weights)
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device = torch.device(torch_device(device))
        self.fp16 = fp16 and self.device.type == "cuda"
        model = build_maskrcnn(num_classes=2, coco_init=False)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        set_maskrcnn_inference_size(model, imgsz)
        model.roi_heads.score_thresh = float(conf)
        model.roi_heads.nms_thresh = float(iou)
        self.model = model.to(self.device).eval()

    def detect_batch(self, crops):
        import torch
        tensors = [torch.from_numpy(np.ascontiguousarray(c[:, :, ::-1]))
                   .to(self.device).permute(2, 0, 1).float().div_(255.0)
                   for c in crops]                      # BGR -> RGB, CHW, [0,1]
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16,
                enabled=self.fp16):
            outs = self.model(tensors)
        return [maskrcnn_output_to_detections(o, self.conf) for o in outs]


def maskrcnn_output_to_detections(out: dict, conf: float
                                  ) -> list[tl.LarvaDetection]:
    """torchvision output dict -> LarvaDetection list (larva = label 1)."""
    boxes = out["boxes"].float().cpu().numpy()
    scores = out["scores"].float().cpu().numpy()
    labels = out["labels"].cpu().numpy()
    keep = (labels == 1) & (scores >= conf)
    if not keep.any():
        return []
    masks = (out["masks"][keep, 0].float() > MASK_THRESHOLD).cpu().numpy()
    polys = masks_to_polygons(masks)
    dets = []
    for (x1, y1, x2, y2), s, poly in zip(boxes[keep], scores[keep], polys):
        dets.append(detection_from_polygon(x1, y1, x2, y2, s,
                                           poly if len(poly) >= 3 else None))
    return dets


BACKENDS = {"ultralytics": UltralyticsSegmenter,
            "torchvision_maskrcnn": MaskRCNNSegmenter}


def load_segmenter(name: str, spec: dict, conf: float, iou: float,
                   device=None, weights=None, imgsz=None) -> Segmenter:
    """Instantiate the backend a `segmenters:` config entry names."""
    cls = BACKENDS[spec["backend"]]
    return cls(name, weights or spec["weights"], conf=conf, iou=iou,
               imgsz=int(imgsz or spec["imgsz"]), device=device)
