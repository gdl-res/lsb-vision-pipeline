#!/usr/bin/env python3
"""Crop full-tray images into per-Petri-dish square crops and remap YOLO
segmentation labels into each crop's coordinate frame.

Implements SPECIFICATION.md sections 4 (Petri detection), 7 (coordinates),
18.1 (CLI), 20 (testable functions) and 21 (provenance).

Pipeline per image:
  1. Run the Petri detector on the full image.
  2. Keep class `petri`, confidence-filter, NMS (handled by the model).
  3. Select rows*cols best boxes (by confidence).
  4. Assign dish IDs in row-major order.
  5. Build a square crop centered on each Petri box.
  6. Remap full-frame YOLO segmentation polygons into each crop (Shapely clip).
  7. Write crop image, crop label .txt (empty file if no larva), and metadata.
  8. Optionally write a debug overlay.

The pure-geometry functions (parsing, grid sorting, crop geometry, polygon
remapping) are import-safe and unit-testable without YOLO/OpenCV installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PETRI_CLASS_ID = 0  # SPECIFICATION.md 3.1: petri detector has a single class `petri`.


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class PetriBox:
    """An axis-aligned Petri detection in full-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def radius(self) -> float:
        return 0.5 * min(self.w, self.h)

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 0 else float("nan")


@dataclass
class CropGeom:
    """Square crop window in full-frame pixel coordinates.

    `x0`/`y0` may be negative when the square extends past the image edge; the
    extractor pads with black so the Petri box stays centered. Remapping uses
    these same (possibly negative) origins, which keeps coordinates consistent.
    """

    x0: int
    y0: int
    size: int


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Rim-occlusion attribution (shared by QC and behavioral features)
# --------------------------------------------------------------------------- #
def _rim_occluded_mask(missing: np.ndarray, r_norm: np.ndarray,
                       rim_r: float) -> np.ndarray:
    """Boolean mask of frames belonging to a missing gap that began with the
    larva last seen at/near the rim (last-seen r_norm >= rim_r) -- i.e.
    missingness attributable to the larva tucking under the dish edge rather
    than a mid-dish detection failure."""
    missing = np.asarray(missing, dtype=bool)
    r = np.asarray(r_norm, dtype=float)
    n = missing.size
    occluded = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        last_r = r[i - 1] if i - 1 >= 0 else np.nan  # last seen before the gap
        if np.isfinite(last_r) and last_r >= rim_r:
            occluded[i:j] = True
        i = j
    return occluded


def rim_occlusion_fraction(missing: np.ndarray, r_norm: np.ndarray,
                           rim_r: float) -> float:
    """Fraction of all frames that are rim-occlusion missing (see
    _rim_occluded_mask). Comparable denominator to missing_fraction, so always
    <= the missing fraction. NaN for an empty sequence."""
    missing = np.asarray(missing, dtype=bool)
    if missing.size == 0:
        return float("nan")
    return float(_rim_occluded_mask(missing, r_norm, rim_r).mean())


def rim_occlusion_events(missing: np.ndarray, r_norm: np.ndarray,
                         rim_r: float) -> int:
    """Number of distinct rim-occlusion episodes (contiguous attributed gaps)
    -- how many times the larva went under the dish edge."""
    occ = _rim_occluded_mask(missing, r_norm, rim_r)
    if occ.size == 0:
        return 0
    return int((occ[1:] & ~occ[:-1]).sum() + (1 if occ[0] else 0))


# --------------------------------------------------------------------------- #
# YOLO segmentation label parsing (SPECIFICATION.md 20)
# --------------------------------------------------------------------------- #
def parse_yolo_seg_line(line: str) -> Optional[tuple[int, np.ndarray]]:
    """Parse one YOLO label line into (class_id, Nx2 normalized polygon).

    Supports polygon segmentation lines (`cls x1 y1 x2 y2 ...`) and falls back
    to bbox lines (`cls xc yc w h`) by converting the box to a 4-point polygon.
    Returns None for blank lines. Coordinates stay in normalized [0, 1] space.
    """
    parts = line.split()
    if not parts:
        return None
    cls = int(float(parts[0]))
    coords = [float(v) for v in parts[1:]]
    if len(coords) == 4:  # bbox: xc yc w h -> rectangle polygon
        xc, yc, w, h = coords
        poly = np.array([
            [xc - w / 2, yc - h / 2],
            [xc + w / 2, yc - h / 2],
            [xc + w / 2, yc + h / 2],
            [xc - w / 2, yc + h / 2],
        ], dtype=float)
        return cls, poly
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None  # not a valid polygon
    poly = np.array(coords, dtype=float).reshape(-1, 2)
    return cls, poly


def load_yolo_seg_label(path: Path) -> list[tuple[int, np.ndarray]]:
    if not Path(path).is_file():
        return []
    items: list[tuple[int, np.ndarray]] = []
    for line in Path(path).read_text().splitlines():
        parsed = parse_yolo_seg_line(line)
        if parsed is not None:
            items.append(parsed)
    return items


def format_yolo_seg_line(cls: int, poly_norm: np.ndarray) -> str:
    flat = " ".join(f"{v:.6f}" for v in np.asarray(poly_norm).reshape(-1))
    return f"{cls} {flat}"


# --------------------------------------------------------------------------- #
# Petri box selection and row-major sorting (SPECIFICATION.md 4, 20)
# --------------------------------------------------------------------------- #
def select_best_boxes(
    boxes: Sequence[PetriBox], rows: int, cols: int
) -> tuple[list[PetriBox], bool]:
    """Keep the rows*cols highest-confidence boxes.

    Returns (selected, complete). `complete` is True only when exactly
    rows*cols boxes are available. Selection is confidence-based and bounded
    (no combinatorial grid search; see SPECIFICATION.md 19).
    """
    n = rows * cols
    ordered = sorted(boxes, key=lambda b: b.conf, reverse=True)
    if len(ordered) >= n:
        return ordered[:n], True
    return ordered, False


def grid_sort(boxes: Sequence[PetriBox], rows: int, cols: int) -> list[PetriBox]:
    """Sort exactly rows*cols boxes into row-major order.

    Bands by y into `rows` groups, then sorts each band left-to-right by x.
    """
    if len(boxes) != rows * cols:
        raise ValueError(
            f"grid_sort needs exactly {rows * cols} boxes, got {len(boxes)}"
        )
    by_y = sorted(boxes, key=lambda b: b.cy)
    out: list[PetriBox] = []
    for r in range(rows):
        band = by_y[r * cols:(r + 1) * cols]
        out.extend(sorted(band, key=lambda b: b.cx))
    return out


def assign_dish_ids(
    boxes: Sequence[PetriBox], rows: int, cols: int
) -> tuple[list[tuple[int, int, int, PetriBox]], bool]:
    """Return [(dish_id, row, col, box), ...] in row-major order.

    When exactly rows*cols boxes are present, uses the grid layout. Otherwise
    falls back to a best-effort (y, x) ordering and reports complete=False.
    """
    if len(boxes) == rows * cols:
        ordered = grid_sort(boxes, rows, cols)
        complete = True
    else:
        ordered = sorted(boxes, key=lambda b: (b.cy, b.cx))
        complete = False
    result = []
    for i, box in enumerate(ordered):
        result.append((i + 1, i // cols, i % cols, box))
    return result, complete


# --------------------------------------------------------------------------- #
# Crop geometry and polygon remapping (SPECIFICATION.md 4, 7)
# --------------------------------------------------------------------------- #
def compute_crop_geom(box: PetriBox, margin: float) -> CropGeom:
    """Square crop centered on the Petri box (SPECIFICATION.md 4).

    crop_size = max(w, h) * (1 + margin), centered on (cx, cy).
    """
    diameter = max(box.w, box.h)
    size = int(round(diameter * (1.0 + margin)))
    x0 = int(round(box.cx - size / 2.0))
    y0 = int(round(box.cy - size / 2.0))
    return CropGeom(x0=x0, y0=y0, size=size)


@dataclass
class DishFit:
    """An ellipse fitted to the physical dish (refined geometry for §26.2)."""

    cx: float
    cy: float
    a: float            # major semi-axis (px)
    b: float            # minor semi-axis (px)
    valid: bool

    @property
    def radius(self) -> float:
        return 0.5 * (self.a + self.b)         # mean semi-axis

    @property
    def aspect_ratio(self) -> float:
        return self.a / self.b if self.b > 0 else float("nan")


def fit_dish_geometry(gray: np.ndarray, box: "PetriBox", expand: float = 1.6,
                      min_area_frac: float = 0.04) -> DishFit:
    """Fit the physical dish by ellipse-fitting the bright disk inside a ROI
    seeded by the YOLO box (SPECIFICATION.md 26.2).

    The YOLO box localizes the dish but is trained on a tighter convention than
    the true rim; the high-contrast white disk gives an accurate center/radius
    (and measured ellipticity). Returns DishFit(valid=False) -- carrying the box
    geometry -- when no plausible disk contour is found, so callers fall back.
    """
    import cv2
    fallback = DishFit(box.cx, box.cy, box.radius, box.radius, False)
    half = max(box.w, box.h) / 2.0 * expand
    x0 = int(max(0, box.cx - half))
    y0 = int(max(0, box.cy - half))
    x1 = int(min(gray.shape[1], box.cx + half))
    y1 = int(min(gray.shape[0], box.cy + half))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return fallback
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rcx, rcy = roi.shape[1] / 2.0, roi.shape[0] / 2.0
    best, best_d = None, float("inf")
    for c in contours:
        if cv2.contourArea(c) < min_area_frac * roi.size:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        d = (m["m10"] / m["m00"] - rcx) ** 2 + (m["m01"] / m["m00"] - rcy) ** 2
        if d < best_d:
            best_d, best = d, c
    if best is None or len(best) < 5:
        return fallback
    (ex, ey), (axis1, axis2), _ = cv2.fitEllipse(best)
    return DishFit(x0 + ex, y0 + ey, max(axis1, axis2) / 2.0,
                   min(axis1, axis2) / 2.0, True)


def extract_square_crop(image: np.ndarray, geom: CropGeom) -> np.ndarray:
    """Extract a square crop, zero-padding regions outside the image."""
    h, w = image.shape[:2]
    if image.ndim == 3:
        crop = np.zeros((geom.size, geom.size, image.shape[2]), dtype=image.dtype)
    else:
        crop = np.zeros((geom.size, geom.size), dtype=image.dtype)
    sx0, sy0 = max(0, geom.x0), max(0, geom.y0)
    sx1, sy1 = min(w, geom.x0 + geom.size), min(h, geom.y0 + geom.size)
    if sx1 <= sx0 or sy1 <= sy0:
        return crop  # crop entirely outside image
    dx0, dy0 = sx0 - geom.x0, sy0 - geom.y0
    crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return crop


def remap_polygon_to_crop(
    poly_norm: np.ndarray,
    img_w: int,
    img_h: int,
    geom: CropGeom,
    min_area_ratio: float = 0.0,
) -> list[np.ndarray]:
    """Clip a full-frame normalized polygon to a crop and renormalize.

    Returns a list of normalized [0,1] polygons in crop space (a single source
    polygon may be split into several pieces by the crop boundary). Pieces whose
    area is below `min_area_ratio` of the crop area are dropped.
    """
    from shapely.geometry import Polygon, box as shapely_box
    from shapely.validation import make_valid

    # Denormalize full-frame polygon to pixels.
    px = poly_norm.copy().astype(float)
    px[:, 0] *= img_w
    px[:, 1] *= img_h
    poly = Polygon(px)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area <= 0:
        return []

    crop_rect = shapely_box(
        geom.x0, geom.y0, geom.x0 + geom.size, geom.y0 + geom.size
    )
    clipped = poly.intersection(crop_rect)
    if clipped.is_empty:
        return []

    geoms = getattr(clipped, "geoms", [clipped])
    crop_area = float(geom.size * geom.size)
    out: list[np.ndarray] = []
    for g in geoms:
        if g.geom_type != "Polygon" or g.area <= 0:
            continue
        if crop_area > 0 and (g.area / crop_area) < min_area_ratio:
            continue
        coords = np.array(g.exterior.coords[:-1], dtype=float)  # drop closing pt
        coords[:, 0] = (coords[:, 0] - geom.x0) / geom.size
        coords[:, 1] = (coords[:, 1] - geom.y0) / geom.size
        out.append(np.clip(coords, 0.0, 1.0))
    return out


# --------------------------------------------------------------------------- #
# Distortion proxy (SPECIFICATION.md 26.2)
# --------------------------------------------------------------------------- #
def distortion_proxy(
    box: PetriBox, img_w: int, img_h: int, tol: float
) -> dict:
    frame_cx, frame_cy = img_w / 2.0, img_h / 2.0
    frame_diag = float(np.hypot(img_w, img_h))
    offset = float(np.hypot(box.cx - frame_cx, box.cy - frame_cy))
    ar = box.aspect_ratio
    return {
        "petri_aspect_ratio": round(ar, 6),
        "petri_distortion_flag": bool(abs(1.0 - ar) > tol),
        "petri_offset_norm": round(offset / (0.5 * frame_diag), 6),
    }


# --------------------------------------------------------------------------- #
# YOLO model wrapper
# --------------------------------------------------------------------------- #
def load_petri_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def detect_petris(model, image_path: Path, conf: float, iou: float,
                  device: Optional[str]) -> list[PetriBox]:
    results = model.predict(
        source=str(image_path), conf=conf, iou=iou,
        device=device, verbose=False,
    )
    boxes: list[PetriBox] = []
    if not results:
        return boxes
    r = results[0]
    if r.boxes is None:
        return boxes
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    tids = (r.boxes.id.cpu().numpy().astype(int)
            if r.boxes.id is not None else [None] * len(xyxy))
    for (x1, y1, x2, y2), c, cl, tid in zip(xyxy, confs, clss, tids):
        if cl != PETRI_CLASS_ID:
            continue
        boxes.append(PetriBox(float(x1), float(y1), float(x2), float(y2),
                              float(c), None if tid is None else int(tid)))
    return boxes


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #
def process_image(
    image_path: Path,
    label_path: Path,
    out_images: Path,
    out_labels: Path,
    out_debug: Optional[Path],
    model,
    args,
) -> tuple[list[dict], Optional[dict]]:
    """Process one full-tray image. Returns (crop_metadata_records, failure)."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return [], {"image": image_path.name, "reason": "unreadable_image"}
    img_h, img_w = img.shape[:2]

    boxes = detect_petris(model, image_path, args.conf, args.iou, args.device)
    selected, complete = select_best_boxes(boxes, args.rows, args.cols)

    if not complete:
        failure = {
            "image": image_path.name,
            "reason": "insufficient_petri_detections",
            "n_detected": len(boxes),
            "n_expected": args.rows * args.cols,
        }
        if args.on_failure == "error":
            raise RuntimeError(f"{image_path.name}: {failure['reason']}")
        if args.on_failure == "skip":
            return [], failure
        # keep-best: continue with what we have

    labels = load_yolo_seg_label(label_path)
    dishes, grid_complete = assign_dish_ids(selected, args.rows, args.cols)

    overlay = img.copy() if (args.debug and out_debug is not None) else None
    records: list[dict] = []
    stem = image_path.stem

    for dish_id, row, col, box in dishes:
        geom = compute_crop_geom(box, args.margin)
        crop = extract_square_crop(img, geom)

        # Remap each full-frame polygon into this crop.
        crop_label_items: list[tuple[int, np.ndarray]] = []
        for cls, poly in labels:
            for piece in remap_polygon_to_crop(
                poly, img_w, img_h, geom, args.min_label_area_ratio
            ):
                crop_label_items.append((cls, piece))

        crop_name = f"{stem}_dish{dish_id:02d}"
        cv2.imwrite(str(out_images / f"{crop_name}.jpg"), crop)
        # Always write a label file; empty dishes get an empty .txt (SPECIFICATION.md 20).
        (out_labels / f"{crop_name}.txt").write_text(
            "\n".join(format_yolo_seg_line(c, p) for c, p in crop_label_items)
        )

        dist = distortion_proxy(box, img_w, img_h, args.distortion_aspect_tol)
        records.append({
            "source_image": image_path.name,
            "crop_name": crop_name,
            "dish_id": dish_id,
            "row": row,
            "col": col,
            "grid_complete": grid_complete,
            "petri_track_id": box.track_id,
            "petri_conf": round(box.conf, 6),
            "petri_x1": round(box.x1, 3), "petri_y1": round(box.y1, 3),
            "petri_x2": round(box.x2, 3), "petri_y2": round(box.y2, 3),
            "petri_cx": round(box.cx, 3), "petri_cy": round(box.cy, 3),
            "petri_w": round(box.w, 3), "petri_h": round(box.h, 3),
            "petri_radius": round(box.radius, 3),
            "crop_x0": geom.x0, "crop_y0": geom.y0, "crop_size": geom.size,
            "n_larva_polygons": len(crop_label_items),
            **dist,
        })

        if overlay is not None:
            cv2.rectangle(overlay, (int(box.x1), int(box.y1)),
                          (int(box.x2), int(box.y2)), (0, 255, 0), 2)
            cv2.rectangle(overlay, (geom.x0, geom.y0),
                          (geom.x0 + geom.size, geom.y0 + geom.size),
                          (255, 0, 0), 2)
            cv2.putText(overlay, str(dish_id), (int(box.cx), int(box.cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if overlay is not None:
        cv2.imwrite(str(out_debug / f"{stem}_overlay.jpg"), overlay)

    return records, (None if complete else failure)


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def iter_images(images_dir: Path, max_images: Optional[int]) -> list[Path]:
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    return files[:max_images] if max_images else files


def write_provenance(out: Path, args, model_path: Path,
                     n_images: int, n_failures: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_crop"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/crop_petris_yolo_seg.py",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.images), str(args.labels)],
        "output_path": str(args.out),
        "petri_model": str(model_path),
        "petri_model_sha256": sha256_file(model_path),
        "n_images": n_images,
        "n_failures": n_failures,
        "parameters": {
            "rows": args.rows, "cols": args.cols, "margin": args.margin,
            "conf": args.conf, "iou": args.iou,
            "min_label_area_ratio": args.min_label_area_ratio,
            "distortion_aspect_tol": args.distortion_aspect_tol,
            "on_failure": args.on_failure,
        },
    }
    with open(out / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--petri-model", required=True, type=Path)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--min-label-area-ratio", type=float, default=0.0,
                   help="Drop remapped polygon pieces smaller than this "
                        "fraction of the crop area.")
    p.add_argument("--distortion-aspect-tol", type=float, default=0.15)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--on-failure", choices=["skip", "keep-best", "error"],
                   default="skip")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_debug = args.out / "debug" if args.debug else None
    for d in (out_images, out_labels, out_debug):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.images, args.max_images)
    if not images:
        print(f"No images found in {args.images}", file=sys.stderr)
        return 1
    print(f"Found {len(images)} image(s). Loading Petri model...")
    model = load_petri_model(str(args.petri_model))

    meta_path = args.out / "crop_metadata.jsonl"
    failures: list[dict] = []
    n_crops = 0
    with open(meta_path, "w") as meta_fh:
        for i, image_path in enumerate(images, 1):
            label_path = args.labels / f"{image_path.stem}.txt"
            records, failure = process_image(
                image_path, label_path, out_images, out_labels,
                out_debug, model, args,
            )
            for rec in records:
                meta_fh.write(json.dumps(rec) + "\n")
            n_crops += len(records)
            if failure is not None:
                failures.append(failure)
            print(f"[{i}/{len(images)}] {image_path.name}: "
                  f"{len(records)} crop(s)"
                  + (f"  FAILED({failure['reason']})" if failure else ""))

    if failures:
        with open(args.out / "failure_report.jsonl", "w") as fh:
            for f in failures:
                fh.write(json.dumps(f) + "\n")

    write_provenance(args.out, args, args.petri_model, len(images), len(failures))
    print(f"\nDone. {n_crops} crops written, {len(failures)} image(s) failed.")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
