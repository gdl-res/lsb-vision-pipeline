#!/usr/bin/env python3
"""Validate a trained YOLO model against a labeled dataset (SPECIFICATION.md 17, 20).

Runs `ultralytics` `val` for the larva segmenter on a held-out split and records
the standard detection/segmentation metrics (precision, recall, mAP@50,
mAP@50-95 for both boxes and masks) so model quality is a *number*, not an
eyeball. The Roboflow `data.yaml` ships with absolute paths from the training
machine; this tool rewrites them to the dataset's real location before running.

Note: the labeled set is full-tray images; the pipeline applies the same model
to Petri crops (sub-regions of the same frames), so these metrics characterize
the detector on its training-domain images.

Usage:
    python scripts/validate_models.py \
      --model weights/larvae/lsb_yolo11s_20260630v1.pt \
      --dataset data4evaluation/dataset_to_evaluate/lsb_20260629_1432imgs_v1.yolov11 \
      --split test --imgsz 1024 --out outputs/model_validation
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import crop_petris_yolo_seg as cp

# ultralytics results_dict key -> clean snake_case column
METRIC_RENAME = {
    "metrics/precision(B)": "box_precision",
    "metrics/recall(B)": "box_recall",
    "metrics/mAP50(B)": "box_map50",
    "metrics/mAP50-95(B)": "box_map50_95",
    "metrics/precision(M)": "mask_precision",
    "metrics/recall(M)": "mask_recall",
    "metrics/mAP50(M)": "mask_map50",
    "metrics/mAP50-95(M)": "mask_map50_95",
    "fitness": "fitness",
}
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


# --------------------------------------------------------------------------- #
# Pure helpers (testable without ultralytics/weights)
# --------------------------------------------------------------------------- #
def find_data_yaml(dataset: Path):
    """Locate the dataset's data.yaml (the *.yaml carrying `names`)."""
    import yaml
    if dataset.is_file():
        return dataset
    candidates = sorted(dataset.glob("*.yaml")) + sorted(dataset.glob("*.yml"))
    for y in candidates:
        try:
            d = yaml.safe_load(y.read_text()) or {}
        except Exception:
            continue
        if "names" in d:
            return y
    raise FileNotFoundError(f"no data.yaml with `names` under {dataset}")


def resolve_dataset_yaml(dataset: Path, out_dir: Path):
    """Rewrite the dataset yaml with paths absolute to its real location.

    Returns (yaml_path, info) where info has names/nc and which splits exist.
    Roboflow layout assumed: train/ valid/ test/ each with images/ + labels/.
    """
    import yaml
    src = find_data_yaml(dataset)
    base = src.parent
    orig = yaml.safe_load(src.read_text()) or {}

    resolved = {"path": str(base.resolve()), "nc": orig.get("nc"),
                "names": orig.get("names")}
    present = []
    for key, sub in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (base / sub / "images").is_dir():
            resolved[key] = f"{sub}/images"
            present.append(key)

    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "_val_data.yaml"
    yaml_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return yaml_path, {"names": resolved["names"], "nc": resolved["nc"],
                       "splits_present": present, "base": base}


def metrics_from_results_dict(rd: dict) -> dict:
    """Flatten ultralytics results_dict to clean, rounded metric columns."""
    out = {}
    for k, col in METRIC_RENAME.items():
        if k in rd and rd[k] is not None:
            out[col] = round(float(rd[k]), 5)
    return out


def count_split_images(base: Path, split: str) -> int:
    d = base / SPLIT_DIR.get(split, split) / "images"
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))


# --------------------------------------------------------------------------- #
# Reporting / provenance
# --------------------------------------------------------------------------- #
def render_report(row: dict, info: dict) -> str:
    lines = ["# Model validation report", "",
             f"- model: `{row['model']}`",
             f"- dataset split: `{row['split']}` "
             f"({row['n_images']} images, class(es) {info['names']})",
             f"- imgsz {row['imgsz']} | conf {row['conf']} | iou {row['iou']}",
             f"- generated: {row['timestamp']}", "",
             "| metric | boxes | masks |", "|---|---|---|"]
    pairs = [("precision", "box_precision", "mask_precision"),
             ("recall", "box_recall", "mask_recall"),
             ("mAP@50", "box_map50", "mask_map50"),
             ("mAP@50-95", "box_map50_95", "mask_map50_95")]
    for label, bk, mk in pairs:
        lines.append(f"| {label} | {row.get(bk, '—')} | {row.get(mk, '—')} |")
    lines += ["", f"- fitness: {row.get('fitness', '—')}", "",
              "_Validation against full-tray training-domain images; the "
              "pipeline applies this model to Petri crops of the same frames._",
              ""]
    return "\n".join(lines)


def write_provenance(out: Path, args, row: dict) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_validate"),
        "timestamp": row["timestamp"],
        "script": "scripts/validate_models.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.model), str(args.dataset)],
        "output_path": str(out),
        "model_sha256": cp.sha256_file(args.model),
        "parameters": {"split": args.split, "imgsz": args.imgsz,
                       "conf": args.conf, "iou": args.iou},
        "metrics": {k: row[k] for k in METRIC_RENAME.values() if k in row},
    }
    with open(out / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path, help="YOLO weights .pt")
    p.add_argument("--dataset", required=True, type=Path,
                   help="dataset dir (with data.yaml) or a yaml path")
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--out", type=Path, default=Path("outputs/model_validation"))
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.001,
                   help="low conf for mAP integration (standard for val)")
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", type=str, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.is_file():
        print(f"Missing model weights: {args.model}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        yaml_path, info = resolve_dataset_yaml(args.dataset, args.out)
    except FileNotFoundError as exc:
        print(f"Cannot resolve dataset: {exc}", file=sys.stderr)
        return 1
    if args.split not in info["splits_present"]:
        print(f"Split '{args.split}' not present (have "
              f"{info['splits_present']})", file=sys.stderr)
        return 1

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    print(f"Validating {args.model.name} on '{args.split}' split "
          f"({count_split_images(info['base'], args.split)} images)...")
    results = model.val(data=str(yaml_path), split=args.split,
                        imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                        device=args.device, project=str(args.out),
                        name="ultralytics", exist_ok=True, plots=True,
                        verbose=False)

    row = {"model": str(args.model), "split": args.split,
           "n_images": count_split_images(info["base"], args.split),
           "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
           "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
           **metrics_from_results_dict(getattr(results, "results_dict", {}))}

    with open(args.out / "validation_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    (args.out / "validation_metrics.json").write_text(json.dumps(row, indent=2))
    (args.out / "validation_report.md").write_text(render_report(row, info))
    write_provenance(args.out, args, row)

    print(f"box mAP@50={row.get('box_map50', '—')} "
          f"mAP@50-95={row.get('box_map50_95', '—')} | "
          f"mask mAP@50={row.get('mask_map50', '—')} "
          f"mAP@50-95={row.get('mask_map50_95', '—')}")
    print(f"Report + metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
