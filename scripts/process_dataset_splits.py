#!/usr/bin/env python3
"""Run the Petri cropper across all splits of a YOLO dataset.

Implements SPECIFICATION.md 18.2. Discovers the train/valid/test splits of a
Roboflow-style YOLO dataset, runs `crop_petris_yolo_seg` on each, and writes a
`data.yaml` for the resulting Petri-crop dataset so the larva segmenter can be
trained on it (SPECIFICATION.md 24).

Input layout (any subset of splits may be present):
    <dataset>/
        data.yaml
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels

Output layout:
    <out>/
        data.yaml
        train/{images,labels,crop_metadata.jsonl,...}
        valid/...
        test/...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import crop_petris_yolo_seg as cp

SPLITS = ("train", "valid", "test")


def discover_splits(dataset: Path) -> list[str]:
    """Return splits that have both an images/ and labels/ directory."""
    found = []
    for split in SPLITS:
        if (dataset / split / "images").is_dir() and \
           (dataset / split / "labels").is_dir():
            found.append(split)
    return found


def build_crop_argv(args, split: str) -> list[str]:
    """Construct the crop_petris_yolo_seg argv for one split."""
    src = args.dataset / split
    argv = [
        "--images", str(src / "images"),
        "--labels", str(src / "labels"),
        "--out", str(args.out / split),
        "--petri-model", str(args.petri_model),
        "--rows", str(args.rows),
        "--cols", str(args.cols),
        "--margin", str(args.margin),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--min-label-area-ratio", str(args.min_label_area_ratio),
        "--distortion-aspect-tol", str(args.distortion_aspect_tol),
        "--on-failure", args.on_failure,
    ]
    if args.device is not None:
        argv += ["--device", args.device]
    if args.max_images is not None:
        argv += ["--max-images", str(args.max_images)]
    if args.debug:
        argv += ["--debug"]
    return argv


def load_class_names(dataset: Path) -> dict[int, str]:
    """Read class names from the source data.yaml; fall back to {0: larva}."""
    src_yaml = dataset / "data.yaml"
    default = {0: "larva"}
    if not src_yaml.is_file():
        return default
    try:
        import yaml
        data = yaml.safe_load(src_yaml.read_text()) or {}
    except Exception:
        return default
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return default


def write_crops_data_yaml(out: Path, splits: list[str],
                          names: dict[int, str]) -> Path:
    """Write a data.yaml for the Petri-crop dataset (paths relative to `out`)."""
    import yaml
    data: dict = {"path": str(out.resolve())}
    if "train" in splits:
        data["train"] = "train/images"
    if "valid" in splits:
        data["val"] = "valid/images"
    if "test" in splits:
        data["test"] = "test/images"
    data["names"] = {int(k): v for k, v in sorted(names.items())}
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return yaml_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--petri-model", required=True, type=Path)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--min-label-area-ratio", type=float, default=0.0)
    p.add_argument("--distortion-aspect-tol", type=float, default=0.15)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--on-failure", choices=["skip", "keep-best", "error"],
                   default="skip")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dataset.is_dir():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    splits = discover_splits(args.dataset)
    if not splits:
        print(f"No train/valid/test splits with images+labels under "
              f"{args.dataset}", file=sys.stderr)
        return 1

    print(f"Discovered splits: {', '.join(splits)}")
    args.out.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n=== Processing split: {split} ===")
        rc = cp.main(build_crop_argv(args, split))
        if rc != 0:
            print(f"Split '{split}' failed (rc={rc})", file=sys.stderr)
            return rc

    names = load_class_names(args.dataset)
    yaml_path = write_crops_data_yaml(args.out, splits, names)
    print(f"\nAll splits done. Crop dataset config: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
