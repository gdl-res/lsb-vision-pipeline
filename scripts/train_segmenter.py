#!/usr/bin/env python3
"""Train one alternative ultralytics segmenter under the fixed protocol.

Part of the segmenter-swap tooling (config/segmenters.yaml). Every model is trained on the same
dataset/split with the production YOLO11s hyper-parameters (imgsz, epochs,
patience, batch, seed, deterministic) read from config/segmenters.yaml
`training:`; only the COCO-pretrained starting checkpoint differs.

Resumable: ultralytics writes last.pt every epoch. If a run directory holds a
last.pt but no done-marker, this script resumes it (`resume=True`) instead of
starting over, so an interrupted run loses at most one epoch. A done-marker
(`_done.json`) is written only after best.pt has been copied to its final
location, so a crash between the two re-does the copy, never the training.

    python scripts/train_segmenter.py --segmenter yolo11s_crop
    python scripts/train_segmenter.py --segmenter yolo11s_crop --epochs 1 --run-name smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crop_petris_yolo_seg as cp  # noqa: E402
import validate_models as vm  # noqa: E402

TRAIN_KEYS = ("imgsz", "epochs", "patience", "batch", "seed", "deterministic",
              "workers", "cache")


def dataset_fingerprint(base: Path) -> str:
    """SHA-256 over every split's sorted (label filename, label content).

    Identifies the exact annotation set a model was trained on; image bytes are
    left out (1.7 GB) since Roboflow names embed a per-image content hash.
    """
    h = hashlib.sha256()
    for split in ("train", "valid", "test"):
        ldir = base / split / "labels"
        if not ldir.is_dir():
            continue
        for t in sorted(ldir.glob("*.txt")):
            h.update(f"{split}/{t.name}\n".encode())
            h.update(t.read_bytes())
    return h.hexdigest()


def run_state(run_dir: Path) -> str:
    """'done' | 'resume' | 'fresh' for a training run directory."""
    if (run_dir / "_done.json").is_file():
        return "done"
    if (run_dir / "weights" / "last.pt").is_file():
        return "resume"
    return "fresh"


def train_kwargs(tcfg: dict, overrides: dict, data_yaml: Path, project: Path,
                 name: str, device) -> dict:
    kw = {k: tcfg[k] for k in TRAIN_KEYS if k in tcfg}
    kw.update({k: v for k, v in overrides.items() if v is not None})
    kw.update(data=str(data_yaml), project=str(project.resolve()), name=name,
              exist_ok=True, device=device, plots=True, verbose=True)
    return kw


def resolve_training_yaml(cfg: dict, spec: dict, name: str,
                          project: Path) -> Path:
    """Dataset yaml for this segmenter. The shared full-frame dataset keeps
    the original `<project>/_val_data.yaml` (resumed runs point at it); a
    segmenter with its own `dataset:` gets a private directory so the two can
    never overwrite each other."""
    if "dataset" not in spec:
        yaml_path, _ = vm.resolve_dataset_yaml(Path(cfg["paths"]["dataset"]),
                                               project)
        return yaml_path
    yaml_path, _ = vm.resolve_dataset_yaml(Path(spec["dataset"]),
                                           project / f"_data_{name}")
    # process_dataset_splits.py writes `names` but no `nc`; ultralytics
    # requires them to agree, so derive nc from names.
    import yaml
    d = yaml.safe_load(yaml_path.read_text())
    if d.get("nc") is None and d.get("names") is not None:
        d["nc"] = len(d["names"])
        yaml_path.write_text(yaml.safe_dump(d, sort_keys=False))
    return yaml_path


def append_provenance(out: Path, record: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "processing_provenance.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    import yaml
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("config/segmenters.yaml"))
    p.add_argument("--segmenter", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--run-name", default=None,
                   help="run dir name (default: the segmenter name); a "
                        "non-default name never installs final weights")
    p.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    p.add_argument("--batch", type=int, default=None, help="OOM fallback")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    spec = cfg["segmenters"][args.segmenter]
    if spec["backend"] != "ultralytics" or not spec.get("trained_here"):
        print(f"{args.segmenter} is not a trainable ultralytics segmenter",
              file=sys.stderr)
        return 2
    out = Path(cfg["paths"]["out"])
    project = out / "train"
    name = args.run_name or args.segmenter
    run_dir = project / name
    final = Path(spec["weights"])
    is_production_run = args.run_name is None

    state = run_state(run_dir)
    print(f"[train_segmenter] {args.segmenter} run={run_dir} state={state}")
    if state == "done":
        return 0

    from ultralytics import YOLO
    if state == "resume":
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        try:
            model.train(resume=True, device=args.device)
        except AssertionError as e:
            # last.pt of a run that already finished (crash after training, before
            # the done-marker): nothing to resume, fall through to installation
            if "nothing to resume" not in str(e):
                raise
            print(f"[train_segmenter] {e}")
    else:
        data_yaml = resolve_training_yaml(cfg, spec, args.segmenter, project)
        kw = train_kwargs(cfg["training"],
                          {"epochs": args.epochs, "batch": args.batch},
                          data_yaml, project, name, args.device)
        model = YOLO(str(spec["pretrained"]))
        model.train(**kw)

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[train_segmenter] no best.pt in {run_dir}", file=sys.stderr)
        return 1
    installed = None
    if is_production_run:
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final)
        installed = str(final)

    import csv
    results = run_dir / "results.csv"
    last_row = {}
    if results.is_file():
        rows = list(csv.DictReader(results.open()))
        if rows:
            last_row = {k.strip(): v for k, v in rows[-1].items()}
    record = {
        "run_id": datetime.now().strftime(f"%Y-%m-%dT%H-%M-%S_train_{name}"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/train_segmenter.py",
        "git_commit": cp.git_commit(),
        "command": " ".join([sys.executable] + sys.argv),
        "segmenter": args.segmenter,
        "pretrained": str(spec["pretrained"]),
        "pretrained_sha256": cp.sha256_file(Path(spec["pretrained"])),
        "dataset": spec.get("dataset", cfg["paths"]["dataset"]),
        "dataset_label_sha256": dataset_fingerprint(
            Path(spec.get("dataset", cfg["paths"]["dataset"]))),
        "training": {**cfg["training"],
                     **{k: v for k, v in {"epochs": args.epochs,
                                          "batch": args.batch}.items()
                        if v is not None}},
        "run_dir": str(run_dir),
        "best_pt": str(best),
        "installed_weights": installed,
        "weights_sha256": cp.sha256_file(best),
        "epochs_completed": last_row.get("epoch"),
        "train_time_s": last_row.get("time"),
        "final_val_mask_map50": last_row.get("metrics/mAP50(M)"),
        "final_val_mask_map50_95": last_row.get("metrics/mAP50-95(M)"),
    }
    append_provenance(out, record)
    (run_dir / "_done.json").write_text(json.dumps(record, indent=2))
    print(f"[train_segmenter] done: {installed or best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
