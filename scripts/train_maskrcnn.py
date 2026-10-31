#!/usr/bin/env python3
"""Fine-tune torchvision Mask R-CNN (R50-FPN v2) on the larva dataset.

Part of the segmenter-swap tooling (config/segmenters.yaml): the non-YOLO segmenter. Same
dataset and split as every YOLO model; schedule and augmentation in
config/segmenters.yaml `maskrcnn_training:`.

Scale jitter is a random square crop (crop_frac x the image's long side,
half of them centered on a larva) resized to imgsz. That spans YOLO's
full-frame training scale and the pipeline's Petri-crop inference scale at a
fixed 640x640 memory cost.

Resumable and reproducible across interruptions:
  * last.pt (model, optimizer, scheduler, AMP scaler, epoch, best AP, history)
    is written atomically (tmp + rename) at the end of every epoch, and the run
    resumes from it by default -- at most one epoch is lost.
  * Data order and augmentation are a pure function of (seed, epoch, index),
    so a resumed run sees the same batches an uninterrupted one would have.
Early stopping on val mask AP50-95 (COCO, pycocotools), validated on full
frames at long side `val_imgsz` (YOLO's val protocol).

    python scripts/train_maskrcnn.py
    python scripts/train_maskrcnn.py --max-epochs 1 --limit-train 40 --run-name smoke
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crop_petris_yolo_seg as cp  # noqa: E402
import seg_coco  # noqa: E402
import segmenters as sg  # noqa: E402
import train_segmenter as ts  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_split(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    idir = dataset / split / "images"
    return [(p, dataset / split / "labels" / (p.stem + ".txt"))
            for p in sorted(idir.iterdir()) if p.suffix.lower() in IMG_EXT]


def preresize(pairs, out_dir: Path, long_side: int) -> list[dict]:
    """Downscale every image once (idempotent); labels are normalized, so
    polygons are re-read at the new size. Returns record dicts."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for img, lab in pairs:
        dst = out_dir / (img.stem + ".jpg")
        if not dst.is_file():
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            s = long_side / max(h, w)
            if s < 1.0:
                im = cv2.resize(im, (round(w * s), round(h * s)),
                                interpolation=cv2.INTER_AREA)
            tmp = dst.with_suffix(".tmp.jpg")
            cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            tmp.replace(dst)
        recs.append({"image": dst, "label": lab, "source": img})
    return recs


def rasterize(polys: list[np.ndarray], h: int, w: int) -> np.ndarray:
    import cv2
    masks = np.zeros((len(polys), h, w), dtype=np.uint8)
    for i, p in enumerate(polys):
        cv2.fillPoly(masks[i], [np.round(p).astype(np.int32)], 1)
    return masks


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    out = np.zeros((len(masks), 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        out[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return out


def sample_crop(rng: np.random.Generator, h: int, w: int,
                polys: list[np.ndarray], frac: tuple[float, float],
                centered_p: float) -> tuple[int, int, int]:
    """Square crop (x0, y0, side) inside the image (padded when side > dim)."""
    side = int(round(rng.uniform(*frac) * max(h, w)))
    side = max(32, side)
    if polys and rng.random() < centered_p:
        c = polys[int(rng.integers(len(polys)))].mean(axis=0)
        cx = c[0] + rng.uniform(-0.35, 0.35) * side
        cy = c[1] + rng.uniform(-0.35, 0.35) * side
    else:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    x0 = int(round(np.clip(cx - side / 2, min(0, w - side), max(0, w - side))))
    y0 = int(round(np.clip(cy - side / 2, min(0, h - side), max(0, h - side))))
    return x0, y0, side


def crop_pad(arr: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """Crop [y0:y0+side, x0:x0+side] of the last two/three dims, zero-padded."""
    h, w = arr.shape[-3:-1] if arr.ndim == 3 and arr.shape[-1] == 3 \
        else arr.shape[-2:]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    if arr.ndim == 3 and arr.shape[-1] == 3:          # image HWC
        out = np.zeros((side, side, 3), dtype=arr.dtype)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    else:                                             # masks NHW
        out = np.zeros((arr.shape[0], side, side), dtype=arr.dtype)
        out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[:, sy0:sy1, sx0:sx1]
    return out


def color_jitter(img: np.ndarray, rng: np.random.Generator, cj: dict
                 ) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cj["hue"], cj["hue"]) * 180) % 180
    hsv[..., 1] *= 1 + rng.uniform(-cj["saturation"], cj["saturation"])
    hsv[..., 2] *= 1 + rng.uniform(-cj["brightness"], cj["brightness"])
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                       cv2.COLOR_HSV2BGR).astype(np.float32)
    c = 1 + rng.uniform(-cj["contrast"], cj["contrast"])
    img = (img - img.mean()) * c + img.mean()
    return np.clip(img, 0, 255).astype(np.uint8)


def make_train_sample(rec: dict, mcfg: dict, seed: int, epoch: int,
                      idx: int):
    """Deterministic in (seed, epoch, idx): resume-safe augmentation."""
    import cv2
    rng = np.random.default_rng([seed, epoch, idx])
    img = cv2.imread(str(rec["image"]))
    h, w = img.shape[:2]
    polys = seg_coco.read_yolo_polygons(rec["label"], w, h)
    full = rasterize(polys, h, w)
    full_area = full.reshape(len(full), -1).sum(1) if len(full) else []
    x0, y0, side = sample_crop(rng, h, w, polys, tuple(mcfg["crop_frac"]),
                               mcfg["instance_centered_fraction"])
    img = crop_pad(img, x0, y0, side)
    masks = crop_pad(full, x0, y0, side) if len(full) else \
        np.zeros((0, side, side), np.uint8)
    S = int(mcfg["imgsz"])
    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
    if len(masks):
        vis = masks.reshape(len(masks), -1).sum(1) / np.maximum(full_area, 1)
        masks = masks[vis >= mcfg["min_instance_visible"]]
    if len(masks):
        masks = np.stack([cv2.resize(m, (S, S),
                                     interpolation=cv2.INTER_NEAREST)
                          for m in masks])
        masks = masks[masks.reshape(len(masks), -1).sum(1) >= 16]
    if rng.random() < mcfg["hflip"]:
        img = img[:, ::-1]
        masks = masks[:, :, ::-1]
    img = color_jitter(np.ascontiguousarray(img), rng, mcfg["color_jitter"])
    return img, np.ascontiguousarray(masks)


def to_tensors(img: np.ndarray, masks: np.ndarray):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(
        2, 0, 1).float().div_(255.0)
    boxes = boxes_from_masks(masks) if len(masks) else np.zeros((0, 4),
                                                                np.float32)
    target = {"boxes": torch.from_numpy(boxes),
              "labels": torch.ones(len(masks), dtype=torch.int64),
              "masks": torch.from_numpy(masks.astype(np.uint8))}
    return t, target


class TrainSet:
    def __init__(self, recs, mcfg, seed):
        self.recs, self.mcfg, self.seed, self.epoch = recs, mcfg, seed, 0

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        img, masks = make_train_sample(self.recs[i], self.mcfg, self.seed,
                                       self.epoch, i)
        return to_tensors(img, masks)


def collate(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# Validation (COCO mask AP on full frames)
# --------------------------------------------------------------------------- #
def build_val(recs) -> tuple[dict, list]:
    import cv2
    images, ims = [], []
    for i, r in enumerate(recs, 1):
        im = cv2.imread(str(r["image"]))
        h, w = im.shape[:2]
        images.append({"id": i, "file_name": r["image"].name, "width": w,
                       "height": h,
                       "polygons": seg_coco.read_yolo_polygons(r["label"],
                                                               w, h)})
        ims.append(im)
    return seg_coco.build_coco_gt(images), ims


def evaluate(model, gt: dict, ims: list, imgsz: int, device,
             conf_floor: float = 0.001) -> dict:
    import torch
    model.eval()
    sg.set_maskrcnn_inference_size(model, imgsz)
    old_thr = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = conf_floor
    preds = []
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for im_meta, im in zip(gt["images"], ims):
            t = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1])).to(
                device).permute(2, 0, 1).float().div_(255.0)
            out = model([t])[0]
            dets = sg.maskrcnn_output_to_detections(out, conf_floor)
            preds += seg_coco.detections_to_coco(
                im_meta["id"], dets, im_meta["height"], im_meta["width"])
    model.roi_heads.score_thresh = old_thr
    return seg_coco.score(gt, preds)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def atomic_save(obj, path: Path) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def lr_lambda_factory(warmup_iters: int, total_iters: int):
    def f(it: int) -> float:
        if it < warmup_iters:
            return 0.001 + (1 - 0.001) * it / max(1, warmup_iters)
        p = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    import torch
    import yaml
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=Path("config/segmenters.yaml"))
    p.add_argument("--device", default="0")
    p.add_argument("--run-name", default=None,
                   help="non-default names never install final weights")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="OOM fallback")
    p.add_argument("--limit-train", type=int, default=None, help="smoke only")
    p.add_argument("--limit-val", type=int, default=None, help="smoke only")
    p.add_argument("--crop-frac", type=float, nargs=2, default=None,
                   help="override scale jitter (retry rule)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    mcfg = dict(cfg["maskrcnn_training"])
    if args.batch:
        mcfg["batch"] = args.batch
    if args.max_epochs:
        mcfg["max_epochs"] = args.max_epochs
    if args.crop_frac:
        mcfg["crop_frac"] = list(args.crop_frac)
    spec = cfg["segmenters"]["maskrcnn"]
    out = Path(cfg["paths"]["out"])
    name = args.run_name or "maskrcnn"
    run_dir = out / "train" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "_done.json").is_file():
        print(f"[train_maskrcnn] {run_dir} already done")
        return 0
    seed = int(mcfg["seed"])
    torch.manual_seed(seed)
    device = torch.device(sg.torch_device(args.device))

    dataset = Path(cfg["paths"]["dataset"])
    data_dir = out / "maskrcnn_data"
    print("[train_maskrcnn] pre-resizing images (once)", flush=True)
    train = preresize(list_split(dataset, "train"), data_dir / "train",
                      int(mcfg["preresize_long_side"]))
    val = preresize(list_split(dataset, "valid"), data_dir / "valid",
                    int(mcfg["preresize_long_side"]))
    if args.limit_train:
        train = train[:args.limit_train]
    if args.limit_val:
        val = val[:args.limit_val]
    gt_val, val_ims = build_val(val)

    model = sg.build_maskrcnn(num_classes=2, coco_init=True)
    S = int(mcfg["imgsz"])
    model.to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=float(mcfg["lr"]),
                          momentum=float(mcfg["momentum"]),
                          weight_decay=float(mcfg["weight_decay"]))
    bs = int(mcfg["batch"])
    iters_per_epoch = math.ceil(len(train) / bs)
    total = iters_per_epoch * int(mcfg["max_epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(int(mcfg["warmup_epochs"]) * iters_per_epoch,
                               total))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mcfg["amp"])
                                  and device.type == "cuda")

    start_epoch, best, best_epoch, history = 0, -1.0, -1, []
    last = run_dir / "last.pt"
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, best_epoch = ck["epoch"] + 1, ck["best"], \
            ck["best_epoch"]
        history = ck["history"]
        print(f"[train_maskrcnn] resuming at epoch {start_epoch} "
              f"(best AP {best:.4f} @ {best_epoch})", flush=True)

    ds = TrainSet(train, mcfg, seed)
    t_start = time.time()
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, int(mcfg["max_epochs"])):
        if epoch - best_epoch > int(mcfg["patience"]) and best_epoch >= 0:
            stop_reason = f"early_stop (no improvement since {best_epoch})"
            break
        ds.epoch = epoch
        model.train()
        sg.set_maskrcnn_inference_size(model, S)   # inputs already SxS
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, generator=g,
            num_workers=int(mcfg["workers"]), collate_fn=collate,
            persistent_workers=False)
        t0, run_loss, n = time.time(), 0.0, 0
        for imgs, targets in loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled()):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if not math.isfinite(loss.item()):
                print("[train_maskrcnn] non-finite loss", file=sys.stderr)
                return 1
            run_loss += loss.item()
            n += 1
        metrics = evaluate(model, gt_val, val_ims, int(mcfg["val_imgsz"]),
                           device)
        ap = metrics["mask_map50_95"]
        row = {"epoch": epoch, "train_loss": round(run_loss / max(1, n), 5),
               "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t0, 1), **metrics}
        history.append(row)
        print(f"[train_maskrcnn] epoch {epoch} loss {row['train_loss']} "
              f"val mask AP {ap:.4f} AP50 {metrics['mask_map50']:.4f} "
              f"({row['epoch_s']}s)", flush=True)
        if ap > best:
            best, best_epoch = ap, epoch
            atomic_save({"model": model.state_dict(), "epoch": epoch,
                         "val": metrics}, run_dir / "best.pt")
        atomic_save({"model": model.state_dict(),
                     "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch,
                     "best": best, "best_epoch": best_epoch,
                     "history": history}, last)
        with open(run_dir / "results.csv", "w") as fh:
            import csv
            w = csv.DictWriter(fh, fieldnames=list(history[-1].keys()))
            w.writeheader()
            w.writerows(history)

    best_pt = run_dir / "best.pt"
    installed = None
    if args.run_name is None:
        Path(spec["weights"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, spec["weights"])
        installed = spec["weights"]
    record = {
        "run_id": datetime.now().strftime(f"%Y-%m-%dT%H-%M-%S_train_{name}"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/train_maskrcnn.py",
        "git_commit": cp.git_commit(),
        "command": " ".join([sys.executable] + sys.argv),
        "segmenter": "maskrcnn",
        "architecture": "torchvision maskrcnn_resnet50_fpn_v2 (COCO_V1 init)",
        "dataset": cfg["paths"]["dataset"],
        "dataset_label_sha256": ts.dataset_fingerprint(dataset),
        "training": mcfg, "stop_reason": stop_reason,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_val_mask_map50_95": best,
        "best_val_mask_map50": next((h["mask_map50"] for h in history
                                     if h["epoch"] == best_epoch), None),
        "train_time_s_this_session": round(time.time() - t_start, 1),
        "run_dir": str(run_dir), "installed_weights": installed,
        "weights_sha256": cp.sha256_file(best_pt),
    }
    ts.append_provenance(out, record)
    (run_dir / "_done.json").write_text(json.dumps(record, indent=2))
    print(f"[train_maskrcnn] done ({stop_reason}); best epoch {best_epoch} "
          f"mask AP {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
