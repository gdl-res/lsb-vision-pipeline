#!/usr/bin/env python3
"""Draft experiment_metadata.csv from video filenames (SPECIFICATION.md 14).

The pipeline must NEVER silently infer treatment from filenames. This tool is
an EXPLICIT, opt-in helper: it parses the encoded condition tokens, expands each
video into one row per dish (treatment is tray-level), and writes a
`*.draft.csv` for a human to review and rename. Every row carries
`parse_confidence` / `parse_warnings`; nothing is auto-applied.

Two naming conventions are supported (auto-detected per file):

  A  20260514_133535000_lsb_iphone17pro_az_3x2petri_video_nostress_nat9
     date_time_project_camera_op_layout_video_<lighting>     (unstressed)
     -- here warm/cold/nat<i> are LIGHTING, not a stressor.
  B  20260623_38C_2h_lucewarm   /   20260623_nostress_lucecold
     date_<38C|nostress>_<duration>_luce<light>
     -- here 38C is heat stress for the given duration; luce* is lighting.

experiment_id groups treated + control videos of one campaign (month + profile)
so the divergence step (SPECIFICATION.md 26.4) has a control reference: a heat tray
has no internal control dish, so the reference is the nostress *videos* of the
same campaign.

Usage:
    python scripts/parse_filenames.py --videos data4evaluation/videos_to_evaluate \
        --out metadata/experiment_metadata.draft.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import crop_petris_yolo_seg as cp
from generate_fair_metadata import EXPERIMENT_METADATA_COLUMNS

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
DUR_S = {"1h": 3600, "2h": 7200, "3h": 10800}
# audit columns appended to the schema in the draft (drop when finalizing)
AUDIT_COLUMNS = ["parse_convention", "parse_confidence", "parse_light_intensity",
                 "parse_warnings"]


# --------------------------------------------------------------------------- #
# Pure parsing (testable, no IO)
# --------------------------------------------------------------------------- #
def parse_stem(stem: str, profile: str = "auto") -> dict:
    """Parse one filename stem into condition fields + warnings/confidence.

    `experiment_id` is NOT set here -- it is assigned campaign-wide by
    assign_experiment_ids once all files are known.
    """
    f = {
        "video_id": stem, "date": "", "recording_start_time": "",
        "phone_model": "", "camera_id": "", "rows": "", "cols": "",
        "treatment": "unknown", "stressor": "", "stressor_intensity": "",
        "temperature_c": "", "stressor_duration_s": "",
        "light_condition": "unknown", "light_intensity": "", "batch_id": "",
        "convention": "", "confidence": "high", "warnings": [], "notes": "",
    }
    s = stem.lower().replace("°", "")
    toks = re.split(r"[_]+", s)

    m = re.match(r"(\d{4})(\d{2})(\d{2})$", toks[0])
    if m:
        f["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        f["warnings"].append("no leading YYYYMMDD date")

    if len(toks) > 1 and re.fullmatch(r"\d{9}", toks[1]):
        t = toks[1]
        f["recording_start_time"] = f"{t[0:2]}:{t[2:4]}:{t[4:6]}"

    cam = re.search(r"iphone(\d+)\s*pro", s)
    if cam:
        f["phone_model"] = f"iPhone {cam.group(1)} Pro"
        f["camera_id"] = f"iphone{cam.group(1)}pro"

    lay = re.search(r"(\d)x(\d)petri", s)
    if lay:
        f["rows"], f["cols"] = lay.group(1), lay.group(2)

    if re.search(r"_(\d)$", stem):
        f["batch_id"] = re.search(r"_(\d)$", stem).group(1)

    has_temp = re.search(r"(\d{2})_?\s*c(?![a-z])", s)   # 38c / 38_c / 38°c
    is_nostress = "nostress" in toks or "control" in toks
    dur = next((d for d in DUR_S if d in toks), None)
    detected = "B" if ("luce" in s or has_temp) else "A"
    f["convention"] = {"a": "A", "b": "B"}.get(profile, detected)

    # --- light_condition (+ intensity) --------------------------------- #
    lm = re.search(r"luce(nat|natural|warm|cold)", s)
    if lm:
        f["light_condition"] = {"nat": "natural", "natural": "natural",
                                "warm": "warm", "cold": "cold"}[lm.group(1)]
    elif "warm" in toks:
        f["light_condition"] = "warm"
    elif "cold" in toks:
        f["light_condition"] = "cold"
    else:
        am = re.search(r"nat(\d+)", s)            # convention A "nat9"
        if am:
            f["light_condition"], f["light_intensity"] = "natural", am.group(1)

    # --- treatment / stressor ------------------------------------------ #
    if has_temp:                                  # convention B heat stress
        f["treatment"], f["stressor"] = "heat", "heat"
        f["temperature_c"] = has_temp.group(1)
        f["stressor_intensity"] = f"{has_temp.group(1)}C"
        if dur:
            f["stressor_duration_s"] = DUR_S[dur]
        else:
            f["confidence"] = "medium"
            f["warnings"].append("heat stress but no 1h/2h/3h duration token")
    elif is_nostress:                             # explicit control
        f["treatment"], f["stressor"] = "control", "none"
    elif f["convention"] == "A":                  # confirmed unstressed lighting test
        f["treatment"], f["stressor"] = "control", "none"
        f["notes"] = "convention-A lighting-variation session (unstressed)"
    else:
        f["treatment"] = "unknown"
        f["confidence"] = "low"
        f["warnings"].append("no recognizable treatment/stress token")

    return f


def assign_experiment_ids(parsed: list[dict], dataset_id: str) -> list[dict]:
    """Assign a campaign-level experiment_id (month + protocol) so each group
    pools its controls + treated videos for divergence (SPECIFICATION.md 26.4)."""
    groups: dict[tuple, list] = defaultdict(list)
    for p in parsed:
        ym = (p["date"].replace("-", "") or "NA")[:6]
        groups[(ym, p["convention"])].append(p)
    for (ym, conv), members in groups.items():
        temps = [m["temperature_c"] for m in members
                 if m["treatment"] == "heat" and m["temperature_c"]]
        if temps:
            label = f"heat{max(set(temps), key=temps.count)}c"
        elif conv == "A":
            label = "lighting"
        else:
            label = "control"
        eid = f"exp_{ym}_{label}"
        for m in members:
            m["experiment_id"] = eid
    return parsed


def control_reference_warnings(parsed: list[dict]) -> list[str]:
    """Flag experiment_id groups that have treated videos but no control to
    serve as the divergence reference (SPECIFICATION.md 26.4)."""
    groups: dict[str, list] = defaultdict(list)
    for p in parsed:
        groups[p.get("experiment_id", "")].append(p)
    out = []
    for eid, g in sorted(groups.items()):
        treated = [p for p in g if p["treatment"] not in ("control", "unknown")]
        ctrl = [p for p in g if p["treatment"] == "control"]
        if treated and not ctrl:
            out.append(f"experiment_id {eid}: {len(treated)} treated video(s) "
                       "but NO control reference -- divergence cannot be scored")
    return out


def to_metadata_rows(parsed: dict, dataset_id: str, n_dishes: int) -> list[dict]:
    """Expand one parsed video into n_dishes schema rows (treatment is
    tray-level, so every dish shares it). Human-only fields stay blank."""
    base = {c: "" for c in EXPERIMENT_METADATA_COLUMNS}
    base.update({
        "dataset_id": dataset_id,
        "experiment_id": parsed.get("experiment_id", ""),
        "video_id": parsed["video_id"],
        "batch_id": parsed["batch_id"],
        "date": parsed["date"],
        "camera_id": parsed["camera_id"],
        "phone_model": parsed["phone_model"],
        "treatment": parsed["treatment"],
        "stressor": parsed["stressor"],
        "stressor_intensity": parsed["stressor_intensity"],
        "stressor_duration_s": parsed["stressor_duration_s"],
        "temperature_c": parsed["temperature_c"],
        "light_condition": parsed["light_condition"],
        "recording_start_time": parsed["recording_start_time"],
        "notes": parsed["notes"],
    })
    audit = {
        "parse_convention": parsed["convention"],
        "parse_confidence": parsed["confidence"],
        "parse_light_intensity": parsed["light_intensity"],
        "parse_warnings": "; ".join(parsed["warnings"]),
    }
    rows = []
    for dish in range(1, n_dishes + 1):
        rows.append({**base, **audit, "dish_id": dish})
    return rows


# --------------------------------------------------------------------------- #
# IO / orchestration
# --------------------------------------------------------------------------- #
def discover_stems(videos: Path, image_sets: Optional[Path]) -> list[str]:
    stems: list[str] = []
    if videos.is_file():
        stems.append(videos.stem)
    elif videos.is_dir():
        stems += [p.stem for p in sorted(videos.iterdir())
                  if p.suffix.lower() in VIDEO_EXTS]
    if image_sets and image_sets.is_dir():
        stems += [d.name for d in sorted(image_sets.iterdir()) if d.is_dir()]
    # de-dup, stable order
    seen, out = set(), []
    for s in stems:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def write_provenance(out: Path, args, n_videos: int, n_rows: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_parse_filenames"),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "script": "scripts/parse_filenames.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.videos)] +
                       ([str(args.image_sets)] if args.image_sets else []),
        "output_path": str(args.out),
        "n_videos": n_videos, "n_rows": n_rows,
        "parameters": {"profile": args.profile, "dataset_id": args.dataset_id,
                       "rows": args.rows, "cols": args.cols},
    }
    prov = args.out.parent / "processing_provenance.jsonl"
    prov.parent.mkdir(parents=True, exist_ok=True)
    with open(prov, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, type=Path,
                   help="video file or directory of videos")
    p.add_argument("--image-sets", type=Path, default=None,
                   help="optional dir whose sub-folders are named by video stem")
    p.add_argument("--out", type=Path,
                   default=Path("experiment_metadata.draft.csv"),
                   help="draft CSV path (review then rename, never auto-apply)")
    p.add_argument("--profile", choices=["auto", "a", "b"], default="auto")
    p.add_argument("--dataset-id", default="wormstress_tmolitor_v1")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=2)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stems = discover_stems(args.videos, args.image_sets)
    if not stems:
        print(f"No videos found at {args.videos}", file=sys.stderr)
        return 1

    parsed = [parse_stem(s, args.profile) for s in stems]
    assign_experiment_ids(parsed, args.dataset_id)

    n_dishes = args.rows * args.cols
    rows = []
    for p in parsed:
        rows.extend(to_metadata_rows(p, args.dataset_id, n_dishes))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = EXPERIMENT_METADATA_COLUMNS + AUDIT_COLUMNS
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    write_provenance(args.out, args, len(parsed), len(rows))

    conf = {"high": 0, "medium": 0, "low": 0}
    for p in parsed:
        conf[p["confidence"]] += 1
    print(f"Parsed {len(parsed)} video(s) -> {len(rows)} dish rows "
          f"({n_dishes}/video) -> {args.out}")
    print(f"  confidence: {conf['high']} high, {conf['medium']} medium, "
          f"{conf['low']} low")
    ref_warn = control_reference_warnings(parsed)
    for w in ref_warn:
        print(f"  WARN {w}", file=sys.stderr)
    flagged = [p for p in parsed if p["confidence"] != "high" or p["warnings"]]
    if flagged:
        print(f"  {len(flagged)} video(s) need review (see parse_warnings "
              "column); medium/low rows should be hand-checked before this "
              "draft is renamed to experiment_metadata.csv.")
    print("  NOTE: this is a DRAFT. Review it, fill the blank human-only fields "
          "(fps, petri_diameter_mm, operator, ...), drop the parse_* audit "
          "columns, then rename to metadata/experiment_metadata.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
