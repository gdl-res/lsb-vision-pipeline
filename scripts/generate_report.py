#!/usr/bin/env python3
"""Human-readable per-run report for a FAIR dataset (SPECIFICATION.md 15.4, 23).

Reads an already-produced FAIR dataset directory (the --out of run_pipeline)
and renders a standalone `report.html` and a `report.md`:

    - a header (dataset id/title/version, video/dish counts);
    - a batch summary when metadata/batch_index.json is present;
    - one curated behavioral row per video x dish, with the QC verdict
      (EXCLUDE rows flagged in red);
    - an image gallery embedding each dish's heatmap and trajectory PNG;
    - the interpretation caveats required by SPECIFICATION.md 23 / 15.

It runs purely on saved tables/images -- no YOLO, no recompute -- so it is
cheap to regenerate after `run_pipeline --skip-tracking`. This is the
companion to the machine-readable FAIR outputs: the page a non-coder opens
to eyeball whether a run is sane.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import crop_petris_yolo_seg as cp

# Curated subset of features_per_video columns shown in the report table.
# (label, column, fmt) -- kept small on purpose; the full feature set lives in
# features/features_per_video.csv.
REPORT_FEATURES = [
    ("Distance (mm)", "total_distance_mm", "{:.1f}"),
    ("Mean speed (mm/s)", "mean_speed_mm_s", "{:.2f}"),
    ("Immobile frac", "immobility_fraction", "{:.2f}"),
    ("Bouts", "movement_bout_count", "{:.0f}"),
    ("Path eff.", "path_efficiency", "{:.2f}"),
    ("Tortuosity", "tortuosity", "{:.2f}"),
    ("Edge frac", "edge_fraction", "{:.2f}"),
    ("Mean r_norm", "mean_r_norm", "{:.2f}"),
    ("Rim occ. frac", "rim_occlusion_fraction", "{:.2f}"),
    ("Rim occ. events", "rim_occlusion_event_count", "{:.0f}"),
    ("Occ. entropy", "occupancy_entropy", "{:.2f}"),
    ("Mean area (px)", "mean_area_px", "{:.0f}"),
    ("Divergence (JS)", "js_divergence_from_control", "{:.3f}"),
]

# Interpretation caveats -- objective behavioral proxies, not welfare/suffering
# claims (SPECIFICATION.md 23, 15.3). Embedded in every report.
CAVEATS = [
    "These are objective behavioral proxies (altered locomotor activity, "
    "spatial occupancy, area dynamics), not measures of welfare, suffering, "
    "or consciousness. Interpretation requires experimental validation.",
    "Centroids track the larva bounding-box/mask centre, not body posture or "
    "head/tail orientation.",
    "Distances/speeds are in millimetres only when a Petri diameter was given "
    "(calibrated); otherwise normalized Petri-radius units or pixels apply.",
    "Track IDs can switch under ambiguous detections; filter by quality_score "
    "and treat dishes flagged for exclusion with caution.",
    "Morphology features (mask area, eccentricity, compactness, aspect ratio) "
    "are lower-confidence than trajectory/position features: the segmenter's "
    "mask boundaries are loose at high IoU (mask mAP@50-95 ~0.70), so treat "
    "absolute shape/area values as approximate and weight per-dish by "
    "morphology_reliability.",
    "Stress-associated deviation is a comparison against a control "
    "distribution; a single dish in isolation is not a stress measurement.",
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset(dataset: Path) -> dict:
    """Gather the tables/metadata a report needs (tolerant of absences)."""
    import pandas as pd

    feat_path = dataset / "features" / "features_per_video.csv"
    qc_path = dataset / "qc" / "qc_summary.csv"
    if not feat_path.is_file():
        raise FileNotFoundError(f"missing {feat_path}")
    feat = pd.read_csv(feat_path)
    qc = pd.read_csv(qc_path) if qc_path.is_file() else pd.DataFrame()

    desc_path = dataset / "dataset_description.json"
    desc = json.loads(desc_path.read_text()) if desc_path.is_file() else {}

    batch_path = dataset / "metadata" / "batch_index.json"
    batch = json.loads(batch_path.read_text()) if batch_path.is_file() else None

    return {"features": feat, "qc": qc, "description": desc, "batch": batch}


def qc_verdict(qc_row) -> tuple[str, str]:
    """(verdict, reason) for one qc_summary row; verdict in {OK, EXCLUDE}."""
    if qc_row is None:
        return "?", "no QC row"
    excl = bool(qc_row.get("exclude_recommended", False))
    reason = qc_row.get("exclude_reason", "")
    if isinstance(reason, float):  # NaN
        reason = ""
    return ("EXCLUDE", reason) if excl else ("OK", "")


def build_rows(data: dict) -> list[dict]:
    """One merged record per (video_id, dish_id), sorted for stable output."""
    import pandas as pd

    feat = data["features"]
    qc = data["qc"]
    qc_idx = {}
    if not qc.empty:
        for _, r in qc.iterrows():
            qc_idx[(r["video_id"], int(r["dish_id"]))] = r

    rows = []
    for _, fr in feat.sort_values(["video_id", "dish_id"]).iterrows():
        key = (fr["video_id"], int(fr["dish_id"]))
        qr = qc_idx.get(key)
        verdict, reason = qc_verdict(qr)
        feats = {}
        for label, col, fmt in REPORT_FEATURES:
            val = fr.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                feats[label] = "—"
            else:
                feats[label] = fmt.format(val)
        rows.append({
            "video_id": fr["video_id"],
            "dish_id": int(fr["dish_id"]),
            "treatment": _treatment(fr, qr),
            "quality_score": _num(fr.get("quality_score")),
            "valid_fraction": _num(fr.get("valid_fraction")),
            "track_switches": _num(fr.get("track_switch_count"), "{:.0f}"),
            "verdict": verdict,
            "reason": reason,
            "features": feats,
        })
    return rows


def _treatment(feat_row, qc_row) -> str:
    for src in (feat_row, qc_row):
        if src is not None:
            t = src.get("treatment")
            if isinstance(t, str) and t:
                return t
    return "—"


def _num(val, fmt: str = "{:.2f}") -> str:
    import pandas as pd
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return fmt.format(float(val))
    except (TypeError, ValueError):
        return str(val)


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #
def heatmap_pngs(dataset: Path, video_id: str, dish_id: int) -> dict:
    """Locate the heatmap and trajectory PNGs for one dish (may be absent)."""
    png_dir = dataset / "heatmaps" / "png"
    stem = f"{video_id}_dish{dish_id:02d}"
    out = {}
    for kind in ("heatmap", "trajectory"):
        p = png_dir / f"{stem}_{kind}.png"
        if p.is_file():
            out[kind] = p
    return out


def img_src(path: Path, dataset: Path, embed: bool) -> str:
    """data: URI (standalone HTML) or a path relative to the dataset root."""
    if embed:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    import os
    return os.path.relpath(path, dataset)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_markdown(data: dict, rows: list[dict]) -> str:
    desc = data["description"]
    title = desc.get("dataset_title") or desc.get("title") or "Behavioral report"
    lines = [f"# {title}", ""]
    lines.append(f"- dataset_id: `{desc.get('dataset_id', '—')}`")
    lines.append(f"- version: {desc.get('dataset_version', '—')}")
    lines.append(f"- generated: {_utcnow()}")
    n_vid = len({r['video_id'] for r in rows})
    lines.append(f"- videos: {n_vid} | dishes: {len(rows)}")
    excl = sum(r["verdict"] == "EXCLUDE" for r in rows)
    lines.append(f"- flagged for exclusion: {excl}/{len(rows)}")
    lines.append("")

    b = data["batch"]
    if b:
        lines += ["## Batch", "",
                  f"- done: {b.get('n_done', 0)} | failed: "
                  f"{b.get('n_failed', 0)} | skipped: {b.get('n_skipped', 0)}",
                  ""]
        failed = [r for r in b.get("runs", []) if r.get("status") == "failed"]
        for r in failed:
            lines.append(f"  - FAILED `{r.get('video')}`: {r.get('error')}")
        if failed:
            lines.append("")

    # behavior table
    feat_labels = [lbl for lbl, _, _ in REPORT_FEATURES]
    header = ["video", "dish", "treatment", "quality", "QC"] + feat_labels
    lines += ["## Behavior per dish", "",
              "| " + " | ".join(header) + " |",
              "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [str(r["video_id"]), str(r["dish_id"]), r["treatment"],
                 r["quality_score"],
                 r["verdict"] + (f" ({r['reason']})" if r["reason"] else "")]
        cells += [r["features"][lbl] for lbl in feat_labels]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## How to read these results", ""]
    for c in CAVEATS:
        lines.append(f"- {c}")
    lines.append("")
    return "\n".join(lines)


_CSS = """
body{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#1a1a1a;}
h1{margin-bottom:.2rem;} .meta{color:#555;margin-bottom:1rem;}
table{border-collapse:collapse;margin:1rem 0;font-size:.9rem;}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;}
th{background:#f0f0f0;} td.l,th.l{text-align:left;}
tr.exclude td{background:#fff0f0;} .excl{color:#b00020;font-weight:bold;}
.ok{color:#0a7d28;} .caveats{background:#fff8e6;border-left:4px solid #e6a700;
padding:.6rem 1rem;margin-top:1.5rem;} .caveats li{margin:.3rem 0;}
.gallery{display:flex;flex-wrap:wrap;gap:1rem;margin:.5rem 0 2rem;}
.dish{border:1px solid #ddd;border-radius:6px;padding:.5rem;}
.dish img{height:200px;display:block;} .dish .cap{font-size:.8rem;color:#555;}
.fail{color:#b00020;}
"""


def render_html(data: dict, rows: list[dict], dataset: Path,
                embed: bool) -> str:
    import html

    desc = data["description"]
    title = desc.get("dataset_title") or desc.get("title") or "Behavioral report"
    n_vid = len({r['video_id'] for r in rows})
    excl = sum(r["verdict"] == "EXCLUDE" for r in rows)

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{html.escape(title)}</title><style>{_CSS}</style>",
             "</head><body>", f"<h1>{html.escape(title)}</h1>",
             "<div class='meta'>",
             f"dataset_id <code>{html.escape(str(desc.get('dataset_id','—')))}"
             f"</code> &middot; version {html.escape(str(desc.get('dataset_version','—')))}"
             f" &middot; generated {_utcnow()}<br>",
             f"{n_vid} video(s), {len(rows)} dish(es) &middot; "
             f"<b>{excl}</b> flagged for exclusion</div>"]

    b = data["batch"]
    if b:
        parts.append(f"<p><b>Batch:</b> {b.get('n_done',0)} done, "
                     f"<span class='fail'>{b.get('n_failed',0)} failed</span>, "
                     f"{b.get('n_skipped',0)} skipped.</p>")
        for r in b.get("runs", []):
            if r.get("status") == "failed":
                parts.append("<p class='fail'>FAILED "
                             f"<code>{html.escape(str(r.get('video')))}</code>: "
                             f"{html.escape(str(r.get('error')))}</p>")

    # behavior table
    feat_labels = [lbl for lbl, _, _ in REPORT_FEATURES]
    head = "".join(f"<th class='l'>{html.escape(h)}</th>"
                   for h in ("video", "dish", "treatment")) + \
        "<th>quality</th><th class='l'>QC</th>" + \
        "".join(f"<th>{html.escape(h)}</th>" for h in feat_labels)
    parts.append("<h2>Behavior per dish</h2><table><tr>" + head + "</tr>")
    for r in rows:
        cls = " class='exclude'" if r["verdict"] == "EXCLUDE" else ""
        verdict_html = (f"<span class='excl'>EXCLUDE</span> "
                        f"{html.escape(r['reason'])}" if r["verdict"] == "EXCLUDE"
                        else "<span class='ok'>OK</span>")
        cells = [f"<td class='l'>{html.escape(str(r['video_id']))}</td>",
                 f"<td>{r['dish_id']}</td>",
                 f"<td class='l'>{html.escape(r['treatment'])}</td>",
                 f"<td>{r['quality_score']}</td>",
                 f"<td class='l'>{verdict_html}</td>"]
        cells += [f"<td>{html.escape(r['features'][lbl])}</td>"
                  for lbl in feat_labels]
        parts.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    parts.append("</table>")

    # image gallery
    parts.append("<h2>Heatmaps &amp; trajectories</h2>")
    for r in rows:
        imgs = heatmap_pngs(dataset, r["video_id"], r["dish_id"])
        if not imgs:
            continue
        parts.append("<div class='gallery'>")
        for kind in ("heatmap", "trajectory"):
            if kind in imgs:
                src = img_src(imgs[kind], dataset, embed)
                parts.append(
                    f"<div class='dish'><img src='{src}' alt='{kind}'>"
                    f"<div class='cap'>{html.escape(str(r['video_id']))} "
                    f"dish {r['dish_id']:02d} — {kind}</div></div>")
        parts.append("</div>")

    parts.append("<div class='caveats'><b>How to read these results</b><ul>")
    for c in CAVEATS:
        parts.append(f"<li>{html.escape(c)}</li>")
    parts.append("</ul></div></body></html>")
    return "".join(parts)


def _utcnow() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Provenance / CLI
# --------------------------------------------------------------------------- #
def write_provenance(dataset: Path, n_rows: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_report"),
        "timestamp": _utcnow(),
        "script": "scripts/generate_report.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(dataset / "features" / "features_per_video.csv")],
        "output_path": str(dataset / "report.html"),
        "n_rows": n_rows,
    }
    prov = dataset / "metadata" / "processing_provenance.jsonl"
    prov.parent.mkdir(parents=True, exist_ok=True)
    with open(prov, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path,
                   help="FAIR dataset dir (the --out of run_pipeline)")
    p.add_argument("--html-out", type=Path, default=None,
                   help="default: <dataset>/report.html")
    p.add_argument("--md-out", type=Path, default=None,
                   help="default: <dataset>/report.md")
    p.add_argument("--link-images", action="store_true",
                   help="reference PNGs by relative path instead of embedding "
                        "them as base64 (smaller, but not standalone)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = args.dataset
    try:
        data = load_dataset(dataset)
    except FileNotFoundError as exc:
        print(f"Cannot build report: {exc}", file=sys.stderr)
        return 1

    rows = build_rows(data)
    embed = not args.link_images
    html_out = args.html_out or dataset / "report.html"
    md_out = args.md_out or dataset / "report.md"
    html_out.write_text(render_html(data, rows, dataset, embed))
    md_out.write_text(render_markdown(data, rows))

    write_provenance(dataset, len(rows))
    excl = sum(r["verdict"] == "EXCLUDE" for r in rows)
    print(f"Report: {len(rows)} dish row(s), {excl} flagged "
          f"-> {html_out} and {md_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
