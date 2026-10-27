#!/usr/bin/env python3
"""Treatment-grouped aggregate report across videos (SPECIFICATION.md 11, 23).

Rolls up `features/features_per_video.csv` (one row per video x dish) by
experimental condition and contrasts each treated condition against its control
within the same experiment -- the step that turns per-dish measurements into a
stress-vs-control *result*.

Treatment is not in the feature table; it is joined from
`metadata/experiment_metadata.csv` on (video_id, dish_id). Conditions are
`treatment` optionally refined by `--factors` (e.g. stressor_duration_s for a
dose-response). For each experiment_id the control rows are the reference, and
every non-control condition is contrasted feature-by-feature with a mean
difference, Hedges' g effect size, and a Mann-Whitney U p-value.

Outputs (under --out):
    aggregate_by_condition.csv   one row per (experiment_id, condition): n + means
    treatment_contrasts.csv      one row per (experiment_id, condition, feature)
    aggregate_report.md / .html  human-readable summary + caveats

IMPORTANT (SPECIFICATION.md 23): these are descriptive behavioral proxies, not proof of
welfare/suffering. Dishes within a tray are pseudo-replicates (the tray/video is
the experimental unit); effect sizes are uncorrected for multiple comparisons.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import crop_petris_yolo_seg as cp
import compute_behavior_features as cf

# features to contrast (skip identifiers / counts that aren't traits)
CONTRAST_FEATURES = [c for c in cf.FEATURE_KEYS if c != "n_frames"] + \
    cf.DIVERGENCE_KEYS
# Detection-quality features are reported SEPARATELY from behavior: if these
# differ by treatment, the behavioral contrasts are confounded (you may be
# comparing how well the larva was tracked, not how it behaved).
QUALITY_FEATURES = {
    "valid_fraction", "mean_larva_conf", "median_larva_conf", "min_larva_conf",
    "missing_fraction", "interpolated_fraction", "track_switch_count",
    "outlier_fraction", "fraction_outside_petri", "quality_score",
}
MERGE_META = ["treatment", "stressor", "stressor_intensity",
              "stressor_duration_s", "temperature_c", "light_condition",
              "experiment_id"]


# --------------------------------------------------------------------------- #
# Statistics (pure)
# --------------------------------------------------------------------------- #
def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def hedges_g(treated, control) -> float:
    """Bias-corrected standardized mean difference (treated - control)."""
    t, c = _finite(treated), _finite(control)
    nt, nc = t.size, c.size
    if nt < 2 or nc < 2:
        return float("nan")
    sp2 = ((nt - 1) * t.var(ddof=1) + (nc - 1) * c.var(ddof=1)) / (nt + nc - 2)
    if sp2 <= 0:
        return float("nan")
    d = (t.mean() - c.mean()) / np.sqrt(sp2)
    correction = 1.0 - 3.0 / (4.0 * (nt + nc) - 9.0)
    return float(d * correction)


def mannwhitney_p(treated, control) -> float:
    """Two-sided Mann-Whitney U p-value; NaN if scipy missing or degenerate."""
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        return float("nan")
    t, c = _finite(treated), _finite(control)
    if t.size < 1 or c.size < 1:
        return float("nan")
    try:
        return float(mannwhitneyu(t, c, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def load_and_merge(features_path: Path, metadata_path: Optional[Path]):
    import pandas as pd

    df = pd.read_csv(features_path)
    if metadata_path and metadata_path.is_file():
        meta = pd.read_csv(metadata_path)
        keep = ["video_id", "dish_id"] + [c for c in MERGE_META
                                          if c in meta.columns]
        df = df.merge(meta[keep], on=["video_id", "dish_id"], how="left",
                      suffixes=("", "_meta"))
        # prefer the curated (campaign) experiment_id from metadata
        if "experiment_id_meta" in df.columns:
            df["experiment_id"] = df["experiment_id_meta"].where(
                df["experiment_id_meta"].notna(), df["experiment_id"])
            df = df.drop(columns=["experiment_id_meta"])
    if "treatment" not in df.columns:
        df["treatment"] = "unknown"
    df["treatment"] = df["treatment"].fillna("unknown").astype(str)
    if "experiment_id" not in df.columns:
        df["experiment_id"] = "all"
    df["experiment_id"] = df["experiment_id"].fillna("all").astype(str)
    return df


def add_condition(df, factors: list[str]):
    """A condition is treatment plus any --factors values (controls collapse to
    just 'control' since their factor cells are blank)."""
    import pandas as pd

    def label(r) -> str:
        parts = [str(r["treatment"])]
        for fac in factors:
            v = r.get(fac)
            if pd.notna(v) and str(v).strip() not in ("", "nan"):
                parts.append(f"{fac}={_fmt(v)}")
        return " | ".join(parts)

    df = df.copy()
    df["condition"] = df.apply(label, axis=1)
    return df


def _fmt(v):
    try:
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else f"{fv:g}"
    except (TypeError, ValueError):
        return str(v)


def feature_columns(df) -> list[str]:
    return [c for c in CONTRAST_FEATURES if c in df.columns]


def behavioral_features(df) -> list[str]:
    return [c for c in feature_columns(df) if c not in QUALITY_FEATURES]


def quality_features(df) -> list[str]:
    return [c for c in feature_columns(df) if c in QUALITY_FEATURES]


# --------------------------------------------------------------------------- #
# Aggregation + contrasts
# --------------------------------------------------------------------------- #
def summarize_conditions(df, feature_cols: list[str]) -> list[dict]:
    import pandas as pd

    rows = []
    for (eid, cond), g in df.groupby(["experiment_id", "condition"],
                                     sort=True):
        row = {"experiment_id": eid, "condition": cond,
               "treatment": g["treatment"].iloc[0],
               "n_dishes": len(g), "n_videos": g["video_id"].nunique()}
        for c in feature_cols:
            row[f"{c}_mean"] = float(pd.to_numeric(g[c], errors="coerce").mean())
        rows.append(row)
    return rows


def compute_contrasts(df, feature_cols: list[str], control: str = "control",
                      group: str = "") -> list[dict]:
    import pandas as pd

    rows = []
    for eid, g in df.groupby("experiment_id", sort=True):
        ctrl = g[g["treatment"] == control]
        if ctrl.empty:
            continue
        treated = g[g["treatment"] != control]
        for cond, cg in treated.groupby("condition", sort=True):
            for c in feature_cols:
                tv = pd.to_numeric(cg[c], errors="coerce")
                cv = pd.to_numeric(ctrl[c], errors="coerce")
                rows.append({
                    "experiment_id": eid, "condition": cond,
                    "feature_group": group, "feature": c,
                    "n_control": int(cv.notna().sum()),
                    "n_treated": int(tv.notna().sum()),
                    "control_mean": _round(cv.mean()),
                    "treated_mean": _round(tv.mean()),
                    "diff": _round(tv.mean() - cv.mean()),
                    "hedges_g": _round(hedges_g(tv, cv), 4),
                    "mannwhitney_p": _round(mannwhitney_p(tv, cv), 5),
                })
    return rows


def filter_min_valid(df, min_valid: float):
    """Drop dishes below a valid-detection floor (for behavioral contrasts only,
    so a poorly-tracked dish can't masquerade as a behavioral change)."""
    import pandas as pd
    if min_valid <= 0 or "valid_fraction" not in df.columns:
        return df
    vf = pd.to_numeric(df["valid_fraction"], errors="coerce").fillna(0.0)
    return df[vf >= min_valid]


def _round(v, nd: int = 4):
    import pandas as pd
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    return round(float(v), nd)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
CAVEATS = [
    "Descriptive behavioral proxies (locomotion, occupancy, rim-seeking, area), "
    "not measures of welfare/suffering; interpretation needs validation.",
    "Dishes within a tray are pseudo-replicates -- the tray/video is the true "
    "experimental unit; treat per-dish n and p-values as exploratory.",
    "Effect sizes (Hedges' g) and p-values are uncorrected for the many "
    "features compared; use them to rank candidates, not to confirm.",
    "Contrasts use the control videos of the SAME experiment_id as reference "
    "(treatment is tray-level, so there is no within-tray control).",
]


def _movers(contrasts, eid, cond, group, n=None):
    rows = [r for r in contrasts if r["experiment_id"] == eid
            and r["condition"] == cond and r["feature_group"] == group
            and np.isfinite(r["hedges_g"])]
    rows.sort(key=lambda r: -abs(r["hedges_g"]))
    return rows[:n] if n else rows


def _md_table(rows: list[dict]) -> list[str]:
    out = ["| feature | control | treated | diff | g | p |",
           "|---|---|---|---|---|---|"]
    for m in rows:
        out.append(f"| {m['feature']} | {m['control_mean']} | "
                   f"{m['treated_mean']} | {m['diff']} | {m['hedges_g']} | "
                   f"{m['mannwhitney_p']} |")
    return out


def render_markdown(summary, contrasts, top: int, min_valid: float) -> str:
    exps = sorted({r["experiment_id"] for r in summary})
    lines = ["# Treatment-grouped aggregate report", "",
             f"- generated: {_utcnow()}",
             f"- experiments: {len(exps)} | conditions: {len(summary)}",
             f"- behavioral contrasts use dishes with valid_fraction >= "
             f"{min_valid}" if min_valid > 0 else
             "- behavioral contrasts use ALL dishes (no --min-valid-fraction)",
             ""]
    for eid in exps:
        conds = [r for r in summary if r["experiment_id"] == eid]
        lines += [f"## {eid}", "",
                  "| condition | n_videos | n_dishes |", "|---|---|---|"]
        for r in sorted(conds, key=lambda r: r["condition"]):
            lines.append(f"| {r['condition']} | {r['n_videos']} | "
                         f"{r['n_dishes']} |")
        lines.append("")
        treated_conds = sorted({r["condition"] for r in contrasts
                                if r["experiment_id"] == eid})
        for cond in treated_conds:
            qual = _movers(contrasts, eid, cond, "detection_quality")
            behav = _movers(contrasts, eid, cond, "behavioral", top)
            if qual:
                lines += [f"### {cond} — detection quality (confound check)", ""]
                lines += _md_table(qual) + [""]
            if behav:
                lines += [f"### {cond} — behavioral, top {len(behav)} by |g|", ""]
                lines += _md_table(behav) + [""]
    lines += ["## How to read these results", ""]
    lines += [f"- {c}" for c in CAVEATS]
    return "\n".join(lines) + "\n"


_CSS = """body{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#1a1a1a}
h1{margin-bottom:.2rem}h2{margin-top:1.6rem}table{border-collapse:collapse;
margin:.5rem 0;font-size:.9rem}th,td{border:1px solid #ccc;padding:3px 8px;
text-align:right}th{background:#f0f0f0}td.l,th.l{text-align:left}
.pos{color:#b00020}.neg{color:#0a4db0}
.caveats{background:#fff8e6;border-left:4px solid #e6a700;padding:.6rem 1rem;
margin-top:1.5rem}.caveats li{margin:.3rem 0}"""


def _html_table(rows: list[dict]) -> str:
    out = ["<table><tr><th class='l'>feature</th><th>control</th>"
           "<th>treated</th><th>diff</th><th>g</th><th>p</th></tr>"]
    for m in rows:
        cls = "pos" if m["hedges_g"] > 0 else "neg"
        out.append(f"<tr><td class='l'>{html.escape(m['feature'])}</td>"
                   f"<td>{m['control_mean']}</td><td>{m['treated_mean']}</td>"
                   f"<td>{m['diff']}</td><td class='{cls}'>{m['hedges_g']}</td>"
                   f"<td>{m['mannwhitney_p']}</td></tr>")
    out.append("</table>")
    return "".join(out)


def render_html(summary, contrasts, top: int, min_valid: float) -> str:
    exps = sorted({r["experiment_id"] for r in summary})
    vnote = (f"behavioral contrasts use dishes with valid_fraction &ge; "
             f"{min_valid}" if min_valid > 0 else
             "behavioral contrasts use all dishes")
    p = ["<!doctype html><html><head><meta charset='utf-8'>",
         f"<title>Aggregate report</title><style>{_CSS}</style></head><body>",
         "<h1>Treatment-grouped aggregate report</h1>",
         f"<div>generated {_utcnow()} &middot; {len(exps)} experiment(s), "
         f"{len(summary)} condition(s) &middot; {vnote}</div>"]
    for eid in exps:
        p.append(f"<h2>{html.escape(eid)}</h2><table>"
                 "<tr><th class='l'>condition</th><th>n_videos</th>"
                 "<th>n_dishes</th></tr>")
        for r in sorted((x for x in summary if x["experiment_id"] == eid),
                        key=lambda r: r["condition"]):
            p.append(f"<tr><td class='l'>{html.escape(r['condition'])}</td>"
                     f"<td>{r['n_videos']}</td><td>{r['n_dishes']}</td></tr>")
        p.append("</table>")
        for cond in sorted({r["condition"] for r in contrasts
                            if r["experiment_id"] == eid}):
            qual = _movers(contrasts, eid, cond, "detection_quality")
            behav = _movers(contrasts, eid, cond, "behavioral", top)
            if qual:
                p.append(f"<h3>{html.escape(cond)} — detection quality "
                         "(confound check)</h3>" + _html_table(qual))
            if behav:
                p.append(f"<h3>{html.escape(cond)} — behavioral, top "
                         f"{len(behav)} by |g|</h3>" + _html_table(behav))
    p.append("<div class='caveats'><b>How to read these results</b><ul>")
    p += [f"<li>{html.escape(c)}</li>" for c in CAVEATS]
    p.append("</ul></div></body></html>")
    return "".join(p)


def _utcnow() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# IO / CLI
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_provenance(out: Path, args, n_cond: int, n_contrast: int) -> None:
    record = {
        "run_id": datetime.now().strftime("%Y-%m-%dT%H-%M-%S_aggregate"),
        "timestamp": _utcnow(),
        "script": "scripts/aggregate_report.py",
        "git_commit": cp.git_commit(),
        "command": " ".join(sys.argv),
        "input_paths": [str(args.features)] +
                       ([str(args.metadata)] if args.metadata else []),
        "output_path": str(args.out),
        "n_conditions": n_cond, "n_contrasts": n_contrast,
        "parameters": {"factors": args.factors,
                       "control_treatment": args.control_treatment,
                       "min_valid_fraction": args.min_valid_fraction,
                       "top": args.top},
    }
    prov = out / "processing_provenance.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    with open(prov, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path,
                   help="features/features_per_video.csv")
    p.add_argument("--metadata", type=Path, default=None,
                   help="experiment_metadata.csv (for treatment labels)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--factors", nargs="*", default=[],
                   help="extra metadata columns refining the condition, e.g. "
                        "stressor_duration_s light_condition")
    p.add_argument("--control-treatment", default="control")
    p.add_argument("--min-valid-fraction", type=float, default=0.0,
                   help="exclude dishes below this valid_fraction from the "
                        "BEHAVIORAL contrasts (detection-quality contrasts "
                        "always use all dishes); 0 = no filtering")
    p.add_argument("--top", type=int, default=12,
                   help="top behavioral features by |Hedges g| to show")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.features.is_file():
        print(f"Missing features table: {args.features}", file=sys.stderr)
        return 1

    df = load_and_merge(args.features, args.metadata)
    if (df["treatment"] == "unknown").all():
        print("No treatment labels found (pass --metadata with a curated "
              "experiment_metadata.csv); nothing to aggregate.",
              file=sys.stderr)
        return 1
    df = add_condition(df, args.factors)
    feats = feature_columns(df)

    summary = summarize_conditions(df, feats)
    # detection-quality contrasts use ALL dishes (the confound is the point);
    # behavioral contrasts use only adequately-detected dishes.
    df_behav = filter_min_valid(df, args.min_valid_fraction)
    contrasts = (
        compute_contrasts(df, quality_features(df), args.control_treatment,
                          group="detection_quality") +
        compute_contrasts(df_behav, behavioral_features(df),
                          args.control_treatment, group="behavioral"))

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "aggregate_by_condition.csv", summary)
    write_csv(args.out / "treatment_contrasts.csv", contrasts)
    (args.out / "aggregate_report.md").write_text(
        render_markdown(summary, contrasts, args.top, args.min_valid_fraction))
    (args.out / "aggregate_report.html").write_text(
        render_html(summary, contrasts, args.top, args.min_valid_fraction))
    write_provenance(args.out, args, len(summary), len(contrasts))

    n_exp = len({r["experiment_id"] for r in summary})
    n_scored = len({(r["experiment_id"], r["condition"]) for r in contrasts})
    print(f"Aggregated {len(df)} dish rows -> {len(summary)} condition(s) "
          f"across {n_exp} experiment(s); {n_scored} treated condition(s) "
          f"contrasted vs control -> {args.out}")
    if not contrasts:
        print("  (no contrasts: no experiment had both control and treated "
              "rows -- check experiment_id grouping in the metadata)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
