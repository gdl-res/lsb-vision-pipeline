"""Tests for the treatment-grouped aggregate report."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aggregate_report as ag  # noqa: E402


# --- statistics ------------------------------------------------------------ #
def test_hedges_g_sign_and_symmetry():
    treated = [3, 4, 5, 6, 7]
    control = [1, 2, 3, 4, 5]
    g = ag.hedges_g(treated, control)
    assert g > 0                                   # treated shifted up
    assert ag.hedges_g(control, treated) == pytest.approx(-g)


def test_hedges_g_degenerate_is_nan():
    assert np.isnan(ag.hedges_g([1], [1, 2, 3]))   # n<2
    assert np.isnan(ag.hedges_g([2, 2, 2], [2, 2, 2]))  # zero pooled variance


def test_mannwhitney_p_separation():
    p_sep = ag.mannwhitney_p([10, 11, 12, 13], [1, 2, 3, 4])
    p_same = ag.mannwhitney_p([1, 2, 3, 4], [1, 2, 3, 4])
    assert p_sep < p_same
    assert 0.0 <= p_sep <= 1.0


# --- assembly -------------------------------------------------------------- #
def _write(tmp_path, with_meta=True):
    import pandas as pd
    feat_rows, meta_rows = [], []
    spec = [("vc1", "control", ""), ("vc2", "control", ""),
            ("vh1", "heat", 3600), ("vh2", "heat", 3600)]
    for vid, treat, dur in spec:
        base = 2.0 if treat == "control" else 6.0     # heat moves the feature
        for dish in (1, 2):
            feat_rows.append({"dataset_id": "ds", "experiment_id": "exp1",
                              "video_id": vid, "dish_id": dish,
                              "mean_speed_mm_s": base + dish * 0.1,
                              "edge_fraction": 0.2 if treat == "control" else 0.8,
                              "valid_fraction": 0.9})
            meta_rows.append({"video_id": vid, "dish_id": dish,
                              "experiment_id": "exp1", "treatment": treat,
                              "stressor_duration_s": dur})
    feats = tmp_path / "features_per_video.csv"
    pd.DataFrame(feat_rows).to_csv(feats, index=False)
    meta = tmp_path / "experiment_metadata.csv"
    if with_meta:
        pd.DataFrame(meta_rows).to_csv(meta, index=False)
    return feats, meta


def test_load_and_merge_brings_treatment(tmp_path):
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    df = ag.load_and_merge(feats, meta)
    assert set(df["treatment"]) == {"control", "heat"}
    assert "stressor_duration_s" in df.columns


def test_add_condition_with_factor(tmp_path):
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    df = ag.add_condition(ag.load_and_merge(feats, meta),
                          ["stressor_duration_s"])
    conds = set(df["condition"])
    assert "control" in conds                       # control collapses
    assert "heat | stressor_duration_s=3600" in conds


def test_contrasts_direction_and_reference(tmp_path):
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    df = ag.add_condition(ag.load_and_merge(feats, meta), [])
    rows = ag.compute_contrasts(df, ag.feature_columns(df))
    speed = next(r for r in rows if r["feature"] == "mean_speed_mm_s")
    assert speed["control_mean"] < speed["treated_mean"]   # heat faster
    assert speed["diff"] > 0 and speed["hedges_g"] > 0
    assert speed["n_control"] == 4 and speed["n_treated"] == 4


def test_no_contrasts_without_control(tmp_path):
    import pandas as pd
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    df = ag.add_condition(ag.load_and_merge(feats, meta), [])
    df = df[df["treatment"] == "heat"]              # drop controls
    assert ag.compute_contrasts(df, ag.feature_columns(df)) == []


# --- end-to-end ------------------------------------------------------------ #
def test_main_writes_outputs(tmp_path):
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    out = tmp_path / "aggregate"
    rc = ag.main(["--features", str(feats), "--metadata", str(meta),
                  "--out", str(out), "--factors", "stressor_duration_s"])
    assert rc == 0
    for name in ("aggregate_by_condition.csv", "treatment_contrasts.csv",
                 "aggregate_report.md", "aggregate_report.html"):
        assert (out / name).is_file()
    with open(out / "treatment_contrasts.csv") as fh:
        contr = list(csv.DictReader(fh))
    # mean_speed has within-group variance -> finite positive g for heat
    assert any(r["feature"] == "mean_speed_mm_s" and float(r["hedges_g"]) > 0
               for r in contr)
    # edge_fraction is constant within group -> diff>0 but g is NaN
    edge = next(r for r in contr if r["feature"] == "edge_fraction")
    assert float(edge["diff"]) > 0 and edge["hedges_g"] in ("nan", "")
    assert "How to read these results" in (out / "aggregate_report.md").read_text()
    assert (out / "processing_provenance.jsonl").is_file()


def test_feature_groups_split_quality_vs_behavioral(tmp_path):
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    out = tmp_path / "agg"
    assert ag.main(["--features", str(feats), "--metadata", str(meta),
                    "--out", str(out)]) == 0
    with open(out / "treatment_contrasts.csv") as fh:
        rows = list(csv.DictReader(fh))
    groups = {r["feature"]: r["feature_group"] for r in rows}
    assert groups["valid_fraction"] == "detection_quality"
    assert groups["mean_speed_mm_s"] == "behavioral"
    assert groups["edge_fraction"] == "behavioral"


def test_min_valid_filters_behavioral_not_quality(tmp_path):
    import pandas as pd
    pytest.importorskip("pandas")
    feats, meta = _write(tmp_path)
    df = ag.add_condition(ag.load_and_merge(feats, meta), [])
    # force the heat dishes to look poorly tracked
    df.loc[df["treatment"] == "heat", "valid_fraction"] = 0.2
    df_b = ag.filter_min_valid(df, 0.5)
    # behavioral side loses the heat dishes -> no treated rows survive
    behav = ag.compute_contrasts(df_b, ag.behavioral_features(df), group="behavioral")
    assert all(r["n_treated"] == 0 for r in behav) or behav == []
    # detection-quality side keeps all dishes and shows the valid_fraction drop
    qual = ag.compute_contrasts(df, ag.quality_features(df),
                                group="detection_quality")
    vf = next(r for r in qual if r["feature"] == "valid_fraction")
    assert vf["n_treated"] == 4 and vf["treated_mean"] < vf["control_mean"]


def test_main_requires_treatment_labels(tmp_path):
    pytest.importorskip("pandas")
    feats, _ = _write(tmp_path, with_meta=False)
    rc = ag.main(["--features", str(feats), "--out", str(tmp_path / "agg")])
    assert rc == 1                                  # no metadata -> nothing to do
