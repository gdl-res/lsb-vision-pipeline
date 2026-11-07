"""Tests for the human-readable report generator (no YOLO, no recompute)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_report as gr  # noqa: E402


def _make_dataset(root, *, with_qc=True, with_batch=False, with_pngs=True):
    import pandas as pd
    (root / "features").mkdir(parents=True)
    (root / "qc").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    (root / "heatmaps" / "png").mkdir(parents=True)

    feat = pd.DataFrame([
        {"dataset_id": "ds", "experiment_id": "e", "video_id": "v1",
         "dish_id": 1, "valid_fraction": 1.0, "total_distance_mm": 28.0,
         "mean_speed_mm_s": 2.8, "immobility_fraction": 0.4,
         "movement_bout_count": 69, "edge_fraction": 0.84, "mean_r_norm": 0.75,
         "mean_area_px": 636, "quality_score": 1.0, "track_switch_count": 0,
         "js_divergence_from_control": float("nan"), "tortuosity": float("nan")},
        {"dataset_id": "ds", "experiment_id": "e", "video_id": "v1",
         "dish_id": 2, "valid_fraction": 0.97, "total_distance_mm": 94.0,
         "mean_speed_mm_s": 9.4, "immobility_fraction": 0.1,
         "movement_bout_count": 32, "edge_fraction": 0.14, "mean_r_norm": 0.34,
         "mean_area_px": 552, "quality_score": 0.97, "track_switch_count": 10,
         "js_divergence_from_control": 0.21, "tortuosity": 1.4},
    ])
    feat.to_csv(root / "features" / "features_per_video.csv", index=False)

    if with_qc:
        qc = pd.DataFrame([
            {"video_id": "v1", "dish_id": 1, "exclude_recommended": False,
             "exclude_reason": "", "treatment": "control"},
            {"video_id": "v1", "dish_id": 2, "exclude_recommended": True,
             "exclude_reason": "track_switch_count>5", "treatment": "heat"},
        ])
        qc.to_csv(root / "qc" / "qc_summary.csv", index=False)

    (root / "dataset_description.json").write_text(json.dumps(
        {"dataset_id": "ds", "dataset_title": "Test dataset",
         "dataset_version": "1.0.0"}))

    if with_batch:
        (root / "metadata" / "batch_index.json").write_text(json.dumps(
            {"n_done": 1, "n_failed": 1, "n_skipped": 0,
             "runs": [{"video": "v1.mov", "status": "done"},
                      {"video": "bad.mov", "status": "failed",
                       "error": "RuntimeError: boom"}]}))

    if with_pngs:
        # 1x1 PNG bytes are enough to base64-embed
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c6300010000050001a5f645400000000049454e44ae"
            "426082")
        for dish in (1, 2):
            for kind in ("heatmap", "trajectory"):
                (root / "heatmaps" / "png" /
                 f"v1_dish{dish:02d}_{kind}.png").write_bytes(png)


def test_build_rows_merges_features_and_qc(tmp_path):
    pytest.importorskip("pandas")
    _make_dataset(tmp_path)
    data = gr.load_dataset(tmp_path)
    rows = gr.build_rows(data)
    assert [r["dish_id"] for r in rows] == [1, 2]
    assert rows[0]["verdict"] == "OK" and rows[0]["treatment"] == "control"
    assert rows[1]["verdict"] == "EXCLUDE"
    assert rows[1]["reason"] == "track_switch_count>5"
    # NaN feature renders as em dash, real value formats
    assert rows[0]["features"]["Divergence (JS)"] == "—"
    assert rows[1]["features"]["Divergence (JS)"] == "0.210"


def test_qc_verdict_none_row():
    assert gr.qc_verdict(None)[0] == "?"


def test_render_markdown_has_table_and_caveats(tmp_path):
    pytest.importorskip("pandas")
    _make_dataset(tmp_path)
    data = gr.load_dataset(tmp_path)
    md = gr.render_markdown(data, gr.build_rows(data))
    assert "Test dataset" in md
    assert "flagged for exclusion: 1/2" in md
    assert "EXCLUDE (track_switch_count>5)" in md
    assert "objective behavioral proxies" in md  # caveat present


def test_render_html_embeds_images_and_flags(tmp_path):
    pytest.importorskip("pandas")
    _make_dataset(tmp_path, with_batch=True)
    data = gr.load_dataset(tmp_path)
    html = gr.render_html(data, gr.build_rows(data), tmp_path, embed=True)
    assert html.count("data:image/png;base64") == 4   # 2 dishes x 2 imgs
    assert "class='exclude'" in html                   # excluded row styled
    assert "1 failed" in html                          # batch summary shown
    assert "RuntimeError: boom" in html                # failure surfaced


def test_render_html_link_images_no_base64(tmp_path):
    pytest.importorskip("pandas")
    _make_dataset(tmp_path)
    data = gr.load_dataset(tmp_path)
    html = gr.render_html(data, gr.build_rows(data), tmp_path, embed=False)
    assert "data:image/png;base64" not in html
    assert "heatmaps/png/v1_dish01_heatmap.png" in html


def test_main_writes_reports_and_provenance(tmp_path):
    pytest.importorskip("pandas")
    _make_dataset(tmp_path)
    rc = gr.main(["--dataset", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report.md").is_file()
    prov = (tmp_path / "metadata" / "processing_provenance.jsonl").read_text()
    assert "generate_report.py" in prov


def test_main_missing_features_errors(tmp_path):
    pytest.importorskip("pandas")
    rc = gr.main(["--dataset", str(tmp_path)])
    assert rc == 1


def test_report_works_without_qc(tmp_path):
    pytest.importorskip("pandas")
    _make_dataset(tmp_path, with_qc=False)
    data = gr.load_dataset(tmp_path)
    rows = gr.build_rows(data)
    # no qc table -> verdict unknown, but report still builds
    assert all(r["verdict"] == "?" for r in rows)
    gr.render_html(data, rows, tmp_path, embed=True)


def test_treatment_falls_back_to_experiment_metadata(tmp_path):
    import pandas as pd
    _make_dataset(tmp_path, with_qc=False)
    pd.DataFrame([{"video_id": "v1", "dish_id": 1, "treatment": "control"},
                  {"video_id": "v1", "dish_id": 2, "treatment": "heat"}]
                 ).to_csv(tmp_path / "metadata" / "experiment_metadata.csv",
                          index=False)
    rows = gr.build_rows(gr.load_dataset(tmp_path))
    assert [r["treatment"] for r in rows] == ["control", "heat"]


def test_header_only_experiment_metadata_gives_dash(tmp_path):
    _make_dataset(tmp_path, with_qc=False)
    (tmp_path / "metadata" / "experiment_metadata.csv").write_text(
        "video_id,dish_id,treatment\n")
    rows = gr.build_rows(gr.load_dataset(tmp_path))
    assert all(r["treatment"] == "—" for r in rows)
