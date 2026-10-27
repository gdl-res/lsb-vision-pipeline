"""Tests for the filename -> experiment_metadata draft parser (no IO needed)."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import parse_filenames as pf  # noqa: E402


# --- convention B (heat stress) -------------------------------------------- #
def test_parse_convention_b_heat():
    f = pf.parse_stem("20260623_38°C_2h_lucewarm")
    assert f["convention"] == "B"
    assert f["treatment"] == "heat" and f["stressor"] == "heat"
    assert f["temperature_c"] == "38"
    assert f["stressor_duration_s"] == 7200
    assert f["light_condition"] == "warm"   # luce* is lighting, not stress
    assert f["confidence"] == "high"


def test_parse_convention_b_underscore_temp_and_replicate():
    f = pf.parse_stem("20260625_38_C_3h_lucenat_2")
    assert f["temperature_c"] == "38"          # 38_C spelling
    assert f["stressor_duration_s"] == 10800
    assert f["light_condition"] == "natural"
    assert f["batch_id"] == "2"


def test_parse_convention_b_nostress_is_control():
    f = pf.parse_stem("20260623_nostress_lucecold")
    assert f["convention"] == "B"
    assert f["treatment"] == "control" and f["stressor"] == "none"
    assert f["light_condition"] == "cold"


def test_heat_without_duration_flagged():
    f = pf.parse_stem("20260623_38°C_lucenat")
    assert f["treatment"] == "heat"
    assert f["stressor_duration_s"] == ""
    assert f["confidence"] == "medium"
    assert any("duration" in w for w in f["warnings"])


# --- convention A (lighting-only, unstressed) ------------------------------ #
def test_parse_convention_a_nostress_light_intensity():
    f = pf.parse_stem("20260514_133535000_lsb_iphone17pro_az_3x2petri_video_nostress_nat9")
    assert f["convention"] == "A"
    assert f["treatment"] == "control"
    assert f["light_condition"] == "natural"
    assert f["light_intensity"] == "9"        # the digit after nat = intensity
    assert f["recording_start_time"] == "13:35:35"
    assert f["phone_model"] == "iPhone 17 Pro"
    assert (f["rows"], f["cols"]) == ("3", "2")


def test_parse_convention_a_warm_is_lighting_not_stress():
    # the critical disambiguation: bare warm in convention A is LIGHTING
    f = pf.parse_stem("20260514_134807000_lsb_iphone17pro_az_3x2petri_video_warm")
    assert f["convention"] == "A"
    assert f["treatment"] == "control"        # NOT heat
    assert f["stressor"] == "none"
    assert f["light_condition"] == "warm"
    assert f["confidence"] == "high"
    assert "lighting" in f["notes"]


def test_profile_forces_convention():
    # forcing profile=b makes a bare-warm name parse as convention B (no temp ->
    # not heat, not nostress -> unknown/low), proving the override is honored
    f = pf.parse_stem("20260514_video_warm", profile="b")
    assert f["convention"] == "B"
    assert f["treatment"] == "unknown" and f["confidence"] == "low"


# --- experiment_id grouping + control reference ---------------------------- #
def test_assign_experiment_ids_groups_campaign():
    parsed = [pf.parse_stem(s) for s in [
        "20260623_38°C_1h_lucenat",         # June heat
        "20260625_38°C_3h_lucewarm",        # June heat (different day, heat-only)
        "20260623_nostress_lucecold",       # June control
        "20260514_133535000_lsb_iphone17pro_az_3x2petri_video_nostress_nat9",  # May
    ]]
    pf.assign_experiment_ids(parsed, "ds")
    eid = {p["video_id"]: p["experiment_id"] for p in parsed}
    # all June files share one campaign id (so Jun-25 heat-only still has a ref)
    assert eid["20260623_38°C_1h_lucenat"] == "exp_202606_heat38c"
    assert eid["20260625_38°C_3h_lucewarm"] == "exp_202606_heat38c"
    assert eid["20260623_nostress_lucecold"] == "exp_202606_heat38c"
    # May lighting session is its own group
    assert eid[parsed[3]["video_id"]] == "exp_202605_lighting"


def test_control_reference_warning_when_no_control():
    parsed = [pf.parse_stem("20260625_38°C_3h_lucewarm")]   # heat only, no control
    pf.assign_experiment_ids(parsed, "ds")
    warns = pf.control_reference_warnings(parsed)
    assert len(warns) == 1 and "NO control reference" in warns[0]


def test_control_reference_ok_with_control():
    parsed = [pf.parse_stem("20260623_38°C_1h_lucenat"),
              pf.parse_stem("20260623_nostress_lucecold")]
    pf.assign_experiment_ids(parsed, "ds")
    assert pf.control_reference_warnings(parsed) == []


# --- dish expansion (tray-level treatment) --------------------------------- #
def test_to_metadata_rows_expands_dishes_shared_treatment():
    p = pf.parse_stem("20260623_38°C_2h_lucewarm")
    pf.assign_experiment_ids([p], "ds")
    rows = pf.to_metadata_rows(p, "ds", n_dishes=6)
    assert len(rows) == 6
    assert [r["dish_id"] for r in rows] == [1, 2, 3, 4, 5, 6]
    # tray-level: every dish shares the treatment
    assert all(r["treatment"] == "heat" for r in rows)
    assert all(r["dataset_id"] == "ds" for r in rows)
    # schema columns all present; human-only field left blank
    for c in pf.EXPERIMENT_METADATA_COLUMNS:
        assert c in rows[0]
    assert rows[0]["petri_diameter_mm"] == ""


# --- end-to-end ------------------------------------------------------------ #
def test_main_writes_draft_with_audit_columns(tmp_path):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("20260623_38°C_1h_lucenat.MOV", "20260623_nostress_lucecold.MOV"):
        (vids / n).write_bytes(b"")
    out = tmp_path / "metadata" / "experiment_metadata.draft.csv"
    rc = pf.main(["--videos", str(vids), "--out", str(out),
                  "--rows", "3", "--cols", "2"])
    assert rc == 0
    with open(out) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames == \
            pf.EXPERIMENT_METADATA_COLUMNS + pf.AUDIT_COLUMNS
    assert len(rows) == 12            # 2 videos x 6 dishes
    heat = [r for r in rows if r["treatment"] == "heat"]
    assert len(heat) == 6 and heat[0]["experiment_id"] == "exp_202606_heat38c"
    # provenance recorded
    assert (out.parent / "processing_provenance.jsonl").is_file()


def test_main_no_videos_errors(tmp_path):
    assert pf.main(["--videos", str(tmp_path / "empty"),
                    "--out", str(tmp_path / "x.csv")]) == 1
