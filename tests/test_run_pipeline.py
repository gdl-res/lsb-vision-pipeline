"""Tests for the end-to-end driver (analysis chain runs without YOLO)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_pipeline as rp  # noqa: E402


# --- helpers --------------------------------------------------------------- #
def test_discover_videos_dir(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mov").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    found = rp.discover_videos(tmp_path)
    assert [p.name for p in found] == ["a.mp4", "b.mov"]


def test_discover_videos_single_file(tmp_path):
    v = tmp_path / "clip.mkv"
    v.write_bytes(b"")
    assert rp.discover_videos(v) == [v]


def test_combine_tables(tmp_path):
    pd = pytest.importorskip("pandas")
    work = tmp_path / "_work"
    out = tmp_path / "out"
    for d in ("petri", "larva"):
        (out / d).mkdir(parents=True)
    for vid in ("v1", "v2"):
        vd = work / vid
        vd.mkdir(parents=True)
        pd.DataFrame({"video_id": [vid], "frame_idx": [0], "dish_id": [1],
                      "petri_cx": [1.0]}).to_csv(vd / "petri_tracks.csv",
                                                 index=False)
        pd.DataFrame({"video_id": [vid, vid], "frame_idx": [0, 1],
                      "dish_id": [1, 1]}).to_csv(vd / "larva_tracks.csv",
                                                 index=False)
        (vd / "larva_masks.jsonl").write_text('{"video_id":"%s"}\n' % vid)
    rp.combine_tables(work, out)
    petri = pd.read_csv(out / "petri" / "petri_tracks.csv")
    larva = pd.read_csv(out / "larva" / "larva_tracks.csv")
    assert len(petri) == 2 and len(larva) == 4
    masks = (out / "larva" / "larva_masks.jsonl").read_text().splitlines()
    assert len(masks) == 2


# --- batch resilience ------------------------------------------------------ #
def _batch_args(tmp_path, videos_dir, **over):
    """A parsed args namespace with real model files, for batch helpers."""
    petri = tmp_path / "petri.pt"
    larva = tmp_path / "larva.pt"
    petri.write_bytes(b"")
    larva.write_bytes(b"")
    argv = ["--out", str(tmp_path / "out"), "--videos", str(videos_dir),
            "--petri-model", str(petri), "--larva-model", str(larva)]
    for k, v in over.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else [flag, str(v)]
    args = rp.build_parser().parse_args(argv)
    (args.out / "metadata").mkdir(parents=True, exist_ok=True)
    return args


def _fake_track_ok(argv):
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "larva_tracks.csv").write_text("video_id\nv\n")
    return 0


def _fake_track_boom(argv):
    raise RuntimeError("boom")


def test_build_track_argv_pins_imgsz_and_geometry():
    import track_larvae_video as tl
    args = rp.build_parser().parse_args(
        ["--videos", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l"])
    assert (args.petri_imgsz, args.larva_imgsz) == (640, 640)   # pinned default
    argv = rp.build_track_argv(Path("v.mp4"), Path("w"), args)
    for flag, val in (("--petri-imgsz", "640"), ("--larva-imgsz", "640"),
                      ("--petri-geometry", "box"), ("--tracker", "bytetrack")):
        assert flag in argv and argv[argv.index(flag) + 1] == val
    # the tracker actually accepts what the driver forwards
    targs = tl.build_parser().parse_args(argv)
    assert (targs.petri_imgsz, targs.larva_imgsz) == (640, 640)


def test_build_track_argv_forwards_selected_tracker():
    """The driver is the production entry point, so a tracker
    choice is useless unless it survives the hop into track_larvae_video."""
    import track_larvae_video as tl
    args = rp.build_parser().parse_args(
        ["--videos", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l", "--tracker", "ocsort"])
    argv = rp.build_track_argv(Path("v.mp4"), Path("w"), args)
    assert argv[argv.index("--tracker") + 1] == "ocsort"
    assert tl.build_parser().parse_args(argv).tracker == "ocsort"


def test_build_track_argv_forwards_selection_policy():
    """The hysteresis jump gate and area gate (SPECIFICATION.md 6) are what stop a
    transient artifact from hijacking a track. If the driver drops them, a batch
    run silently uses different defaults than the single-video run it is meant
    to reproduce."""
    import track_larvae_video as tl
    args = rp.build_parser().parse_args(
        ["--videos", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l", "--max-jump-px", "80",
         "--max-jump-growth-px", "7", "--max-jump-cap-px", "300",
         "--area-min-px", "150", "--area-max-px", "9000",
         "--low-conf-threshold", "0.5"])
    argv = rp.build_track_argv(Path("v.mp4"), Path("w"), args)
    targs = tl.build_parser().parse_args(argv)
    assert (targs.max_jump_px, targs.max_jump_growth_px,
            targs.max_jump_cap_px) == (80.0, 7.0, 300.0)
    assert (targs.area_min_px, targs.area_max_px) == (150.0, 9000.0)
    assert targs.low_conf_threshold == 0.5


def test_driver_selection_policy_defaults_match_the_tracker():
    """Mirrored defaults, so forwarding them changes nothing for existing runs."""
    import track_larvae_video as tl
    d = rp.build_parser().parse_args(
        ["--videos", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l"])
    t = tl.build_parser().parse_args(
        ["--video", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l"])
    for name in ("low_conf_threshold", "area_min_px", "area_max_px",
                 "max_jump_px", "max_jump_growth_px", "max_jump_cap_px"):
        assert getattr(d, name) == getattr(t, name), name


def test_build_track_argv_omits_unset_area_gates():
    """area gates are None by default and must not be forwarded as 'None'."""
    args = rp.build_parser().parse_args(
        ["--videos", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l"])
    argv = rp.build_track_argv(Path("v.mp4"), Path("w"), args)
    assert "--area-min-px" not in argv and "--area-max-px" not in argv


def test_build_track_argv_forwards_tracker_config_override(tmp_path):
    import track_larvae_video as tl
    cfg = tmp_path / "tuned.yaml"
    cfg.write_text("tracker_type: botsort\n")
    args = rp.build_parser().parse_args(
        ["--videos", "v", "--out", "o", "--petri-model", "p",
         "--larva-model", "l", "--tracker", "botsort",
         "--tracker-config", str(cfg)])
    argv = rp.build_track_argv(Path("v.mp4"), Path("w"), args)
    assert argv[argv.index("--tracker-config") + 1] == str(cfg)
    assert tl.build_parser().parse_args(argv).tracker_config == cfg


def test_track_one_success_writes_sentinel(tmp_path, monkeypatch):
    vids = tmp_path / "vids"
    vids.mkdir()
    v = vids / "clip.mp4"
    v.write_bytes(b"")
    args = _batch_args(tmp_path, vids)
    monkeypatch.setattr(rp.tl, "main", _fake_track_ok)
    rec = rp.track_one(v, args.out / "_work", args)
    assert rec["status"] == "done"
    sentinel = args.out / "_work" / "clip" / rp.DONE_SENTINEL
    assert sentinel.is_file()
    assert not (args.out / "_work" / "clip" / rp.FAILURE_FILE).exists()


def test_track_one_failure_isolated(tmp_path, monkeypatch):
    vids = tmp_path / "vids"
    vids.mkdir()
    v = vids / "bad.mp4"
    v.write_bytes(b"")
    args = _batch_args(tmp_path, vids)
    monkeypatch.setattr(rp.tl, "main", _fake_track_boom)
    rec = rp.track_one(v, args.out / "_work", args)  # does not raise
    assert rec["status"] == "failed"
    assert "boom" in rec["error"]
    fail = json.loads(
        (args.out / "_work" / "bad" / rp.FAILURE_FILE).read_text())
    assert "traceback" in fail
    assert not (args.out / "_work" / "bad" / rp.DONE_SENTINEL).exists()


def test_batch_skip_existing_resumes(tmp_path, monkeypatch):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("a.mp4", "b.mp4"):
        (vids / n).write_bytes(b"")
    args = _batch_args(tmp_path, vids, skip_existing=True)
    work = args.out / "_work"
    calls = []

    def counting(argv):
        calls.append(argv[argv.index("--out") + 1])
        return _fake_track_ok(argv)

    monkeypatch.setattr(rp.tl, "main", counting)
    rp.run_tracking_batch(rp.discover_videos(vids), work, args)
    assert len(calls) == 2          # first pass tracks both
    rp.run_tracking_batch(rp.discover_videos(vids), work, args)
    assert len(calls) == 2          # second pass skips both (resume)


def test_batch_continues_past_failure_and_indexes(tmp_path, monkeypatch):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("ok1.mp4", "bad.mp4", "ok2.mp4"):
        (vids / n).write_bytes(b"")
    args = _batch_args(tmp_path, vids)

    def selective(argv):
        if "bad" in argv[argv.index("--out") + 1]:
            raise RuntimeError("boom")
        return _fake_track_ok(argv)

    monkeypatch.setattr(rp.tl, "main", selective)
    # discover_videos sorts -> bad, ok1, ok2
    records = rp.run_tracking_batch(rp.discover_videos(vids), work=args.out /
                                    "_work", args=args)
    assert [r["status"] for r in records] == ["failed", "done", "done"]
    rp.write_batch_index(args.out, records)
    index = json.loads(
        (args.out / "metadata" / "batch_index.json").read_text())
    assert index["n_done"] == 2 and index["n_failed"] == 1


def test_batch_fail_fast_aborts(tmp_path, monkeypatch):
    vids = tmp_path / "vids"
    vids.mkdir()
    for n in ("bad.mp4", "ok.mp4"):
        (vids / n).write_bytes(b"")
    args = _batch_args(tmp_path, vids, fail_fast=True)
    monkeypatch.setattr(rp.tl, "main", _fake_track_boom)
    records = rp.run_tracking_batch(rp.discover_videos(vids),
                                    args.out / "_work", args)
    assert len(records) == 1 and records[0]["status"] == "failed"


def test_preflight_blocks_on_missing_models(tmp_path):
    args = rp.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--videos", str(tmp_path)])
    assert args.petri_model is None
    assert rp.preflight([], args) == 2   # petri + larva both missing


def test_preflight_passes_with_models_and_no_videos(tmp_path):
    args = _batch_args(tmp_path, tmp_path)
    assert rp.preflight([], args) == 0


def test_gather_provenance(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "kinematics").mkdir()
    (tmp_path / "kinematics" / "processing_provenance.jsonl").write_text(
        '{"step":"kin"}\n')
    (tmp_path / "qc").mkdir()
    (tmp_path / "qc" / "processing_provenance.jsonl").write_text(
        '{"step":"qc"}\n')
    rp.gather_provenance(tmp_path)
    merged = (tmp_path / "metadata" /
              "processing_provenance.jsonl").read_text().splitlines()
    assert len(merged) == 2


# --- end-to-end (skip tracking) -------------------------------------------- #
def _make_combined_tables(out):
    import pandas as pd
    (out / "petri").mkdir(parents=True, exist_ok=True)
    (out / "larva").mkdir(parents=True, exist_ok=True)
    track_rows, petri_rows = [], []
    n = 60
    for i in range(n):
        for dish in (1, 2):
            cx = 200.0 + (i % 20)
            track_rows.append({
                "dataset_id": "wormstress_tmolitor_v1", "experiment_id": "exp1",
                "video_id": "v1", "frame_idx": i, "time_s": i / 10.0,
                "dish_id": dish, "larva_track_id": dish * 1000 + 1,
                "valid_detection": True, "larva_conf": 0.8,
                "cx_global_raw": cx, "cy_global_raw": 200.0,
                "x_norm_raw": (cx - 200.0) / 100.0, "y_norm_raw": 0.0,
                "inside_petri": True, "mask_area_px": 100.0,
                "mask_area_norm_petri": 0.003, "mask_aspect_ratio": 2.0,
                "mask_eccentricity": 0.8, "mask_compactness": 0.5,
                "track_switch": False,
            })
            petri_rows.append({
                "dataset_id": "wormstress_tmolitor_v1", "experiment_id": "exp1",
                "video_id": "v1", "frame_idx": i, "time_s": i / 10.0,
                "dish_id": dish, "petri_cx": 200.0, "petri_cy": 200.0,
                "petri_radius": 100.0, "petri_valid": True, "petri_conf": 0.9,
            })
    pd.DataFrame(track_rows).to_csv(out / "larva" / "larva_tracks.csv",
                                    index=False)
    pd.DataFrame(petri_rows).to_csv(out / "petri" / "petri_tracks.csv",
                                    index=False)


def test_end_to_end_skip_tracking(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    pytest.importorskip("yaml")
    out = tmp_path / "dataset"
    _make_combined_tables(out)

    rc = rp.main(["--out", str(out), "--skip-tracking",
                  "--petri-diameter-mm", "90", "--window-size-s", "3.0",
                  "--heatmap-bins", "30"])
    assert rc == 0

    # every stage produced its artifact in the FAIR layout
    assert (out / "kinematics" / "larva_kinematics.csv").is_file()
    assert (out / "heatmaps" / "npy" / "v1_dish01_counts.npy").is_file()
    assert (out / "features" / "features_per_window.csv").is_file()
    assert (out / "features" / "features_per_video.csv").is_file()
    # divergence post-processor ran (no metadata -> audit with no_metadata)
    assert (out / "features" / "divergence_from_control.csv").is_file()
    assert (out / "qc" / "qc_summary.csv").is_file()
    assert (out / "data_dictionary.csv").is_file()
    assert (out / "README.md").is_file()
    assert (out / "processing_config.yaml").is_file()
    # provenance from sub-steps merged centrally
    prov = (out / "metadata" / "processing_provenance.jsonl")
    assert prov.is_file() and len(prov.read_text().splitlines()) >= 3


def test_end_to_end_divergence_with_metadata(tmp_path):
    import pandas as pd
    pytest.importorskip("scipy")
    pytest.importorskip("yaml")
    out = tmp_path / "dataset"
    _make_combined_tables(out)
    # dish 1 = control, dish 2 = heat; pooled control reference exists
    meta = pd.DataFrame([
        {"video_id": "v1", "dish_id": 1, "treatment": "control",
         "experiment_id": "exp1"},
        {"video_id": "v1", "dish_id": 2, "treatment": "heat",
         "experiment_id": "exp1"},
    ])
    meta_path = tmp_path / "experiment_metadata.csv"
    meta.to_csv(meta_path, index=False)

    rc = rp.main(["--out", str(out), "--skip-tracking",
                  "--petri-diameter-mm", "90", "--window-size-s", "3.0",
                  "--heatmap-bins", "30", "--metadata", str(meta_path)])
    assert rc == 0

    audit = pd.read_csv(out / "features" / "divergence_from_control.csv")
    assert (audit["divergence_status"] == "ok").all()
    # treated dish was scored against a real control reference (not NaN)
    vid = pd.read_csv(out / "features" / "features_per_video.csv")
    treated = vid[vid["dish_id"] == 2]["js_divergence_from_control"]
    assert treated.notna().all()
    # the metadata-gated aggregate step ran (heat contrasted vs control)
    contrasts = pd.read_csv(out / "aggregate" / "treatment_contrasts.csv")
    assert (contrasts["condition"] == "heat").any()
    assert "feature_group" in contrasts.columns


def test_aggregate_skipped_without_metadata(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    pytest.importorskip("yaml")
    out = tmp_path / "dataset"
    _make_combined_tables(out)
    rc = rp.main(["--out", str(out), "--skip-tracking",
                  "--petri-diameter-mm", "90", "--window-size-s", "3.0",
                  "--heatmap-bins", "30"])     # no --metadata
    assert rc == 0
    assert not (out / "aggregate" / "treatment_contrasts.csv").exists()
