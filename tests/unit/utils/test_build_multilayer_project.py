"""Headless multi-layer project builder: contracts and project.json (heavy steps stubbed)."""

import json
import sys
import types

import h5py
import numpy as np
import pytest

import scripts.build_multilayer_project as bmp
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.core.layer_status import CALIBRATION_OWNER_KEY, find_path_conflicts
from src.core.project import ProjectManager


def _sim_run(tmp_path, name, anchor_slots, status="complete"):
    run = tmp_path / name
    run.mkdir()
    (run / "video.mp4").write_bytes(b"v")
    (run / "manifest.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    cal = MultiAnchorCalibration()
    for fid in anchor_slots:
        cal.add_anchor(fid, np.array([[0.5, 0, 2905000.0 + fid], [0, -0.5, 6175000.0]]))
    cal.save(str(run / "calibration.json"))
    return run


def _fake_db(path, keyframes, n=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        kp = np.zeros(n, np.int16)
        kp[list(keyframes)] = 100
        f.create_group("local_features").create_dataset("kp_counts", data=kp)


def test_parse_args_rejects_case_duplicates_and_bad_altitude(tmp_path):
    with pytest.raises(SystemExit):
        bmp.parse_args(
            ["--project", str(tmp_path), "--layer", "Low", "500", "a", "--layer", "low", "600", "b"]
        )
    with pytest.raises(SystemExit):
        bmp.parse_args(["--project", str(tmp_path), "--layer", "x", "-5", "a"])


def test_incomplete_recording_is_refused(tmp_path):
    run = _sim_run(tmp_path, "r", [0], status="running")
    with pytest.raises(bmp.BuildError, match="not 'complete'"):
        bmp.check_recording(bmp.SimulatorRun(run))


def test_anchor_contract(tmp_path):
    run = _sim_run(tmp_path, "r", [0, 4, 8])
    db = tmp_path / "db" / "database.h5"
    _fake_db(db, keyframes=[0, 4, 8, 9])
    assert bmp.check_anchor_contract(run / "calibration.json", db) == 3
    _fake_db(db, keyframes=[0, 8])
    with pytest.raises(bmp.BuildError, match=r"\[4\]"):
        bmp.check_anchor_contract(run / "calibration.json", db)


def test_main_builds_every_layer_and_a_loadable_project(tmp_path, monkeypatch):
    runs = {
        "main": _sim_run(tmp_path, "r1000", [0, 5]),
        "2": _sim_run(tmp_path, "r2000", [0, 3]),
    }
    built, propagated = [], []
    monkeypatch.setattr(
        bmp,
        "build_database",
        lambda video, db, config, manager: (built.append(db), _fake_db(db, range(10))),
    )
    monkeypatch.setattr(
        bmp, "propagate", lambda db, cal, config, manager: propagated.append((db, cal))
    )
    stub = types.ModuleType("src.models.model_manager")
    stub.ModelManager = lambda config: object()
    monkeypatch.setitem(sys.modules, "src.models.model_manager", stub)

    project = tmp_path / "proj"
    argv = [
        "--project",
        str(project),
        "--layer",
        "main",
        "1000",
        str(runs["main"]),
        "--layer",
        "2",
        "2000",
        str(runs["2"]),
    ]
    assert bmp.main(argv) == 0
    assert len(built) == 2 and len(propagated) == 2

    pm = ProjectManager()
    assert pm.load_project(str(project))
    sources = pm.settings.source_configs()
    assert [s.source_id for s in sources] == ["main", "2"]
    assert {s.area_id for s in sources} == {"area_main"}
    assert find_path_conflicts(sources, project) == []
    owner = json.loads((project / "sources/2/calibration.json").read_text(encoding="utf-8"))
    assert owner[CALIBRATION_OWNER_KEY] == "2"

    # A second run without --resume must not touch the finished project.
    assert bmp.main(argv) == 1
    # --resume skips the finished layers entirely.
    built.clear()
    propagated.clear()
    with h5py.File(project / "sources/main/database.h5", "a") as f:
        f.create_group("calibration").create_dataset("frame_affine", data=np.zeros((10, 2, 3)))
    assert bmp.main([*argv, "--resume"]) == 0
    assert built == []
    assert [db.parent.name for db, _ in propagated] == ["2"]


def _write_gt(run, slots):
    records = [
        {"slot": s, "affine": [[0.5, 0, 2905000.0 + s], [0, -0.5, 6175000.0]], "rmse_m": 0.1 * s}
        for s in slots
    ]
    (run / "ground_truth.json").write_text(json.dumps({"slots": records}), encoding="utf-8")


def test_resnap_moves_missed_anchor_onto_nearest_keyframe_from_ground_truth(tmp_path):
    run = _sim_run(tmp_path, "r", [0, 4, 8, 12])
    _write_gt(run, range(20))
    db = tmp_path / "db" / "database.h5"
    _fake_db(db, keyframes=[0, 5, 8], n=20)
    out = tmp_path / "db" / "resnapped.json"
    moves = bmp.resnap_anchors(run / "calibration.json", db, bmp.SimulatorRun(run), out)
    assert moves == [(4, 5), (12, None)]  # 12 has no keyframe within 3 slots
    assert bmp.check_anchor_contract(out, db) == 3
    cal = MultiAnchorCalibration()
    cal.load(str(out))
    moved = next(a for a in cal.anchors if a.frame_id == 5)
    assert moved.affine_matrix[0, 2] == pytest.approx(2905005.0)
    notes = json.loads(out.read_text(encoding="utf-8"))["anchors"][1]["qa_data"]["notes"]
    assert "re-snapped from slot 4" in notes


def test_resnap_never_stacks_two_anchors_on_one_keyframe(tmp_path):
    run = _sim_run(tmp_path, "r", [4, 6])
    _write_gt(run, range(10))
    db = tmp_path / "db" / "database.h5"
    _fake_db(db, keyframes=[5], n=10)
    out = tmp_path / "db" / "resnapped.json"
    moves = bmp.resnap_anchors(run / "calibration.json", db, bmp.SimulatorRun(run), out)
    assert moves == [(4, 5), (6, None)]
    assert bmp.check_anchor_contract(out, db) == 1


def test_config_overrides_reach_nested_keys_and_check_types():
    from scripts.propagation_sweep import apply_overrides

    config = {
        "graph_optimization": {"isotropy_weight": 200.0},
        "models": {"vlad": {"enabled": False, "vocab_path": None, "pca_dim": 256}},
    }
    result = apply_overrides(
        config,
        {
            "graph_optimization.isotropy_weight": 10,
            "models.vlad.enabled": True,
            "models.vlad.vocab_path": "models/v1.npz",
        },
    )
    assert result["models"]["vlad"] == {
        "enabled": True,
        "vocab_path": "models/v1.npz",
        "pca_dim": 256,
    }
    assert result["graph_optimization"]["isotropy_weight"] == 10
    assert config["models"]["vlad"]["enabled"] is False  # input untouched
    for bad in (
        {"models.vlad.no_such": 1},
        {"models": 1},
        {"models.vlad": {}},
        {"models.vlad.enabled": 1},
        {"models.vlad.pca_dim": "many"},
    ):
        with pytest.raises(ValueError):
            apply_overrides(config, bad)
