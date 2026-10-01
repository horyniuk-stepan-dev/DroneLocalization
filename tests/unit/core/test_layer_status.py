import json
from types import SimpleNamespace

import numpy as np

from src.core.layer_status import (
    CALIBRATION_OWNER_KEY,
    LayerState,
    compute_layer_status,
    find_layer_conflicts,
    find_path_conflicts,
    propagation_matches_anchors,
)


def _anchor(fid, shift=0.0):
    m = np.array([[0.5, 0.0, 1000.0 + shift], [0.0, -0.5, 2000.0]], dtype=np.float64)
    return SimpleNamespace(frame_id=fid, affine_matrix=m)


def _cal(*anchors, owner=None):
    extra = {CALIBRATION_OWNER_KEY: owner} if owner else {}
    return SimpleNamespace(anchors=list(anchors), extra_metadata=extra)


def _db(n=10, propagated=False, valid=None, anchors_json=None):
    fv = None
    if propagated:
        fv = np.zeros(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return SimpleNamespace(
        get_num_frames=lambda: n,
        is_propagated=propagated,
        frame_valid=fv,
        propagation_anchors_json=anchors_json,
    )


def _src(sid="layer", **kw):
    d = {
        "source_id": sid,
        "area_id": "area_main",
        "enabled": True,
        "database_file": f"sources/{sid}/database.h5",
        "calibration_file": f"sources/{sid}/calibration.json",
    }
    d.update(kw)
    return d


def _json(*anchors):
    return json.dumps(
        [{"frame_id": a.frame_id, "affine_matrix": a.affine_matrix.tolist()} for a in anchors]
    )


def test_state_progression_follows_live_objects(tmp_path):
    src = _src()
    assert compute_layer_status(src, tmp_path).state is LayerState.NO_DB

    db_path = tmp_path / "sources/layer/database.h5"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"x")
    # File on disk but the manager did not load it (schema mismatch, error).
    assert compute_layer_status(src, tmp_path).state is LayerState.DB_NOT_LOADED

    db = _db()
    cal = _cal()
    assert compute_layer_status(src, tmp_path, database=db, calibration=cal).state is (
        LayerState.NO_CALIBRATION
    )

    # First anchor: status changes immediately, without any file check.
    cal.anchors.append(_anchor(3))
    status = compute_layer_status(src, tmp_path, database=db, calibration=cal)
    assert status.state is LayerState.NOT_PROPAGATED
    assert status.anchors_text == "1"

    # Propagation reloads the same loader object in place.
    db.is_propagated = True
    db.frame_valid = np.array([True] * 7 + [False] * 3)
    db.propagation_anchors_json = _json(_anchor(3))
    status = compute_layer_status(src, tmp_path, database=db, calibration=cal)
    assert status.state is LayerState.READY
    assert status.gps_text == "7/10"


def test_anchor_edit_after_propagation_is_stale():
    db = _db(propagated=True, anchors_json=_json(_anchor(3)))
    cal = _cal(_anchor(3, shift=5.0))
    assert compute_layer_status(_src(), None, database=db, calibration=cal).state is (
        LayerState.STALE
    )
    cal.anchors.append(_anchor(8))
    assert compute_layer_status(_src(), None, database=db, calibration=cal).state is (
        LayerState.STALE
    )


def test_propagation_without_anchor_record_is_not_flagged():
    # Older propagation (no anchors_json) or a script-built DB: unknown, not stale.
    db = _db(propagated=True, anchors_json=None)
    status = compute_layer_status(_src(), None, database=db, calibration=_cal(_anchor(1)))
    assert status.state is LayerState.READY
    assert propagation_matches_anchors(None, []) is None
    assert propagation_matches_anchors("{broken", []) is None


def test_propagated_db_without_calibration_file_is_ready():
    db = _db(propagated=True, valid=[True, False])
    status = compute_layer_status(_src(), None, database=db, calibration=_cal())
    assert status.state is LayerState.READY


def test_disabled_wins():
    status = compute_layer_status(_src(enabled=False), None, database=_db(), calibration=None)
    assert status.state is LayerState.DISABLED
    assert status.anchors_text == "—"


def test_path_conflicts_detect_shared_files_and_folders(tmp_path):
    ok = [_src("a"), _src("b")]
    assert find_path_conflicts(ok, tmp_path) == []

    same_cal = [_src("a"), _src("b", calibration_file="sources/a/calibration.json")]
    kinds = [c.message for c in find_path_conflicts(same_cal, tmp_path)]
    assert len(kinds) == 1 and "calibration_file" in kinds[0]

    # Shared folder means shared vectors.lance even with distinct .h5 names.
    same_dir = [_src("a"), _src("b", database_file="sources/a/database_b.h5")]
    msgs = [c.message for c in find_path_conflicts(same_dir, tmp_path)]
    assert len(msgs) == 1 and "database folder" in msgs[0]

    # Same DB file is reported once, not again as a shared folder.
    same_db = [_src("a"), _src("b", database_file="sources/a/database.h5")]
    msgs = [c.message for c in find_path_conflicts(same_db, tmp_path)]
    assert len(msgs) == 1 and "database_file" in msgs[0]

    # Windows paths are case-insensitive.
    case = [_src("a"), _src("b", calibration_file="SOURCES/A/calibration.json")]
    assert len(find_path_conflicts(case, tmp_path)) == 1


def test_layer_conflicts_owner_range_and_duplicates(tmp_path):
    sources = [_src("main"), _src("low")]
    cals = {
        "main": _cal(_anchor(0), _anchor(900), owner="main"),
        "low": _cal(_anchor(0), _anchor(900), owner="main"),
    }
    conflicts = find_layer_conflicts(sources, tmp_path, cals, {"main": 1688, "low": 75})
    kinds = sorted(c.kind for c in conflicts)
    assert kinds == ["duplicate", "owner", "range"]
    rng = next(c for c in conflicts if c.kind == "range")
    assert rng.source_ids == ("low",) and "[900]" in rng.message


def test_clean_project_has_no_conflicts(tmp_path):
    sources = [_src("main"), _src("low")]
    cals = {
        "main": _cal(_anchor(0), _anchor(900), owner="main"),
        "low": _cal(_anchor(0, shift=1.0), _anchor(40), owner="low"),
        "empty": _cal(),
    }
    assert find_layer_conflicts(sources, tmp_path, cals, {"main": 1688, "low": 75}) == []
