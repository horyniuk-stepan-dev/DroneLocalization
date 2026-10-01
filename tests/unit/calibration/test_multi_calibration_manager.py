import json

import numpy as np

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.calibration.multi_calibration_manager import (
    MultiCalibrationManager,
    save_layer_calibration,
)
from src.core.layer_status import CALIBRATION_OWNER_KEY
from src.core.project_video_source import ProjectVideoSource


def _src(sid, enabled=True):
    return ProjectVideoSource(
        source_id=sid,
        area_id="area_main",
        video_path=f"{sid}.mp4",
        database_file=f"sources/{sid}/database.h5",
        calibration_file=f"sources/{sid}/calibration.json",
        enabled=enabled,
    )


def _write_cal(project_dir, sid, frame_ids):
    cal = MultiAnchorCalibration()
    for fid in frame_ids:
        m = np.array([[0.5, 0.0, 100.0 + fid], [0.0, -0.5, 200.0]], dtype=np.float64)
        cal.add_anchor(fid, m)
    save_layer_calibration(cal, project_dir / f"sources/{sid}/calibration.json", sid)


def test_disabled_layer_is_loaded_from_disk_not_created_empty(tmp_path):
    _write_cal(tmp_path, "low", [0, 10, 20])
    low = _src("low", enabled=False)
    manager = MultiCalibrationManager()
    manager.load_all([_src("main"), low], tmp_path)
    assert "low" not in manager

    # Old code path: get() returned an empty calibration whose first save
    # overwrote the 3 anchors on disk.
    cal = manager.get_or_load(low, tmp_path)
    assert [a.frame_id for a in cal.anchors] == [0, 10, 20]
    assert manager.get_or_load(low, tmp_path) is cal  # cached, same live object


def test_unreadable_file_is_not_cached_as_empty(tmp_path):
    path = tmp_path / "sources/main/calibration.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ not json", encoding="utf-8")
    manager = MultiCalibrationManager()
    manager.load_all([_src("main")], tmp_path)
    assert "main" not in manager
    assert path.read_text(encoding="utf-8") == "{ not json"


def test_save_stamps_owner_and_load_preserves_it(tmp_path):
    _write_cal(tmp_path, "main", [5])
    raw = json.loads((tmp_path / "sources/main/calibration.json").read_text(encoding="utf-8"))
    assert raw[CALIBRATION_OWNER_KEY] == "main"

    manager = MultiCalibrationManager()
    manager.load_all([_src("main")], tmp_path)
    assert manager.get("main").extra_metadata[CALIBRATION_OWNER_KEY] == "main"


def test_set_and_discard(tmp_path):
    manager = MultiCalibrationManager()
    replacement = MultiAnchorCalibration()
    manager.set("low", replacement)
    assert manager.get("low") is replacement
    manager.discard("low")
    assert "low" not in manager
