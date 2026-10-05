"""Video-slot mapping and map-aligned mosaics, independent of learned models."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.geometry.coordinates import CoordinateConverter
from src.geometry.local_panorama import (
    LocalPanoramaLayout,
    load_local_panorama_metadata,
    render_local_panorama,
)


def database():
    return SimpleNamespace(
        converter=CoordinateConverter("LOCAL"),
        metadata={"frame_width": 24, "frame_height": 24, "frame_step": 2},
        frame_valid=np.array([True, False, True]),
        frame_affine=np.array(
            [[[1, 0, 0], [0, -1, 24]], [[1, 0, 24], [0, -1, 24]], [[1, 0, 48], [0, -1, 24]]],
            dtype=float,
        ),
    )


@pytest.fixture
def reference_video(tmp_path):
    path = tmp_path / "reference.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (24, 24))
    assert writer.isOpened()
    for i in range(5):
        frame = np.full((24, 24, 3), (255, 0, 0), dtype=np.uint8)
        if i == 0:
            frame[:] = (0, 0, 255)
            frame[12:] = 0  # black is valid content, not transparency
        if i == 4:
            frame[:] = (0, 255, 0)
        writer.write(frame)
    writer.release()
    return path


def test_video_slots_geometry_alpha_and_cancellation(reference_video):
    db = database()
    layout = LocalPanoramaLayout.from_database(db, max_size=144)
    assert layout.canvas_size == (144, 48)
    assert layout.corners_yx == [[24, 0], [24, 72], [0, 72], [0, 0]]
    # A concurrent database change must not alter the worker's geometry snapshot.
    db.frame_affine[:] = 0
    image = render_local_panorama(reference_video, layout)
    assert image.shape == (48, 144, 4)
    assert image[10, 20, 2] > 220  # reference frame 0, top/red
    assert image[38, 20, :3].max() < 20  # reference frame 0, bottom/black
    assert image[38, 20, 3] == 255
    assert image[20, 120, 1] > 220  # DB slot 2 -> video frame 4
    assert image[20, 72, 3] == 0  # invalid slot remains a transparent gap
    assert render_local_panorama(reference_video, layout, running=lambda: False) is None


def test_worker_saves_registered_png_and_rejects_wrong_map(reference_video, tmp_path):
    from src.workers.local_panorama_worker import LocalPanoramaWorker

    layout = LocalPanoramaLayout.from_database(database(), max_size=144)
    output = tmp_path / "panoramas" / "map.png"
    worker = LocalPanoramaWorker(reference_video, output, layout)
    completed, errors = [], []
    worker.completed.connect(completed.append)
    worker.error.connect(errors.append)
    worker.run()
    assert not errors and completed == [str(output)]
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED).shape == (48, 144, 4)
    assert load_local_panorama_metadata(output, layout) == layout.corners_yx
    assert (
        json.loads(output.with_suffix(".png.json").read_text())["coordinate_units"] == "arbitrary"
    )
    changed = database()
    changed.frame_affine[2, 0, 2] += 1
    with pytest.raises(ValueError, match="версії"):
        load_local_panorama_metadata(output, LocalPanoramaLayout.from_database(changed))


def test_wrong_video_is_not_painted_with_reference_coordinates(reference_video, tmp_path):
    from src.workers.local_panorama_worker import LocalPanoramaWorker

    worker = LocalPanoramaWorker(
        reference_video,
        tmp_path / "bad.png",
        LocalPanoramaLayout.from_database(database()),
        "wrong-hash",
    )
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert len(errors) == 1 and "не відповідає" in errors[0]
    assert not (tmp_path / "bad.png").exists()


def test_missing_sampling_metadata_is_not_guessed():
    db = database()
    del db.metadata["frame_step"]
    with pytest.raises(ValueError, match="frame_step"):
        LocalPanoramaLayout.from_database(db)


def test_gui_generation_and_reopen_use_saved_geometry(reference_video, tmp_path, monkeypatch):
    from unittest.mock import Mock

    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
    from src.gui.mixins import panorama_mixin as ui

    class Harness(ui.PanoramaMixin):
        def _get_current_source_id(self):
            return "fpv"

        def _source_config(self, sid):
            return SimpleNamespace(video_path=str(reference_video))

        def _layer_ops_busy(self, action):
            return False

    window = Harness()
    window.database = database()
    window.database.db_path = str(tmp_path / "database.h5")
    window.database.is_propagated = True
    window.calibration = MultiAnchorCalibration(CoordinateConverter("LOCAL"))
    window.project_manager = SimpleNamespace(is_loaded=False)
    window.control_panel = Mock()
    window.status_bar = Mock()
    window.map_widget = Mock()
    window.on_db_progress = Mock()
    window.repaint = Mock()
    window._localize_panorama_corners = Mock(side_effect=AssertionError("No matching is needed"))
    output = tmp_path / "panorama.png"
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *a: (str(output), "PNG (*.png)"))
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *a: (str(output), "PNG (*.png)"))
    errors = Mock()
    monkeypatch.setattr(ui.QMessageBox, "critical", errors)
    monkeypatch.setattr(ui.QMessageBox, "warning", errors)

    def synchronous_start(worker):
        worker.run()
        worker.finished.emit()

    monkeypatch.setattr(ui.LocalPanoramaWorker, "start", synchronous_start)
    window.on_generate_panorama()
    errors.assert_not_called()
    assert output.exists()
    payload = window.map_widget.set_panorama_overlay.call_args.args
    assert payload[0].startswith("data:image/png;base64,")
    np.testing.assert_allclose(
        np.array(payload[1:]).reshape(4, 2),
        LocalPanoramaLayout.from_database(window.database).corners_yx,
    )
    window.map_widget.reset_mock()
    window.on_show_panorama()
    errors.assert_not_called()
    window.map_widget.set_panorama_overlay.assert_called_once()
    window._localize_panorama_corners.assert_not_called()
