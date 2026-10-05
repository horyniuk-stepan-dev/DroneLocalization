"""Exercise the actual offline canvas and its asynchronous Qt bridge."""

from src.gui.widgets.map_widget import MapWidget


def test_relative_button_starts_local_pipeline_without_mutating_calibration(qtbot, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from PyQt6.QtCore import Qt

    import src.gui.main_window as main
    import src.gui.mixins.calibration_mixin as calibration_ui

    monkeypatch.setattr(main, "ModelManager", Mock())
    monkeypatch.setattr(main, "CoordinatesBroker", Mock())
    monkeypatch.setattr(main.MainWindow, "_restore_debug_visibility", lambda self: None)
    monkeypatch.setattr(calibration_ui, "FeatureMatcher", Mock())
    worker_factory = Mock()
    worker_factory.return_value.isRunning.return_value = False
    monkeypatch.setattr(calibration_ui, "CalibrationPropagationWorker", worker_factory)

    class TestWindow(main.MainWindow):
        def closeEvent(self, event):
            # Production closeEvent persists user settings. This isolated GUI
            # test has no real workers/services and must not write user_config.
            event.accept()

    window = TestWindow()
    qtbot.addWidget(window)
    window.database = SimpleNamespace(is_propagated=False, get_num_frames=lambda: 20)
    original_calibration = window.calibration
    qtbot.mouseClick(window.control_panel.btn_relative_map, Qt.MouseButton.LeftButton)
    worker_factory.assert_called_once()
    work_calibration = worker_factory.call_args.kwargs["calibration"]
    assert work_calibration.converter.mode == "LOCAL" and not work_calibration.anchors
    assert window.calibration is original_calibration
    worker_factory.return_value.start.assert_called_once()
    window._propagation_dialog.close()
    window._propagation_dialog = None
    window.database = None


def test_local_map_ready_queue_and_canvas(qtbot, tmp_path):
    widget = MapWidget()
    qtbot.addWidget(widget)
    widget.resize(900, 650)
    widget.set_coordinate_mode("LOCAL")
    widget.show_verification_markers(
        [
            {"lat": 20.0, "lon": 25.0, "label": "0"},
            {"lat": 50.0, "lon": 55.0, "label": "1"},
            {"lat": 70.0, "lon": 75.0, "label": "2"},
        ]
    )
    widget.update_marker(50.0, 55.0)
    widget.add_trajectory_point(20.0, 25.0)
    widget.add_trajectory_point(50.0, 55.0)
    widget.update_fov([(60, 45), (60, 65), (40, 65), (40, 45)])
    widget.show()
    qtbot.waitUntil(lambda: widget._map_ready, timeout=15000)
    seen = []
    widget.page().runJavaScript(
        "JSON.stringify({mode: document.title, markers: verification.length, point: marker, trail: path.length, fov: fov.length, remoteImages: document.images.length})",
        seen.append,
    )
    qtbot.waitUntil(lambda: bool(seen), timeout=5000)
    import json

    state = json.loads(seen[0])
    assert state == {
        "mode": "Локальна карта",
        "markers": 3,
        "point": [55, 50],
        "trail": 2,
        "fov": 4,
        "remoteImages": 0,
    }
    assert not widget._pending_updates
    qtbot.wait(250)  # Chromium compositor finishes its first paint.
    screenshot = tmp_path / "local_map.png"
    assert widget.grab().save(str(screenshot))
    print(f"LOCAL_MAP_SCREENSHOT={screenshot}")


def test_panorama_overlay_affine_render_zoom_toggle_and_reset(qtbot):
    import base64
    import json

    import cv2
    import numpy as np

    widget = MapWidget()
    qtbot.addWidget(widget)
    widget.resize(900, 650)
    widget.set_coordinate_mode("LOCAL")
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[:20, :20] = [0, 0, 255, 255]
    img[20:, 20:] = [0, 255, 0, 255]
    _, buf = cv2.imencode(".png", img)
    uri = "data:image/png;base64," + base64.b64encode(buf).decode()
    # A rotated and sheared affine footprint (transport is y,x).
    widget.set_panorama_overlay(uri, 80, 20, 60, 60, 30, 45, 50, 5)
    widget.show()
    qtbot.waitUntil(lambda: widget._map_ready, timeout=15000)

    def js(expression):
        values = []
        widget.page().runJavaScript(expression, values.append)
        qtbot.waitUntil(lambda: bool(values), timeout=5000)
        return values[0]

    qtbot.waitUntil(lambda: js("panorama !== null"), timeout=5000)
    # Pick non-grid pixels in the red top-left and green bottom-right quarters.
    sample = """JSON.stringify([[26.25,67.5], [38.75,42.5]].map(p => {
        const s = screen(p), d = window.devicePixelRatio || 1;
        return Array.from(ctx.getImageData(Math.round(s[0]*d), Math.round(s[1]*d), 1, 1).data);
    }))"""
    pixels = json.loads(js(sample))
    assert pixels[0][0] > 220 and pixels[0][1] < 30
    assert pixels[1][1] > 220 and pixels[1][0] < 30
    js("span = 180; center = [40, 45]; draw(); true")
    pixels = json.loads(js(sample))
    assert pixels[0][0] > 220 and pixels[1][1] > 220
    js("document.getElementById('panorama-visible').click(); true")
    assert js("showPanorama") is False
    widget.clear_trajectory()
    assert js("panorama !== null") is True
    widget.set_coordinate_mode("LOCAL", reset=True)
    qtbot.waitUntil(lambda: widget._map_ready, timeout=15000)
    assert js("panorama === null") is True
