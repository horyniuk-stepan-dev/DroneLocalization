"""Relative FPV maps: gauge, persistence, disconnected footage and output semantics."""

import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.database.database_loader import DatabaseLoader
from src.geometry.coordinates import CoordinateConverter
from src.geometry.relative_map import normalize_relative_map
from src.workers.propagation_pipeline import PropagationPipeline


def test_normalization_preserves_geometry_and_fits_footprints():
    a = np.array([[2.0, 0.0, -400.0], [0.0, -2.0, 200.0]])
    b = a.copy()
    b[:, 2] += [200.0, 40.0]
    result, transform = normalize_relative_map({0: a, 1: b}, 200, 100)
    corners = np.array([[0, 0, 1], [200, 0, 1], [200, 100, 1], [0, 100, 1]])
    points = np.concatenate([corners @ matrix.T for matrix in result.values()])
    assert points.min() >= -1e-10 and points.max() <= 100 + 1e-10
    assert np.max(np.ptp(points, axis=0)) == pytest.approx(100)
    np.testing.assert_allclose(result[1][:, 2] - result[0][:, 2], np.array([200, 40]) / 6)
    assert transform[0, 0] == transform[1, 1]
    assert np.linalg.det(result[0][:, :2]) < 0  # image Y down -> map Y up


def test_local_converter_roundtrip_without_projection():
    converter = CoordinateConverter.from_metadata(CoordinateConverter("LOCAL").export_metadata())
    assert converter.mode == "LOCAL" and converter.is_initialized
    assert converter.reference_gps is None
    assert converter.metric_to_gps(123, -47) == (-47, 123)
    assert converter.gps_to_metric(-47, 123) == (123, -47)
    np.testing.assert_array_equal(converter.metric_to_gps_array([123], [-47]), [[-47], [123]])


def make_database(path, n=4):
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs.update(frame_width=200, frame_height=100, num_frames=n)
        desc = f.create_group("global_descriptors")
        desc.create_dataset("descriptors", data=np.eye(n, dtype=np.float32))
        desc.create_dataset("frame_poses", data=np.repeat(np.eye(3)[None], n, axis=0))
    return DatabaseLoader(str(path))


def run_pipeline(db, monkeypatch, connected=True):
    cal = MultiAnchorCalibration(CoordinateConverter("LOCAL"))
    pipeline = PropagationPipeline(
        db,
        cal,
        Mock(),
        config={"graph_optimization": {"terrain_scale_prior": True, "export_geojson": True}},
        completed_callback=Mock(),
        error_callback=Mock(),
    )
    grid = np.array(
        [[x, y] for x in np.linspace(60, 170, 6) for y in np.linspace(15, 65, 6)], dtype=np.float32
    )
    features = {i: {"keypoints": grid - [20 * i, -10 * i], "frame_id": i} for i in range(4)}
    monkeypatch.setattr(pipeline, "_prefetch_features", lambda n: features)

    def match(a, b):
        if not connected or max(a["frame_id"], b["frame_id"]) >= 3:
            return np.empty((0, 2)), np.empty((0, 2))
        return a["keypoints"], b["keypoints"]

    pipeline.matcher.match.side_effect = match
    monkeypatch.setattr(pipeline, "_detect_loop_closures", lambda *args: 0)
    pipeline._propagate()
    return pipeline


def test_relative_graph_save_reload_and_no_fabricated_gap(tmp_path, monkeypatch):
    path = tmp_path / "fpv.h5"
    db = make_database(path)
    try:
        pipeline = run_pipeline(db, monkeypatch)
        pipeline._error_cb.assert_not_called()
        pipeline._completed_cb.assert_called_once()
        assert not pipeline.calibration.anchors
        assert db.converter.mode == "LOCAL" and db.is_propagated
        np.testing.assert_array_equal(db.frame_valid, [True, True, True, False])
        assert db.get_frame_affine(3) is None
        assert db.frame_gps is None and db.spatial_index is None
        center = np.array([100, 50, 1])
        positions = np.array([db.get_frame_affine(i) @ center for i in range(3)])
        np.testing.assert_allclose(
            np.diff(positions, axis=0), [[20 / 2.4, 10 / 2.4]] * 2, atol=1e-4
        )
        assert db.db_file["calibration"].attrs["num_anchors"] == 0
        assert json.loads(db.db_file["calibration"].attrs["anchors_json"]) == []
        before = db.frame_affine.copy()
    finally:
        db.close()
    reopened = DatabaseLoader(str(path))
    try:
        assert reopened.converter.mode == "LOCAL"
        np.testing.assert_allclose(reopened.frame_affine, before)
        assert not list(tmp_path.glob("*.geojson"))
    finally:
        reopened.close()


def test_unconnected_video_does_not_commit_a_map(tmp_path, monkeypatch):
    db = make_database(tmp_path / "empty.h5")
    try:
        pipeline = run_pipeline(db, monkeypatch, connected=False)
        pipeline._error_cb.assert_called_once()
        pipeline._completed_cb.assert_not_called()
        assert "calibration" not in db.db_file
    finally:
        db.close()


def test_localization_returns_xy_and_uses_selected_local_map_only(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "fixtures"))
    from localizer_fakes import FakeDatabase, FakeMatcher, _CombinedExtractor, synthetic_frame

    from src.localization.localizer import Localizer

    db = FakeDatabase()
    db.converter = CoordinateConverter("LOCAL")
    loc = Localizer(
        db,
        _CombinedExtractor(),
        FakeMatcher(),
        MultiAnchorCalibration(db.converter),
        config={"tracking": {"smoother_enabled": True}},
        ref_frame_width=640,
        ref_frame_height=480,
        db_manager=Mock(),
        calib_manager=Mock(),
    )
    result = loc.localize_frame(synthetic_frame(1))
    assert result["success"], result
    assert loc.db_manager is None and loc._smoother is None
    assert result["coordinate_kind"] == "local_planar"
    assert result["coordinate_units"] == "arbitrary"
    assert result["x"] == result["lon"] and result["y"] == result["lat"]
    # A metric speed limit must not suppress motion in arbitrary units.
    loc.outlier_detector.is_outlier = Mock(return_value=True)
    result = loc.localize_frame(synthetic_frame(1))
    assert result["success"]
    loc.outlier_detector.is_outlier.assert_not_called()
    flow = loc.localize_optical_flow(5, 3, dt=1.0, rot_width=640, rot_height=480)
    assert flow["success"] and flow["coordinate_kind"] == "local_planar"


def test_network_local_coordinates_are_never_gps():
    from config import NetworkApiConfig
    from src.network.coordinates_broker import CoordinatesBroker

    broker = CoordinatesBroker(NetworkApiConfig(enabled=False))
    broker.on_location_found(48.0, 26.0, 0.9, 80)
    broker.set_coordinate_mode("LOCAL")
    assert broker.get_last_position() is None and not broker.get_history()
    broker.on_location_found(33.0, 120.0, 0.9, 80)
    pos = broker.get_last_position()
    assert pos["x"] == 120 and pos["y"] == 33
    assert "lat" not in pos and "lon" not in pos
    broker.on_objects_gps_updated(
        [SimpleNamespace(track_id=1, class_name="car", lat=33.0, lon=120.0, confidence=0.8)]
    )
    obj = broker.get_last_objects()[0]
    assert obj["x"] == 120 and "lat" not in obj and obj["coordinate_kind"] == "local_planar"
    broker.set_coordinate_mode("WEB_MERCATOR")
    broker.on_location_found(48.0, 26.0, 0.9, 80)
    assert broker.get_last_position()["lat"] == 48


@pytest.mark.parametrize("objects", [False, True])
def test_exports_label_xy_and_refuse_geographic_formats(tmp_path, objects):
    from src.core.export_results import ResultExporter

    rows = [
        dict(
            x=80.0,
            y=30.0,
            lat=30.0,
            lon=80.0,
            coordinate_kind="local_planar",
            coordinate_units="arbitrary",
        )
    ]
    path = tmp_path / "local.csv"
    export = ResultExporter.export_objects_csv if objects else ResultExporter.export_csv
    export(rows, str(path))
    with path.open(encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))[0]
    assert result["x"] == "80.0" and "lat" not in result and "lon" not in result
    export = ResultExporter.export_objects_geojson if objects else ResultExporter.export_geojson
    with pytest.raises(ValueError, match="no GPS"):
        export(rows, str(tmp_path / "bad.geojson"))
    with pytest.raises(ValueError, match="no GPS"):
        ResultExporter.export_kml(rows, str(tmp_path / "bad.kml"))
