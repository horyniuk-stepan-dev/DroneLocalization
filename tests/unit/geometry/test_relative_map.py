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
        attrs = db.db_file["calibration"].attrs
        assert attrs["optimizer"] == "pose_graph_lm+affine_point_irls"
        assert attrs["frame_rmse_kind"] == "shared_point_alignment"
        report = json.loads(attrs["local_alignment_json"])
        assert report["frames"] == 3 and report["after_p95_px"] < 1e-3
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


def test_projective_pipeline_roundtrip_preserves_image_geometry(tmp_path, monkeypatch):
    from src.geometry.frame_transform import get_frame_transform, project_points

    path = tmp_path / "projective.h5"
    db = make_database(path, n=4)
    truth = {
        0: np.eye(3),
        1: np.array([[1.0, 0.08, 20], [0.02, 1.01, -8], [0.0002, -0.0001, 1]]),
        2: np.array([[1.03, -0.03, 40], [-0.01, 0.98, 5], [-0.0001, 0.0002, 1]]),
    }
    world = np.array([[x, y] for x in np.linspace(55, 150, 10) for y in np.linspace(30, 80, 8)])
    features = {
        fid: {"keypoints": project_points(np.linalg.inv(H), world)} for fid, H in truth.items()
    }
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        db,
        MultiAnchorCalibration(CoordinateConverter("LOCAL")),
        matcher,
        config={"propagation": {"high_accuracy": True, "subpixel_refinement": False}},
        error_callback=Mock(),
        completed_callback=Mock(),
    )
    monkeypatch.setattr(pipeline, "_prefetch_features", lambda n: features)
    monkeypatch.setattr(pipeline, "_detect_loop_closures", lambda *a: 0)
    pipeline._propagate()
    pipeline._error_cb.assert_not_called()
    pipeline._completed_cb.assert_called_once()
    np.testing.assert_array_equal(db.frame_valid, [True, True, True, False])
    assert db.db_file["calibration"].attrs["transform_model"] == "projective"
    before = db.frame_homography.copy()
    db.close()
    reopened = DatabaseLoader(str(path))
    try:
        np.testing.assert_array_equal(reopened.frame_homography, before)
        heldout = np.array([[20, 15], [100, 50], [180, 90]])
        gauge = get_frame_transform(reopened, 0)
        for fid in truth:
            H = np.linalg.inv(gauge) @ get_frame_transform(reopened, fid)
            np.testing.assert_allclose(
                project_points(H, heldout), project_points(truth[fid], heldout), atol=0.01
            )
        assert get_frame_transform(reopened, 3) is None
    finally:
        reopened.close()


def test_corrupt_projective_calibration_is_not_silently_read_as_affine(tmp_path):
    path = tmp_path / "corrupt.h5"
    make_database(path, n=1).close()
    with h5py.File(path, "a") as f:
        group = f.create_group("calibration")
        group["frame_affine"] = np.eye(3)[None, :2]
        group["frame_valid"] = [True]
        group["frame_homography"] = np.zeros((1, 3, 3))
    db = DatabaseLoader(str(path))
    try:
        assert not db.is_propagated and db.frame_homography is None
    finally:
        db.close()


def test_projective_geographic_stage_preserves_anchor_and_exports_final_centers(
    tmp_path, monkeypatch
):
    from src.geometry.frame_transform import get_frame_transform, project_points

    path = tmp_path / "geographic.h5"
    db = make_database(path, n=3)
    calibration = MultiAnchorCalibration(CoordinateConverter("WEB_MERCATOR"))
    anchor = np.array([[0.7, 0.0, 3e6], [0.0, -0.7, 6e6]])
    calibration.add_anchor(0, anchor.copy())
    truth = {0: np.eye(3), 1: np.array([[1.0, 0.08, 20], [0.02, 1.01, -8], [0.0002, -0.0001, 1]])}
    grid = np.array([[x, y] for x in np.linspace(55, 150, 10) for y in np.linspace(30, 80, 8)])
    features = {
        fid: {"keypoints": project_points(np.linalg.inv(H), grid)} for fid, H in truth.items()
    }
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        db,
        calibration,
        matcher,
        config={"propagation": {"high_accuracy": True, "subpixel_refinement": False}},
        error_callback=Mock(),
        completed_callback=Mock(),
    )
    monkeypatch.setattr(pipeline, "_prefetch_features", lambda n: features)
    monkeypatch.setattr(pipeline, "_detect_loop_closures", lambda *a: 0)
    try:
        pipeline._propagate()
        pipeline._error_cb.assert_not_called()
        pipeline._completed_cb.assert_called_once()
        np.testing.assert_array_equal(calibration.anchors[0].affine_matrix, anchor)
        np.testing.assert_allclose(db.get_frame_affine(0), anchor)
        np.testing.assert_allclose(get_frame_transform(db, 2)[2], [0, 0, 1])
        exported = json.loads((tmp_path / "geographic_graph.geojson").read_text(encoding="utf-8"))
        point = next(f for f in exported["features"] if f["properties"].get("frame_id") == 1)
        center = project_points(get_frame_transform(db, 1), [[100, 50]])[0]
        lat, lon = calibration.converter.metric_to_gps(*center)
        np.testing.assert_allclose(point["geometry"]["coordinates"], [lon, lat], atol=1e-10)
        edge = next(f for f in exported["features"] if f["geometry"]["type"] == "LineString")
        assert edge["properties"]["residual"] is not None
        assert edge["properties"]["residual_units"] == "original_image_pixels"
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["pruned_orphan", "invalid_models", "diagnostic_horizon"])
def test_geo_refinement_failure_does_not_abort_atomic_save(tmp_path, monkeypatch, failure):
    from src.geometry.pose_graph.optimizer import PoseGraphOptimizer

    db = make_database(tmp_path / f"{failure}.h5", n=3)
    calibration = MultiAnchorCalibration(CoordinateConverter("WEB_MERCATOR"))
    calibration.add_anchor(0, np.array([[0.7, 0.0, 3e6], [0.0, -0.7, 6e6]]))
    grid = np.array([[x, y] for x in np.linspace(30, 170, 10) for y in np.linspace(20, 80, 8)])
    features = {i: {"keypoints": grid - [10 * i, 0]} for i in range(3)}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        db,
        calibration,
        matcher,
        config={"propagation": {"high_accuracy": True, "subpixel_refinement": False}},
        error_callback=Mock(),
        completed_callback=Mock(),
    )
    monkeypatch.setattr(pipeline, "_prefetch_features", lambda n: features)
    monkeypatch.setattr(pipeline, "_detect_loop_closures", lambda *a: 0)
    monkeypatch.setattr(pipeline, "_local_overlap_links", lambda *a: [])
    if failure == "pruned_orphan":
        original = PoseGraphOptimizer.optimize

        def prune(self, *args, **kwargs):
            results = original(self, *args, **kwargs)
            self._edges[:] = [e for e in self.edges if 2 not in (e.from_id, e.to_id)]
            return results

        monkeypatch.setattr(PoseGraphOptimizer, "optimize", prune)
    elif failure == "invalid_models":

        def reject(*args, **kwargs):
            raise ValueError("both relative models rejected")

        monkeypatch.setattr("src.geometry.relative_projective.refine_projective_map", reject)
    else:

        def unsafe(*args, **kwargs):
            raise ValueError("center prediction outside reference chart")

        monkeypatch.setattr("src.geometry.frame_transform.project_frame_points", unsafe)
    try:
        pipeline._propagate()
        pipeline._error_cb.assert_not_called()
        pipeline._completed_cb.assert_called_once()
        assert db.is_propagated and db.frame_valid.all()
        if failure == "pruned_orphan":
            assert pipeline._relative_alignment_report["unrefined_frame_ids"] == [2]
            assert not db.is_frame_georef_supported(2)
            np.testing.assert_array_equal(db.frame_homography[2, 2], [0, 0, 1])
        elif failure == "invalid_models":
            assert db.frame_homography is None
            assert pipeline._relative_alignment_report["status"] == "fallback"
            assert db.db_file["calibration"].attrs["optimizer"] == "pose_graph_lm"
        else:
            assert db.frame_homography is not None
            assert np.isfinite(db.db_file["calibration/frame_disagreement"][:]).all()
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
