"""Accuracy checks use known geometry and points not used for optimization."""

from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.geometry.frame_transform import (
    affine_at_center,
    get_frame_transform,
    project_points,
    valid_footprint,
)
from src.geometry.pair_refinement import pair_quality, refine_subpixel
from src.geometry.point_spread import inlier_spread
from src.geometry.relative_alignment import PointLink
from src.geometry.relative_map import normalize_relative_map
from src.geometry.relative_projective import initialize_from_pairs, refine_projective_map
from src.geometry.transformations import GeometryTransforms


def scene():
    truth = {
        0: np.eye(3),
        1: np.array([[1.0, 0.08, 20], [0.02, 1.01, -8], [0.0002, -0.0001, 1]]),
        2: np.array([[1.03, -0.03, 40], [-0.01, 0.98, 5], [-0.0001, 0.0002, 1]]),
    }
    rng = np.random.default_rng(7)
    links = []
    for first, second in [(0, 1), (1, 2), (0, 2)]:
        world = rng.uniform([55, 30], [150, 80], (80, 2))
        p = project_points(np.linalg.inv(truth[first]), world)
        q = project_points(np.linalg.inv(truth[second]), world)
        links.append(PointLink(first, second, p, q))
    initial = {fid: H.copy() for fid, H in truth.items()}
    for fid in (1, 2):
        initial[fid][2, :2] = 0
        initial[fid][:2, 2] += [3, -2]
    return truth, initial, links


def test_projective_adjustment_recovers_heldout_geometry_and_preserves_gauge():
    truth, initial, links = scene()
    result, report, _ = refine_projective_map(initial, links, 200, 100, {0})
    grid = np.array([[x, y] for x in [10, 100, 190] for y in [10, 50, 90]])
    for fid in truth:
        error = np.linalg.norm(
            project_points(result[fid], grid) - project_points(truth[fid], grid), axis=1
        )
        assert error.max() < 0.03
    np.testing.assert_array_equal(result[0], truth[0])
    assert report["heldout_after_p95_px"] < 0.01
    assert report["heldout_points"] > 0
    assert report["selected_model"] == "projective"


def test_model_selection_prefers_full_affine_for_affine_observations():
    rng = np.random.default_rng(123)
    H = np.array([[1.0, 0.06, 10], [0.02, 0.99, -4], [0, 0, 1.0]])
    points = rng.uniform([30, 20], [170, 80], (120, 2))
    observed = project_points(H, points) + rng.normal(0, 0.08, points.shape)
    links = [PointLink(1, 0, points, observed)]
    seed = H.copy()
    seed[2, :2] = [0.0001, -0.0001]
    result, report, _ = refine_projective_map({0: np.eye(3), 1: seed}, links, 200, 100, {0})
    assert report["selected_model"] == "affine"
    np.testing.assert_array_equal(result[1][2], [0, 0, 1])


def test_full_frame_validation_preserves_supported_shear():
    points = np.array([[x, y] for x in np.linspace(10, 30, 10) for y in np.linspace(10, 30, 8)])
    H = np.array([[1.0, 4.0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    estimated, mask = GeometryTransforms.estimate_homography(
        points, project_points(H, points), frame_size=(200, 100), fallback_to_affine=False
    )
    assert estimated is not None and mask.sum() == len(points)
    np.testing.assert_allclose(
        project_points(estimated, points), project_points(H, points), atol=1e-4
    )


def test_subpixel_source_rejects_changed_video(tmp_path):
    from src.geometry.pair_refinement import SourceFrames

    source = tmp_path / "changed.mp4"
    source.write_bytes(b"wrong recording")
    with pytest.raises(ValueError, match="changed"):
        SourceFrames(
            {
                "source_path": str(source),
                "source_sha256": "0" * 64,
                "frame_step": 30,
                "frame_width": 200,
                "frame_height": 100,
            }
        )


def test_full_composition_has_no_shear_chain_drift():
    H = np.array([[1.0, 0.02, 20], [0, 1.0, 10], [0, 0, 1.0]])
    transforms = {(i, i - 1): H for i in range(1, 21)}
    result = initialize_from_pairs(
        transforms, [(i - 1, i, 1) for i in range(1, 21)], {0: np.eye(3)}, 1920, 1080
    )
    center = [[960.0, 540.0]]
    np.testing.assert_allclose(
        project_points(result[20], center),
        project_points(np.linalg.matrix_power(H, 20), center),
        atol=1e-8,
    )


@pytest.mark.parametrize("scale", [1e-20, -1e-20, 1e20, -1e20])
def test_homography_scale_does_not_change_validity_or_projection(scale):
    H = scene()[0][1]
    assert valid_footprint(H * scale, 200, 100, orientation=1)
    np.testing.assert_allclose(
        project_points(H * scale, [[100, 50]]), project_points(H, [[100, 50]])
    )


def test_geographic_origin_does_not_change_relative_solution():
    truth, initial, links = scene()
    map_gauge = np.array([[0.7, 0, 3e6], [0, -0.7, 6e6], [0, 0, 1.0]])
    result, report, _ = refine_projective_map(
        {fid: map_gauge @ H for fid, H in initial.items()}, links, 200, 100, {0}
    )
    for fid, H in result.items():
        np.testing.assert_allclose(
            project_points(np.linalg.inv(map_gauge) @ H, [[100, 50]]),
            project_points(truth[fid], [[100, 50]]),
            atol=0.03,
        )
    assert report["after_p95_px"] < 0.01


def test_sparse_bad_observations_do_not_move_heldout_geometry():
    truth, initial, links = scene()
    rng = np.random.default_rng(42)
    for link in links:
        link.second_points += rng.normal(0, 0.1, link.second_points.shape)
        link.second_points[2:5] += [8, -6]
    result, _, _ = refine_projective_map(initial, links, 200, 100, {0})
    for fid, H in result.items():
        assert (
            np.linalg.norm(project_points(H, [[100, 50]]) - project_points(truth[fid], [[100, 50]]))
            < 0.3
        )


def test_accuracy_matcher_does_not_modify_shared_model_configuration():
    from src.localization.matcher import FeatureMatcher

    matcher = FeatureMatcher.__new__(FeatureMatcher)
    matcher.lightglue = SimpleNamespace(
        conf=SimpleNamespace(depth_confidence=0.95, width_confidence=0.99)
    )
    features = {"descriptors": np.zeros((8, 128))}

    def check(*args, model):
        assert model is not matcher.lightglue
        assert model.conf.depth_confidence == model.conf.width_confidence == -1
        raise RuntimeError("inference failed")

    matcher._lightglue_match = check
    with pytest.raises(RuntimeError, match="inference failed"):
        matcher.match_accuracy(features, features)
    assert matcher.lightglue.conf.depth_confidence == 0.95
    assert matcher.lightglue.conf.width_confidence == 0.99


def test_disconnected_refinement_rejected():
    _, initial, links = scene()
    with pytest.raises(ValueError, match="connect"):
        refine_projective_map(initial, links[:1], 200, 100, {0})


def test_cancellation_is_not_swallowed():
    _, initial, links = scene()

    def cancel():
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        refine_projective_map(initial, links, 200, 100, {0}, check_running=cancel)


def test_all_fixed_anchors_remain_exact():
    truth, _, links = scene()
    result, _, _ = refine_projective_map(truth, links, 200, 100, set(truth))
    for fid in truth:
        np.testing.assert_array_equal(result[fid], truth[fid])


@pytest.mark.parametrize("angle", [0, 25, 45, 80, 135])
def test_spread_rejects_collinearity_at_every_angle(angle):
    t = np.linspace(-300, 300, 100)
    direction = [np.cos(np.radians(angle)), np.sin(np.radians(angle))]
    points = t[:, None] * direction + [700, 500]
    assert inlier_spread(points, 1920, 1080, principal=True) == 0


def test_pair_gate_checks_both_images_and_horizons():
    p = np.array([[x, y] for x in np.linspace(10, 190, 10) for y in np.linspace(10, 90, 5)])
    ok, _ = pair_quality(np.eye(3), p, p, len(p), 200, 100)
    assert ok
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.01, 0, 1]])
    ok, qa = pair_quality(H, p, p, len(p), 200, 100)
    assert not ok and qa["rejection"] == "invalid_footprint"
    ok, _ = pair_quality(np.eye(3), p, p, len(p) * 10, 200, 100)
    assert not ok


def test_full_geometry_survives_normalization_and_compatibility_projection():
    truth, _, _ = scene()
    result, normalization = normalize_relative_map(truth, 200, 100)
    p = np.array([[100, 50], [200, 100], [0, 0]])
    for fid in truth:
        np.testing.assert_allclose(
            project_points(result[fid], p),
            project_points(normalization, project_points(truth[fid], p)),
        )
        np.testing.assert_allclose(
            GeometryTransforms.apply_affine(p, result[fid]), project_points(result[fid], p)
        )
        tangent = affine_at_center(result[fid], 200, 100)
        np.testing.assert_allclose(
            project_points(tangent, p[:1]), project_points(result[fid], p[:1])
        )
    db = SimpleNamespace(
        get_frame_affine=lambda fid: affine_at_center(result[fid], 200, 100),
        frame_homography=np.stack([result[i] for i in range(3)]),
    )
    np.testing.assert_array_equal(get_frame_transform(db, 1), result[1])


def test_subpixel_refinement_improves_known_translation():
    rng = np.random.default_rng(8)
    image = cv2.GaussianBlur(rng.integers(0, 256, (200, 300), dtype=np.uint8), (5, 5), 0.8)
    shift = np.array([3.25, -2.5])
    moved = cv2.warpAffine(image, np.column_stack([np.eye(2), shift]), (300, 200))
    first = cv2.goodFeaturesToTrack(image, 80, 0.01, 12).reshape(-1, 2)
    first = first[
        (first[:, 0] > 20) & (first[:, 0] < 270) & (first[:, 1] > 25) & (first[:, 1] < 170)
    ]
    initial = first + shift + [0.4, -0.3]
    refined, count = refine_subpixel(image, moved, first, initial)
    assert count > len(first) * 0.8
    assert np.median(np.linalg.norm(refined - first - shift, axis=1)) < 0.1


def test_accuracy_temporal_graph_contains_redundant_edges():
    from src.geometry.pose_graph.optimizer import PoseGraphOptimizer
    from src.workers.propagation_pipeline import PropagationPipeline

    grid = np.array([[x, y] for x in np.linspace(30, 170, 8) for y in np.linspace(20, 80, 6)])
    features = {i: {"keypoints": grid - [10 * i, 0]} for i in range(4)}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        SimpleNamespace(metadata={"frame_width": 200, "frame_height": 100}),
        SimpleNamespace(converter=SimpleNamespace(mode="LOCAL")),
        matcher,
        {"propagation": {"high_accuracy": True, "temporal_neighbors": 3}},
    )
    pipeline._relative_feature_ids = {id(f): i for i, f in features.items()}
    graph = PoseGraphOptimizer(200, 100)
    for i in features:
        graph.add_node(i)
    assert pipeline._build_temporal_edges(graph, features, 4) == 6
    assert len(pipeline._pair_homographies) == 6


def test_spatial_search_checks_close_neighbors_without_duplicating_temporal_edges():
    from src.geometry.pose_graph.optimizer import PoseGraphOptimizer
    from src.workers.propagation_pipeline import PropagationPipeline

    grid = np.array([[x, y] for x in np.linspace(30, 170, 8) for y in np.linspace(20, 80, 6)])
    features = {i: {"keypoints": grid - [10 * i, 0]} for i in range(4)}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        SimpleNamespace(
            metadata={"frame_width": 200, "frame_height": 100},
            global_descriptors=np.ones((4, 3), dtype=np.float32),
        ),
        SimpleNamespace(converter=SimpleNamespace(mode="LOCAL")),
        matcher,
        {
            "propagation": {"high_accuracy": True, "temporal_neighbors": 1},
            "graph_optimization": {"loop_closure_min_frame_gap": 10},
        },
    )
    pipeline._relative_feature_ids = {id(f): i for i, f in features.items()}
    graph = PoseGraphOptimizer(200, 100)
    for i in features:
        graph.add_node(i)
    assert pipeline._build_temporal_edges(graph, features, 4) == 3
    assert pipeline._detect_loop_closures(graph, features, 4) == 3
    assert len({tuple(sorted((e.from_id, e.to_id))) for e in graph.edges}) == 6


def test_rotation_retry_validates_original_coordinates_and_keeps_actual_inlier_ratio(monkeypatch):
    from src.workers.propagation_pipeline import PropagationPipeline

    grid = np.array([[x, y] for x in np.linspace(30, 170, 40) for y in np.linspace(20, 80, 40)])
    features = {i: {"keypoints": grid.copy()} for i in range(2)}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        SimpleNamespace(metadata={"frame_width": 200, "frame_height": 100}),
        SimpleNamespace(converter=SimpleNamespace(mode="LOCAL")),
        matcher,
        {"propagation": {"high_accuracy": True}},
    )
    monkeypatch.setattr(
        "src.localization.rotation_geometry.temporal_retry_angles", lambda *a, **kw: [90]
    )
    result = pipeline._temporal_rotation_retry(features[1], features[0], 0, 1)
    assert result is not None
    assert pipeline._pair_diagnostics[(1, 0)]["inlier_ratio"] == 1
    assert pipeline._pair_diagnostics[(1, 0)]["inliers"] == len(grid)
    np.testing.assert_allclose(
        project_points(pipeline._pair_homographies[(1, 0)], [[100, 50]]), [[100, 50]], atol=1e-4
    )
