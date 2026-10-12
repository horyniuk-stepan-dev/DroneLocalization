"""Regressions from the independent accuracy review."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.geometry.frame_transform import project_frame_points, project_points
from src.geometry.pair_refinement import pair_quality
from src.geometry.point_spread import inlier_spread
from src.geometry.relative_alignment import PointLink
from src.geometry.relative_projective import connected_links, symmetric_transfer


@pytest.mark.parametrize("dof", [6, 8])
def test_analytic_symmetric_jacobian_matches_independent_finite_difference(dof):
    rng = np.random.default_rng(9)
    H = np.array([np.eye(3), [[1.1, 0.04, 0.2], [-0.03, 0.95, -0.1], [0.02, -0.04, 1.0]]])
    if dof == 6:
        H[:, 2, :2] = 0
    p = np.column_stack([rng.uniform(-0.4, 0.4, (30, 2)), np.ones(30)])
    q = p.copy()
    ia, ib = np.zeros(30, dtype=int), np.ones(30, dtype=int)
    _, ja, jb = symmetric_transfer(H, ia, ib, p, q, 1280, dof, jacobian=True)
    for camera, block in enumerate((ja, jb)):
        for column in range(dof):
            plus, minus = H.copy(), H.copy()
            plus[camera].flat[column] += 1e-6
            minus[camera].flat[column] -= 1e-6
            numeric = (
                symmetric_transfer(plus, ia, ib, p, q, 1280, dof)
                - symmetric_transfer(minus, ia, ib, p, q, 1280, dof)
            ) / 2e-6
            np.testing.assert_allclose(block[:, :, column], numeric, rtol=1e-6, atol=1e-6)


def test_principal_spread_is_opt_in_for_live_compatibility():
    points = np.column_stack([np.arange(100), np.arange(100)])
    assert inlier_spread(points, 100, 100) == pytest.approx(np.std(np.arange(100)) / 100)
    assert inlier_spread(points, 100, 100, principal=True) == 0


@pytest.mark.parametrize("scale", [1.0, -1e-20, 1e20])
def test_extrapolated_points_cannot_cross_reference_horizon(scale):
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.004, 0, 1.0]]) * scale
    assert np.isfinite(project_frame_points(H, [[100, 50]], 200, 100)).all()
    for point in ([300, 50], [240, 50]):
        with pytest.raises(ValueError, match="safe reference"):
            project_frame_points(H, [point], 200, 100)


def test_fov_drops_polygon_crossing_reference_horizon():
    from src.localization.result_builder import ResultBuilder

    builder = ResultBuilder({}, 3.0)
    reference = np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.004, 0, 1.0]])
    query = np.array([[1.0, 0, 150], [0, 1.0, 0], [0, 0, 1.0]])
    converter = Mock()
    result = builder.build_fov(
        query,
        reference,
        100,
        100,
        None,
        converter,
        0.0,
        0.0,
        0.0,
        0.0,
        [0.0, 0.0],
        1,
        reference_size=(200, 100),
    )
    assert result == []
    converter.metric_to_gps.assert_not_called()


def grid():
    return np.array([[x, y] for x in np.linspace(30, 170, 8) for y in np.linspace(20, 80, 6)])


def pipeline():
    from src.workers.propagation_pipeline import PropagationPipeline

    matcher = Mock()
    matcher.match.return_value = grid(), grid() + [10, 0]
    return PropagationPipeline(
        SimpleNamespace(metadata={"frame_width": 200, "frame_height": 100}),
        SimpleNamespace(converter=SimpleNamespace(mode="LOCAL")),
        matcher,
        {"propagation": {"high_accuracy": True}},
    )


def test_corrupt_link_is_dropped_and_pruned_orphan_is_not_refined():
    points = grid()
    valid = PointLink(0, 1, points, points)
    invalid = PointLink(1, 2, points[:4], points[:4])
    initial, links = connected_links({i: np.eye(3) for i in range(3)}, [valid, invalid], {0})
    assert set(initial) == {0, 1} and links == [valid]


def test_coherent_repeated_field_alias_is_rejected_against_temporal_seed():
    p = pipeline()
    points = grid()
    wrong = np.array([[1.0, 0, 50], [0, 1.0, 0], [0, 0, 1.0]])
    observed = project_points(wrong, points)
    # A perfect, spatially broad consensus passes pairwise RANSAC quality.
    assert pair_quality(wrong, points, observed, len(points), 200, 100)[0]
    assert not p._consistent_with_map({0: np.eye(3), 1: np.eye(3)}, 0, 1, points, observed)


def test_video_read_failure_keeps_verified_feature_edge():
    p = pipeline()
    a, b = {}, {}
    p._relative_feature_ids = {id(a): 0, id(b): 1}
    p._source_frames = Mock()
    p._source_frames.get.side_effect = ValueError("corrupt video frame")
    assert p._match_and_build_edge(a, b) is not None
    assert p._pair_diagnostics[(0, 1)]["subpixel_points"] == 0


def test_rotation_sample_cannot_hide_low_original_inlier_ratio(monkeypatch):
    p = pipeline()
    monkeypatch.setattr(
        "src.localization.rotation_geometry.temporal_retry_angles", lambda *a, **kw: [0]
    )

    def match(*args):
        p._last_inlier_pair = grid(), grid()
        return np.eye(3), 100, 0.0, 1000, 0.15

    monkeypatch.setattr(p, "_match_and_build_edge", match)
    assert p._temporal_rotation_retry({"keypoints": grid()}, {}, 0, 1) is None
    assert p._pair_diagnostics[(1, 0)]["rejection"] == "inlier_ratio"


def test_optional_subpixel_preserves_cancellation():
    from src.workers.propagation_pipeline import PropagationCancelledError

    p = pipeline()
    a, b = {}, {}
    p._relative_feature_ids = {id(a): 0, id(b): 1}
    p._source_frames = Mock()
    p._source_frames.get.side_effect = PropagationCancelledError()
    with pytest.raises(PropagationCancelledError):
        p._match_and_build_edge(a, b)


def test_long_map_windows_keep_distant_loop_constraints_and_fixed_anchors():
    from src.geometry.relative_projective import refine_projective_map

    rng = np.random.default_rng(12)
    points = rng.uniform([20, 20], [170, 80], (30, 2))
    truth = {i: np.array([[1.0, 0, i * 5], [0, 1.0, 0], [0, 0, 1.0]]) for i in range(70)}
    seed = {i: H.copy() for i, H in truth.items()}
    for i in range(1, 69):
        seed[i][0, 2] += 2 * np.sin(i)
    links = [PointLink(i, i + 1, points, points - [5, 0]) for i in range(69)]
    links.append(PointLink(0, 69, points, points - [345, 0]))
    result, report, _ = refine_projective_map(seed, links, 200, 100, {0, 69}, max_nfev=30)
    for fid in (0, 69):
        np.testing.assert_array_equal(result[fid], truth[fid])
    assert report["solver"] == "overlapping_block_coordinate"
    assert report["links"] == 70
    assert report["after_p95_px"] < report["before_p95_px"] * 0.1
    assert report["window_frames"] == 32
