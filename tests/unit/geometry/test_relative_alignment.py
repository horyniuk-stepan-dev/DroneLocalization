"""Point alignment must correct geometry, preserve its gauge and reject bad input."""

import numpy as np
import pytest

from src.geometry.relative_alignment import (
    PointLink,
    nearby_frame_pairs,
    refine_relative_affines,
    sample_point_pair,
)


def scene():
    rng = np.random.default_rng(41)
    truth = {
        0: np.array([[1.0, 0.0, -100], [0.0, -1.0, 50]]),
        1: np.array([[1.1, 0.08, -65], [0.04, -0.9, 52]]),
        2: np.array([[0.95, -0.06, -20], [-0.03, -1.05, 47]]),
    }
    links = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        world = rng.uniform([-40, -25], [60, 25], (100, 2))
        p = (world - truth[i][:, 2]) @ np.linalg.inv(truth[i][:, :2]).T
        q = (world - truth[j][:, 2]) @ np.linalg.inv(truth[j][:, :2]).T
        links.append(PointLink(i, j, p, q))
    initial = {fid: a.copy() for fid, a in truth.items()}
    for fid in (1, 2):
        initial[fid][0, 1] = initial[fid][1, 0] = 0
        initial[fid][:, 2] += [8, -5]
    return initial, truth, links


def test_recovers_shear_and_translation_without_moving_gauge():
    initial, truth, links = scene()
    untouched = {fid: a.copy() for fid, a in initial.items()}
    result, report, errors = refine_relative_affines(initial, links, 200, 100, 0)
    assert report["after_median_px"] < 0.01
    assert report["after_p95_px"] < report["before_p95_px"] / 100
    np.testing.assert_array_equal(result[0], initial[0])
    for fid in truth:
        np.testing.assert_allclose(result[fid], truth[fid], atol=0.03)
        np.testing.assert_array_equal(initial[fid], untouched[fid])
        assert errors[fid] < 0.02


def test_robust_to_a_minority_of_bad_correspondences():
    initial, truth, links = scene()
    links[1].second_points[:10] += [60, -30]
    result, report, _ = refine_relative_affines(initial, links, 200, 100, 0)
    assert report["after_median_px"] < 0.5
    # Check physical displacement on a held-out grid, not affine coefficients
    # that mix dimensionless scale with pixel translations.
    world = np.array([[x, y] for x in [-40, 10, 60] for y in [-25, 0, 25]])
    for fid in truth:
        pixels = (world - truth[fid][:, 2]) @ np.linalg.inv(truth[fid][:, :2]).T
        errors = np.linalg.norm(pixels @ result[fid][:, :2].T + result[fid][:, 2] - world, axis=1)
        assert np.median(errors) < 1
        assert np.max(errors) < 2


def test_disconnected_points_fail_instead_of_fabricating_geometry():
    initial, _, links = scene()
    with pytest.raises(ValueError, match="connect every frame"):
        refine_relative_affines(initial, links[:1], 200, 100, 0)


def test_cancellation_propagates():
    initial, _, links = scene()

    def cancelled():
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        refine_relative_affines(initial, links, 200, 100, 0, cancelled)


def test_collinear_correspondences_are_rejected():
    initial, _, links = scene()
    links[0].first_points[:, 1] = 0
    with pytest.raises(ValueError, match="non-collinear"):
        refine_relative_affines(initial, links, 200, 100, 0)


def test_sampling_keeps_correspondences_and_extent():
    rng = np.random.default_rng(10)
    points = rng.uniform([0, 0], [200, 100], (2000, 2))
    first, second = sample_point_pair(points, points + [12, -8])
    assert len(first) == 96
    np.testing.assert_allclose(second - first, np.tile([12, -8], (96, 1)))
    assert np.ptp(first[:, 0]) > 195 and np.ptp(first[:, 1]) > 95


def test_mirrored_solution_is_rejected():
    initial, _, links = scene()
    points = links[0].first_points
    reflected = points * [-1, 1] + [200, 0]
    with pytest.raises(ValueError, match="fold or excessively distort"):
        refine_relative_affines(
            {0: initial[0], 1: initial[0]}, [PointLink(0, 1, points, reflected)], 200, 100, 0
        )


def test_downweighted_graph_link_cannot_regain_full_influence():
    initial, truth, links = scene()
    bad = PointLink(0, 2, links[2].first_points, links[2].second_points + [4, 0], weight=0.001)
    result, _, _ = refine_relative_affines(initial, [*links, bad], 200, 100, 0)
    world = np.array([[0.0, 0.0]])
    pixels = (world - truth[2][:, 2]) @ np.linalg.inv(truth[2][:, :2]).T
    error = np.linalg.norm(pixels @ result[2][:, :2].T + result[2][:, 2] - world)
    assert error < 0.02


def test_rotation_retry_caches_points_in_original_image_coordinates():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.geometry.coordinates import CoordinateConverter
    from src.workers.propagation_pipeline import PropagationPipeline

    grid = np.array([[x, y] for x in np.linspace(30, 170, 6) for y in np.linspace(20, 80, 6)])
    query = {"keypoints": grid}
    reference = {"keypoints": grid + [12, 5]}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    pipeline = PropagationPipeline(
        SimpleNamespace(metadata={"frame_width": 200, "frame_height": 100}),
        SimpleNamespace(converter=CoordinateConverter("LOCAL")),
        matcher,
    )
    pipeline._relative_feature_ids = {id(query): 2, id(reference): 0}
    assert pipeline._temporal_rotation_retry(query, reference, 0, 2) is not None
    first, second = pipeline._relative_point_pairs[(2, 0)]
    np.testing.assert_allclose(first, query["keypoints"], atol=1e-4)
    np.testing.assert_allclose(second, reference["keypoints"], atol=1e-4)


def test_spatial_proposals_exclude_known_temporal_and_distant_frames():
    affines = {
        fid: np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0]])
        for fid, x in [(0, 0), (1, 20), (20, 70), (40, 90), (60, 10000)]
    }
    pairs = nearby_frame_pairs(affines, 200, 100, {(20, 0)}, min_frame_gap=3, limit=2)
    assert (0, 40) in pairs
    assert (0, 20) not in pairs  # known even with reverse input order
    assert (0, 1) not in pairs  # temporal neighbor
    assert not any(60 in pair for pair in pairs)
    assert len(pairs) == len(set(pairs))
    assert nearby_frame_pairs(affines, 200, 100, set(), 3, limit=0) == []


@pytest.mark.parametrize("corrupt", [False, True])
def test_extra_overlap_is_verified_before_it_can_move_the_map(corrupt):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.geometry.coordinates import CoordinateConverter
    from src.workers.propagation_pipeline import PropagationPipeline

    grid = np.array([[x, y] for x in np.linspace(30, 170, 6) for y in np.linspace(15, 85, 6)])
    features = {0: {"keypoints": grid}, 20: {"keypoints": grid - [30, 0]}}
    matcher = Mock()
    matcher.match.side_effect = lambda a, b: (a["keypoints"], b["keypoints"])
    if corrupt:
        # A scene that would require a reflection must never close a map seam.
        features[20]["keypoints"] = grid * [-1, 1] + [200, 0]
    pipeline = PropagationPipeline(
        SimpleNamespace(metadata={"frame_width": 200, "frame_height": 100}),
        SimpleNamespace(converter=CoordinateConverter("LOCAL")),
        matcher,
    )
    affines = {0: np.eye(2, 3), 20: np.array([[1.0, 0.0, 60], [0.0, 1.0, 0]])}
    links = pipeline._local_overlap_links(affines, features, set())
    if corrupt:
        assert not links
    else:
        assert len(links) == 1
        result, _, _ = refine_relative_affines(affines, links, 200, 100, 0)
        assert result[20][0, 2] == pytest.approx(30.0, abs=0.01)
    np.testing.assert_array_equal(affines[20], [[1, 0, 60], [0, 1, 0]])
