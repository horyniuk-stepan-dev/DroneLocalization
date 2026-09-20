import numpy as np

from src.geometry.calibration_provenance import CalibrationOrigin as Origin
from src.geometry.calibration_provenance import GeoreferenceStatus as Status
from src.geometry.calibration_provenance import anchored_graph_support, classify_calibration


class Edge:
    def __init__(self, a, b, weight=1.0):
        self.from_id = a
        self.to_id = b
        self.weight = weight


def test_interpolation_does_not_masquerade_as_optimized_support():
    valid = np.ones(7, dtype=bool)
    supported = np.array([False, True, False, False, True, False, False])
    origins, distances, statuses = classify_calibration(valid, supported, [1])
    np.testing.assert_array_equal(
        origins,
        [
            Origin.EXTRAPOLATED,
            Origin.ANCHOR,
            Origin.INTERPOLATED,
            Origin.INTERPOLATED,
            Origin.OPTIMIZED,
            Origin.EXTRAPOLATED,
            Origin.EXTRAPOLATED,
        ],
    )
    np.testing.assert_array_equal(distances, [1, 0, 1, 1, 0, 1, 2])
    np.testing.assert_array_equal(
        statuses,
        [
            Status.PROVISIONAL,
            Status.SUPPORTED,
            Status.PROVISIONAL,
            Status.PROVISIONAL,
            Status.SUPPORTED,
            Status.PROVISIONAL,
            Status.PROVISIONAL,
        ],
    )


def test_no_support_never_invents_provenance_or_zero_error():
    origins, distances, statuses = classify_calibration([True, False], [False, False], [])
    np.testing.assert_array_equal(origins, [Origin.UNKNOWN, Origin.UNKNOWN])
    np.testing.assert_array_equal(distances, [-1, -1])
    np.testing.assert_array_equal(statuses, [Status.UNKNOWN, Status.UNKNOWN])


def test_invalid_gap_overrides_interpolated_availability():
    origins, _, statuses = classify_calibration(
        [True] * 5,
        [True, False, False, False, True],
        [0, 4],
        invalid_ids={1, 2, 3},
    )
    assert origins[2] == Origin.INTERPOLATED
    np.testing.assert_array_equal(
        statuses,
        [Status.SUPPORTED, Status.INVALID, Status.INVALID, Status.INVALID, Status.SUPPORTED],
    )


def test_graph_support_preserves_both_anchored_sides_of_a_broken_chain():
    supported, components, anchors = anchored_graph_support(
        range(7),
        [Edge(0, 1), Edge(1, 2), Edge(4, 5), Edge(5, 6)],
        [0, 6],
    )

    assert supported == {0, 1, 2, 4, 5, 6}
    assert components[3] >= 0
    assert anchors[int(components[3])] == []


def test_optimized_node_without_anchor_path_is_only_provisional():
    valid = np.ones(4, dtype=bool)
    optimized = np.ones(4, dtype=bool)
    supported = np.array([True, True, False, False])

    origins, _, statuses = classify_calibration(
        valid, optimized, [0], supported=supported
    )

    assert origins[2] == Origin.OPTIMIZED
    np.testing.assert_array_equal(
        statuses,
        [Status.SUPPORTED, Status.SUPPORTED, Status.PROVISIONAL, Status.PROVISIONAL],
    )
