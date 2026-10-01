"""Straight-leg fallback must be opt-in and reject sparse/curved anchor paths."""

import numpy as np
import pytest

from src.geometry.anchor_linear_model import (
    interpolate_linear_anchor_intervals,
    linear_anchor_intervals,
)


def _affine(cx: float, cy: float) -> np.ndarray:
    # Mirrored map transform: its translation places the image centre at cx,cy.
    return np.array([[2.0, 0.0, cx - 100.0], [0.0, -2.0, cy + 50.0]])


def test_three_consistent_long_intervals_support_straight_leg():
    anchors = {
        0: _affine(0, 0),
        20: _affine(40, 0),
        40: _affine(80.1, 0),
        60: _affine(120, 0),
        61: _affine(121, 10),  # short interval marks a turn
        81: _affine(121, 50),
    }
    intervals = linear_anchor_intervals(anchors, 100, 50)
    assert intervals == [(0, 20), (20, 40), (40, 60)]
    predicted = interpolate_linear_anchor_intervals(anchors, intervals, 100, 50)
    assert len(predicted) == 61
    np.testing.assert_array_equal(predicted[0], anchors[0])
    np.testing.assert_array_equal(predicted[60], anchors[60])
    centre = predicted[10][:, :2] @ np.array([50.0, 25.0]) + predicted[10][:, 2]
    np.testing.assert_allclose(centre, [20.0, 0.0], atol=1e-9)
    assert np.linalg.det(predicted[10][:, :2]) < 0
    assert 61 not in predicted


def test_short_or_curved_anchor_runs_are_not_certified():
    short = {0: _affine(0, 0), 8: _affine(8, 0), 16: _affine(16, 0), 24: _affine(24, 0)}
    assert linear_anchor_intervals(short, 100, 50) == []
    curved = {
        0: _affine(0, 0),
        20: _affine(40, 0),
        40: _affine(60, 20),
        60: _affine(60, 60),
    }
    assert linear_anchor_intervals(curved, 100, 50) == []


def test_rejects_stationary_and_incompatible_reflection():
    stationary = {i: _affine(0, 0) for i in (0, 20, 40, 60)}
    assert linear_anchor_intervals(stationary, 100, 50) == []
    anchors = {i: _affine(i, 0) for i in (0, 20, 40, 60)}
    anchors[60][1, 1] = 2.0
    anchors[60][1, 2] = -50.0
    intervals = linear_anchor_intervals(anchors, 100, 50)
    assert intervals == []
    predicted = interpolate_linear_anchor_intervals(anchors, intervals, 100, 50)
    assert 41 not in predicted


def test_local_straight_runs_survive_an_outlier_gap():
    centres = [0, 20, 40, 60, 100, 120, 140, 160]
    anchors = {20 * i: _affine(float(x), 0.0) for i, x in enumerate(centres)}
    assert linear_anchor_intervals(anchors, 100, 50) == [
        (0, 20), (20, 40), (40, 60), (80, 100), (100, 120), (120, 140)
    ]


def test_rejects_nonpositive_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        linear_anchor_intervals({}, 100, 50, min_run_intervals=0)
