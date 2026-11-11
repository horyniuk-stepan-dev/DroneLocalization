"""VisualScaleTrack (vision-only plan 2.1)."""

import numpy as np
import pytest

from src.localization.visual_altitude import VisualScaleTrack, flow_scale


def _sim(s, angle_deg=0.0):
    a = np.radians(angle_deg)
    return np.array([[s * np.cos(a), -s * np.sin(a), 3.0], [s * np.sin(a), s * np.cos(a), -2.0]])


def test_flow_scale_rotation_invariant_and_bounded():
    assert flow_scale(_sim(1.2, 30)) == pytest.approx(1.2)
    assert flow_scale(_sim(5.0)) is None
    assert flow_scale(None) is None


def test_descent_lowers_gsd_and_survives_failures():
    tr = VisualScaleTrack(max_age_s=60)
    assert tr.estimate(0.0) is None
    tr.on_keyframe(0.0, 0.8)
    tr.on_flow(_sim(1.25))  # ground grew by 25 %: lower
    assert tr.estimate(0.5) == pytest.approx(0.64)
    tr.on_rebase(_sim(1.25))  # chain anchor moved to that frame
    tr.on_flow(_sim(1.0 / 0.8))
    assert tr.estimate(10.0) == pytest.approx(0.512)
    assert tr.estimate(61.0) is None  # expired


def test_new_keyframe_resets_base():
    tr = VisualScaleTrack()
    tr.on_keyframe(0.0, 1.0)
    tr.on_flow(_sim(0.5))
    tr.on_keyframe(1.0, 0.7)
    tr.on_flow(_sim(1.0))
    assert tr.estimate(1.2) == pytest.approx(0.7)
