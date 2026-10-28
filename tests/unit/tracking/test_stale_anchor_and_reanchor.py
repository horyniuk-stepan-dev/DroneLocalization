"""Failure cascade after a failed keyframe (route3min, 2026-10-28).

1. KeyframeFlow.max_anchor_age_s: LK must not keep tracking from an anchor that
   is older than the limit (a failed keyframe does not move the anchor).
2. TrajectoryFilter(reanchor_velocity_from_fixes=True): re-anchoring to a good
   fix must not derive velocity from a garbage optical-flow sample.
"""

import numpy as np

from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.keyframe_flow import KeyframeFlow


def _img():
    rng = np.random.default_rng(0)
    return (rng.random((120, 160)) * 255).astype(np.uint8)


def test_anchor_expires_after_limit():
    f = KeyframeFlow(max_anchor_age_s=1.5)
    f.anchor(_img(), t=10.0)
    if not f.has_anchor:  # featureless corner case guard
        return
    assert f.expire(11.0) is False and f.has_anchor
    assert f.expire(11.6) is True and not f.has_anchor


def test_no_limit_is_legacy():
    f = KeyframeFlow()
    f.anchor(_img(), t=0.0)
    had = f.has_anchor
    assert f.expire(1e6) is False and f.has_anchor == had


def _feed(kf, garbage_offset):
    """50 m/s along x: trusted keyframe fixes at 1 Hz, OF in between; the last
    OF sample before the 3rd keyframe is garbage."""
    for k in range(3):
        t0 = float(k)
        kf.update((50.0 * t0, 0.0), dt=1.0, trusted_max_offset_m=5.0, timestamp=t0)
        for j in range(1, 6):
            t = t0 + j / 6.0
            x = 50.0 * t + (garbage_offset if (k == 2 and j == 5) else 0.0)
            kf.update((x, 0.0), dt=1 / 6, noise_scale=1.5, timestamp=t)
    # good fix far from the dragged filter -> re-anchor
    kf.update((150.0, 0.0), dt=1 / 6, trusted_max_offset_m=5.0, timestamp=3.0)
    return float(kf.kf.x[2, 0])


def test_legacy_reanchor_velocity_is_poisoned_by_garbage_of():
    v = _feed(TrajectoryFilter(process_noise=8.0), garbage_offset=300.0)
    assert abs(v - 50.0) > 500.0


def test_fix_based_reanchor_velocity_survives_garbage_of():
    kf = TrajectoryFilter(process_noise=8.0, reanchor_velocity_from_fixes=True)
    v = _feed(kf, garbage_offset=300.0)
    assert kf.last_update_reanchored
    assert abs(v - 50.0) < 1.0
