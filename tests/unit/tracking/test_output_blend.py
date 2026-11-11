"""OutputBlend (vision-only plan 2.2): re-anchor without a jump."""

import math

from src.tracking.output_blend import _R, OutputBlend

LAT0, LON0 = 48.42, 26.18
K = math.radians(1.0) * _R


def _ll(x, y):
    return LAT0 + y / K, LON0 + x / (K * math.cos(math.radians(LAT0)))


def _xy(lat, lon):
    return (lon - LON0) * K * math.cos(math.radians(LAT0)), (lat - LAT0) * K


def _run(blend, drift_per_step, kf_every=6, steps=60, v=50.0, dt=1 / 6):
    """Truth moves east at v; OF drifts, keyframes are exact. Returns max step error."""
    worst = 0.0
    prev_out = prev_true = None
    drift = 0.0
    for k in range(steps):
        t = k * dt
        true_x = v * t
        is_kf = k % kf_every == 0
        drift = 0.0 if is_kf else drift + drift_per_step
        lat, lon, _ = blend.apply(*_ll(true_x + drift, 0.0), t, is_kf)
        out = _xy(lat, lon)
        if prev_out is not None:
            est = (out[0] - prev_out[0], out[1] - prev_out[1])
            tru = (true_x - prev_true, 0.0)
            worst = max(worst, math.hypot(est[0] - tru[0], est[1] - tru[1]))
        prev_out, prev_true = out, true_x
    return worst


def test_without_blend_reanchor_is_a_jump_with_blend_it_is_not():
    # 3 m/step drift -> 15 m at the next keyframe
    big = OutputBlend(max_step_m=1e9, min_m=1e9)  # never absorbs: legacy
    assert _run(big, 3.0) > 14.0
    assert _run(OutputBlend(max_step_m=3.0), 3.0) < 10.0


def test_relocation_beyond_max_offset_is_reported():
    b = OutputBlend(max_step_m=3.0, max_offset_m=40.0)
    b.apply(*_ll(0.0, 0.0), 0.0, True)
    b.apply(*_ll(8.0, 0.0), 1 / 6, False)
    lat, lon, off = b.apply(*_ll(200.0, 0.0), 2 / 6, True)
    assert off == 0.0
    assert abs(_xy(lat, lon)[0] - 200.0) < 0.5


def test_offset_decays_to_zero():
    b = OutputBlend(max_step_m=3.0)
    b.apply(*_ll(0.0, 0.0), 0.0, True)
    b.apply(*_ll(10.0, 0.0), 0.2, False)
    _, _, off = b.apply(*_ll(5.0, 0.0), 0.4, True)  # fix 10 m behind the continued output
    assert 5.0 < off <= 12.0
    for k in range(10):
        _, _, off = b.apply(*_ll(5.0 + 0.0 * k, 0.0), 0.6 + 0.2 * k, False)
    assert off == 0.0


def test_no_absorption_after_a_gap():
    b = OutputBlend(max_step_m=3.0, max_gap_s=0.5)
    b.apply(*_ll(0.0, 0.0), 0.0, True)
    b.apply(*_ll(8.0, 0.0), 1 / 6, False)
    lat, lon, off = b.apply(*_ll(20.0, 0.0), 1.0, True)  # 0.83 s later, 20 m off the guess
    assert off == 0.0
    assert abs(_xy(lat, lon)[0] - 20.0) < 0.5


def test_no_absorption_after_dead_reckoning():
    b = OutputBlend(max_step_m=3.0)
    b.apply(*_ll(0.0, 0.0), 0.0, True)
    b.apply(*_ll(8.0, 0.0), 1 / 6, False)
    lat, lon, off = b.apply(*_ll(4.0, 0.0), 2 / 6, True, absorb=False)
    assert off == 0.0
    assert abs(_xy(lat, lon)[0] - 4.0) < 0.5
