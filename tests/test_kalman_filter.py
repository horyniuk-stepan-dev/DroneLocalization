"""Unit tests for TrajectoryFilter — Kalman filter used for live trajectory
smoothing.

Regression coverage for the two-point velocity seed (2026-07-18): before this
fix the filter started with vx=vy=0 and needed several updates to "learn" the
true velocity, which on a fast, roughly-constant-speed trajectory (~150 m/s,
observed on a simulator flight) made the filtered output lag the raw fixes by
tens of metres for the first few keyframes — and corrupted the speed-history
baseline of OutlierDetector (which consumes filtered_pt, not the raw fix),
triggering spurious z-score rejections. See docs/REMAINING_WORK_PLAN.md /
session notes for the full derivation.
"""

import numpy as np
import pytest

from src.tracking.kalman_filter import TrajectoryFilter


class TestTwoPointVelocitySeed:
    """The second update() call should seed velocity from the first two raw
    points instead of leaving it at zero."""

    def test_first_update_returns_raw_measurement(self):
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        out = kf.update((100.0, 200.0), dt=1.0)
        assert out == (100.0, 200.0)
        assert kf.kf.x[2, 0] == 0.0
        assert kf.kf.x[3, 0] == 0.0

    def test_second_update_seeds_velocity_from_raw_delta(self):
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        kf.update((0.0, 0.0), dt=1.0)
        kf.update((150.0, 0.0), dt=1.0)  # 150 m in 1 s = 150 m/s
        # No measurement noise in this synthetic pair -> residual is exactly
        # zero after the seed, so velocity should land on the true value.
        assert kf.kf.x[2, 0] == pytest.approx(150.0, abs=1e-6)
        assert kf.kf.x[3, 0] == pytest.approx(0.0, abs=1e-6)

    def test_second_update_output_matches_raw_no_lag(self):
        """Before the fix: v0=0 meant the filtered output lagged the raw
        fix on the very first real update (large initial P made position
        converge fast, but velocity — and therefore the NEXT prediction —
        started from zero). After the fix this lag is eliminated."""
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        kf.update((2901576.6, 6167739.3), dt=1.0)
        out = kf.update((2901576.5, 6167865.9), dt=1.0)
        raw = np.array([2901576.5, 6167865.9])
        lag = float(np.linalg.norm(np.array(out) - raw))
        assert lag < 0.01

    def test_seed_applies_once_not_on_every_update(self):
        """A third update on a trajectory that keeps the same constant
        velocity must not re-seed (velocity should evolve through the normal
        predict/update cycle, not get overwritten again)."""
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        kf.update((0.0, 0.0), dt=1.0)
        kf.update((150.0, 0.0), dt=1.0)
        v_after_seed = float(kf.kf.x[2, 0])
        kf.update((300.0, 0.0), dt=1.0)
        # Still tracking ~150 m/s constant velocity — should stay close, not
        # jump discontinuously the way a repeated "seed" would.
        assert kf.kf.x[2, 0] == pytest.approx(v_after_seed, abs=5.0)

    def test_reset_reseeds_on_next_session(self):
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        kf.update((0.0, 0.0), dt=1.0)
        kf.update((150.0, 0.0), dt=1.0)
        kf.reset()
        assert kf._prev_raw is None
        # New session, different velocity — must seed again, not carry over
        # the old one or skip seeding because of stale internal state.
        kf.update((1000.0, 1000.0), dt=1.0)
        kf.update((1000.0, 1080.0), dt=1.0)  # 80 m/s in y this time
        assert kf.kf.x[3, 0] == pytest.approx(80.0, abs=1e-6)
        assert kf.kf.x[2, 0] == pytest.approx(0.0, abs=1e-6)


class TestFilterStillSmooths:
    """Guard against a degenerate implementation that just echoes raw input
    forever — the filter must still damp a single noisy measurement."""

    def test_single_noisy_point_is_damped_not_echoed(self):
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        kf.update((0.0, 0.0), dt=1.0)
        kf.update((100.0, 0.0), dt=1.0)  # establishes ~100 m/s, low P by now
        # A single measurement that jumps far off the established trend.
        out = kf.update((500.0, 0.0), dt=1.0)
        # Constant-velocity prediction alone would land at 200.0; a filter
        # that just echoed raw input would land at 500.0. The true filtered
        # output must sit strictly between the two.
        assert 200.0 < out[0] < 500.0


class TestShiftUnaffectedBySeed:
    """shift() (used by the sliding-window smoother) must keep working
    exactly as before — it only touches position, never velocity."""

    def test_shift_moves_position_only(self):
        kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
        kf.update((0.0, 0.0), dt=1.0)
        kf.update((150.0, 0.0), dt=1.0)
        v_before = (float(kf.kf.x[2, 0]), float(kf.kf.x[3, 0]))
        kf.shift(5.0, -3.0)
        assert kf.kf.x[0, 0] == pytest.approx(155.0)
        assert kf.kf.x[1, 0] == pytest.approx(-3.0)
        assert (float(kf.kf.x[2, 0]), float(kf.kf.x[3, 0])) == pytest.approx(v_before)
