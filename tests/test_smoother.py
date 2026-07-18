"""Tests for SlidingWindowSmoother (RESEARCH 3.1) and TrajectoryFilter.shift.

Pure numpy/filterpy — runnable in the sandbox. The solver is verified against
an independent brute-force least-squares of the same stacked system.

The Scenario driver mirrors the real pipeline's semantics:

- OF positions are ANCHOR-REBASED: in the live system an OF metric point is
  the center pushed through the last accepted keyframe's homography, so it
  equals `anchor_fix + true_displacement_since_anchor` — a wrong anchor fix
  offsets the whole OF track by the same amount and the offset cancels in
  edge deltas. Feeding ground-truth OF instead would violate the invariant
  OF(t_anchor) == anchor_fix and tear the graph apart at tampered nodes.
- dt semantics follow the worker: keyframe dt = time since the previous
  keyframe (rejected included); OF dt = time since the last successful
  localization (previous OF sample or the anchoring accepted keyframe).
"""

import time

import numpy as np
import pytest

from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.smoother import SlidingWindowSmoother

FPS = 30.0
FRAME = 1.0 / FPS
KF_DT = 6 * FRAME  # keyframe cadence 0.2 s


def make_smoother(**kw) -> SlidingWindowSmoother:
    defaults = dict(
        window=60,
        huber_k=1.2,
        fix_sigma_base_m=5.0,
        odom_sigma_base_m=3.0,
        max_correction_m=50.0,
        entry_prior_sigma_m=15.0,
        irls_iterations=4,
    )
    defaults.update(kw)
    return SlidingWindowSmoother(**defaults)


class Scenario:
    """Feeds the smoother keyframe fixes + per-frame OF with worker-accurate
    timing and anchor-rebased OF positions (see module docstring)."""

    def __init__(
        self,
        sm,
        pos_fn,
        rng=None,
        fix_noise=0.0,
        of_noise=0.0,
        conf=0.8,
        of_quality=0.9,
    ):
        self.sm = sm
        self.pos = lambda t: np.asarray(pos_fn(t), dtype=np.float64)
        self.rng = rng
        self.fix_noise = fix_noise
        self.of_noise = of_noise
        self.conf = conf
        self.of_quality = of_quality
        self.t_kf = 0.0  # video time of the previous keyframe
        self.t_success = 0.0  # video time of the last successful localization
        self.anchor_z = None  # raw fix of the anchoring accepted keyframe
        self.anchor_truth = None

    def _noise(self, s):
        if self.rng is None or s <= 0:
            return np.zeros(2)
        return self.rng.normal(0.0, s, 2)

    def _feed_of(self, t_until):
        """OF frames on the video grid in (t_kf, t_until)."""
        ts = self.t_kf + FRAME
        while ts < t_until - 1e-9:
            if self.anchor_z is not None:
                xy = (
                    self.anchor_z
                    + (self.pos(ts) - self.anchor_truth)
                    + self._noise(self.of_noise)
                )
                self.sm.note_of(xy, dt=ts - self.t_success, quality=self.of_quality)
                self.t_success = ts
            ts += FRAME

    def fix(
        self,
        accepted=True,
        tamper=None,
        kf_xy="truth",
        dt=KF_DT,
        with_of=True,
        source_id=None,
    ):
        """Advance one keyframe interval. Returns the smoother correction.

        kf_xy: "truth" -> KF closely follows truth; array-like -> explicit
        (e.g. frozen front-end); None only makes sense with accepted=False.
        """
        t_next = self.t_kf + dt
        if with_of:
            self._feed_of(t_next)
        truth = self.pos(t_next)
        z = truth + self._noise(self.fix_noise)
        if tamper is not None:
            z = z + np.asarray(tamper, dtype=np.float64)

        if accepted:
            kf = (
                truth.copy()
                if isinstance(kf_xy, str)
                else np.asarray(kf_xy, dtype=np.float64)
            )
            corr = self.sm.add_fix(
                z,
                dt=dt,
                confidence=self.conf,
                source_id=source_id,
                accepted=True,
                kf_xy=kf,
            )
            self.t_success = t_next
            self.anchor_z = z.copy()  # OF re-anchors at the RAW fix
            self.anchor_truth = truth.copy()
        else:
            corr = self.sm.add_fix(
                z, dt=dt, confidence=self.conf, source_id=source_id, accepted=False
            )
        self.t_kf = t_next
        return corr

    def run(self, n_nodes, tamper=None, accepted_fn=None, kf_xy="truth"):
        tamper = tamper or {}
        out = []
        for k in range(n_nodes):
            accepted = True if accepted_fn is None else accepted_fn(k)
            out.append(self.fix(accepted=accepted, tamper=tamper.get(k), kf_xy=kf_xy))
        return out


# ── solver correctness ───────────────────────────────────────────────────────


def brute_force_solve(sm) -> np.ndarray:
    """Independent stacked least-squares of the smoother's current window
    (huber weights taken as 1 — call with huber_k large enough)."""
    nodes = sm._nodes
    uid_to_idx = {node.uid: i for i, node in enumerate(nodes)}
    n = len(nodes)
    rows, rhs = [], []
    for i, node in enumerate(nodes):
        w = 1.0 / node.sigma**2
        r = np.zeros(n)
        r[i] = np.sqrt(w)
        rows.append(r)
        rhs.append(np.sqrt(w) * node.z)
        if node.prior is not None:
            pw = 1.0 / node.prior[1] ** 2
            r = np.zeros(n)
            r[i] = np.sqrt(pw)
            rows.append(r)
            rhs.append(np.sqrt(pw) * node.prior[0])
    for e in sm._edges:
        ia, ib = uid_to_idx[e.a], uid_to_idx[e.b]
        r = np.zeros(n)
        r[ib], r[ia] = np.sqrt(e.weight), -np.sqrt(e.weight)
        rows.append(r)
        rhs.append(np.sqrt(e.weight) * e.delta)
    A = np.stack(rows)
    B = np.stack(rhs)  # (m, 2)
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    return sol


def test_pure_ls_matches_brute_force():
    rng = np.random.default_rng(7)
    # huber_k huge -> weights stay 1 -> plain weighted LS
    sm = make_smoother(huber_k=1e9, irls_iterations=3)
    Scenario(
        sm, lambda t: (12.0 * t, 3.0 * t), rng=rng, fix_noise=2.0, of_noise=0.3
    ).run(25)
    p = sm.solve()
    expected = brute_force_solve(sm)
    np.testing.assert_allclose(p, expected, atol=1e-8)


def test_solve_deterministic_and_stateless():
    rng = np.random.default_rng(3)
    sm = make_smoother()
    Scenario(sm, lambda t: (10.0 * t, 0.0), rng=rng, fix_noise=2.0, of_noise=0.3).run(
        20
    )
    p1 = sm.solve()
    p2 = sm.solve()
    np.testing.assert_array_equal(p1, p2)


def test_empty_window_returns_none():
    sm = make_smoother()
    assert sm.solve() is None


# ── robustness ───────────────────────────────────────────────────────────────


def test_huber_suppresses_gross_outlier():
    rng = np.random.default_rng(11)
    sm = make_smoother()
    Scenario(sm, lambda t: (10.0 * t, 0.0), rng=rng, fix_noise=1.0, of_noise=0.2).run(
        21, tamper={10: (0.0, 300.0)}
    )
    p = sm.solve()
    outlier_uid = sm._nodes[10].uid
    # The poisoned node stays on the true line...
    assert abs(p[10][1]) < 3.0, f"deviation {p[10][1]:.2f} m"
    # ...and its fix weight is crushed.
    assert sm.last_fix_weights[outlier_uid] < 0.05


def test_wrong_fix_cluster_outvoted():
    """3 consecutive biased fixes (wrong tile lock) vs consistent OF chain —
    the case the binary Z-score front-end handles worst."""
    rng = np.random.default_rng(13)
    sm = make_smoother()
    off = (80.0, 0.0)
    Scenario(
        sm, lambda t: (15.0 * t, 5.0 * t), rng=rng, fix_noise=1.0, of_noise=0.2
    ).run(20, tamper={8: off, 9: off, 10: off})
    p = sm.solve()
    for k in (8, 9, 10):
        truth = np.array([15.0, 5.0]) * (k + 1) * KF_DT
        dev = np.linalg.norm(p[k] - truth)
        assert dev < 15.0, f"node {k}: deviation {dev:.1f} m"


def test_relocation_recovery_direction_and_clamp():
    """Fixes+OF consistently move to a new area while the front-end KF is
    frozen (Z-score false rejections) — correction must point there, clamped."""
    rng = np.random.default_rng(17)
    sm = make_smoother(max_correction_m=50.0)
    t_move = 8 * KF_DT

    def pos(t):
        if t <= t_move:
            return (0.0, 0.0)
        return (0.0, min(200.0, 100.0 * (t - t_move)))

    corr = Scenario(sm, pos, rng=rng, fix_noise=1.0, of_noise=0.3).run(
        16, kf_xy=(0.0, 0.0)
    )  # front-end frozen at the origin
    last = corr[-1]
    assert last is not None
    assert last[1] > 0, "correction must point toward the new position"
    assert np.linalg.norm(last) == pytest.approx(50.0, abs=1e-6), "clamp expected"


def test_rejected_fixes_enter_window_without_correction():
    rng = np.random.default_rng(19)
    sm = make_smoother()
    corr = Scenario(
        sm, lambda t: (10.0 * t, 0.0), rng=rng, fix_noise=1.0, of_noise=0.2
    ).run(10, accepted_fn=lambda k: k not in (6, 7))
    assert sm.num_nodes == 10  # rejected fixes are still nodes
    assert corr[6] is None and corr[7] is None  # but produce no correction


# ── window mechanics ─────────────────────────────────────────────────────────


def test_window_slide_bounds_and_continuity():
    rng = np.random.default_rng(23)
    sm = make_smoother(window=60)
    corr = Scenario(
        sm, lambda t: (20.0 * t, -7.0 * t), rng=rng, fix_noise=2.0, of_noise=0.3
    ).run(150)
    assert sm.num_nodes == 60
    assert sm.num_edges <= 59
    applied = [c for c in corr[10:] if c is not None]
    assert applied, "corrections expected in steady state"
    for c in applied:
        assert np.all(np.isfinite(c))
        # noisy-but-consistent data: corrections stay at noise scale
        assert np.linalg.norm(c) < 6.0


def test_entry_prior_set_after_slide():
    rng = np.random.default_rng(29)
    sm = make_smoother(window=10)
    Scenario(sm, lambda t: (10.0 * t, 0.0), rng=rng, fix_noise=1.0, of_noise=0.2).run(
        15
    )
    assert sm._nodes[0].prior is not None


def test_source_switch_resets_window():
    sm = make_smoother()
    for k in range(6):
        sm.add_fix(
            (k * 2.0, 0.0),
            dt=KF_DT,
            confidence=0.8,
            source_id="A",
            kf_xy=(k * 2.0, 0.0),
        )
    assert sm.num_nodes == 6
    sm.add_fix(
        (1000.0, 1000.0),
        dt=KF_DT,
        confidence=0.8,
        source_id="B",
        kf_xy=(1000.0, 1000.0),
    )
    assert sm.num_nodes == 1
    assert sm.num_edges == 0


def test_min_nodes_gate():
    sm = make_smoother()
    corrs = []
    for k in range(5):
        corrs.append(
            sm.add_fix(
                (k * 2.0, 0.0), dt=KF_DT, confidence=0.8, kf_xy=(k * 2.0 + 1.0, 1.0)
            )
        )
    assert all(c is None for c in corrs[:4])
    assert corrs[4] is not None  # 5th node, kf offset (1,1) -> correction


# ── odometry edge construction ───────────────────────────────────────────────


def test_of_edge_delta_extrapolated():
    sm = make_smoother()
    v = np.array([30.0, 0.0])
    sc = Scenario(sm, lambda t: v * t)
    sc.fix()
    sc.fix()
    assert sm.num_edges == 1
    np.testing.assert_allclose(sm._edges[0].delta, v * KF_DT, atol=0.1)


def test_no_of_no_edge():
    sm = make_smoother()
    sm.add_fix((0.0, 0.0), dt=KF_DT, confidence=0.9, kf_xy=(0.0, 0.0))
    sm.add_fix((2.0, 0.0), dt=KF_DT, confidence=0.9, kf_xy=(2.0, 0.0))
    assert sm.num_edges == 0


def test_stale_of_no_edge():
    sm = make_smoother()
    sm.add_fix((0.0, 0.0), dt=KF_DT, confidence=0.9, kf_xy=(0.0, 0.0))
    sm.note_of((0.5, 0.0), dt=FRAME, quality=0.9)
    # next keyframe a full second later -> the OF sample is stale
    sm.add_fix((10.0, 0.0), dt=1.0, confidence=0.9, kf_xy=(10.0, 0.0))
    assert sm.num_edges == 0


def test_of_before_first_fix_ignored():
    sm = make_smoother()
    sm.note_of((5.0, 5.0), dt=FRAME, quality=0.9)  # no anchor yet
    sm.add_fix((0.0, 0.0), dt=KF_DT, confidence=0.9, kf_xy=(0.0, 0.0))
    assert sm.num_nodes == 1
    assert sm.num_edges == 0


def test_edge_across_rejected_node():
    """OF track survives a rejected keyframe: edges exist on both sides of it
    (anchored at the last accepted keyframe)."""
    sm = make_smoother()
    v = np.array([10.0, 0.0])
    sc = Scenario(sm, lambda t: v * t)
    sc.fix()  # accepted anchor
    sc.fix(accepted=False)  # rejected: OF track continues through it
    sc.fix()  # accepted again
    assert sm.num_edges == 2
    for e in sm._edges:
        np.testing.assert_allclose(e.delta, v * KF_DT, atol=0.15)


def test_wrong_anchor_offset_cancels_in_edges():
    """A wrong-tile ACCEPTED fix offsets both the anchor and its OF track;
    edge deltas must stay clean (the offset cancels)."""
    sm = make_smoother()
    v = np.array([10.0, 0.0])
    sc = Scenario(sm, lambda t: v * t)
    sc.fix()
    sc.fix(tamper=(0.0, 120.0))  # wrong tile: fix AND subsequent OF offset
    sc.fix()
    assert sm.num_edges == 2
    for e in sm._edges:
        np.testing.assert_allclose(e.delta, v * KF_DT, atol=0.15)


# ── performance ──────────────────────────────────────────────────────────────


def test_solve_budget_120_nodes():
    rng = np.random.default_rng(31)
    sm = make_smoother(window=120)
    Scenario(
        sm, lambda t: (25.0 * t, 10.0 * t), rng=rng, fix_noise=2.0, of_noise=0.3
    ).run(130)
    assert sm.num_nodes == 120
    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        sm.solve()
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    assert avg_ms < 50.0, f"solve too slow: {avg_ms:.2f} ms"


# ── TrajectoryFilter.shift ───────────────────────────────────────────────────


def test_trajectory_filter_shift():
    tf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
    tf.update((100.0, 200.0), dt=0.2)
    tf.update((102.0, 201.0), dt=0.2)
    x_before = tf.kf.x.copy()
    tf.shift(5.0, -3.0)
    assert tf.kf.x[0, 0] == pytest.approx(x_before[0, 0] + 5.0)
    assert tf.kf.x[1, 0] == pytest.approx(x_before[1, 0] - 3.0)
    # velocities untouched
    assert tf.kf.x[2, 0] == pytest.approx(x_before[2, 0])
    assert tf.kf.x[3, 0] == pytest.approx(x_before[3, 0])


def test_trajectory_filter_shift_before_init_is_noop():
    tf = TrajectoryFilter()
    tf.shift(5.0, 5.0)  # must not raise, must not mark initialized
    assert not tf.is_initialized
