"""Robust odometry edges in SlidingWindowSmoother (LOCALIZER_IMPROVEMENT_PLAN 1.2).

A collapsed OF track produces one odometry edge that contradicts the fixes.
Quadratic edges (legacy, odom_huber_k=0) pass that error into the window at
full weight; Huber-IRLS on the normalized edge residual suppresses it.
"""

import numpy as np

from src.tracking.smoother import SlidingWindowSmoother

V = 10.0  # m/s along x


def _truth(t):
    return np.array([V * t, 0.0])


def _build(odom_huber_k, n=20, irls_iterations=4):
    sm = SlidingWindowSmoother(
        window=60, odom_huber_k=odom_huber_k, irls_iterations=irls_iterations
    )
    t = 0.0
    sm.add_fix(_truth(t), dt=1.0, confidence=1.0, kf_xy=_truth(t))
    for _ in range(n - 1):
        sm.note_of(_truth(t + 0.5), dt=0.5, quality=1.0)
        sm.note_of(_truth(t + 0.9), dt=0.4, quality=1.0)
        t += 1.0
        sm.add_fix(_truth(t), dt=1.0, confidence=1.0, kf_xy=_truth(t))
    return sm


def _errors(sm):
    p = sm.solve()
    truth = np.stack([_truth(float(i)) for i in range(sm.num_nodes)])
    return np.linalg.norm(p - truth, axis=1)


def test_clean_edges_built_and_exact():
    sm = _build(0.0)
    assert sm.num_edges == sm.num_nodes - 1
    assert _errors(sm).max() < 1e-6


def test_flag_off_is_bitwise_legacy():
    a = SlidingWindowSmoother(window=60)
    b = SlidingWindowSmoother(window=60, odom_huber_k=0.0)
    for sm in (a, b):
        t = 0.0
        sm.add_fix(_truth(t) + [1.0, -2.0], dt=1.0, confidence=0.6, kf_xy=_truth(t))
        for k in range(8):
            sm.note_of(_truth(t + 0.5) + [0.3 * k, 0.0], dt=0.5, quality=0.8)
            sm.note_of(_truth(t + 0.9), dt=0.4, quality=0.8)
            t += 1.0
            sm.add_fix(_truth(t) + [(-1) ** k * 3.0, 1.0], dt=1.0, confidence=0.7)
    assert np.array_equal(a.solve(), b.solve())
    assert b.last_edge_weights == [1.0] * b.num_edges


def _corrupt(sm, bad=9):
    sm._edges[bad].delta = sm._edges[bad].delta + np.array([60.0, 0.0])
    return sm


def test_collapsed_edge_is_suppressed_at_default_iterations():
    err_legacy = _errors(_corrupt(_build(0.0)))
    robust = _corrupt(_build(1.2))
    err_robust = _errors(robust)
    assert err_legacy.max() > 10.0  # the failure the flag exists for
    assert err_robust.max() < 0.5 * err_legacy.max()
    assert robust.last_edge_weights[9] < 0.2


def test_more_iterations_release_neighbour_edges():
    # At 4 IRLS iterations the edges next to the collapsed one are still
    # partly down-weighted (measured 0.57); 8 iterations converge.
    err_legacy = _errors(_corrupt(_build(0.0, irls_iterations=8)))
    robust = _corrupt(_build(1.2, irls_iterations=8))
    err_robust = _errors(robust)
    assert err_robust.max() < 0.25 * err_legacy.max()
    w = robust.last_edge_weights
    assert w[9] < 0.2
    assert min(x for i, x in enumerate(w) if i != 9) > 0.9


def test_hostile_odom_huber_k_disables_instead_of_breaking():
    for k in (-1.0, float("nan"), 0.0):
        assert SlidingWindowSmoother(odom_huber_k=k).odom_huber_k == 0.0
    assert SlidingWindowSmoother(odom_huber_k=0.01).odom_huber_k == 0.1
