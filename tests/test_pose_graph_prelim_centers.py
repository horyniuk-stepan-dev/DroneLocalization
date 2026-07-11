"""Дистанційний префільтр (Етап 2.2): preliminary_centers — прикидка центрів
BFS-ланцюгом ЛИШЕ по temporal-ребрах від якорів, ДО матчингу/оптимізації.
"""

import numpy as np

from src.geometry.pose_graph.model_5dof import _state_to_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

CX, CY = 640.0, 360.0  # 1280x720


def _anchor(cx_m, cy_m):
    return _state_to_affine(np.array([cx_m, cy_m, 0.0, 0.0, 0.0]), CX, CY, sign=1.0)


def _rel(dx, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _chain(n=7, step=200.0, spatial=False):
    opt = PoseGraphOptimizer(1280, 720)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        opt.add_edge(i, i + 1, _rel(step), weight=1.0, edge_type="temporal")
    if spatial:
        opt.add_edge(0, n - 1, _rel(step * (n - 1)), weight=1.0, edge_type="spatial")
    return opt


def test_prelim_centers_follow_temporal_chain():
    opt = _chain(n=7, step=200.0)
    centers = opt.preliminary_centers({0: _anchor(0.0, 0.0)})
    assert set(centers) == set(range(7))
    for i in range(7):
        assert np.allclose(centers[i], [200.0 * i, 0.0], atol=1e-6)


def test_prelim_only_temporal_edges():
    # spatial-ребро НЕ має впливати на прикидку (лише temporal-ланцюг)
    opt = _chain(n=7, step=200.0, spatial=True)
    centers = opt.preliminary_centers({0: _anchor(0.0, 0.0)})
    assert np.allclose(centers[6], [1200.0, 0.0], atol=1e-6)


def test_prelim_distance_prefilter_semantics():
    opt = _chain(n=7, step=200.0)
    centers = opt.preliminary_centers({0: _anchor(0.0, 0.0)})
    # frame diag ~1469 px * scale 1 m/px; margin 2 → threshold ~2939 m
    thr = 2.0 * np.hypot(1280, 720) * 1.0
    far = np.linalg.norm(centers[0] - centers[6])  # 1200 m
    assert far < thr  # сусідні по лінії ще нижче порога (правильно не ріжемо)
    # штучно далека пара — за порогом
    assert np.linalg.norm(np.array([0.0, 0.0]) - np.array([5000.0, 0.0])) > thr


def test_prelim_non_mutating_and_empty():
    opt = _chain(n=5)
    free_before = {k: v.copy() for k, v in opt._free_nodes.items()}
    opt.preliminary_centers({0: _anchor(0.0, 0.0)})
    for k, v in free_before.items():
        np.testing.assert_array_equal(opt._free_nodes[k], v)
    assert opt.preliminary_centers({}) == {}
    assert opt.preliminary_centers({999: _anchor(0.0, 0.0)}) == {}  # seed not a node


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
