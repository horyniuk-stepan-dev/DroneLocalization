"""Odometry-consistency (PCM-lite, Етап 2.3): spatial-ребро, несумісне з
temporal-ланцюгом, отримує вагу ×factor (не викидання).
"""
import numpy as np

from src.geometry.pose_graph.model_5dof import _state_to_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

W, H = 1280, 720
CX, CY = W / 2.0, H / 2.0
STEP = 200.0


def _anchor(cx_m, cy_m):
    return _state_to_affine(np.array([cx_m, cy_m, 0.0, 0.0, 0.0]), CX, CY, sign=1.0)


def _rel(dx, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _chain(n=9):
    opt = PoseGraphOptimizer(W, H)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        opt.add_edge(i, i + 1, _rel(STEP), weight=1.0, edge_type="temporal")
    return opt


def _prelim(opt):
    return opt.preliminary_states({0: _anchor(0.0, 0.0)})


def test_consistent_edge_keeps_full_weight():
    opt = _chain()
    ps = _prelim(opt)
    # ребро (0,6), що ставить node6 туди ж, куди й ланцюг → сумісне
    specs = [{"i": 0, "j": 6, "similarity": _rel(STEP * 6, 0.0)}]
    f = opt.odometry_consistency_factors(specs, ps, W, H,
                                         margin=1.5, drift_frac=0.25, factor=0.3)
    assert f == [1.0]


def test_inconsistent_edge_downweighted():
    opt = _chain()
    ps = _prelim(opt)
    # аліас: node6 зсунуто на 4000 м вбік від ланцюга → несумісно
    specs = [{"i": 0, "j": 6, "similarity": _rel(STEP * 6, 4000.0)}]
    f = opt.odometry_consistency_factors(specs, ps, W, H,
                                         margin=1.5, drift_frac=0.25, factor=0.3)
    assert f == [0.3]


def test_mixed_specs():
    opt = _chain()
    ps = _prelim(opt)
    specs = [
        {"i": 0, "j": 5, "similarity": _rel(STEP * 5, 0.0)},       # сумісне
        {"i": 1, "j": 7, "similarity": _rel(STEP * 6, 5000.0)},    # аліас
        {"i": 2, "j": 8, "similarity": _rel(STEP * 6, 0.0)},       # сумісне
    ]
    f = opt.odometry_consistency_factors(specs, ps, W, H, factor=0.25)
    assert f == [1.0, 0.25, 1.0]


def test_missing_prelim_is_conservative():
    opt = _chain()
    ps = _prelim(opt)
    # j=999 немає у прикидці → судити не можемо → повна вага
    specs = [{"i": 0, "j": 999, "similarity": _rel(STEP * 6, 9999.0)}]
    f = opt.odometry_consistency_factors(specs, ps, W, H)
    assert f == [1.0]
    assert opt.odometry_consistency_factors([], ps, W, H) == []
    assert opt.odometry_consistency_factors(specs, {}, W, H) == [1.0]


def test_drift_tolerance_grows_with_chain_length():
    """Однаковий боковий зсув: короткий ланцюг ріже, дуже довгий — толерує більше.
    Перевіряємо, що допуск монотонно зростає з |i−j| (компенсація дрейфу)."""
    opt = _chain(n=9)
    ps = _prelim(opt)
    lateral = 2600.0  # трохи більше за базовий допуск (~1.5 діагоналі ≈ 2203 м)
    near = opt.odometry_consistency_factors(
        [{"i": 5, "j": 6, "similarity": _rel(STEP, lateral)}], ps, W, H)
    far = opt.odometry_consistency_factors(
        [{"i": 0, "j": 8, "similarity": _rel(STEP * 8, lateral)}], ps, W, H)
    assert near == [0.3]        # короткий ланцюг: поза допуском
    assert far == [1.0]         # довгий ланцюг: у межах збільшеного допуску


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
