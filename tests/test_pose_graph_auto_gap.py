"""Авто min_frame_gap із геометрії руху (Етап 2.1).

gap_min = ceil(overlap · frame_diag_px / median_disp_px), median_disp з
temporal-ребер (нормований на span ребра). Дуги/викиди не мусять ламати медіану.
"""

import math

import numpy as np

from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

W, H = 1280, 720
DIAG = math.hypot(W, H)


def _rel(dx, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _opt_with_temporal(disps, spans=None):
    opt = PoseGraphOptimizer(W, H)
    n = len(disps) + 1
    for i in range(n):
        opt.add_node(i)
    fid = 0
    for k, d in enumerate(disps):
        span = 1 if spans is None else spans[k]
        opt.add_edge(fid, fid + span, _rel(d), weight=1.0, edge_type="temporal")
        fid += span
    return opt


def test_auto_gap_basic():
    disp = 288.0
    opt = _opt_with_temporal([disp] * 5)
    gap = opt.estimate_min_loop_gap(W, H, k_overlap=1.0)
    assert gap == math.ceil(DIAG / disp)  # ~6


def test_auto_gap_span_normalized():
    # ребро через 2 слоти з подвійним зсувом → той самий рух за слот
    opt = _opt_with_temporal([288.0, 576.0, 288.0], spans=[1, 2, 1])
    gap = opt.estimate_min_loop_gap(W, H, k_overlap=1.0)
    assert gap == math.ceil(DIAG / 288.0)


def test_auto_gap_overlap_factor_scales():
    opt = _opt_with_temporal([300.0] * 4)
    g1 = opt.estimate_min_loop_gap(W, H, k_overlap=1.0)
    g2 = opt.estimate_min_loop_gap(W, H, k_overlap=2.0)
    assert g2 >= 2 * g1 - 1  # приблизно вдвічі більший


def test_auto_gap_none_without_temporal():
    opt = PoseGraphOptimizer(W, H)
    opt.add_node(0)
    opt.add_node(50)
    opt.add_edge(0, 50, _rel(10.0), weight=1.0, edge_type="spatial")
    assert opt.estimate_min_loop_gap(W, H) is None


def test_auto_gap_robust_to_outlier():
    # одна дуга з величезним зсувом не має завищити медіану
    opt = _opt_with_temporal([288.0, 288.0, 288.0, 5000.0, 288.0])
    gap = opt.estimate_min_loop_gap(W, H, k_overlap=1.0)
    assert gap == math.ceil(DIAG / 288.0)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
