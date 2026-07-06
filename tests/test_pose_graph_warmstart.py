"""Тест теплого старту (Етап 4.2): x0 з попереднього розв'язку, без BFS з нуля."""

import numpy as np

from src.geometry.affine_utils import compose_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer


def _sim(tx, ty, scale=1.0, angle_deg=0.0):
    return compose_affine(tx, ty, scale, np.radians(angle_deg))


def _ring():
    opt = PoseGraphOptimizer()
    opt.fix_node(0, _sim(0.0, 0.0))
    for i in range(1, 5):
        opt.add_node(i)
    for i in range(4):
        opt.add_edge(i, i + 1, _sim(100.0, 0.0), weight=1.0, edge_type="temporal")
    opt.add_edge(4, 0, _sim(-400.0, 0.0), weight=2.0, edge_type="spatial")
    return opt


class TestWarmStart:
    def test_warm_start_matches_bfs_solution(self):
        # 1) базовий розв'язок через BFS
        base = _ring()
        base.initialize_from_bfs()
        r_bfs = base.optimize(max_iterations=80, tolerance=1e-12, use_analytic_jac=True)

        # 2) новий оптимізатор, тепла ініціалізація з r_bfs (БЕЗ BFS)
        warm = _ring()
        seeded = warm.warm_start_from_affines(r_bfs)
        assert seeded == 4, f"мали засіяти 4 вільні вузли, засіяно {seeded}"
        r_warm = warm.optimize(max_iterations=80, tolerance=1e-12, use_analytic_jac=True)

        for k in r_bfs:
            np.testing.assert_allclose(r_bfs[k], r_warm[k], atol=1e-4,
                                       err_msg=f"теплий старт дав інший розв'язок на {k}")

    def test_warm_start_skips_anchors(self):
        opt = _ring()
        # афіни містять і якір 0 — його чіпати не можна
        affines = {0: _sim(999.0, 999.0), 1: _sim(100.0, 0.0)}
        seeded = opt.warm_start_from_affines(affines)
        assert seeded == 1  # тільки вузол 1, якір 0 пропущено
        assert 0 in opt._fixed_nodes
