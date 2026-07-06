"""
Верифікація аналітичного якобіана (Етап 4.1 плану).

Гарантія неполомки: на випадкових станах аналітичний J має збігатися з
чисельним (central-difference) якобіаном _residuals_vec. Та сама модель —
лише точніші/швидші градієнти. Плюс: розв'язок із use_analytic_jac=True
збігається з дефолтним (2-point FD) у межах толерансу.
"""

import numpy as np

from src.geometry.affine_utils import compose_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer


def _make_d(X_fixed, free_indices, edges_from, edges_to, edge_from_free,
            edge_to_free, n_free, edge_params, cx=960.0, sign=1.0):
    ne = len(edges_from)
    return {
        "X_full": X_fixed.copy(),
        "free_indices": list(free_indices),
        "edges_from": np.array(edges_from, dtype=np.int32),
        "edges_to": np.array(edges_to, dtype=np.int32),
        "dtx": np.array(edge_params["dtx"], dtype=np.float64),
        "dty": np.array(edge_params["dty"], dtype=np.float64),
        "log_dsx": np.array(edge_params["log_dsx"], dtype=np.float64),
        "log_dsy": np.array(edge_params["log_dsy"], dtype=np.float64),
        "dtheta": np.array(edge_params["dtheta"], dtype=np.float64),
        "weights": np.array(edge_params["weights"], dtype=np.float64),
        "cx": cx, "sign": sign, "n_edges": ne, "n_free": n_free,
        "edge_from_free": np.array(edge_from_free, dtype=np.int64),
        "edge_to_free": np.array(edge_to_free, dtype=np.int64),
    }


def _num_jac(opt, x, d, eps=1e-6):
    n = x.size
    m = opt._residuals_vec(x.copy(), d).size
    J = np.zeros((m, n))
    for k in range(n):
        xp = x.copy()
        xp[k] += eps
        fp = opt._residuals_vec(xp, d).copy()
        xm = x.copy()
        xm[k] -= eps
        fm = opt._residuals_vec(xm, d).copy()
        J[:, k] = (fp - fm) / (2 * eps)
    return J


def _max_rel_diff(Ja, Jn):
    denom = np.maximum(np.abs(Jn), 1.0)
    return float(np.max(np.abs(Ja - Jn) / denom))


class TestAnalyticJacobianVsFiniteDiff:
    def test_two_free_nodes_one_edge_random_states(self):
        opt = PoseGraphOptimizer()
        rng = np.random.default_rng(7)
        for _ in range(20):
            d = _make_d(
                X_fixed=np.zeros((2, 5)),
                free_indices=[0, 1], edges_from=[0], edges_to=[1],
                edge_from_free=[0], edge_to_free=[1], n_free=2,
                edge_params={"dtx": [rng.uniform(-20, 20)], "dty": [rng.uniform(-20, 20)],
                             "log_dsx": [rng.uniform(-0.3, 0.3)], "log_dsy": [rng.uniform(-0.3, 0.3)],
                             "dtheta": [rng.uniform(-1, 1)], "weights": [rng.uniform(0.3, 3.0)]},
            )
            x = rng.uniform(-2, 2, size=10)
            x[2] = rng.uniform(-0.5, 0.5)  # log-scales moderate
            x[3] = rng.uniform(-0.5, 0.5)
            x[7] = rng.uniform(-0.5, 0.5)
            x[8] = rng.uniform(-0.5, 0.5)
            Ja = opt._jacobian_vec(x, d).toarray()
            Jn = _num_jac(opt, x, d)
            assert _max_rel_diff(Ja, Jn) < 1e-5, "аналітичний J розійшовся з FD"

    def test_with_fixed_node(self):
        """Одне ребро fixed(0) -> free(1): from-стовпців немає, тільки to."""
        opt = PoseGraphOptimizer()
        rng = np.random.default_rng(11)
        X_fixed = np.zeros((2, 5))
        X_fixed[0] = [1.0, -2.0, 0.1, -0.2, 0.3]  # фіксований вузол 0
        d = _make_d(
            X_fixed=X_fixed, free_indices=[1], edges_from=[0], edges_to=[1],
            edge_from_free=[-1], edge_to_free=[0], n_free=1,
            edge_params={"dtx": [7.0], "dty": [-3.0], "log_dsx": [0.05],
                         "log_dsy": [-0.1], "dtheta": [0.2], "weights": [1.5]},
        )
        x = rng.uniform(-1, 1, size=5)
        Ja = opt._jacobian_vec(x, d).toarray()
        Jn = _num_jac(opt, x, d)
        assert Ja.shape == (5 * 1 + 1, 5)
        assert _max_rel_diff(Ja, Jn) < 1e-5

    def test_multi_edge_chain(self):
        opt = PoseGraphOptimizer()
        rng = np.random.default_rng(3)
        # 3 free вузли, 2 ребра 0->1, 1->2
        d = _make_d(
            X_fixed=np.zeros((3, 5)),
            free_indices=[0, 1, 2], edges_from=[0, 1], edges_to=[1, 2],
            edge_from_free=[0, 1], edge_to_free=[1, 2], n_free=3,
            edge_params={"dtx": [5.0, -4.0], "dty": [2.0, 3.0],
                         "log_dsx": [0.02, -0.03], "log_dsy": [0.01, 0.04],
                         "dtheta": [0.1, -0.15], "weights": [1.0, 2.0]},
        )
        x = rng.uniform(-1, 1, size=15)
        Ja = opt._jacobian_vec(x, d).toarray()
        Jn = _num_jac(opt, x, d)
        assert Ja.shape == (5 * 2 + 3, 15)
        assert _max_rel_diff(Ja, Jn) < 1e-5


def _sim(tx, ty, scale, angle_deg):
    return compose_affine(tx, ty, scale, np.radians(angle_deg))


class TestAnalyticJacGivesSameSolution:
    def _ring(self):
        opt = PoseGraphOptimizer()
        opt.fix_node(0, _sim(0.0, 0.0, 1.0, 0.0))
        for i in range(1, 5):
            opt.add_node(i)
        step = _sim(100.0, 0.0, 1.0, 0.0)
        for i in range(4):
            opt.add_edge(i, i + 1, step, weight=1.0, edge_type="temporal")
        opt.add_edge(4, 0, _sim(-400.0, 0.0, 1.0, 0.0), weight=2.0, edge_type="spatial")
        opt.initialize_from_bfs()
        return opt

    def test_analytic_matches_fd_solution(self):
        r_fd = self._ring().optimize(max_iterations=80, tolerance=1e-12,
                                     use_analytic_jac=False)
        r_an = self._ring().optimize(max_iterations=80, tolerance=1e-12,
                                     use_analytic_jac=True)
        for k in r_fd:
            np.testing.assert_allclose(r_fd[k], r_an[k], atol=1e-4,
                                       err_msg=f"розв'язки FD vs analytic різні на вузлі {k}")
