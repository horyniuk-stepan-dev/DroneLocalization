"""LOO-валідація якорів (Етап 1.2, read-only): ланцюг сусідніх якорів
передбачає стан якоря; велика розбіжність → warning.

Перевіряє: (1) добрий якір на консистентному ланцюзі → normal;
(2) навмисно зсунутий якір → warning із розбіжністю ≈ зсуву;
(3) метод не мутує стан оптимізатора; працює і для fix_node, і для add_anchor.
"""

import numpy as np

from src.geometry.pose_graph.model_5dof import _state_to_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

CX, CY = 960.0, 540.0


def _anchor_affine(cx_m, cy_m):
    return _state_to_affine(np.array([cx_m, cy_m, 0.0, 0.0, 0.0]), CX, CY, sign=1.0)


def _step(dx, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _chain(anchor_specs, n=7, step=100.0):
    """anchor_specs: list of (node_id, cx_m, cy_m). Ланцюг темпоральних ребер +100."""
    opt = PoseGraphOptimizer(1920, 1080)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        opt.add_edge(i, i + 1, _step(step), weight=1.0, edge_type="temporal")
    for nid, cx_m, cy_m in anchor_specs:
        opt.fix_node(nid, _anchor_affine(cx_m, cy_m))
    return opt


def test_loo_good_anchors_all_normal():
    # усі якорі на істинній прямій (100*i, 0)
    opt = _chain([(0, 0, 0), (3, 300, 0), (6, 600, 0)])
    loo = opt.leave_one_out_anchor_check(threshold_m=5.0)
    assert set(loo) == {0, 3, 6}
    for fid, v in loo.items():
        assert v["reachable"] and v["flag"] == "normal", (fid, v)
        assert v["disagreement_m"] < 1e-6


def test_loo_flags_bad_anchor():
    # Щільні якорі; #3 зсунутий +40 м. Добрі сусіди #2 і #4 з обох боків
    # ізолюють кривий якір: тільки #3 конфліктує з ОБОМА боками.
    opt = _chain([(0, 0, 0), (2, 200, 0), (3, 300, 40), (4, 400, 0), (6, 600, 0)])
    loo = opt.leave_one_out_anchor_check(threshold_m=5.0)
    assert loo[3]["flag"] == "warning"
    assert abs(loo[3]["disagreement_m"] - 40.0) < 1e-6
    # добрі якорі (в т.ч. безпосередні сусіди кривого) лишаються normal
    for fid in (0, 2, 4, 6):
        assert loo[fid]["flag"] == "normal", (fid, loo[fid])


def test_loo_does_not_mutate_state():
    opt = _chain([(0, 0, 0), (3, 300, 40), (6, 600, 0)])
    free_before = {k: v.copy() for k, v in opt._free_nodes.items()}
    init_before = set(opt._initialized_nodes)
    opt.leave_one_out_anchor_check()
    assert set(opt._initialized_nodes) == init_before
    for k, v in free_before.items():
        np.testing.assert_array_equal(opt._free_nodes[k], v)


def test_loo_single_anchor_returns_empty():
    opt = _chain([(0, 0, 0)])
    assert opt.leave_one_out_anchor_check() == {}


def test_loo_works_for_soft_anchors():
    opt = PoseGraphOptimizer(1920, 1080)
    for i in range(7):
        opt.add_node(i)
    for i in range(6):
        opt.add_edge(i, i + 1, _step(100.0), weight=1.0, edge_type="temporal")
    for nid, cxm, cym in [(0, 0, 0), (2, 200, 0), (3, 300, 40), (4, 400, 0), (6, 600, 0)]:
        opt.add_anchor(nid, _anchor_affine(cxm, cym), sigma_m=5.0)
    loo = opt.leave_one_out_anchor_check(threshold_m=5.0)
    assert loo[3]["flag"] == "warning" and abs(loo[3]["disagreement_m"] - 40.0) < 1e-6
    assert loo[2]["flag"] == "normal" and loo[4]["flag"] == "normal"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
