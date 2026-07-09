"""М'які якорі (Етап 1.1): унарний фактор замість жорсткого fix_node.

Перевіряє:
1. Аналітичний якобіан із анкер-рядками збігається з чисельним (FD) — валідність
   вручну виписаного блоку d(w_a·Δstate)/d(state)=w_a·I.
2. М'який якір із σ→floor відтворює розв'язок fix_node (сим-інваріант: бенчмарк
   на GT-якорях σ≈0 не змінюється).
3. КЛЮЧОВЕ: хибний якір (RMSE великий) із soft_anchors «переголосовується»
   узгодженим ланцюгом → median-похибка МЕНША, ніж із жорстким fix_node.
"""
import numpy as np

from src.geometry.pose_graph.model_5dof import _state_to_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

CX, CY = 960.0, 540.0


def _anchor_affine(cx_m, cy_m, log_sx=0.0, log_sy=0.0, theta=0.0):
    return _state_to_affine(
        np.array([cx_m, cy_m, log_sx, log_sy, theta]), CX, CY, sign=1.0
    )


def _step(dx, dy=0.0):
    """Відносна афінна ребра: центр дитини = центр батька + (dx, dy)."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _center(affine):
    return affine[:2, :2] @ np.array([CX, CY]) + affine[:2, 2]


# ── 1. Якобіан із анкерами vs FD ─────────────────────────────────────────────


def _num_jac(opt, x, d, eps=1e-6):
    m = opt._residuals_vec(x.copy(), d).size
    J = np.zeros((m, x.size))
    for k in range(x.size):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        J[:, k] = (opt._residuals_vec(xp, d) - opt._residuals_vec(xm, d)) / (2 * eps)
    return J


def test_analytic_jacobian_with_anchors_matches_fd():
    opt = PoseGraphOptimizer(1920, 1080)
    rng = np.random.default_rng(5)
    # 2 вільні вузли, 1 ребро 0->1, анкери на ОБОХ вузлах
    d = {
        "X_full": np.zeros((2, 5)),
        "free_indices": [0, 1],
        "edges_from": np.array([0], dtype=np.int32),
        "edges_to": np.array([1], dtype=np.int32),
        "dtx": np.array([12.0]), "dty": np.array([-5.0]),
        "log_dsx": np.array([0.05]), "log_dsy": np.array([-0.03]),
        "dtheta": np.array([0.2]), "weights": np.array([1.4]),
        "cx": 960.0, "sign": 1.0, "n_edges": 1, "n_free": 2,
        "edge_from_free": np.array([0], dtype=np.int64),
        "edge_to_free": np.array([1], dtype=np.int64),
        "anchor_var_idx": np.array([0, 1], dtype=np.int64),
        "anchor_states": np.array([[1.0, -2.0, 0.1, -0.1, 0.3],
                                   [3.0, 1.0, -0.05, 0.2, -0.2]]),
        "anchor_w": np.array([30.0, 5.0]),
        "n_anch": 2,
    }
    for _ in range(15):
        x = rng.uniform(-2, 2, size=10)
        x[[2, 3, 7, 8]] = rng.uniform(-0.4, 0.4, size=4)
        Ja = opt._jacobian_vec(x, d).toarray()
        Jn = _num_jac(opt, x, d)
        assert Ja.shape == Jn.shape == (5 * 1 + 2 + 5 * 2, 10)
        denom = np.maximum(np.abs(Jn), 1.0)
        assert float(np.max(np.abs(Ja - Jn) / denom)) < 1e-5


# ── 2. σ→floor відтворює fix_node ────────────────────────────────────────────


def _build_chain(anchor_fn, n=7, step=100.0, tw=1.0, spatial=False):
    opt = PoseGraphOptimizer(1920, 1080)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        opt.add_edge(i, i + 1, _step(step), weight=tw, edge_type="temporal")
    if spatial:
        # Реалістичне підкріплення прямої ноги cross-leg-замиканнями, які
        # брекетять хибний вузол консенсусом добрих сусідів.
        opt.add_edge(0, n - 1, _step(step * (n - 1)), weight=tw, edge_type="spatial")
        opt.add_edge(1, n - 2, _step(step * (n - 3)), weight=tw, edge_type="spatial")
        opt.add_edge(2, n - 3, _step(step * (n - 5)), weight=tw, edge_type="spatial")
    anchor_fn(opt)
    opt.initialize_from_bfs()
    return opt


def test_soft_anchor_near_hard_matches_fix_node():
    n, step = 7, 100.0

    def hard(opt):
        opt.fix_node(0, _anchor_affine(0.0, 0.0))
        opt.fix_node(n - 1, _anchor_affine(step * (n - 1), 0.0))

    def soft(opt):
        opt.add_anchor(0, _anchor_affine(0.0, 0.0), sigma_m=0.0,
                       base_w=200.0, sigma_floor=0.05)
        opt.add_anchor(n - 1, _anchor_affine(step * (n - 1), 0.0), sigma_m=0.0,
                       base_w=200.0, sigma_floor=0.05)

    r_hard = _build_chain(hard, n, step).optimize(max_iterations=100, tolerance=1e-12)
    r_soft = _build_chain(soft, n, step).optimize(max_iterations=100, tolerance=1e-12)
    for i in range(n):
        d = np.linalg.norm(_center(r_hard[i]) - _center(r_soft[i]))
        assert d < 0.05, f"вузол {i}: розбіжність soft↔hard {d:.4f} м"


# ── 3. Хибний якір: soft переголосовує, hard гне граф ────────────────────────


def test_soft_anchors_outvote_bad_anchor():
    """Пряма лінія nodes 0..6 @ (100i, 0). Якір #3 хибний (+40 м вбік), із
    великим RMSE. Ланцюг + два добрі кінці мають його переголосувати."""
    n, step = 7, 100.0
    gt = {i: np.array([step * i, 0.0]) for i in range(n)}
    bad_id, bad_offset = 3, 40.0

    def hard(opt):
        opt.fix_node(0, _anchor_affine(0.0, 0.0))
        opt.fix_node(n - 1, _anchor_affine(step * (n - 1), 0.0))
        opt.fix_node(bad_id, _anchor_affine(step * bad_id, bad_offset))  # хибний

    def soft(opt):
        # добрі кінці — довірені (σ мала); хибний — з великим RMSE (σ велика)
        opt.add_anchor(0, _anchor_affine(0.0, 0.0), sigma_m=0.05)
        opt.add_anchor(n - 1, _anchor_affine(step * (n - 1), 0.0), sigma_m=0.05)
        opt.add_anchor(bad_id, _anchor_affine(step * bad_id, bad_offset), sigma_m=120.0)

    kw = dict(tw=3.0, spatial=True)
    opt_kw = dict(max_iterations=100, tolerance=1e-10, use_analytic_jac=True)
    r_hard = _build_chain(hard, n, step, **kw).optimize(**opt_kw)
    r_soft = _build_chain(soft, n, step, **kw).optimize(**opt_kw)

    e_hard = np.array([np.linalg.norm(_center(r_hard[i]) - gt[i]) for i in range(n)])
    e_soft = np.array([np.linalg.norm(_center(r_soft[i]) - gt[i]) for i in range(n)])

    # хибний вузол: жорсткий пришпилений на 40 м; м'який підтягнутий ланцюгом
    assert e_hard[bad_id] > 35.0, f"hard bad-node err {e_hard[bad_id]:.1f} (очікували ~40)"
    assert e_soft[bad_id] < e_hard[bad_id] * 0.5, (
        f"soft не переголосував: {e_soft[bad_id]:.1f} vs hard {e_hard[bad_id]:.1f}")
    # загальна якість краща
    assert np.median(e_soft) < np.median(e_hard), (
        f"median soft {np.median(e_soft):.2f} !< hard {np.median(e_hard):.2f}")
    print(f"median err: hard={np.median(e_hard):.2f} м, soft={np.median(e_soft):.2f} м; "
          f"bad-node: hard={e_hard[bad_id]:.1f} м, soft={e_soft[bad_id]:.1f} м")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
