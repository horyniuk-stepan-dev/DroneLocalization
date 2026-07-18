"""Кінематичний prior (Етап 7.1 CALIBRATION_IMPROVEMENT_PLAN).

Слабкі фактори другої різниці центрів: r = w_eff·(α·a + (1−α)·c − b).

1. w=0 → структура резидуалів і розв'язок незмінні (контракт 5E+N).
2. Трійки: α та вага правильно масштабуються на нееквідистантних гепах.
3. Нульовий резидуал на зваженій інтерполяції.
4. Аналітичний якобіан kin-блоку == центральна FD (обов'язкова звірка за планом).
5. «Провисання» середини між якорями на отруєних слабких ребрах ↓ у рази.
6. Консистентна дуга з сильними ребрами не деградує (< кількох px).
"""

import numpy as np

from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

CX, CY = 960.0, 540.0  # дефолтний конструктор: frame 1920×1080


def _t_affine(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _fix_at(opt: PoseGraphOptimizer, fid: int, x: float, y: float) -> None:
    opt.fix_node(fid, _t_affine(x - CX, y - CY))


def _center(affine: np.ndarray) -> np.ndarray:
    return affine[:, :2] @ np.array([CX, CY]) + affine[:, 2]


def _chain(bump: bool) -> PoseGraphOptimizer:
    """Ланцюг 0..8 по прямій y=0, крок 100 px; якорі 0 і 8 fixed.

    bump=True: слабкі (w=0.05) «отруєні» ребра 3→4 (+40 вгору) і 4→5 (−40) —
    взаємно консистентний горб, який ніщо, крім prior, не прибирає.
    """
    opt = PoseGraphOptimizer()
    _fix_at(opt, 0, 0.0, 0.0)
    _fix_at(opt, 8, 800.0, 0.0)
    for i in range(8):
        if bump and i == 3:
            opt.add_edge(3, 4, _t_affine(100.0, 40.0), weight=0.05)
        elif bump and i == 4:
            opt.add_edge(4, 5, _t_affine(100.0, -40.0), weight=0.05)
        else:
            opt.add_edge(i, i + 1, _t_affine(100.0, 0.0), weight=1.0)
        if i + 1 != 8:
            opt.add_node(i + 1)
    opt.initialize_from_bfs()
    return opt


# ── 1. w=0: структура і розв'язок незмінні ───────────────────────────────────


def test_disabled_prior_keeps_solution_and_structure():
    res_default = _chain(bump=True).optimize(max_iterations=50)
    res_zero = _chain(bump=True).optimize(max_iterations=50, kinematic_prior_weight=0.0)
    assert set(res_default) == set(res_zero)
    for fid in res_default:
        np.testing.assert_allclose(res_default[fid], res_zero[fid], atol=1e-12)


def test_no_triples_when_weight_zero():
    opt = _chain(bump=False)
    ids, al, w = opt._build_kinematic_triples({i: i - 1 for i in range(1, 8)}, 0.0)
    assert ids == [] and al == [] and w == []


# ── 2. Трійки: α і вага на гепах ─────────────────────────────────────────────


def test_triples_alpha_and_gap_downweight():
    opt = PoseGraphOptimizer()
    _fix_at(opt, 0, 0.0, 0.0)
    _fix_at(opt, 5, 500.0, 0.0)
    opt.add_edge(0, 1, _t_affine(100.0, 0.0), weight=1.0)
    opt.add_edge(1, 2, _t_affine(100.0, 0.0), weight=1.0)
    opt.add_edge(2, 5, _t_affine(300.0, 0.0), weight=1.0)
    opt.add_node(1)
    opt.add_node(2)
    opt.initialize_from_bfs()

    id_to_var = {1: 0, 2: 1}
    ids, alphas, weights = opt._build_kinematic_triples(id_to_var, 1.0)
    assert ids == [(0, 1, 2), (1, 2, 5)]
    # (0,1,2): h1=h2=1 → α=0.5, w_eff=2/(1+1)=1.0 (еквідистантний еталон)
    assert np.isclose(alphas[0], 0.5) and np.isclose(weights[0], 1.0)
    # (1,2,5): h1=1, h2=3 → α=3/4, w_eff=2/4=0.5 — геп послаблює prior
    assert np.isclose(alphas[1], 0.75) and np.isclose(weights[1], 0.5)


def test_all_fixed_triples_skipped():
    opt = PoseGraphOptimizer()
    for i, x in enumerate((0.0, 100.0, 200.0)):
        _fix_at(opt, i, x, 0.0)
    opt.add_edge(0, 1, _t_affine(100.0, 0.0), weight=1.0)
    opt.add_edge(1, 2, _t_affine(100.0, 0.0), weight=1.0)
    ids, _, _ = opt._build_kinematic_triples({}, 1.0)
    assert ids == []


# ── 3–4. Резидуали і якобіан на ручному d ────────────────────────────────────


def _kin_d(n_nodes: int = 4):
    """Мінімальний d: 1 ребро + kin-трійки на вузлах 0..3 (0 fixed, 1..3 вільні),
    нееквідистантність через фіктивні ваги/α — беремо формули напряму."""
    X_full = np.zeros((n_nodes, 5), dtype=np.float64)
    d = {
        "X_full": X_full,
        "free_indices": [1, 2, 3],
        "edges_from": np.array([0], dtype=np.int32),
        "edges_to": np.array([1], dtype=np.int32),
        "dtx": np.array([100.0]),
        "dty": np.array([0.0]),
        "log_dsx": np.array([0.0]),
        "log_dsy": np.array([0.0]),
        "dtheta": np.array([0.0]),
        "weights": np.array([1.0]),
        "cx": CX,
        "sign": 1.0,
        "n_edges": 1,
        "n_free": 3,
        "edge_from_free": np.array([-1], dtype=np.int64),
        "edge_to_free": np.array([0], dtype=np.int64),
        # трійки (0,1,2) α=0.5 w=1.0 та (1,2,3) α=0.75 w=0.5; вузол 0 fixed
        "kin_ia": np.array([0, 1], dtype=np.int64),
        "kin_ib": np.array([1, 2], dtype=np.int64),
        "kin_ic": np.array([2, 3], dtype=np.int64),
        "kin_fa": np.array([-1, 0], dtype=np.int64),
        "kin_fb": np.array([0, 1], dtype=np.int64),
        "kin_fc": np.array([1, 2], dtype=np.int64),
        "kin_alpha": np.array([0.5, 0.75]),
        "kin_w": np.array([1.0, 0.5]),
        "n_kin": 2,
    }
    return d


def test_kin_residual_zero_on_weighted_interp():
    opt = PoseGraphOptimizer()
    d = _kin_d()
    # Вузол 0 fixed у (0,0); вільні 1..3 так, щоб обидві трійки лягали на інтерп.
    # (0,1,2) α=.5: c1 = (c0+c2)/2; (1,2,3) α=.75: c2 = .75·c1 + .25·c3
    c0 = np.array([0.0, 0.0])
    c1 = np.array([100.0, 10.0])
    c2 = 2.0 * c1 - c0  # з першої умови
    c3 = (c2 - 0.75 * c1) / 0.25  # з другої
    x = np.zeros(15)
    x[0:2], x[5:7], x[10:12] = c1, c2, c3
    res = opt._residuals_vec(x, d)
    assert res.shape[0] == 5 * 1 + 3 + 2 * 2
    np.testing.assert_allclose(res[-4:], 0.0, atol=1e-9)

    # Зсув b другої трійки (вузол 2) на δ → останні два резидуали = −w·δ
    delta = np.array([4.0, -6.0])
    x2 = x.copy()
    x2[5:7] += delta
    res2 = opt._residuals_vec(x2, d)
    np.testing.assert_allclose(res2[-2:], -0.5 * delta, atol=1e-9)


def test_kin_jacobian_matches_fd():
    opt = PoseGraphOptimizer()
    d = _kin_d()
    rng = np.random.default_rng(11)
    x = rng.normal(scale=50.0, size=15)
    x[2::5] = rng.normal(scale=0.1, size=3)  # log-масштаби помірні
    x[3::5] = rng.normal(scale=0.1, size=3)
    x[4::5] = rng.normal(scale=0.2, size=3)

    Ja = opt._jacobian_vec(x, d).toarray()

    eps = 1e-6
    m = opt._residuals_vec(x.copy(), d).size
    Jn = np.zeros((m, x.size))
    for k in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        Jn[:, k] = (opt._residuals_vec(xp, d) - opt._residuals_vec(xm, d)) / (2 * eps)

    denom = np.maximum(np.abs(Jn), 1.0)
    assert float(np.max(np.abs(Ja - Jn) / denom)) < 1e-6


# ── 5–6. Поведінка e2e ───────────────────────────────────────────────────────


def test_midpoint_sag_reduced_between_anchors():
    base = _chain(bump=True).optimize(max_iterations=80, use_analytic_jac=True)
    y4_base = abs(_center(base[4])[1])
    assert y4_base > 30.0, f"очікували провисання ~40 px, маємо {y4_base:.1f}"

    kin = _chain(bump=True).optimize(
        max_iterations=80, use_analytic_jac=True, kinematic_prior_weight=1.0
    )
    y4_kin = abs(_center(kin[4])[1])
    assert y4_kin < 0.4 * y4_base, (
        f"prior не гасить провисання: |y4| {y4_base:.1f} → {y4_kin:.1f}"
    )
    # Сильніший prior не слабший (сатурація ~9 px: гасимо кривину, не амплітуду)
    kin2 = _chain(bump=True).optimize(
        max_iterations=80, use_analytic_jac=True, kinematic_prior_weight=2.0
    )
    assert abs(_center(kin2[4])[1]) <= y4_kin + 0.1
    # Якорі лишаються на місці
    np.testing.assert_allclose(_center(kin[0]), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_center(kin[8]), [800.0, 0.0], atol=1e-6)


def test_analytic_and_fd_agree_with_prior():
    a = _chain(bump=True).optimize(
        max_iterations=80, use_analytic_jac=True, kinematic_prior_weight=0.5
    )
    b = _chain(bump=True).optimize(
        max_iterations=80, use_analytic_jac=False, kinematic_prior_weight=0.5
    )
    for fid in a:
        np.testing.assert_allclose(a[fid], b[fid], atol=1e-3)


def _arc() -> PoseGraphOptimizer:
    """Консистентна Г-подібна траєкторія (поворот) із сильними ребрами."""
    pts = [(0, 0), (100, 0), (200, 0), (300, 0), (300, 100), (300, 200), (300, 300)]
    opt = PoseGraphOptimizer()
    _fix_at(opt, 0, *map(float, pts[0]))
    _fix_at(opt, 6, *map(float, pts[6]))
    for i in range(6):
        dx = float(pts[i + 1][0] - pts[i][0])
        dy = float(pts[i + 1][1] - pts[i][1])
        opt.add_edge(i, i + 1, _t_affine(dx, dy), weight=1.0)
        if i + 1 != 6:
            opt.add_node(i + 1)
    opt.initialize_from_bfs()
    return opt


def test_consistent_arc_not_degraded():
    base = _arc().optimize(max_iterations=80, use_analytic_jac=True)
    kin = _arc().optimize(
        max_iterations=80, use_analytic_jac=True, kinematic_prior_weight=0.02
    )
    shifts = [
        float(np.linalg.norm(_center(kin[fid]) - _center(base[fid]))) for fid in base
    ]
    assert max(shifts) < 5.0, f"дуга деградувала: max зсув {max(shifts):.2f} px"
