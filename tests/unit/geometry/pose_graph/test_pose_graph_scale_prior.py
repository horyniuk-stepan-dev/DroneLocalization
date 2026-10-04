"""Unary log-scale prior block of PoseGraphOptimizer (terrain model).

Contract: no priors -> residual layout and solution unchanged; with priors the
analytic Jacobian matches finite differences and the sparsity pattern covers it;
a prior pulls a free node's mean log-scale to its target.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.geometry.affine_utils import compose_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

W, H = 1280, 720


def _chain(n=5, scale=0.8, step_px=300.0, soft=False):
    opt = PoseGraphOptimizer(W, H, isotropy_weight=10.0)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        rel = np.array([[1.0, 0.0, step_px], [0.0, 1.0, 0.0]])
        opt.add_edge(i, i + 1, rel, weight=5.0)
    a0 = compose_affine(0.0, 0.0, scale, 0.0)
    if soft:
        opt.add_anchor(0, a0, sigma_m=1.0)
    else:
        opt.fix_node(0, a0)
    opt.initialize_from_bfs()
    return opt


def _capture(opt, **kw):
    seen = {}

    def fake(fun, x0, args=(), jac=None, **kwargs):
        seen.update(fun=fun, x0=x0.copy(), d=args[0], jac=jac, sparsity=kwargs.get("jac_sparsity"))
        return SimpleNamespace(x=x0, cost=0.0, nfev=1, status=1, message="captured")

    with mock.patch("src.geometry.pose_graph.optimizer.least_squares", side_effect=fake):
        opt.optimize(**kw)
    return seen


def test_no_priors_keeps_layout_and_solution():
    a = _chain()
    b = _chain()
    b.set_scale_priors({})
    ra = a.optimize(use_analytic_jac=True)
    rb = b.optimize(use_analytic_jac=True)
    for fid in ra:
        np.testing.assert_array_equal(ra[fid], rb[fid])
    seen = _capture(_chain(), use_analytic_jac=True)
    n_edges, n_free = 4, 4
    assert seen["fun"](seen["x0"], seen["d"]).size == 5 * n_edges + n_free


def test_priors_add_one_row_each_and_jacobian_matches_fd():
    opt = _chain(soft=True)
    opt.set_scale_priors({2: (np.log(0.9), 3.0), 4: (np.log(0.7), 10.0), 99: (0.0, 1.0)})
    seen = _capture(opt, use_analytic_jac=True, kinematic_prior_weight=0.5)
    fun, d = seen["fun"], seen["d"]
    rng = np.random.default_rng(3)
    x = seen["x0"] + rng.normal(0, 0.05, seen["x0"].size)
    r = fun(x, d)
    assert d["n_sp"] == 2  # node 99 does not exist
    Ja = seen["jac"](x, d).toarray()
    assert Ja.shape == (r.size, x.size)
    eps = 1e-6
    Jn = np.zeros_like(Ja)
    for k in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        Jn[:, k] = (fun(xp, d) - fun(xm, d)) / (2 * eps)
    assert np.max(np.abs(Ja - Jn) / np.maximum(np.abs(Jn), 1.0)) < 1e-5

    seen_fd = _capture(opt, use_analytic_jac=False, kinematic_prior_weight=0.5)
    sp = seen_fd["sparsity"].toarray() != 0
    assert sp.shape == Ja.shape
    assert not np.any((np.abs(Ja) > 1e-12) & ~sp)


def test_prior_pulls_mean_log_scale_to_target():
    opt = _chain(n=3)
    target = np.log(0.6)
    opt.set_scale_priors({2: (target, 1000.0)})
    opt.optimize(use_analytic_jac=True)
    st = opt.node_states()[2]
    assert abs(0.5 * (st[2] + st[3]) - target) < 0.01
    # without the prior the chain keeps the anchor scale 0.8
    free = _chain(n=3)
    free.optimize(use_analytic_jac=True)
    st0 = free.node_states()[2]
    assert abs(0.5 * (st0[2] + st0[3]) - np.log(0.8)) < 1e-3


def test_node_states_and_is_free():
    opt = _chain(n=3)
    states = opt.node_states()
    assert set(states) == {0, 1, 2}
    assert not opt.is_free(0) and opt.is_free(1)
    states[1][0] = 1e9
    assert opt.node_states()[1][0] != 1e9  # copies
