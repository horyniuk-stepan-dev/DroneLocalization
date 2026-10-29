import numpy as np
import pytest

from src.geometry.georef_fit import apply, fit_similarity
from src.geometry.georef_graph import (
    build_local_graph,
    fit_graph_georef,
    graph_from_layer,
)


def _sim(scale, deg, t):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.hstack([scale * R, np.asarray(t, float)[:, None]])


def _path(n=400):
    s = np.linspace(0, 1, n)
    # L-shaped flight in local units
    x = np.where(s < 0.5, s * 200, 100.0)
    y = np.where(s < 0.5, 0.0, (s - 0.5) * 200)
    return np.column_stack([x, y]), s


def _drifted_truth(local, s, M, drift=0.3):
    """True geo positions: the local map's scale drifts by `drift` along the
    path (integrated, like monocular VO)."""
    steps = np.diff(local, axis=0)
    k = 1.0 + drift * s[1:]
    true_local = np.vstack([local[:1], local[:1] + np.cumsum(steps * k[:, None], axis=0)])
    return apply(M, true_local)


def test_consistent_points_reproduce_global_similarity():
    local, _ = _path()
    g = build_local_graph(np.arange(len(local)), local, footprint=4.0)
    M = _sim(2.5, 20, [5e5, 6e6])
    idx = [0, 150, 399]
    fit = fit_graph_georef(g, local[idx], apply(M, local[idx]))
    np.testing.assert_allclose(fit.matrix, M, atol=1e-6)
    for i in (0, 100, 250, 399):
        np.testing.assert_allclose(fit.frame_matrices[i], M, atol=1e-4)
    assert fit.max_correction < 1e-3


def test_corrects_scale_drift_better_than_global():
    local, s = _path()
    g = build_local_graph(np.arange(len(local)), local, footprint=4.0)
    M = _sim(3.0, -35, [4e6, 6e6])
    truth = _drifted_truth(local, s, M)
    idx = [0, 80, 160, 240, 320, 399]
    rng = np.random.default_rng(1)
    dst = truth[idx] + rng.normal(0, 0.5, (len(idx), 2))
    fit = fit_graph_georef(g, local[idx], dst)
    M0 = fit_similarity(local[idx], dst)
    err_global = np.linalg.norm(apply(M0, local) - truth, axis=1)
    err_graph = np.linalg.norm(fit.map_points(local) - truth, axis=1)
    assert err_graph.max() < 0.25 * err_global.max()
    assert fit.loo_residuals is not None
    assert np.sqrt(np.mean(fit.loo_residuals**2)) < np.sqrt(
        np.mean(fit.global_loo_residuals**2)
    )
    assert fit.scale_range[1] / fit.scale_range[0] > 1.1


def test_frame_matrices_match_map_points_at_centres():
    local, s = _path()
    g = build_local_graph(np.arange(len(local)), local, footprint=4.0)
    M = _sim(3.0, 10, [1e5, 2e5])
    truth = _drifted_truth(local, s, M)
    idx = [0, 130, 260, 399]
    fit = fit_graph_georef(g, local[idx], truth[idx])
    via_mats = np.einsum("nij,nj->ni", fit.frame_matrices[:, :, :2], local) + fit.frame_matrices[
        :, :, 2
    ]
    np.testing.assert_allclose(via_mats, fit.map_points(local), atol=0.5)


def test_mirrored_map():
    local, s = _path()
    g = build_local_graph(np.arange(len(local)), local, footprint=4.0)
    M = _sim(2.0, 15, [3e5, 1e5]) @ np.diag([1.0, -1.0, 1.0])
    truth = _drifted_truth(local, s, M)
    idx = [0, 100, 200, 300, 399]
    fit = fit_graph_georef(g, local[idx], truth[idx], rotation_sign=-1)
    assert fit.reflected
    assert np.all(np.linalg.det(fit.frame_matrices[:, :, :2]) < 0)
    err = np.linalg.norm(fit.map_points(local) - truth, axis=1)
    assert err.max() < 0.25 * np.linalg.norm(apply(fit.matrix, local) - truth, axis=1).max()


def test_disconnected_component_warns_and_keeps_global():
    a, _ = _path(200)
    b = a + [1000.0, 1000.0]  # far away, no edges
    centers = np.vstack([a, b])
    g = build_local_graph(np.arange(400), centers, footprint=4.0)
    # break the temporal link between the two pieces
    g.pairs = g.pairs[~((g.pairs[:, 0] == 199) & (g.pairs[:, 1] == 200))]
    M = _sim(2.0, 0, [0, 0])
    idx = [0, 100, 199]
    fit = fit_graph_georef(g, centers[idx], apply(M, centers[idx]))
    assert any("200 кадрів" in w for w in fit.warnings)
    c = centers[300]
    got = fit.frame_matrices[300][:, :2] @ c + fit.frame_matrices[300][:, 2]
    np.testing.assert_allclose(got, M[:, :2] @ c + M[:, 2], atol=0.1)


def test_three_points_warn_no_check():
    local, _ = _path()
    g = build_local_graph(np.arange(len(local)), local, footprint=4.0)
    fit = fit_graph_georef(g, local[[0, 200, 399]], local[[0, 200, 399]] * 2)
    assert fit.loo_residuals is None and fit.warnings


def test_too_few_points():
    local, _ = _path()
    g = build_local_graph(np.arange(len(local)), local, footprint=4.0)
    with pytest.raises(ValueError):
        fit_graph_georef(g, local[[0, 1]], local[[0, 1]])


def test_graph_from_layer():
    w, h = 640, 480
    n = 50
    A = np.zeros((n, 3, 3))
    for i in range(n):
        A[i, :2] = [[0.01, 0, i * 2.0], [0, 0.01, 0]]
        A[i, 2, 2] = 1
    valid = np.ones(n, bool)
    valid[7] = False
    g = graph_from_layer(A[:, :2], valid, w, h)
    assert g.n == n - 1 and 7 not in g.frame_ids
    assert g.footprint == pytest.approx(0.01 * np.sqrt(w * h))


def test_revisit_edges_link_passes():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    loop = np.column_stack([np.cos(t), np.sin(t)]) * 50
    two = np.vstack([loop, loop + 0.3])  # second pass over the same ground
    g = build_local_graph(np.arange(600), two, footprint=4.0)
    cross = np.abs(g.pairs[:, 0] - g.pairs[:, 1]) >= 10
    assert cross.sum() > 200
