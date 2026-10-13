"""Optional priors of the planar adjustment: soft anchor points, scale, perspective.

Every check uses known geometry; the image observations are synthetic projections
of shared map points, and errors are measured against the truth, never against the
solver's own residuals.
"""

import numpy as np
import pytest

from src.geometry.frame_transform import affine_at_center, project_points
from src.geometry.relative_alignment import PointLink
from src.geometry.relative_projective import (
    FramePriors,
    _log_scale_terms,
    _point_prior_terms,
    _refine_map,
    refine_projective_map,
)

W, H = 1280, 720
QA = np.array([[640, 360], [256, 144], [1024, 144], [1024, 576], [256, 576]], float)


def frame_truth(x, y, scale=0.5, persp=(0.0, 0.0)):
    """Pixel -> map homography: nadir-like frame centred at (x, y), optional tilt."""
    S = np.array([[scale, 0, -scale * W / 2], [0, -scale, scale * H / 2], [0, 0, 1.0]])
    P = np.array([[1, 0, 0], [0, 1, 0], [persp[0], persp[1], 1.0]])
    C = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1.0]])
    Cinv = np.linalg.inv(C)
    M = np.array([[1, 0, x], [0, 1, y], [0, 0, 1.0]]) @ S @ C @ P @ Cinv
    return M / (M[2] @ [W / 2, H / 2, 1])


def chain(n, step=160.0, persp=None, noise=0.3, seed=0, region=None, scale_drift=1.0, points=200):
    rng = np.random.default_rng(seed)
    truth = {}
    for k in range(n):
        tilt = persp(k) if persp else (0.0, 0.0)
        truth[k] = frame_truth(k * step, 0.0, persp=tilt)
    drifted = {
        k: frame_truth(
            k * step, 0.0, scale=0.5 * scale_drift**k, persp=persp(k) if persp else (0, 0)
        )
        for k in range(n)
    }
    links = []
    for a in range(n):
        for b in (a + 1, a + 2):
            if b >= n:
                continue
            src = drifted if scale_drift != 1.0 else truth
            world = rng.uniform([a * step - 250, -150], [b * step + 250, 150], (3000, 2))
            p = project_points(np.linalg.inv(src[a]), world)
            q = project_points(np.linalg.inv(src[b]), world)
            ok = (
                (p > 5).all(1)
                & (p < [W - 5, H - 5]).all(1)
                & (q > 5).all(1)
                & (q < [W - 5, H - 5]).all(1)
            )
            if region is not None:
                ok &= region(a, p) & region(b, q)
            p, q = p[ok][:points], q[ok][:points]
            if len(p) < 20:
                continue
            links.append(
                PointLink(
                    a, b, p + rng.normal(0, noise, p.shape), q + rng.normal(0, noise, q.shape)
                )
            )
    return truth, links


def centre_errors(result, truth, pts=None):
    pts = np.array([[W / 2, H / 2]]) if pts is None else pts
    return {
        fid: float(
            np.linalg.norm(
                project_points(result[fid], pts) - project_points(truth[fid], pts), axis=1
            ).max()
        )
        for fid in result
    }


def test_without_priors_results_are_unchanged():
    truth, links = chain(6, persp=lambda k: (2e-5 * (k % 2), -1e-5))
    initial = {k: affine_at_center(T, W, H) for k, T in truth.items()}
    initial[0] = truth[0]
    a = refine_projective_map(initial, links, W, H, {0})
    b = refine_projective_map(initial, links, W, H, {0}, priors=FramePriors())
    for fid in a[0]:
        np.testing.assert_array_equal(a[0][fid], b[0][fid])
    assert a[1]["selected_model"] == b[1]["selected_model"]
    assert "priors" not in a[1] and "priors" not in b[1]


def test_prior_derivatives_match_finite_differences():
    rng = np.random.default_rng(3)
    Hm = np.array([[1.02, 0.03, 0.4], [-0.02, 0.97, -0.3], [0.04, -0.03, 1.0]])
    pn = np.column_stack([rng.uniform(-0.5, 0.5, (5, 2)), np.ones(5)])
    target = rng.normal(0, 0.3, (5, 2))
    r, d = _point_prior_terms(Hm, pn, target, 7.0)
    ls, g = _log_scale_terms(Hm)
    eps = 1e-7
    for i in range(8):
        dH = np.zeros(9)
        dH[i] = eps
        Hp, Hn = Hm + dH.reshape(3, 3), Hm - dH.reshape(3, 3)
        num = (
            _point_prior_terms(Hp, pn, target, 7.0)[0] - _point_prior_terms(Hn, pn, target, 7.0)[0]
        ) / (2 * eps)
        np.testing.assert_allclose(d[:, i], num, rtol=1e-5, atol=1e-6)
        num_s = (_log_scale_terms(Hp)[0] - _log_scale_terms(Hn)[0]) / (2 * eps)
        assert g[i] == pytest.approx(num_s, rel=1e-5, abs=1e-7)


def tilted_scene(n=10):
    # Terrain slope: every frame is a genuine homography (planar tilted ground).
    return chain(n, persp=lambda k: (3e-5, -2e-5), noise=0.3, seed=11)


def test_soft_anchor_points_beat_fixed_affine_anchors():
    truth, links = tilted_scene()
    anchors = (0, 9)
    # Fixed anchors: the best 5-point affine of a projective truth (what a surveyed
    # affine anchor stores) -- biased away from the image geometry.
    fitted = {}
    for a in anchors:
        X = np.column_stack([QA, np.ones(5)])
        fitted[a] = np.vstack(
            [np.linalg.lstsq(X, project_points(truth[a], QA), rcond=None)[0].T, [0, 0, 1]]
        )
    initial = {k: affine_at_center(truth[k], W, H) for k in truth}
    hard_init = {**initial, **fitted}
    hard, _, _ = _refine_map(
        hard_init,
        links,
        W,
        H,
        set(anchors),
        model="projective",
        max_nfev=200,
        check_running=lambda: None,
    )
    priors = FramePriors(points={a: (QA, project_points(truth[a], QA), 1.0) for a in anchors})
    soft, report, _ = _refine_map(
        hard_init,
        links,
        W,
        H,
        set(),
        model="projective",
        max_nfev=200,
        check_running=lambda: None,
        priors=priors,
    )
    grid = np.array([[x, y] for x in (0, 640, 1280) for y in (0, 360, 720)], float)
    e_hard = max(centre_errors(hard, truth, grid).values())
    e_soft = max(centre_errors(soft, truth, grid).values())
    assert e_soft < 0.5 * e_hard
    assert (
        report["priors"]["anchor_points_rms_m_after"]
        < report["priors"]["anchor_points_rms_m_before"]
    )
    # The anchor frames themselves now follow the image geometry: no anchor/free step.
    step_err = [
        np.linalg.norm(
            (project_points(soft[b], [[W / 2, H / 2]]) - project_points(soft[a], [[W / 2, H / 2]]))
            - (
                project_points(truth[b], [[W / 2, H / 2]])
                - project_points(truth[a], [[W / 2, H / 2]])
            )
        )
        for a, b in ((0, 1), (8, 9))
    ]
    assert max(step_err) < 0.2


def test_perspective_prior_limits_one_sided_extrapolation():
    # Affine truth; frame 3 is only observed in its left third (one-sided support).
    def region(fid, pts):
        return pts[:, 0] < 420 if fid == 3 else np.ones(len(pts), bool)

    truth, links = chain(7, noise=1.5, seed=5, region=region)
    initial = {k: truth[k].copy() for k in truth}
    corners = np.array([[W, 0], [W, H]], float)

    def run(priors):
        return _refine_map(
            initial,
            links,
            W,
            H,
            {0, 6},
            model="projective",
            max_nfev=200,
            check_running=lambda: None,
            priors=priors,
        )[0]

    free_err = centre_errors(run(None), truth, corners)[3]
    reg_err = centre_errors(run(FramePriors(perspective_sigma_px=5.0)), truth, corners)[3]
    assert reg_err < 0.5 * free_err


def test_scale_prior_counteracts_chain_scale_drift():
    # Image links imply a 1 %/frame scale drift (e.g. relief); the DEM says constant.
    truth, links = chain(12, noise=0.2, seed=2, scale_drift=1.01)
    initial = {k: truth[k].copy() for k in truth}

    def run(priors):
        return _refine_map(
            initial,
            links,
            W,
            H,
            {0},
            model="affine",
            max_nfev=300,
            check_running=lambda: None,
            priors=priors,
        )[0]

    target = np.log(0.5)
    free = run(None)
    tied = run(FramePriors(log_scale={k: (target, 0.002) for k in range(1, 12)}))

    def drift(res):
        tangent = affine_at_center(res[11], W, H)[:, :2]
        return abs(0.5 * np.log(abs(np.linalg.det(tangent))) - target)

    assert drift(free) > 0.05
    assert drift(tied) < 0.5 * drift(free)


def test_global_sparse_lm_carries_soft_anchors_along_a_long_chain():
    # 67 free frames: the windowed trust-region default cannot carry the anchor
    # corrections across fixed window boundaries; one global sparse LM solve can.
    truth, links = chain(67, persp=lambda k: (2e-5, 1e-5), noise=0.3, seed=9, points=40)
    initial = {k: affine_at_center(truth[k], W, H) for k in truth}
    anchors = (0, 33, 66)
    priors = FramePriors(points={a: (QA, project_points(truth[a], QA), 1.0) for a in anchors})
    hard = {**initial, **{a: truth[a] for a in anchors}}
    ideal, _, _ = refine_projective_map(
        hard, links, W, H, set(anchors), window_frames=0, solver="lm_sparse"
    )
    soft, report, _ = refine_projective_map(
        initial, links, W, H, set(), priors=priors, window_frames=0, solver="lm_sparse"
    )
    assert report["converged"] and report["solver_method"] == "lm_sparse"
    e_soft, e_ideal = centre_errors(soft, truth), centre_errors(ideal, truth)
    # Soft anchors with exact points match exact hard anchors (noise-limited).
    assert np.median(list(e_soft.values())) < 1.2 * np.median(list(e_ideal.values())) + 0.05
    assert max(e_soft.values()) < 1.3 * max(e_ideal.values()) + 0.2
    assert (
        report["priors"]["anchor_points_rms_m_after"]
        < report["priors"]["anchor_points_rms_m_before"]
    )


def test_priors_reach_window_solves():
    truth, links = chain(70, noise=0.3, seed=4, points=24)
    initial = {k: truth[k].copy() for k in truth}
    anchors = (0, 69)
    priors = FramePriors(points={a: (QA, project_points(truth[a], QA), 1.0) for a in anchors})
    _, report, _ = refine_projective_map(
        initial, links, W, H, set(), priors=priors, max_nfev=20, solver="lm_sparse"
    )
    assert report["solver"] == "overlapping_block_coordinate"
    assert report["priors"]["point_frames"] == 2
