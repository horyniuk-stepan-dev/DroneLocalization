import numpy as np
import pytest

from src.geometry.georef_fit import (
    AFFINE,
    apply,
    compose_affine,
    compose_homography,
    fit_georef,
    fit_similarity,
    required_rotation_sign,
)


def _sim(scale, deg, t):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.hstack([scale * R, np.asarray(t, float)[:, None]])


def test_two_points_exact():
    M = _sim(3.7, 31.0, [4e6, 6e6])
    src = np.array([[10.0, 20.0], [80.0, 55.0]])
    fit = fit_georef(src, apply(M, src))
    np.testing.assert_allclose(fit.matrix, M, atol=1e-6)
    assert fit.loo_residuals is None and fit.warnings


def test_noise_and_loo_flags_outlier():
    rng = np.random.default_rng(0)
    M = _sim(2.0, -120.0, [1000, -500])
    src = rng.uniform(0, 100, (6, 2))
    dst = apply(M, src) + rng.normal(0, 0.2, (6, 2))
    dst[3] += [30, 0]
    fit = fit_georef(src, dst)
    assert np.nanargmax(fit.loo_residuals) == 3
    assert abs(fit.scale - 2.0) < 0.2


def test_rotation_sign_forced_and_contradiction():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    mirrored = src * [1, -1]
    M = fit_similarity(src, mirrored, rotation_sign=-1)
    np.testing.assert_allclose(apply(M, src), mirrored, atol=1e-9)
    src4 = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [7.0, 3.0]])
    fit = fit_georef(src4, src4 * [1, -1], rotation_sign=1)
    assert any("дзеркальн" in w for w in fit.warnings)


def test_collinear_warning_and_affine_minimum():
    src = np.array([[0.0, 0.0], [1, 1], [2, 2.0001]])
    assert fit_georef(src, src).warnings
    with pytest.raises(ValueError):
        fit_georef(src, src, kind=AFFINE)


def test_compose_matches_sequential_application():
    M = _sim(1.5, 45, [7, 8])
    A = np.array([[0.1, 0.02, -3.0], [0.01, -0.1, 4.0]])
    p = np.array([[100.0, 50.0]])
    np.testing.assert_allclose(apply(compose_affine(M, A), p), apply(M, apply(A, p)))
    H = np.vstack([A, [1e-5, 0, 1]])
    q = compose_homography(M, H) @ [100, 50, 1]
    r = H @ [100, 50, 1]
    np.testing.assert_allclose(q[:2] / q[2], apply(M, [r[:2] / r[2]])[0])


def test_gauge_convention_needs_proper_rotation():
    gauge = np.array([[1.0, 0.0, -960], [0.0, -1.0, 540]])  # relative_gauge: det < 0
    assert required_rotation_sign([gauge]) == 1
    assert required_rotation_sign([gauge * [[1], [-1]]]) == -1
