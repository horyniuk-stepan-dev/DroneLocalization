"""FOV-remap unit tests (IMPLEMENTATION_PLAN, Фаза 1.2).

Verifies crop_to_affine() against the actual ScaleManager.normalize() output and
the end-to-end composition property used in Localizer.localize_frame:

    H_ransac (normalized→ref)  @  A (rotated→normalized)  ==  H_true (rotated→ref)

Pure numpy/cv2 — runs in the sandbox (no torch).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.localization.scale_manager import ScaleManager, crop_to_affine

W, H = 640, 360


def _frame() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def _sm() -> ScaleManager:
    return ScaleManager({})


def _apply(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 to (N, 2) points (column convention, like perspectiveTransform)."""
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = (A @ hom.T).T
    return out[:, :2] / out[:, 2:3]


# ── crop_to_affine per branch ────────────────────────────────────────────────


def test_crop_branch_maps_crop_box_to_full_frame():
    """r > 1: A maps the crop box exactly onto the normalized frame."""
    norm, ci = _sm().normalize(_frame(), 2.0)
    n_h, n_w = norm.shape[:2]
    A = crop_to_affine(ci, n_w, n_h)
    box = np.array(
        [
            [ci.crop_x, ci.crop_y],
            [ci.crop_x + ci.crop_w, ci.crop_y + ci.crop_h],
        ],
        dtype=np.float64,
    )
    mapped = _apply(A, box)
    np.testing.assert_allclose(mapped, [[0.0, 0.0], [n_w, n_h]], atol=1e-9)


def test_downscale_branch_scales_full_frame():
    """r < 0.85: A is a pure per-axis scale onto the smaller frame."""
    norm, ci = _sm().normalize(_frame(), 0.5)
    n_h, n_w = norm.shape[:2]
    A = crop_to_affine(ci, n_w, n_h)
    mapped = _apply(A, np.array([[0.0, 0.0], [W, H]], dtype=np.float64))
    np.testing.assert_allclose(mapped, [[0.0, 0.0], [n_w, n_h]], atol=1e-9)
    assert A[0, 1] == A[1, 0] == 0.0
    assert A[0, 2] == A[1, 2] == 0.0


def test_tolerance_band_yields_identity():
    """0.85 ≤ r ≤ 1.18: normalize is a no-op and A must be the identity."""
    norm, ci = _sm().normalize(_frame(), 1.1)
    assert ci.resize_scale == 1.0
    A = crop_to_affine(ci, norm.shape[1], norm.shape[0])
    np.testing.assert_allclose(A, np.eye(3), atol=0.0)


def test_degenerate_crop_yields_identity():
    from src.localization.scale_manager import CropInfo

    ci = CropInfo(scale_r=2.0, crop_x=0, crop_y=0, crop_w=0, crop_h=0, resize_scale=2.0)
    np.testing.assert_allclose(crop_to_affine(ci, W, H), np.eye(3))


@pytest.mark.parametrize("r", [0.5, 0.7, 1.4, 2.0])
def test_inverse_is_exact(r):
    norm, ci = _sm().normalize(_frame(), r)
    n_h, n_w = norm.shape[:2]
    A = crop_to_affine(ci, n_w, n_h)
    A_inv = crop_to_affine(ci, n_w, n_h, inverse=True)
    np.testing.assert_allclose(A_inv @ A, np.eye(3), atol=1e-12)
    rng = np.random.default_rng(7)
    pts = rng.uniform([0, 0], [W, H], size=(50, 2))
    np.testing.assert_allclose(_apply(A_inv, _apply(A, pts)), pts, atol=1e-9)


# ── end-to-end composition property (what localize_frame does) ───────────────


def _similarity(scale: float, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a) * scale, np.sin(a) * scale
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("r", [0.5, 0.7, 1.4, 2.0])
def test_composition_recovers_true_homography(r):
    """RANSAC sees normalized pixels → finds H_true @ A⁻¹; composing back with A
    must recover the rotated-frame homography exactly (centers, corners, matrix)."""
    norm, ci = _sm().normalize(_frame(), r)
    n_h, n_w = norm.shape[:2]
    A = crop_to_affine(ci, n_w, n_h)
    A_inv = crop_to_affine(ci, n_w, n_h, inverse=True)

    H_true = _similarity(r, 3.0, 120.0, -40.0)  # rotated frame → ref px
    H_ransac = H_true @ A_inv  # what RANSAC finds on the normalized frame
    composed = H_ransac @ A

    np.testing.assert_allclose(composed, H_true, atol=1e-8)

    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    np.testing.assert_allclose(
        _apply(composed, corners), _apply(H_true, corners), atol=1e-6
    )
    center = np.array([[W / 2.0, H / 2.0]])
    np.testing.assert_allclose(
        _apply(composed, center), _apply(H_true, center), atol=1e-6
    )


def test_uncomposed_center_bias_documents_the_bug():
    """Regression guard: WITHOUT the composition, feeding the rotated-frame
    center into H_ransac lands ~(1−r)/2·frame off — the bug being fixed."""
    r = 0.5
    norm, ci = _sm().normalize(_frame(), r)
    n_h, n_w = norm.shape[:2]
    A_inv = crop_to_affine(ci, n_w, n_h, inverse=True)
    H_true = _similarity(r, 0.0, 0.0, 0.0)
    H_ransac = H_true @ A_inv

    center_rot = np.array([[W / 2.0, H / 2.0]])
    wrong = _apply(H_ransac, center_rot)
    true = _apply(H_true, center_rot)
    # bias ≈ r·(1/resize_scale − 1)·W/2 = W/2·(r/s − r); for r=s=0.5 → W/2·(1−r)
    assert np.linalg.norm(wrong - true) > 100.0


def test_prior_recovers_r_after_composition():
    """update_from_homography on the composed H must measure the true GSD ratio
    (uncomposed H would collapse the EMA prior to ≈ 1)."""
    pytest.importorskip("scipy")
    for r in (0.5, 2.0):
        sm = _sm()
        norm, ci = sm.normalize(_frame(), r)
        n_h, n_w = norm.shape[:2]
        A = crop_to_affine(ci, n_w, n_h)
        A_inv = crop_to_affine(ci, n_w, n_h, inverse=True)
        composed = (_similarity(r, 0.0, 15.0, 8.0) @ A_inv) @ A
        sm.update_from_homography(composed, W, H)
        assert sm.prior == pytest.approx(r, rel=0.05), f"r={r}: prior={sm.prior}"
