"""Pure geometry of the continuous query rotation (src/localization/query_rotation.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.localization.query_rotation import (
    angle_distance_deg,
    next_prior_deg,
    norm_deg,
    residual_angle_deg,
    rotate_view,
    rotation_matrix,
    scan_angles,
    view_mask,
)

W, H = 640, 360


def test_scan_angles():
    assert scan_angles(45) == [0, 45, 90, 135, 180, 225, 270, 315]
    assert len(scan_angles(30)) == 12
    assert scan_angles(0) == [0.0]
    assert scan_angles(400) == [0.0]


def test_norm_and_distance():
    assert norm_deg(-10) == 350
    assert norm_deg(720) == 0
    assert angle_distance_deg(350, 10) == pytest.approx(20)
    assert angle_distance_deg(90, 270) == pytest.approx(180)


@pytest.mark.parametrize("x,y", [(320, 180), (250, 120), (400, 230)])
def test_rotation_matrix_matches_warp(x, y):
    """A marker at (x, y) lands where rotation_matrix says it does."""
    img = np.zeros((H, W), np.uint8)
    cv2.circle(img, (x, y), 4, 255, -1)
    R = rotation_matrix(37.0, W, H)
    view = cv2.warpAffine(img, R[:2], (W, H))
    ys, xs = np.nonzero(view > 128)
    u, v, s = R @ [x, y, 1.0]
    assert s == 1.0
    assert abs(xs.mean() - u) < 1.0 and abs(ys.mean() - v) < 1.0
    # rotation about the centre keeps the centre
    np.testing.assert_allclose(R @ [W / 2, H / 2, 1], [W / 2, H / 2, 1], atol=1e-9)


@pytest.mark.parametrize("true", [0, 17, 37, 90, 133, 200, 315])
def test_next_prior_recovers_true_angle_from_any_scan_angle(true):
    for applied in scan_angles(45):
        H_view_to_ref = rotation_matrix(true, W, H) @ np.linalg.inv(rotation_matrix(applied, W, H))
        assert angle_distance_deg(next_prior_deg(applied, H_view_to_ref), true) < 1e-6


def test_residual_ignores_scale_and_translation():
    R = rotation_matrix(25, W, H)
    S = np.diag([1.3, 1.3, 1.0])
    T = np.array([[1, 0, 40], [0, 1, -15], [0, 0, 1.0]])
    assert residual_angle_deg(T @ S @ R) == pytest.approx(residual_angle_deg(R), abs=1e-9)


def test_rotate_view_zero_is_identity():
    img = np.zeros((H, W, 3), np.uint8)
    view, mask, valid = rotate_view(img, 0.0)
    assert view is img and mask is None and valid.min() == 255


@pytest.mark.parametrize("angle", [30, 90, 211])
def test_view_mask_matches_rotate_view(angle):
    img = np.full((H, W, 3), 100, np.uint8)
    static = np.full((H, W), 255, np.uint8)
    static[:40] = 0
    _, m1, valid = rotate_view(img, angle, static, erode_px=6)
    m2 = view_mask((H, W), angle, static, erode_px=6)
    np.testing.assert_array_equal(m1, m2)
    # border pixels of a rotated view never count as valid
    assert valid[0, 0] == 0 and valid[-1, -1] == 0
