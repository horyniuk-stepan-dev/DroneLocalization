"""Ротаційна робастність temporal-ребер (Етап 5) — чисті хелпери
src.localization.rotation_geometry + композиція H_true = H_r · R(θ), яку
використовує воркер. (Сам matcher потребує torch → тестуємо геометрію.)
"""

import numpy as np
import pytest

from src.localization.rotation_geometry import (
    chain_relative_angle_deg,
    rotate_keypoints,
    rotation_homography,
    temporal_retry_angles,
)

CX, CY = 640.0, 360.0


def _apply_H(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))])
    q = (H @ p.T).T
    return q[:, :2] / q[:, 2:3]


def test_rotation_homography_matches_rotate_keypoints():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1280, size=(20, 2))
    for deg in [10, 37, 90, 180, -45]:
        th = np.radians(deg)
        via_H = _apply_H(rotation_homography(th, CX, CY), pts)
        via_kp = rotate_keypoints(pts, th, CX, CY)
        np.testing.assert_allclose(via_H, via_kp, atol=1e-9)


def test_rotate_keypoints_90_about_center():
    p = np.array([[CX + 100.0, CY]])  # праворуч від центру
    r = rotate_keypoints(p, np.radians(90), CX, CY)
    np.testing.assert_allclose(r[0], [CX, CY + 100.0], atol=1e-9)  # → вгору
    assert rotate_keypoints(np.empty((0, 2)), 1.0, CX, CY).shape == (0, 2)


def test_chain_relative_angle():
    def cum(deg):
        th = np.radians(deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, 5.0], [s, c, -3.0], [0, 0, 1.0]])

    # from=30°, to=115° → to відносно from = 85°
    ang = chain_relative_angle_deg(cum(30), cum(115))
    assert ang == pytest.approx(85.0, abs=1e-6)
    # вироджені пози → None
    assert chain_relative_angle_deg(np.zeros((3, 3)), cum(10)) is None
    assert chain_relative_angle_deg(cum(10), np.zeros((3, 3))) is None


def test_temporal_retry_angles():
    # з ланцюгом: кут ланцюга перший, далі fallback k·90 (у (−180,180], без 0)
    a = temporal_retry_angles(85.0, use_chain=True)
    assert a[0] == pytest.approx(85.0)
    assert a == pytest.approx([85.0, 90.0, -180.0, -90.0])
    # кут ланцюга ≈ одному з k·90 → дедуп (180 поглинає fallback-180)
    assert temporal_retry_angles(180.0, use_chain=True) == pytest.approx([-180.0, 90.0, -90.0])
    # без ланцюга: чистий перебір k·90
    assert temporal_retry_angles(None, use_chain=False) == [90.0, -180.0, -90.0]
    # кут ≈0 пропускається (первинний матч уже пробував 0)
    assert temporal_retry_angles(0.0, use_chain=True) == [90.0, -180.0, -90.0]


def test_composition_recovers_true_homography():
    """Воркер: rotated_query = R(θ)·query, matcher дає H_r (rotated→ref),
    H_true = H_r · R(θ) має відновити справжню query→ref гомографію."""
    rng = np.random.default_rng(3)
    # справжня query→ref (similarity: поворот 25° + масштаб 1.1 + зсув)
    th_t = np.radians(25.0)
    sc = 1.1
    H = np.array(
        [
            [sc * np.cos(th_t), -sc * np.sin(th_t), 40.0],
            [sc * np.sin(th_t), sc * np.cos(th_t), -15.0],
            [0, 0, 1.0],
        ]
    )
    query = rng.uniform(100, 1000, size=(30, 2))
    ref = _apply_H(H, query)

    theta = np.radians(25.0)  # кут, яким воркер повертає query
    R = rotation_homography(theta, CX, CY)
    H_r = H @ np.linalg.inv(R)  # matcher бачить rotated_query→ref
    # sanity: H_r справді відображає повернуті точки у ref
    np.testing.assert_allclose(
        _apply_H(H_r, rotate_keypoints(query, theta, CX, CY)), ref, atol=1e-6
    )
    # композиція воркера відновлює справжню H
    H_true = H_r @ R
    np.testing.assert_allclose(_apply_H(H_true, query), ref, atol=1e-6)
    np.testing.assert_allclose(H_true / H_true[2, 2], H / H[2, 2], atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
