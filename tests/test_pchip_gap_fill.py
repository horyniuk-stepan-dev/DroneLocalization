"""PCHIP-заповнення пропущених кадрів (Етап 4) — center-базові 5-DoF хелпери
affine_utils.build_5dof_pchip / sample_5dof_pchip.

Гейт плану: на дузі розвороту PCHIP-центр ближче до істини, ніж посегментна
лінійна (яку робить поточний воркер). Плюс: точні control-points, паритет із
MultiAnchorCalibration, clamp за діапазоном.
"""

import numpy as np

from src.geometry.affine_utils import (
    build_5dof_pchip,
    compose_affine_5dof,
    decompose_affine_5dof,
    sample_5dof_pchip,
    unwrap_angles,
)
from src.geometry.pose_graph.model_5dof import _state_to_affine

CX, CY = 640.0, 360.0
REF = (CX, CY)


def _affine_center(px, py, angle_rad, s=0.5):
    """Афінна, центр якої (px,py), поворот angle, ізотропний масштаб s."""
    return _state_to_affine(np.array([px, py, np.log(s), np.log(s), angle_rad]), CX, CY, sign=1.0)


def _center(M):
    return M[:2, :2] @ np.array([CX, CY]) + M[:, 2]


def _arc(n=25, R=1000.0):
    """Дуга 90°: центр кадру рухається по колу, курс = дотична."""
    ids = list(range(n))
    affines, true_centers = [], []
    for i in range(n):
        th = (np.pi / 2) * i / (n - 1)
        px, py = R * np.sin(th), R * (1 - np.cos(th))
        affines.append(_affine_center(px, py, th))
        true_centers.append(np.array([px, py]))
    return ids, affines, true_centers


def test_pchip_reproduces_control_points():
    ids, affines, _ = _arc()
    interp, sign, rng = build_5dof_pchip(ids, affines, REF)
    for k, i in enumerate(ids):
        M = sample_5dof_pchip(interp, sign, rng, REF, i)
        np.testing.assert_allclose(M, affines[k], atol=1e-7)


def test_pchip_matches_multi_anchor_calibration():
    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration

    ids, affines, _ = _arc(n=9)
    mac = MultiAnchorCalibration()
    mac.set_frame_size(int(CX * 2), int(CY * 2))
    for i, a in zip(ids, affines):
        mac.add_anchor(i, a)
    interp, sign, rng = build_5dof_pchip(ids, affines, mac._reference_pixel())
    for q in [0.5, 2.3, 4.7, 7.9]:
        M_helper = sample_5dof_pchip(interp, sign, rng, mac._reference_pixel(), q)
        M_mac = mac._get_interpolated_matrix(q)
        np.testing.assert_allclose(M_helper, M_mac, atol=1e-9)


def test_pchip_clamps_outside_range():
    ids, affines, _ = _arc(n=5)
    interp, sign, rng = build_5dof_pchip(ids, affines, REF)
    lo = sample_5dof_pchip(interp, sign, rng, REF, -10.0)
    hi = sample_5dof_pchip(interp, sign, rng, REF, 999.0)
    np.testing.assert_allclose(lo, affines[0], atol=1e-7)
    np.testing.assert_allclose(hi, affines[-1], atol=1e-7)


def _worker_linear_fill(ids_valid, affines_valid, query_ids):
    """Копія посегментної лінійної інтерполяції воркера (raw tx,ty,sx,sy,angle)."""
    out = {}
    va = {i: a for i, a in zip(ids_valid, affines_valid)}
    ids_valid = sorted(ids_valid)
    for q in query_ids:
        left = max([i for i in ids_valid if i <= q])
        right = min([i for i in ids_valid if i >= q])
        if left == right:
            out[q] = va[left]
            continue
        cl = np.array(decompose_affine_5dof(va[left]), float)
        cr = np.array(decompose_affine_5dof(va[right]), float)
        ang = unwrap_angles([cl[4], cr[4]])
        cl[4], cr[4] = ang
        t = (q - left) / (right - left)
        cm = cl * (1 - t) + cr * t
        out[q] = compose_affine_5dof(
            cm[0],
            cm[1],
            cm[2],
            cm[3],
            cm[4],
            sign=-1.0 if np.linalg.det(va[left][:2, :2]) < 0 else 1.0,
        )
    return out


def test_pchip_beats_worker_linear_on_arc():
    ids, affines, true_c = _arc(n=25, R=1000.0)
    valid = ids[::4]  # рідкі control-точки (кожна 4-та)
    missing = [i for i in ids if i not in valid]
    va = [affines[i] for i in valid]

    interp, sign, rng = build_5dof_pchip(valid, va, REF)
    e_pchip = [
        np.linalg.norm(_center(sample_5dof_pchip(interp, sign, rng, REF, i)) - true_c[i])
        for i in missing
    ]
    lin = _worker_linear_fill(valid, va, missing)
    e_lin = [np.linalg.norm(_center(lin[i]) - true_c[i]) for i in missing]

    m_pchip, m_lin = float(np.median(e_pchip)), float(np.median(e_lin))
    print(f"arc center err (median): worker-linear={m_lin:.2f} м, PCHIP={m_pchip:.2f} м")
    assert m_pchip < m_lin, f"PCHIP не кращий на дузі: {m_pchip:.2f} vs {m_lin:.2f}"
    assert m_pchip < m_lin * 0.5  # відчутне покращення


def test_straight_segment_no_regression():
    # Пряма ділянка: і лінійна, і PCHIP мають бути ~точні (без деградації)
    n = 13
    ids = list(range(n))
    affines = [_affine_center(100.0 * i, 0.0, 0.0) for i in ids]
    true_c = [np.array([100.0 * i, 0.0]) for i in ids]
    valid = ids[::3]
    missing = [i for i in ids if i not in valid]
    va = [affines[i] for i in valid]
    interp, sign, rng = build_5dof_pchip(valid, va, REF)
    e = [
        np.linalg.norm(_center(sample_5dof_pchip(interp, sign, rng, REF, i)) - true_c[i])
        for i in missing
    ]
    assert max(e) < 1e-6  # на прямій PCHIP точний


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
