"""
Утиліти декомпозиції/складання ізотропних афінних матриць.

Єдине джерело істини для decompose/compose — використовується в:
  - src.calibration.multi_anchor_calibration
  - src.workers.calibration_propagation_worker (графова оптимізація)
  - src.geometry.pose_graph_optimizer
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def decompose_affine(M: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """
    Розкладає афінну матрицю 2x3 на компоненти:
    (tx, ty, scale, angle_rad).

    Для афінної матриці вигляду:
        [s*cos(a)  -s*sin(a)  tx]
        [s*sin(a)   s*cos(a)  ty]
    scale = sqrt(det(R_part)), angle = atan2(M[1,0], M[0,0]).
    При наявності шуму (незначний зсув / анізотропний масштаб)
    беремо ізотропне наближення через норму першого стовпця.
    """
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    s_x = float(np.linalg.norm(M[:2, 0]))
    s_y = float(np.linalg.norm(M[:2, 1]))
    scale = (s_x + s_y) * 0.5
    if scale < 1e-9:
        scale = 1e-9
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, scale, angle


def compose_affine(tx: float, ty: float, scale: float, angle: float) -> NDArray[np.float64]:
    """Збирає афінну матрицю 2x3 з компонентів перенесення, масштабу та кута (рад)."""
    c = np.cos(angle) * scale
    s = np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def unwrap_angles(angles: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Розгортає масив кутів (рад) для уникнення стрибків ±π при інтерполяції."""
    return np.unwrap(angles)


def decompose_affine_5dof(M: NDArray[np.float64]) -> tuple[float, float, float, float, float]:
    """
    Розкладає афінну матрицю 2x3 на 5 компонентів для збереження анізотропії:
    (tx, ty, sx, sy, angle_rad).
    """
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    sx = float(np.linalg.norm(M[:2, 0]))
    sy = float(np.linalg.norm(M[:2, 1]))
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, sx, sy, angle


def compose_affine_5dof(
    tx: float, ty: float, sx: float, sy: float, angle: float, sign: float = 1.0
) -> NDArray[np.float64]:
    """
    Збирає афінну матрицю 2x3 з незалежними масштабами X та Y.
    sign = -1.0 додає відображення по осі Y (необхідно для систем координат де Y-вниз мапиться на Y-вверх).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c * sx, -s * sign * sy, tx], [s * sx, c * sign * sy, ty]], dtype=np.float64)


# ── Центр-базова 5-DoF PCHIP-інтерполяція (Етап 4) ───────────────────────────
# Єдине джерело форми, яку використовують MultiAnchorCalibration (інтерполяція
# між якорями) та CalibrationPropagationWorker (заповнення пропущених кадрів).
# Інтерполюється (rx, ry, sx, sy, angle), де (rx, ry) — МЕТРИЧНА позиція опорного
# пікселя (центр кадру), а знак det зберігається окремо: пряма інтерполяція
# tx/ty при зміні кута дає «гойдання» центру, тому кодуємо саме центр.


def build_5dof_pchip(
    ids: Any, affines: Any, ref_px: tuple[float, float], log_scale: bool = False
) -> tuple[Any, float, tuple[float, float] | None]:
    """Будує shape-preserving PCHIP над (rx, ry, sx, sy, angle) валідних матриць.

    Повертає (interp, sign, (lo, hi)). interp=None, якщо <2 вузлів. sign — знак det
    більшості матриць (Y-flip), що відновлюється при композиції.

    log_scale=True (RESEARCH_INTEGRATION_PLAN 1.3): інтерполюються log(sx), log(sy)
    замість sx, sy — геодезично коректна форма для масштабу (лінійна інтерполяція
    масштабу не є геодезичною у групі подібностей). sample_5dof_pchip МУСИТЬ бути
    викликаний з тим самим значенням log_scale.
    """
    from scipy.interpolate import PchipInterpolator

    ids = np.asarray(ids, dtype=np.float64)
    affines = [np.asarray(a, dtype=np.float64) for a in affines]
    n = len(affines)
    if n < 2:
        return None, -1.0, None

    dets = np.array([np.linalg.det(a[:2, :2]) for a in affines], dtype=np.float64)
    n_neg = int(np.sum(dets < 0))
    sign = -1.0 if n_neg * 2 >= n else 1.0

    cx, cy = float(ref_px[0]), float(ref_px[1])
    comps = np.zeros((n, 5), dtype=np.float64)
    for i, a in enumerate(affines):
        _, _, sx, sy, angle = decompose_affine_5dof(a)
        comps[i, 0] = a[0, 0] * cx + a[0, 1] * cy + a[0, 2]  # rx
        comps[i, 1] = a[1, 0] * cx + a[1, 1] * cy + a[1, 2]  # ry
        if log_scale:
            comps[i, 2] = np.log(max(sx, 1e-12))
            comps[i, 3] = np.log(max(sy, 1e-12))
        else:
            comps[i, 2] = sx
            comps[i, 3] = sy
        comps[i, 4] = angle
    comps[:, 4] = unwrap_angles(comps[:, 4])

    interp = PchipInterpolator(ids, comps, extrapolate=False)
    return interp, sign, (float(ids[0]), float(ids[-1]))


def sample_5dof_pchip(
    interp: Any,
    sign: float,
    rng: tuple[float, float] | None,
    ref_px: tuple[float, float],
    frame_id: float,
    log_scale: bool = False,
) -> NDArray[np.float64] | None:
    """Афінна 2x3 для frame_id із PCHIP (clamp за межами діапазону вузлів).

    log_scale має збігатися зі значенням, переданим у build_5dof_pchip.
    """
    if interp is None or rng is None:
        return None
    fid = float(np.clip(frame_id, rng[0], rng[1]))
    comps = interp(fid)
    if comps is None or np.any(np.isnan(comps)):
        return None
    rx, ry, sx, sy, angle = comps
    if log_scale:
        sx = float(np.exp(np.clip(sx, np.log(1e-6), np.log(1e6))))
        sy = float(np.exp(np.clip(sy, np.log(1e-6), np.log(1e6))))
    else:
        sx = float(np.clip(sx, 1e-6, 1e6))
        sy = float(np.clip(sy, 1e-6, 1e6))
    M = compose_affine_5dof(0.0, 0.0, sx, sy, float(angle), sign=sign)
    cx, cy = float(ref_px[0]), float(ref_px[1])
    M[0, 2] = float(rx) - (M[0, 0] * cx + M[0, 1] * cy)
    M[1, 2] = float(ry) - (M[1, 0] * cx + M[1, 1] * cy)
    return M
