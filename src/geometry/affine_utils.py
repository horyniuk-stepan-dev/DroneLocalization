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


def compose_affine_5dof(tx: float, ty: float, sx: float, sy: float, angle: float, sign: float = 1.0) -> NDArray[np.float64]:
    """
    Збирає афінну матрицю 2x3 з незалежними масштабами X та Y.
    sign = -1.0 додає відображення по осі Y (необхідно для систем координат де Y-вниз мапиться на Y-вверх).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c * sx, -s * sign * sy, tx], [s * sx, c * sign * sy, ty]], dtype=np.float64)
