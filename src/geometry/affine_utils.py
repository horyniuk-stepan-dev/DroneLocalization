"""
Утиліти декомпозиції/складання ізотропних афінних матриць.

Єдине джерело істини для decompose/compose — використовується в:
  - src.calibration.multi_anchor_calibration
  - src.workers.calibration_propagation_worker
"""

import numpy as np


def decompose_affine(M: np.ndarray) -> tuple[float, float, float, float]:
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
    # Ізотропний масштаб: середнє між нормами обох стовпців матриці обертання
    s_x = float(np.linalg.norm(M[:2, 0]))
    s_y = float(np.linalg.norm(M[:2, 1]))
    scale = (s_x + s_y) * 0.5
    if scale < 1e-9:
        scale = 1e-9
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, scale, angle


def compose_affine(tx: float, ty: float, scale: float, angle: float) -> np.ndarray:
    """Збирає афінну матрицю 2x3 з компонентів перенесення, масштабу та кута (рад)."""
    c = np.cos(angle) * scale
    s = np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Розгортає масив кутів (рад) для уникнення стрибків ±π при інтерполяції."""
    return np.unwrap(angles)
