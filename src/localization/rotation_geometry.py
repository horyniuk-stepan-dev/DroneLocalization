"""Frame-rotation geometry shared by localization and optical flow.

np.rot90(frame, k=K) rotates the frame counter-clockwise by K*90 degrees. These
helpers map a displacement vector / point measured in the ORIGINAL frame into the
rotated frame's coordinate system (where the homography H was built).

Values verified numerically against np.rot90 (pixel-centred convention):
  k=1 (90):  (x,y) -> (y, W-1-x)      vector (dx,dy) -> (dy, -dx)
  k=2 (180): (x,y) -> (W-1-x, H-1-y)  vector (dx,dy) -> (-dx, -dy)
  k=3 (270): (x,y) -> (H-1-y, x)      vector (dx,dy) -> (-dy, dx)
"""

from __future__ import annotations

# angle: (a, b, c, d) -> new_dx = a*dx + b*dy, new_dy = c*dx + d*dy
_ROTATION_VEC: dict[int, tuple[int, int, int, int]] = {
    0: (1, 0, 0, 1),
    90: (0, 1, -1, 0),
    180: (-1, 0, 0, -1),
    270: (0, -1, 1, 0),
}


def _rotate_point_np90(x: float, y: float, w: float, h: float, angle: int) -> tuple[float, float]:
    """Map point (x, y) of a w×h frame into np.rot90(frame, k=angle//90) coords."""
    if angle == 90:
        return y, (w - 1.0) - x
    if angle == 180:
        return (w - 1.0) - x, (h - 1.0) - y
    if angle == 270:
        return (h - 1.0) - y, x
    return x, y


# ── Загальна ротація фіч + одометричний кут ланцюга (Етап 5: rotation-retry) ──
# Для temporal-матчингу без heading-hold: коли сусідні кадри сильно повернуті,
# матч падає. Повертаємо keypoints query на кут із ланцюга frame_poses (готовий
# одометричний пріор БД) або перебором k·90°, і повторюємо матч. Отриману
# гомографію H_r (rotated_query→ref) компонуємо назад: H_true = H_r · R(θ).

import numpy as np


def rotation_homography(angle_rad: float, cx: float, cy: float) -> np.ndarray:
    """3x3 гомографія повороту точок на angle_rad НАВКОЛО (cx, cy)."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, cx - c * cx + s * cy],
            [s, c, cy - s * cx - c * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_keypoints(kpts: np.ndarray, angle_rad: float, cx: float, cy: float) -> np.ndarray:
    """Повертає Nx2 keypoints на angle_rad навколо (cx, cy). Дескриптори не чіпаємо."""
    kpts = np.asarray(kpts, dtype=np.float64)
    if kpts.size == 0:
        return kpts.copy()
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    x = kpts[:, 0] - cx
    y = kpts[:, 1] - cy
    return np.stack([c * x - s * y + cx, s * x + c * y + cy], axis=1)


def chain_relative_angle_deg(pose_from: np.ndarray, pose_to: np.ndarray) -> float | None:
    """Відносний поворот (град) кадру `to` відносно `from` із кумулятивних
    3x3 chain-поз БД (frame_poses). None, якщо поза вироджена (нулі/сингулярна).
    Кут прикладається до query, щоб вирівняти його орієнтацію з референсом."""
    pf = np.asarray(pose_from, dtype=np.float64)
    pt = np.asarray(pose_to, dtype=np.float64)
    if pf.shape != (3, 3) or pt.shape != (3, 3) or not np.any(pf) or not np.any(pt):
        return None
    try:
        rel = np.linalg.inv(pf) @ pt  # H_{from→to}
    except np.linalg.LinAlgError:
        return None
    return float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))


def temporal_retry_angles(chain_angle_deg: float | None, use_chain: bool = True) -> list[float]:
    """Кути (град) для повторного temporal-матчу (Етап 5): кут ланцюга (якщо є),
    далі fallback-перебір k·90°. Кут ≈0 пропускаємо (первинний матч уже пробував 0)."""
    raw: list[float] = []
    if use_chain and chain_angle_deg is not None:
        raw.append(float(chain_angle_deg))
    raw += [90.0, 180.0, 270.0]

    def _norm(a: float) -> float:
        return ((a + 180.0) % 360.0) - 180.0  # у (−180, 180]

    out: list[float] = []
    for a in raw:
        na = _norm(a)
        if abs(na) < 1e-6:
            continue  # 0° уже пробували у первинному матчі
        if not any(abs(_norm(na - b)) < 1.0 for b in out):
            out.append(na)
    return out
