"""Примітиви вибору keyframe-ів (чисті, без torch/Qt).

Витягнуто дослівно з ``DatabaseBuilder`` (IMPROVEMENT_PLAN п.1.3, розбиття
``database_builder`` на модулі). Рішення «чи це keyframe» залежить лише від
міжкадрової гомографії H і порогів руху — тому логіка headless-тестована.
Матчер інжектується параметром (``compute_inter_frame_homography``), а не
береться з ``self``, тож і цей шлях піддається юніт-тесту з фейковим матчером.

Семантику вибору («keyframe вибірково», на відміну від «поза завжди») цей модуль
НЕ змінює — він лише виносить обчислення; оркестрація (коли писати pose, коли
save_frame_data, ідентичність frame_id↔slot) лишається в ``DatabaseBuilder``.
"""

from __future__ import annotations

import numpy as np

from src.geometry.transformations import GeometryTransforms

# Дефолти = поточні дефолти config (database.*), щоб виклики без порогів
# зберігали поведінку білдера.
DEFAULT_MIN_TRANSLATION_PX = 15.0
DEFAULT_MIN_ROTATION_DEG = 1.5
DEFAULT_INTER_FRAME_MIN_MATCHES = 15
DEFAULT_INTER_FRAME_RANSAC_THRESH = 3.0


def is_significant_motion(
    H: np.ndarray,
    frame_w: int,
    frame_h: int,
    min_translation_px: float = DEFAULT_MIN_TRANSLATION_PX,
    min_rotation_deg: float = DEFAULT_MIN_ROTATION_DEG,
) -> bool:
    """True, якщо гомографія ``H`` (frame_b → frame_a) відповідає значному руху.

    Трансляція центру кадру через H ≥ ``min_translation_px`` АБО кут із лінійної
    частини H ≥ ``min_rotation_deg``. Вироджена H (|det| < 1e-6) → True (щоб не
    застрягнути на битій матриці). Логіка збережена 1:1 з DatabaseBuilder.
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    p_src = np.array([cx, cy, 1.0], dtype=np.float64)
    p_dst = H.astype(np.float64) @ p_src
    p_dst /= p_dst[2]
    translation = np.linalg.norm(p_dst[:2] - np.array([cx, cy]))

    if translation >= min_translation_px:
        return True

    A = H[:2, :2].astype(np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return True  # вироджена матриця → вважаємо рухом
    angle_deg = abs(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return bool(angle_deg >= min_rotation_deg)


def compute_inter_frame_homography(
    matcher,
    fa: dict,
    fb: dict,
    *,
    min_matches: int = DEFAULT_INTER_FRAME_MIN_MATCHES,
    ransac_thresh: float = DEFAULT_INTER_FRAME_RANSAC_THRESH,
    homography_backend: str = "opencv",
    use_mad_ransac: bool = True,
    mad_k_factor: float = 2.5,
) -> np.ndarray | None:
    """H(fb → fa) як 3×3 float64, або None.

    ``matcher.match(fa, fb)`` → відповідності → RANSAC-гомографія. None, якщо
    матчів або inlier-ів менше за ``min_matches``. Матчер інжектується (не
    створюється тут) — лінива ініціалізація/стан лишаються у виклику білдера.
    """
    mkpts_a, mkpts_b = matcher.match(fa, fb)
    if len(mkpts_a) < min_matches:
        return None

    H, mask = GeometryTransforms.estimate_homography(
        mkpts_a,
        mkpts_b,
        ransac_threshold=ransac_thresh,
        backend=homography_backend,
        use_mad_ransac=use_mad_ransac,
        mad_k_factor=mad_k_factor,
    )

    if H is None or int(np.sum(mask)) < min_matches:
        return None

    return H.astype(np.float64)
