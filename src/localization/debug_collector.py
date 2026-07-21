"""Opt-in колектор даних для вікон візуалізації моделей (debug views).

`Localizer.localize_frame(collector=...)` заповнює ці поля ЛИШЕ коли колектор
переданий (тобто відкрите хоча б одне вікно matches/dino/depth). За
замовчуванням `collector=None` — жодного додаткового коштування у гарячому
шляху локалізації.

Поля-запити (`want_*`) ставить worker перед викликом; поля-виходи заповнює
Localizer у міру проходження кадру. Часткове заповнення — норма: якщо кадр
відхилено рано (немає кандидатів), `rotated_frame` лишиться None і відповідні
вікна просто не оновляться цього keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DebugCollector:
    # ── Запити від worker-а: що саме рахувати (дороге — лише за потреби) ─────
    want_matches: bool = False   # keypoints / inlier-матчі / RMSE
    want_dino_pca: bool = False  # патч-токени DINO для PCA-візуалізації (окремий forward)
    want_depth: bool = False     # depth-мапа (окремий GPU-прохід)

    # ── Вихід: повернутий + GSD-нормалізований кадр (RGB) ───────────────────
    # У просторі саме цього кадру лежать keypoints/mkpts та патч-токени.
    rotated_frame: np.ndarray | None = None
    query_features: dict | None = None  # {'keypoints', 'descriptors', 'image_size', ...}

    # ── Вихід: точки / матчі ────────────────────────────────────────────────
    mkpts_q_inliers: np.ndarray | None = None  # (M, 2) query-side inliers
    mkpts_r_inliers: np.ndarray | None = None  # (M, 2) reference-side inliers
    total_matches: int = 0
    inliers: int = 0
    rmse: float = 0.0
    # ADDENDUM 1.1: просторовий розкид інлаєрів, min(σx,σy)/min(W,H).
    # None = порахувати не вдалося. Норма ≈ 0.29, колапс у кут < 0.05.
    spread: float | None = None

    # ── Вихід: retrieval / rotation панель ──────────────────────────────────
    candidate_id: int = -1
    retrieval_candidates: list = field(default_factory=list)  # [(frame_id, score), ...]
    global_angle: int = 0
    scale: float = 1.0
    global_score: float = 0.0

    # ── Вихід: DINO PCA ─────────────────────────────────────────────────────
    patch_tokens: np.ndarray | None = None    # (N, D) на CPU
    patch_grid: tuple | None = None           # (h_p, w_p)

    # ── Вихід: depth ────────────────────────────────────────────────────────
    depth_map: np.ndarray | None = None       # (H, W) float32, відносна глибина
    depth_scale: float | None = None          # відносний масштаб (1 / median depth)
