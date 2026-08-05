"""Opt-in data collector for model debug views.

`Localizer.localize_frame(collector=...)` fills these fields ONLY when collector
is provided (i.e., at least one debug window matches/dino/depth is open). By
default `collector=None` — zero additional overhead in hot localization path.

Request fields (`want_*`) are set by worker before call; output fields are populated by
Localizer as frame processing proceeds. Partial population is normal: if a frame
is rejected early (no candidates), `rotated_frame` remains None and corresponding
windows simply do not update for this keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DebugCollector:
    # -- Worker requests: what to compute (expensive items only on demand) -----
    want_matches: bool = False  # keypoints / inlier-matches / RMSE
    want_dino_pca: bool = False  # DINO patch tokens for PCA visualization (separate forward)
    want_depth: bool = False  # depth map (separate GPU pass)

    # -- Output: rotated + GSD-normalized frame (RGB) -------------------
    # Keypoints/mkpts and patch tokens lie in the space of this frame.
    rotated_frame: np.ndarray | None = None
    query_features: dict | None = None  # {'keypoints', 'descriptors', 'image_size', ...}

    # -- Output: points / matches ----------------------------------------
    mkpts_q_inliers: np.ndarray | None = None  # (M, 2) query-side inliers
    mkpts_r_inliers: np.ndarray | None = None  # (M, 2) reference-side inliers
    total_matches: int = 0
    inliers: int = 0
    rmse: float = 0.0
    # ADDENDUM 1.1: spatial inlier spread, min(sigma_x, sigma_y)/min(W, H).
    # None = computation failed. Normal ~ 0.29, corner collapse < 0.05.
    spread: float | None = None

    # -- Output: retrieval / rotation panel ------------------------------
    candidate_id: int = -1
    retrieval_candidates: list = field(default_factory=list)  # [(frame_id, score), ...]
    global_angle: int = 0
    scale: float = 1.0
    global_score: float = 0.0

    # -- Output: DINO PCA ------------------------------------------------
    patch_tokens: np.ndarray | None = None  # (N, D) on CPU
    patch_grid: tuple | None = None  # (h_p, w_p)

    # -- Output: depth ---------------------------------------------------
    depth_map: np.ndarray | None = None  # (H, W) float32, relative depth
    depth_scale: float | None = None  # relative scale (1 / median depth)
