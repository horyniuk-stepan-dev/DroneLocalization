"""Localization, tracking and homography configuration."""

from pydantic import BaseModel


class ConfidenceConfig(BaseModel):
    inlier_weight: float = 0.7
    stability_weight: float = 0.3
    rmse_norm_m: float = 10.0
    disagreement_norm_m: float = 5.0
    confidence_max_inliers: int = 80


class LocalizationConfig(BaseModel):
    min_matches: int = 12
    min_inliers_accept: int = 10
    # 0.75 = рекомендація Lowe's ratio test; 0.85 пропускало забагато хибних
    # збігів (конфіг перекривав фікс "БАГ 4" у matcher.py)
    ratio_threshold: float = 0.75
    ransac_threshold: float = 3.0
    retrieval_top_k: int = 12
    early_stop_inliers: int = 40
    retrieval_only_min_score: float = 0.90
    auto_rotation: bool = True
    # A3: темпоральний prior кута — пробуємо лише останній вдалий кут
    # (1 DINO-forward замість 4). Якщо retrieval-score prior-кута нижчий за цей
    # поріг — повний батчований скан 4 кутів.
    rotation_rescan_min_score: float = 0.70
    enable_lightglue_fallback: bool = True
    # ── RESEARCH 2.2: аварійний SIFT+LightGlue фолбек ──
    # Одноразовий перезапуск матчингу через SIFT, коли ALIKED дав
    # < min_matches inliers (in-plane rotation / екстремальна похилість).
    # Вимагає БД, збудованої з database.store_sift_features=True.
    sift_fallback: bool = False
    sift_fallback_max_candidates: int = 3
    fallback_extractor: str = "aliked"
    confidence: ConfidenceConfig = ConfidenceConfig()
    use_patchify: bool = False  # Мультимасштабний retrieval через патч-дескриптори DINOv2 (14× DINOv2 forward passes — повільно на слабких GPU)
    patchify_grids: list[list[int]] = [[1, 1], [2, 2], [3, 3]]  # 1+4+9 = 14 патчів
    patchify_batch_size: int = (
        1  # Кількість патчів за один DINOv2 інференс (1 = послідовно, 4-7 = батч)
    )
    patchify_merge_weight: float = 0.4
    # ── Scale-invariance (ScaleManager) ──
    # GSD-ratio pyramid for altitude-invariant localization (r = query_alt / db_alt).
    # Scanned when no temporal prior exists (bootstrap / out-of-coverage).
    scale_pyramid: list[float] = [0.5, 0.7, 1.0, 1.4, 2.0]
    # Minimum retrieval score on the prior-scale level to accept it;
    # below this → full pyramid rescan (analogous to rotation_rescan_min_score).
    scale_rescan_min_score: float = 0.65
    # EMA smoothing factor for the temporal scale prior (higher = faster tracking).
    scale_prior_ema: float = 0.7
    # Use DepthAnythingV2 depth_scales (if available in DB) to reorder the pyramid.
    scale_use_depth_hint: bool = True
    # How often (in keyframes) to recompute the depth-based scale hint.
    depth_hint_every_n: int = 30


class TrackingConfig(BaseModel):
    kalman_process_noise: float = 2.0
    kalman_measurement_noise: float = 5.0
    outlier_window: int = 10
    # ВИПРАВЛЕНО: 150.0 фактично ВИМИКАЛО Z-score фільтр (z>150 не буває).
    # 4.0 = класичний статистичний поріг для викидів.
    outlier_threshold_std: float = 4.0
    # ВИПРАВЛЕНО: 1000 м/с (3.6 млн км/год) не фільтрувало нічого.
    # 120 м/с покриває будь-який реальний дрон із запасом.
    max_speed_mps: float = 120.0
    max_consecutive_outliers: int = 3
    process_fps: float = 1.0
    keyframe_interval: int = 30


class HomographyConfig(BaseModel):
    backend: str = "opencv"  # "poselib" | "opencv"
    ransac_threshold: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.99
    use_mad_ransac: bool = True  # Адаптивне уточнення порогу через MAD
    mad_k_factor: float = 2.5
