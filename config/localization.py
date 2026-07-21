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

    # ── ADDENDUM 1.1: просторовий розкид інлаєрів → confidence. Дефолт off. ──
    # Інлаєрів може бути досить, але всі в одному куті кадру → гомографія
    # ill-conditioned (OrthoTrack §3.4, «spatial collapse»). Не відкидаємо кадр
    # (на межі покриття скупчення легітимне), а множимо confidence — далі він
    # їде в Kalman R (B2), тож слабкий фікс отримує меншу вагу, а не бан.
    spread_confidence_enabled: bool = False
    # Розкид, вище якого штрафу нема. 0.15 ≈ половина рівномірного покриття
    # (σ/L = 1/√12 ≈ 0.289 для точок, розкиданих по всьому кадру).
    spread_ref: float = 0.15
    # Нижня межа множника — вироджена хмара не обнуляє confidence.
    spread_floor: float = 0.35

    # ── ADDENDUM 2.1: каскадний recovery замість повного добутку. Дефолт off. ──
    # off = ПОТОЧНА поведінка: 4 кути × 5 масштабів = 20 ViT-forward одним батчем.
    # on: етап 1 — 4 кути на ОДНОМУ масштабі (prior → hint-найближчий → 1.0);
    # етап 2 (лише якщо скор етапу 1 < rotation_rescan_min_score) — РЕШТА
    # комбінацій. Найгірший випадок лишається 20, типовий стає 4.
    recovery_cascade: bool = False


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

    # ── RESEARCH 3.1: ковзний віконний back-end smoother (дефолт: вимкнено) ──
    # Вікно keyframe-фіксів + OF-одометрії, Huber-IRLS вага замість бінарного
    # Z-score-відкидання; корекція KF зсувом (TrajectoryFilter.shift).
    smoother_enabled: bool = False
    smoother_window: int = 60
    smoother_huber_k: float = 1.2
    smoother_fix_sigma_base_m: float = 5.0
    smoother_odom_sigma_base_m: float = 3.0
    smoother_max_correction_m: float = 50.0
    smoother_entry_prior_sigma_m: float = 15.0
    smoother_irls_iterations: int = 4
    # Fixed-lag servo (v2 після живого прогону): корекція по вузлу з лагом,
    # deadband проти смикання на шумі, гейн + ліміт кроку проти телепортів.
    smoother_correction_lag: int = 10
    smoother_deadband_m: float = 2.0
    smoother_gain: float = 0.25
    smoother_max_step_m: float = 3.0

    # ── ADDENDUM 1.2: forward-backward перевірка optical flow. Дефолт off. ──
    # Зараз LK фільтрується лише за status==1; треки, що «сповзли», доживають
    # до estimateAffinePartial2D. RANSAC там їх відкидає, тож на трансформацію
    # вплив малий — але вони роздувають знаменник inlier_ratio і ЗАНИЖУЮТЬ
    # flow_quality, який іде в Kalman R. Це фікс чесності метрики.
    of_fb_check: bool = False
    of_fb_max_px: float = 2.0


class HomographyConfig(BaseModel):
    backend: str = "opencv"  # "poselib" | "opencv"
    ransac_threshold: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.99
    use_mad_ransac: bool = True  # Адаптивне уточнення порогу через MAD
    mad_k_factor: float = 2.5
