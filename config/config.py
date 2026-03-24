# config/config.py
#
# Єдиний конфіг для всього застосунку.
# Залишені ключі, які реально використовуються поточним кодом.


def get_cfg(config: dict, path: str, default=None):
    """Централізований доступ до конфігу з dot-path.

    Приклад: get_cfg(config, 'tracking.outlier_threshold_std', 25.0)
    """
    keys = path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


APP_CONFIG = {
    # ══════════════════════════════════════════════════════════════════════════
    # DINOv2 — глобальний дескриптор для пошуку схожих кадрів у базі
    # Читають: DatabaseBuilder, FeatureExtractor, ModelManager
    # ══════════════════════════════════════════════════════════════════════════
    "dinov2": {
        "descriptor_dim": 1024,
        "input_size": 336,
    },
    # ══════════════════════════════════════════════════════════════════════════
    # База даних — DatabaseBuilder
    # ══════════════════════════════════════════════════════════════════════════
    "database": {
        "frame_step": 3,
        "prefetch_queue_size": 32,
        "keypoint_video_scale": 0.5,
        "inter_frame_min_matches": 15,
        "inter_frame_ransac_thresh": 3.0,
    },
    # ══════════════════════════════════════════════════════════════════════════
    # Локалізація — Localizer + CalibrationPropagationWorker
    # ══════════════════════════════════════════════════════════════════════════
    "localization": {
        "min_matches": 12,
        "min_inliers_accept": 10,
        "ratio_threshold": 0.85,
        "ransac_threshold": 3.0,
        "retrieval_top_k": 12,
        "early_stop_inliers": 40,
        "retrieval_only_min_score": 0.90,
        "auto_rotation": True,
        "enable_lightglue_fallback": True,
        "fallback_extractor": "aliked",
        # Налаштування впевненості (Fix 8)
        "confidence": {
            "inlier_weight": 0.7,
            "stability_weight": 0.3,
            "rmse_norm_m": 10.0,
            "disagreement_norm_m": 5.0,
            "confidence_max_inliers": 80,
        },
    },
    # ══════════════════════════════════════════════════════════════════════════
    # Tracking — Kalman + OutlierDetector + RealtimeTrackingWorker
    # ══════════════════════════════════════════════════════════════════════════
    "tracking": {
        "kalman_process_noise": 2.0,
        "kalman_measurement_noise": 5.0,
        # Історія для OutlierDetector
        "outlier_window": 10,
        "outlier_threshold_std": 25.0,
        "max_speed_mps": 40.0,
        "process_fps": 1.0,
    },
    # ══════════════════════════════════════════════════════════════════════════
    # Препроцесинг зображення
    # ══════════════════════════════════════════════════════════════════════════
    "preprocessing": {
        "clahe_clip_limit": 3.0,
        "clahe_tile_grid": [8, 8],
        "histogram_matching": True,
        "reference_image_path": "config/reference_style.png",
    },
    # ══════════════════════════════════════════════════════════════════════════
    # GUI
    # ══════════════════════════════════════════════════════════════════════════
    "gui": {
        "video_fps": 30,
    },
    # ══════════════════════════════════════════════════════════════════════════
    # Моделі
    # ══════════════════════════════════════════════════════════════════════════
    "models": {
        "use_cuda": True,
        "yolo": {
            "model_path": "yolo11x-seg.pt",
            "vram_required_mb": 1200.0,
            "description": "YOLOv11x-seg (Extra Large) for dynamic object masking",
        },
        "xfeat": {
            "hub_repo": "verlab/accelerated_features",
            "hub_model": "XFeat",
            "top_k": 2048,
            "vram_required_mb": 300.0,
        },
        "aliked": {
            "max_keypoints": 4096,
            "detection_threshold": 0.2,
            "vram_required_mb": 400.0,
        },
        "superpoint": {
            "nms_radius": 4,
            "max_keypoints": 4096,
            "vram_required_mb": 500.0,
        },
        "lightglue": {
            "depth_confidence": -1,
            "width_confidence": -1,
            "vram_required_mb": 1000.0,
        },
        "dinov2": {
            "hub_repo": "facebookresearch/dinov2",
            "hub_model": "dinov2_vitl14",
            "vram_required_mb": 1600.0,
        },
        # EXPERIMENTAL: CESP (Cross-Enhancement Spatial Pyramid) для DINOv2.
        # Потребує навчених ваг (weights_path) для production use.
        # Без ваг використовує random projection — не впливає на якість.
        # load_cesp() реалізовано в ModelManager, інтеграція — у FeatureExtractor.
        "cesp": {
            "enabled": False,
            "weights_path": None,
            "scales": [1, 2, 4],
        },
        "vram_management": {
            "max_vram_ratio": 0.8,
            "default_required_mb": 2000.0,
        },
        "performance": {
            "propagation_max_workers": 4,
            "fp16_enabled": True,
        },
    },
    "projection": {
        "default_mode": "WEB_MERCATOR",
        "strict_projection": True,
        "fallback_to_webmercator": True,
        "anchor_rmse_threshold_m": 3.0,
        "anchor_max_error_m": 5.0,
        "propagation_disagreement_threshold_m": 2.0,
        "localizer_sample_points": 9,  # 3x3 grid
        "localizer_expected_spread_m": 150.0,  # Очікуваний розмах для дрона на 100-150м
        # confidence_max_inliers — єдине визначення в localization.confidence
    },
}
