# config/config.py
#
# Єдиний конфіг для всього застосунку.
# Залишені ключі, які реально використовуються поточним кодом.

APP_CONFIG = {

    # ══════════════════════════════════════════════════════════════════════════
    # DINOv2 — глобальний дескриптор для пошуку схожих кадрів у базі
    # Читають: DatabaseBuilder, FeatureExtractor, ModelManager
    # ══════════════════════════════════════════════════════════════════════════
    'dinov2': {
        # Має збігатися з моделлю у models.dinov2.hub_model
        'descriptor_dim': 1024,

        # FeatureExtractor resize перед DINOv2
        'input_size': 336,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Локалізація — Localizer + CalibrationPropagationWorker
    # ══════════════════════════════════════════════════════════════════════════
    'localization': {
        'min_matches': 12,
        'min_inliers_accept': 10,
        'ratio_threshold': 0.85, # Більш строгий ratio test для XFeat
        'ransac_threshold': 3.0,
        'retrieval_top_k': 12,
        'early_stop_inliers': 40,
        'retrieval_only_min_score': 0.90,

        # Перебір 0 / 90 / 180 / 270
        'auto_rotation': True,

        # Fallback на SuperPoint + LightGlue при слабкому XFeat
        'enable_lightglue_fallback': True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Tracking — Kalman + OutlierDetector + RealtimeTrackingWorker
    # ══════════════════════════════════════════════════════════════════════════
    'tracking': {
        'kalman_process_noise': 2.0,
        'kalman_measurement_noise': 5.0,

        # Історія для OutlierDetector
        'outlier_window': 10,

        'outlier_threshold_std': 25.0,
        'max_speed_mps': 200.0,
        'process_fps': 1.0,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Препроцесинг зображення
    # ══════════════════════════════════════════════════════════════════════════
    'preprocessing': {
        'clahe_clip_limit': 3.0,
        'clahe_tile_grid': [8, 8],
        'histogram_matching': True,
        'reference_image_path': 'config/reference_style.png',
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GUI
    # ══════════════════════════════════════════════════════════════════════════
    'gui': {
        'video_fps': 30,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Моделі
    # ══════════════════════════════════════════════════════════════════════════
    'models': {
        'use_cuda': True,

        'yolo': {
            'model_path': 'yolo11x-seg.pt',
            'vram_required_mb': 1200.0,
            'description': 'YOLOv11x-seg (Extra Large) for dynamic object masking',
        },

        'xfeat': {
            'hub_repo': 'verlab/accelerated_features',
            'hub_model': 'XFeat',
            'top_k': 2048,
            'vram_required_mb': 300.0,
        },

        'superpoint': {
            'nms_radius': 4,
            'max_keypoints': 4096,
            'vram_required_mb': 500.0,
        },

        'lightglue': {
            'depth_confidence': -1,
            'width_confidence': -1,
            'vram_required_mb': 1000.0,
        },

        'dinov2': {
            'hub_repo': 'facebookresearch/dinov2',
            'hub_model': 'dinov2_vitl14',
            'vram_required_mb': 1600.0,
        },

        'vram_management': {
            'max_vram_ratio': 0.8,
            'default_required_mb': 2000.0,
        },
    },
    'projection': {
        'default_mode': 'WEB_MERCATOR',
        'strict_projection': True,
        'fallback_to_webmercator': True,
        'anchor_rmse_threshold_m': 3.0,
        'anchor_max_error_m': 5.0,
        'propagation_disagreement_threshold_m': 2.0,
        'localizer_sample_points': 9,  # 3x3 grid
        'localizer_expected_spread_m': 150.0, # Очікуваний розмах для дрона на 100-150м
        'confidence_max_inliers': 80,
    },
}
