# config/config.py
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent

APP_CONFIG = {
    # ── XFeat (SuperPoint adapter) ────────────────────────────────────────
    'superpoint': {
        'nms_radius':          4,
        'keypoint_threshold':  0.005,
        'max_keypoints':       2048,
    },

    # ── DINOv2 global descriptor ──────────────────────────────────────────
    'dinov2': {
        'descriptor_dim': 384,   # ViT-S/14
    },

    # ── LightGlue matcher ─────────────────────────────────────────────────
    'lightglue': {
        'depth_confidence':   -1,   # -1 = disabled (faster, less pruning)
        'width_confidence':   -1,
        'filter_threshold':    0.1,
    },

    # ── Tracking / Kalman + OutlierDetector ───────────────────────────────
    'tracking': {
        'kalman_process_noise':       0.1,
        'kalman_measurement_noise':   10.0,

        # OutlierDetector — були 100000.0, що фактично вимикало детектор
        'outlier_threshold_std':       3.0,   # Z-score поріг (3σ = стандартний)
        'outlier_max_speed_mps':      80.0,   # макс. швидкість дрону (≈288 км/год)
        'outlier_window_size':        10,     # кількість точок для статистики
        'outlier_max_consecutive':     5,     # авто-скид після N відхилень підряд

        # Скільки кадрів відео реально локалізувати за секунду
        # 1.0 = 1 кадр/сек при будь-якому FPS відео
        'process_fps':                1.0,
    },

    # ── Localization ──────────────────────────────────────────────────────
    'localization': {
        'min_matches':       15,
        'ransac_threshold':   3.0,
        'retrieval_top_k':    5,
    },

    # ── GUI ───────────────────────────────────────────────────────────────
    'gui': {
        'video_fps': 30,   # частота оновлення preview у UI
    },

    # ── Image preprocessing ───────────────────────────────────────────────
    'preprocessing': {
        'clahe_clip_limit':   2.0,    # було захардковано 3.0 — агресивно для drone
        'clahe_tile_grid':   [8, 8],
        'histogram_matching': False,  # увімкнути якщо є reference_image_path
        # 'reference_image_path': 'assets/reference.jpg',
    },
}
