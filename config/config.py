# config/config.py

APP_CONFIG = {
    # Accelerated Features (XFeat) - основний локальний екстрактор
    'xfeat': {
        'max_keypoints': 2048,
        'detection_threshold': 0.05,
    },

    # SuperPoint - залишаємо для сумісності з LightGlue (якщо знадобиться)
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048,
    },

    # DINOv2 - глобальний дескриптор для пошуку кадрів у базі
    'dinov2': {
        'model_type': 'dinov2_vits14',
        'descriptor_dim': 384
    },

    # YOLO11 - сегментація для відсікання рухомих об'єктів
    'yolo': {
        'model_path': 'yolo11n-seg.pt',
        'conf_threshold': 0.25,
        # Класи COCO для маскування (транспорт, люди, тварини)
        'dynamic_classes': [0, 1, 2, 3, 4]#, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 33, 36, 37]
    },

    'lightglue': {
        'depth_confidence': -1,
        'width_confidence': -1,
        'filter_threshold': 0.1,
    },

    'tracking': {
        # Збільшено для кращого відстеження на високих швидкостях
        'kalman_process_noise': 2.0,
        'kalman_measurement_noise': 5.0,
        # Поріг відсікання аномалій у метрах за секунду (1 км/с)
        'max_speed_mps': 1000.0,
        'outlier_threshold_std': 3.0,
        'outlier_window': 10,
        # Бажана частота обробки кадрів локалізатором (FPS)
        'process_fps': 10.0
    },

    'localization': {
        'min_matches': 5,
        'ransac_threshold': 7.0,
        'top_k_candidates': 5,
        # Автоматичне обертання кадру (0, 90, 180, 270) при втраті локалізації
        'auto_rotation': True
    },

    'gui': {
        'video_fps': 1
    },

    'preprocessing': {
        'clahe_clip_limit': 3.0,
        'clahe_grid_size': (8, 8),
        'histogram_matching': True,
        'reference_image_path': "config/reference_style.png"
    }
}