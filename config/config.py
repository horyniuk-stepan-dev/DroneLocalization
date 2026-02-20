# config/config.py

APP_CONFIG = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048,
    },
    'netvlad': {
        'descriptor_dim': 4096  # <--- Змінити тут з 32768 на 4096
    },
    'lightglue': {
        'depth_confidence': -1,
        'width_confidence': -1,
        'filter_threshold': 0.1,
    },
    'depth_anything': {
        'encoder': 'vits'
    },
    'tracking': {
        'kalman_process_noise': 0.1,
        'kalman_measurement_noise': 10.0,
        'outlier_threshold_std': 100.0
    },
    'localization': {
        'min_matches': 15,
        'ransac_threshold': 3.0,
        'retrieval_top_k': 5
    },
    'gui': {
        'video_fps': 30
    }
}