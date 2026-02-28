# config/config.py

APP_CONFIG = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048,
    },
    'dinov2': {
        'descriptor_dim': 384
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
        'outlier_threshold_std': 100000.0
    },
    'localization': {
        'min_matches': 15,
        'ransac_threshold': 3.0,
        'retrieval_top_k': 5
    },
    'gui': {
        'video_fps': 30
    },
    'preprocessing': {
        'histogram_matching': True,
        'reference_image_path':"E:/Dip/gsdfg/Ne/DroneLocalization/config/reference_style.png"
    }
}