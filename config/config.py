# config/config.py
#
# Єдиний конфіг для всього застосунку.
# Кожен параметр підписаний: який клас його читає і що він робить.
#
# Класи, що читають конфіг:
#   DatabaseBuilder                  → секція 'dinov2'
#   ImagePreprocessor                → секція 'preprocessing'
#   Localizer                        → секції 'localization', 'tracking'
#   CalibrationPropagationWorker     → секція 'localization'
#   ModelManager                     → секція 'lightglue'
#   RealtimeTrackingWorker           → секції 'tracking', 'gui'

APP_CONFIG = {

    # ══════════════════════════════════════════════════════════════════════════
    # DINOv2 — глобальний дескриптор для пошуку схожих кадрів у базі
    # Читає: DatabaseBuilder
    # ══════════════════════════════════════════════════════════════════════════
    'dinov2': {
        # Розмірність вектора-дескриптора.
        # dinov2_vits14 → 384, dinov2_vitb14 → 768, dinov2_vitl14 → 1024
        # ⚠️ Має збігатися з моделлю у ModelManager.load_dinov2()
        # Читає: DatabaseBuilder.__init__ → HDF5 розмір datasets
        'descriptor_dim': 1024,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # LightGlue — точний матчер ключових точок
    # Читає: ModelManager.load_lightglue()
    # ══════════════════════════════════════════════════════════════════════════
    'lightglue': {
        # Ранній вихід по глибині графа (-1 = вимкнено, 0.95 = агресивний)
        # Читає: ModelManager → LightGlue(depth_confidence=...)
        'depth_confidence': -1,

        # Ранній вихід по ширині графа (-1 = вимкнено)
        # Читає: ModelManager → LightGlue(width_confidence=...)
        'width_confidence': -1,

        # Мінімальний score для визнання збігу хорошим (0..1)
        # Читає: ModelManager → LightGlue(filter_threshold=...)
        'filter_threshold': 0.1,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Локалізація — параметри пайплайну Localizer + CalibrationPropagationWorker
    # Читає: Localizer.__init__, CalibrationPropagationWorker.__init__
    # ══════════════════════════════════════════════════════════════════════════
    'localization': {
        # Мінімальна кількість збігів для прийняття гомографії
        # Читає: Localizer, CalibrationPropagationWorker
        'min_matches': 8,

        # Мінімальна кількість інлієрів для прийняття результату (інакше fallback)
        'min_inliers_accept': 8,

        # Поріг відношення Lowe (чим ближче до 1.0, тим більше збігів пропускає)
        'ratio_threshold': 0.95,

        # Поріг репроєкційної помилки для RANSAC/MAGSAC++ (пікселі)
        # Більше значення = більше "шумних" збігів вважатимуться валідними
        'ransac_threshold': 5.0,

        # Кількість кандидатів з DINOv2-пошуку для перебору
        'retrieval_top_k': 8,

        # Кількість inliers для дострокового виходу з перебору ротацій/кандидатів
        'early_stop_inliers': 20,

        # Масштабний множник впевненості: confidence = inliers / confidence_max_inliers
        'confidence_max_inliers': 50,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Tracking — Kalman фільтр і детектор аномалій
    # Читає: Localizer.__init__ (створює TrajectoryFilter і OutlierDetector),
    #        RealtimeTrackingWorker.__init__
    # ══════════════════════════════════════════════════════════════════════════
    'tracking': {
        # Шум процесу Kalman-фільтра (q). Більше → фільтр довіряє вимірюванням,
        # менше → більш плавна траєкторія, але запізнення реакції на різкий маневр
        # Читає: Localizer → TrajectoryFilter(process_noise=...)
        'kalman_process_noise': 2.0,

        # Шум вимірювань Kalman-фільтра (r, метри²). Більше → більше згладжування
        # Читає: Localizer → TrajectoryFilter(measurement_noise=...)
        'kalman_measurement_noise': 5.0,

        # Поріг Z-score для відкидання стрибків координат
        # Читає: Localizer → OutlierDetector(threshold_std=...)
        'outlier_threshold_std': 300.0,

        # Максимальна фізично можлива швидкість дрона (м/с). Все вище — аномалія.
        # 1000 м/с = фактично вимкнено перевірку швидкості
        # Читає: Localizer → OutlierDetector(max_speed_mps=...)
        'max_speed_mps': 1000.0,

        # Бажана частота обробки кадрів локалізатором (кадрів/секунду)
        # frame_step = round(video_fps / process_fps), dt = frame_step / video_fps
        # Читає: RealtimeTrackingWorker.__init__
        'process_fps': 1.0,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Препроцесинг зображень — CLAHE + histogram matching
    # Читає: ImagePreprocessor.__init__
    # ══════════════════════════════════════════════════════════════════════════
    'preprocessing': {
        # Ліміт контрастного підсилення CLAHE (вищий = агресивніше)
        # Читає: ImagePreprocessor → cv2.createCLAHE(clipLimit=...)
        'clahe_clip_limit': 3.0,

        # Розмір тайлу CLAHE у пікселях [ширина, висота]
        # ⚠️ Ключ ОБОВ'ЯЗКОВО 'clahe_tile_grid' — саме так читає ImagePreprocessor
        # Читає: ImagePreprocessor → cv2.createCLAHE(tileGridSize=...)
        'clahe_tile_grid': [8, 8],

        # Вирівнювання гістограми за еталонним зображенням (нормалізація погоди/освітлення)
        # Читає: ImagePreprocessor.__init__ + preprocess()
        'histogram_matching': True,

        # Шлях до еталонного зображення для histogram matching
        # None або відсутній ключ → histogram_matching вимикається автоматично
        # Читає: ImagePreprocessor._load_reference()
        'reference_image_path': 'config/reference_style.png',
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GUI — параметри інтерфейсу
    # Читає: RealtimeTrackingWorker.__init__
    # ══════════════════════════════════════════════════════════════════════════
    'gui': {
        # Цільовий FPS відображення відео у VideoWidget
        # Читає: RealtimeTrackingWorker → frame_time = 1.0 / target_fps
        'video_fps': 30,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Моделі — назви файлів, репозиторії та вимоги до VRAM
    # Читає: ModelManager
    # ══════════════════════════════════════════════════════════════════════════
    'models': {
        'use_cuda': True,  # Дозволяє вимкнути CUDA навіть якщо вона доступна
        'yolo': {
            'model_path': 'yolo11x-seg.pt',
            'vram_required_mb': 1200.0,
            'description': 'YOLOv11x-seg (Extra Large) for dynamic object masking'
        },
        'xfeat': {
            'hub_repo': 'verlab/accelerated_features',
            'hub_model': 'XFeat',
            'top_k': 2048,
            'vram_required_mb': 300.0
        },
        'superpoint': {
            'nms_radius': 4,
            'max_keypoints': 4096,
            'vram_required_mb': 500.0
        },
        'lightglue': {
            'depth_confidence': -1,
            'width_confidence': -1,
            'vram_required_mb': 1000.0
        },
        'dinov2': {
            'hub_repo': 'facebookresearch/dinov2',
            'hub_model': 'dinov2_vitl14',
            'vram_required_mb': 1600.0
        },
        'vram_management': {
            'enabled': True,
            'max_vram_ratio': 0.8,
            'default_required_mb': 2000.0
        }
    },
}