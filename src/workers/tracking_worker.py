import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config.config import get_cfg
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealtimeTrackingWorker(QThread):
    """Real-time localization worker thread (Optimized for XFeat + YOLO11)"""

    frame_ready = pyqtSignal(np.ndarray)
    location_found = pyqtSignal(float, float, float, int)
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    fov_found = pyqtSignal(list)
    objects_detected = pyqtSignal(object)  # list[TrackedObject]
    objects_gps_updated = pyqtSignal(object)  # list[ObjectGPS]

    def __init__(self, video_source: str, localizer, model_manager=None, config=None):
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.model_manager = model_manager
        self.config = config or {}
        self._stop_event = threading.Event()

        # S3-3: Інтервал ключових кадрів для локалізації
        self.keyframe_interval = get_cfg(self.config, "tracking.keyframe_interval", 5)
        # Зберігаємо process_fps для метрик UI, але логіка базується на кадрах
        self.process_fps = get_cfg(
            self.config, "tracking.process_fps", 30.0 / self.keyframe_interval
        )
        self.tracking_config = get_cfg(self.config, "object_tracking", {})

    def run(self):
        # Fix #3: Скидаємо стан трекера при кожному новому старті сесії
        if hasattr(self.localizer, "trajectory_filter"):
            self.localizer.trajectory_filter.reset()
        if hasattr(self.localizer, "outlier_detector"):
            self.localizer.outlier_detector.window.clear()
            self.localizer.outlier_detector._consecutive_outliers = 0
        if hasattr(self.localizer, "_consecutive_failures"):
            self.localizer._consecutive_failures = 0

        if self.model_manager:
            self.model_manager.pin(["aliked", "lightglue_aliked", "dinov2"])

        from src.tracking.object_tracker import ObjectTracker
        from src.tracking.object_projector import ObjectProjector
        
        object_tracker = None
        object_projector = None
        
        is_tracking_enabled = False
        if isinstance(self.tracking_config, dict):
            is_tracking_enabled = self.tracking_config.get("enabled", False)
        else:
            is_tracking_enabled = getattr(self.tracking_config, "enabled", False)
            
        if is_tracking_enabled:
            tracker_cfg = self.tracking_config if isinstance(self.tracking_config, dict) else self.tracking_config.model_dump()
            try:
                object_tracker = ObjectTracker(tracker_cfg)
                object_projector = ObjectProjector(self.localizer.calibration)
                logger.info("Object tracking enabled")
            except Exception as e:
                logger.error(f"Failed to initialize object tracking: {e}")

        # Fix 6: Pre-warm fallback моделей при старті трекінгу
        threading.Thread(target=self._prewarm_fallback_models, daemon=True).start()

        logger.info(f"Starting tracking from source: {self.video_source}")

        yolo_wrapper = None
        if self.model_manager:
            try:
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
                logger.success("YOLO loaded for dynamic object masking in tracking loop")
            except Exception as e:
                logger.error(
                    f"Failed to load YOLO for tracking: {e} | "
                    f"device={self.model_manager.device}. "
                    f"Dynamic object masking will be unavailable. "
                    f"Tracking cannot proceed without YOLO.",
                    exc_info=True,
                )
                self.error.emit(f"YOLO не вдалося завантажити: {e}")
                return

        from src.video.video_source import VideoSource, VideoSourceConfig
        
        if isinstance(self.video_source, VideoSource):
            video_src = self.video_source
        else:
            v_config = VideoSourceConfig(source=str(self.video_source))
            video_src = VideoSource(v_config)

        if not video_src.is_opened:
            logger.error(
                f"Failed to open video source: {self.video_source}. "
                f"Check that the source is available."
            )
            self.error.emit(f"Не вдалося відкрити відеоджерело: {self.video_source}")
            return

        video_fps = video_src.fps
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        # Замість time-based інтервалу використовуємо frame-based:
        frame_idx = 0
        prev_gray_for_of = None
        prev_pts_for_of = None
        last_tracked_objects = []  # Кеш об'єктів з останнього ключового кадру для OF-кадрів

        # Зберігаємо останній час локалізації саме за ВІДЕО-часом, а не за процесорним
        last_localization_video_time = -1.0
        # Час останнього ОБРОБЛЕНОГО keyframe-а (навіть якщо він був відхилений як outlier)
        # Це потрібно для коректного dt в outlier_detector: якщо всі keyframe-и
        # відхиляються, last_localization_video_time залишається -1, і dt = 0.033s,
        # що штучно завищує швидкість у 5× (keyframe_interval=5).
        last_keyframe_video_time = -1.0

        stream_start_time = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()

            ret, frame = video_src.read()
            if not ret:
                logger.info("End of video stream or connection lost.")
                self.status_update.emit("Відеопотік завершено або втрачено.")
                break

            if video_src.is_live:
                current_video_time_sec = time.time() - stream_start_time
            else:
                # Отримуємо поточний час САМОГО ВІДЕО у секундах
                current_video_time_sec = video_src._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                # Fallback: деякі кодеки повертають 0 — рахуємо за номером кадру
                if current_video_time_sec <= 0:
                    current_video_time_sec = video_src._cap.get(cv2.CAP_PROP_POS_FRAMES) * frame_duration_sec

            # 1. Завжди відправляємо кадр в GUI для плавного відтворення (сирий BGR)
            self.frame_ready.emit(frame)

            # S3-3: Optical Flow Pipeline
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = frame_idx % self.keyframe_interval == 0

            # Розрахунок dt — різний для KF та OF
            if is_keyframe or prev_pts_for_of is None:
                # Для ключових кадрів: dt = час від ПОПЕРЕДНЬОГО ключового кадру
                # (навіть якщо він був відхилений як outlier)
                if last_keyframe_video_time < 0:
                    calculated_dt = self.keyframe_interval * frame_duration_sec
                else:
                    calculated_dt = current_video_time_sec - last_keyframe_video_time
                    if calculated_dt <= 0:
                        calculated_dt = self.keyframe_interval * frame_duration_sec
            else:
                # Для OF-кадрів: dt = час від останньої УСПІШНОЇ локалізації
                if last_localization_video_time < 0:
                    calculated_dt = frame_duration_sec
                else:
                    calculated_dt = current_video_time_sec - last_localization_video_time
                    if calculated_dt <= 0:
                        calculated_dt = frame_duration_sec

            loc_result = {"success": False, "error": "Not processed"}
            start_process = time.time()

            if is_keyframe or prev_pts_for_of is None:
                # ====== HEAVY KEYFRAME LOCALIZATION ======
                # Для обробки YOLO та анізотропних дескрипторів потрібен RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                static_mask = None
                detections = []
                if yolo_wrapper:
                    static_mask, detections = yolo_wrapper.detect_and_mask(frame_rgb)

                try:
                    loc_result = self.localizer.localize_frame(
                        frame_rgb, static_mask=static_mask, dt=calculated_dt
                    )
                except Exception as e:
                    logger.error(f"Localization exception on keyframe: {e}", exc_info=True)
                    loc_result = {"success": False, "error": str(e)}

                # Завжди оновлюємо час останнього keyframe, навіть якщо він rejected
                last_keyframe_video_time = current_video_time_sec

                if loc_result.get("success"):
                    # Зберігаємо стан для OF на наступні кадри
                    prev_gray_for_of = curr_gray
                    # Трекаємо гарні точки (corners) для стабільного OF
                    prev_pts_for_of = cv2.goodFeaturesToTrack(
                        curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, mask=None
                    )
                    
                if object_tracker and detections is not None:
                    tracked_objects = object_tracker.update(detections, frame.shape)
                    # Кешуємо останні відстежені об'єкти для OF-кадрів
                    last_tracked_objects = tracked_objects if tracked_objects else last_tracked_objects
                    if tracked_objects:
                        self.objects_detected.emit(tracked_objects)
                        
                        if object_projector and getattr(self.localizer, "_last_state", None):
                            H = self.localizer._last_state.get("H")
                            affine = self.localizer._last_state.get("affine")
                            angle = self.localizer._last_state.get("global_angle", 0)
                            
                            if H is not None and affine is not None:
                                objects_gps = object_projector.project_objects(
                                    tracked_objects, H, affine, angle, frame.shape[1], frame.shape[0]
                                )
                                if objects_gps:
                                    obj_summary = ", ".join([f"{obj.class_name} #{obj.track_id}" for obj in objects_gps])
                                    logger.info(f"Tracked {len(objects_gps)} objects (KF): {obj_summary}")
                                    self.objects_gps_updated.emit(objects_gps)
            else:
                # ====== OPTICAL FLOW TRACKING ======
                if prev_pts_for_of is not None and len(prev_pts_for_of) > 10:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray_for_of,
                        curr_gray,
                        prev_pts_for_of,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                    )
                    good_new = curr_pts[status == 1]
                    good_old = prev_pts_for_of[status == 1]

                    if len(good_new) > 10:
                        # Зсув у пікселях
                        flow_vectors = good_new - good_old
                        dx_px, dy_px = np.median(flow_vectors, axis=0)

                        try:
                            loc_result = self.localizer.localize_optical_flow(
                                dx_px,
                                dy_px,
                                dt=calculated_dt,
                                rot_width=frame.shape[1],
                                rot_height=frame.shape[0],
                            )
                        except Exception as e:
                            logger.error(f"OF Localization error: {e}")
                            loc_result = {"success": False, "error": str(e)}

                        # Оновлюємо стан так, щоб OF завжди рахувався ВІД КЛЮЧОВОГО КАДРУ,
                        # Це усуває проблему накопичення помилок (drift).
                        # Тому prev_gray_for_of та prev_pts_for_of не оновлюються тут!
                        
                        # На OF-кадрах: повторно emit останні відомі об'єкти для візуальної
                        # безперервності (YOLO не запускається, тому нових детекцій немає)
                        if object_tracker and last_tracked_objects:
                            self.objects_detected.emit(last_tracked_objects)
                    else:
                        prev_pts_for_of = None  # Втрата точок — наступний кадр стане ключовим
                else:
                    prev_pts_for_of = None

            if loc_result.get("success") and loc_result.get("matched_frame", -1) != -1:
                self.location_found.emit(
                    loc_result["lat"],
                    loc_result["lon"],
                    loc_result["confidence"],
                    loc_result["inliers"],
                )
                if loc_result.get("fov_polygon"):
                    self.fov_found.emit(loc_result["fov_polygon"])

                track_type = "OF" if loc_result.get("is_of") else "KF"
                method_txt = (
                    "Схожість" if loc_result.get("fallback_mode") == "retrieval_only" else "Inliers"
                )
                score = loc_result.get("global_score", loc_result["inliers"])

                self.status_update.emit(
                    f"[{track_type}] Знайдено ({method_txt}: {score:.2f}, Кадр: {loc_result['matched_frame']})"
                )

                last_localization_video_time = current_video_time_sec
            elif not loc_result.get("success") and loc_result.get("error") != "Not processed":
                self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

            process_duration = time.time() - start_process
            self.fps_updated.emit(1.0 / process_duration if process_duration > 0 else 0)

            frame_idx += 1

            # 3. Синхронізація відтворення (тільки для файлів)
            if not video_src.is_live:
                elapsed_in_loop = time.time() - loop_start
                sleep_time = frame_duration_sec - elapsed_in_loop
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))

        video_src.release()
        logger.info("Tracking worker thread finished cleanly.")

    def _prewarm_fallback_models(self):
        """Завантажує моделі заздалегідь, делегуючи у ModelManager."""
        try:
            if not self.model_manager:
                return
            logger.info("Tracking pre-warming centralized models...")
            self.model_manager.prewarm()
            logger.success("Tracking pre-warming successful")
        except Exception as e:
            logger.warning(
                f"Model pre-warming failed: {e}. "
                f"Models will be loaded on first use (slower first localization).",
                exc_info=True,
            )

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._stop_event.set()
        if not self.wait(5000):  # чекаємо максимум 5 секунд
            logger.warning("Tracking worker did not finish within 5 seconds.")
        else:
            logger.info("Tracking worker successfully stopped.")
