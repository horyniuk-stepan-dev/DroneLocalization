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

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(
                f"Failed to open video source: {self.video_source}. "
                f"Check that the file exists and is a valid video format (MP4/H.264 recommended)."
            )
            self.error.emit(f"Не вдалося відкрити відео: {self.video_source}")
            return

        # Визначаємо натуральну швидкість відео (зазвичай 30 FPS)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        # Замість time-based інтервалу використовуємо frame-based:
        frame_idx = 0
        prev_gray_for_of = None
        prev_pts_for_of = None

        # Зберігаємо останній час локалізації саме за ВІДЕО-часом, а не за процесорним
        last_localization_video_time = -1.0

        while not self._stop_event.is_set():
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                self.status_update.emit("Відеопотік завершено.")
                break

            # Отримуємо поточний час САМОГО ВІДЕО у секундах (не залежить від швидкості комп'ютера)
            current_video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # Fallback: деякі кодеки повертають 0 — рахуємо за номером кадру
            if current_video_time_sec <= 0:
                current_video_time_sec = cap.get(cv2.CAP_PROP_POS_FRAMES) * frame_duration_sec

            # 1. Завжди відправляємо кадр в GUI для плавного відтворення (сирий BGR)
            self.frame_ready.emit(frame)

            # S3-3: Optical Flow Pipeline
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = frame_idx % self.keyframe_interval == 0

            # Розрахунок dt
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
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                try:
                    loc_result = self.localizer.localize_frame(
                        frame_rgb, static_mask=static_mask, dt=calculated_dt
                    )
                except Exception as e:
                    logger.error(f"Localization exception on keyframe: {e}", exc_info=True)
                    loc_result = {"success": False, "error": str(e)}

                if loc_result.get("success"):
                    # Зберігаємо стан для OF на наступні кадри
                    prev_gray_for_of = curr_gray
                    # Трекаємо гарні точки (corners) для стабільного OF
                    prev_pts_for_of = cv2.goodFeaturesToTrack(
                        curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, mask=None
                    )
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

            # 3. Синхронізація відтворення: щоб відео не "пролітало" за секунду,
            # змушуємо потік почекати, імітуючи реальну швидкість відео (1x)
            elapsed_in_loop = time.time() - loop_start
            sleep_time = frame_duration_sec - elapsed_in_loop
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

        cap.release()
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
