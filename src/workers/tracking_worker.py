import cv2
import time
import threading
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

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

        # Скільки кадрів розпізнавати за одну секунду ВІДЕО.
        # 1.0 = 1 кадр в секунду; 2.0 = кожні 0.5 секунд; 0.5 = кожні 2 секунди відео.
        # Ти можеш змінити це число прямо тут для тестів:
        self.process_fps = self.config.get('tracking', {}).get('process_fps', 1.0)

    def run(self):
        logger.info(f"Starting tracking from source: {self.video_source}")

        yolo_wrapper = None
        if self.model_manager:
            try:
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
                logger.success("YOLO loaded for dynamic object masking in tracking loop")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                self.error.emit(f"YOLO load error: {e}")
                return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.video_source}")
            return

        # Визначаємо натуральну швидкість відео (зазвичай 30 FPS)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        # Інтервал обробки у секундах відео (наприклад, 1.0 / 2.0 = 0.5 секунд)
        process_interval_sec = 1.0 / self.process_fps if self.process_fps > 0 else 1.0

        # Ставимо від'ємний час, щоб гарантовано обробити найперший кадр
        last_process_video_time = -process_interval_sec
        last_localization_real_time = time.time()

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

            # 1. Завжди відправляємо кадр в GUI для плавного відтворення
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame_rgb)

            # 2. Локалізація (спрацьовує тільки якщо відео пройшло заданий інтервал)
            if current_video_time_sec - last_process_video_time >= process_interval_sec:
                start_process = time.time()

                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                # Розраховуємо реальний dt для фільтра Калмана
                current_real_time = time.time()
                calculated_dt = current_real_time - last_localization_real_time

                # БЛОК TRY-EXCEPT для запобігання "зависанню на першому кадрі"
                try:
                    loc_result = self.localizer.localize_frame(frame_rgb, static_mask=static_mask, dt=calculated_dt)
                except Exception as e:
                    logger.error(f"Localization exception: {e}", exc_info=True)
                    loc_result = {"success": False, "error": str(e)}

                if loc_result.get("success"):
                    self.location_found.emit(
                        loc_result["lat"], loc_result["lon"],
                        loc_result["confidence"], loc_result["inliers"]
                    )
                    if "fov_polygon" in loc_result and loc_result["fov_polygon"] is not None:
                        self.fov_found.emit(loc_result["fov_polygon"])

                    if loc_result.get("fallback_mode") == "retrieval_only":
                         self.status_update.emit(
                            f"Приблизно (Схожість: {loc_result.get('global_score', 0):.2f}, Кадр: {loc_result['matched_frame']})"
                        )
                    else:
                        self.status_update.emit(
                            f"Знайдено (Inliers: {loc_result['inliers']}, Кадр: {loc_result['matched_frame']})"
                        )
                else:
                    self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

                # Оновлюємо завжди — інакше dt накопичується і Kalman робить стрибок
                last_localization_real_time = current_real_time

                last_process_video_time = current_video_time_sec

                # Рахуємо швидкість самого алгоритму
                process_duration = time.time() - start_process
                self.fps_updated.emit(1.0 / process_duration if process_duration > 0 else 0)

            # 3. Синхронізація відтворення: щоб відео не "пролітало" за секунду,
            # змушуємо потік почекати, імітуючи реальну швидкість відео (1x)
            elapsed_in_loop = time.time() - loop_start
            sleep_time = frame_duration_sec - elapsed_in_loop
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

        cap.release()
        logger.info("Tracking worker thread finished cleanly.")

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._stop_event.set()
        self.wait(5000)  # чекаємо максимум 5 секунд