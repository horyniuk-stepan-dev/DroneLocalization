import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealtimeTrackingWorker(QThread):
    """Real-time localization worker thread"""

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
        self._is_running = True

        # FPS для швидкості оновлення графічного інтерфейсу
        self.target_fps = self.config.get('gui', {}).get('video_fps', 30)

        # НОВИЙ ПАРАМЕТР: Скільки кадрів реально розпізнавати за секунду відео (за замовчуванням 1.0)
        self.process_fps = self.config.get('tracking', {}).get('process_fps', 1.0)

    def run(self):
        from src.models.wrappers.yolo_wrapper import YOLOWrapper
        yolo_wrapper = None
        if self.model_manager is not None:
            try:
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
            except Exception as e:
                logger.warning(f"Failed to get YOLO from model_manager: {e}")

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.video_source}")
            return

        # Визначаємо оригінальний FPS відео
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0

        # Математика пропусків та dt на основі параметра process_fps
        frame_step = max(1, int(video_fps / self.process_fps))
        frames_to_skip = frame_step - 1

        # Точний проміжок часу між обробленими кадрами для фільтра Калмана
        calculated_dt = float(frame_step) / video_fps

        logger.info(f"Video original FPS: {video_fps}, Target Process FPS: {self.process_fps}")
        logger.info(f"Skipping {frames_to_skip} frames per cycle. Kalman dt will be {calculated_dt:.3f}s")

        frame_time = 1.0 / self.target_fps

        while self._is_running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Швидко прокручуємо непотрібні кадри без декодування
            for _ in range(frames_to_skip):
                cap.grab()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                # Використовуємо стабільне розраховане dt
                loc_result = self.localizer.localize_frame(frame_rgb, static_mask=static_mask, dt=calculated_dt)

                if loc_result.get("success"):
                    self.location_found.emit(
                        loc_result["lat"], loc_result["lon"],
                        loc_result["confidence"], loc_result["inliers"]
                    )
                    if "fov_polygon" in loc_result:
                        self.fov_found.emit(loc_result["fov_polygon"])
                    self.status_update.emit(
                        f"Знайдено (Inliers: {loc_result['inliers']}, Кадр: {loc_result['matched_frame']})")
                else:
                    self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

                self.frame_ready.emit(frame_rgb)

            except Exception as e:
                self.error.emit(f"Processing error: {str(e)}")

            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            self.fps_updated.emit(current_fps)

            sleep_time = frame_time - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def stop(self):
        self._is_running = False