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
        self.target_fps = self.config.get('gui', {}).get('video_fps', 30)

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

        frame_time = 1.0 / self.target_fps
        last_frame_time = time.time()

        while self._is_running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                current_time = time.time()
                dt = current_time - last_frame_time
                last_frame_time = current_time

                loc_result = self.localizer.localize_frame(frame_rgb, static_mask=static_mask, dt=dt)

                if loc_result.get("success"):
                    self.location_found.emit(
                        loc_result["lat"], loc_result["lon"],
                        loc_result["confidence"], loc_result["inliers"]
                    )
                    if "fov_polygon" in loc_result:
                        self.fov_found.emit(loc_result["fov_polygon"])
                    self.status_update.emit(f"Знайдено (Inliers: {loc_result['inliers']}, Кадр: {loc_result['matched_frame']})")
                else:
                    self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

                # Передаємо сирий масив замість QPixmap
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