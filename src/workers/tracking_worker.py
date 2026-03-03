import cv2
import time
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
        self._is_running = True

        # FPS для швидкості відображення відео в GUI (зазвичай 30)
        self.target_fps = self.config.get('gui', {}).get('video_fps', 1)

        # Скільки кадрів реально розпізнавати за секунду.
        # Завдяки XFeat можна сміливо ставити 10.0 замість 1.0!
        self.process_fps = self.config.get('tracking', {}).get('process_fps', 1.0)

    def run(self):
        logger.info(f"Starting real-time tracking from source: {self.video_source}")

        yolo_wrapper = None
        if self.model_manager:
            try:
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
                logger.success("YOLO loaded for dynamic object masking in tracking loop")
            except Exception as e:
                logger.error(f"Failed to load YOLO in tracking worker: {e}")
                self.error.emit(f"YOLO load error: {e}")
                return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.video_source}")
            return

        process_interval = 1.0 / self.process_fps if self.process_fps > 0 else 0
        gui_interval = 1.0 / self.target_fps

        last_process_time = 0
        last_gui_time = 0

        # Час останньої УСПІШНОЇ локалізації для точного фільтра Калмана
        last_localization_time = time.time()

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                self.status_update.emit("Відеопотік завершено.")
                break

            current_time = time.time()

            # 1. Оновлюємо картинку в графічному інтерфейсі (плавно)
            if current_time - last_gui_time >= gui_interval:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
                last_gui_time = current_time

            # 2. Блок важких обчислень: Локалізація
            if current_time - last_process_time >= process_interval:
                start_process = time.time()

                frame_rgb_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb_proc)

                # Розраховуємо реальний dt (скільки секунд пройшло з останніх відомих координат)
                calculated_dt = current_time - last_localization_time

                loc_result = self.localizer.localize_frame(frame_rgb_proc, static_mask=static_mask, dt=calculated_dt)

                if loc_result.get("success"):
                    self.location_found.emit(
                        loc_result["lat"], loc_result["lon"],
                        loc_result["confidence"], loc_result["inliers"]
                    )
                    if "fov_polygon" in loc_result:
                        self.fov_found.emit(loc_result["fov_polygon"])

                    self.status_update.emit(
                        f"Знайдено (Inliers: {loc_result['inliers']}, Кадр: {loc_result['matched_frame']})"
                    )
                    # Оновлюємо час успішної локалізації тільки якщо ми дійсно знайшли позицію
                    last_localization_time = current_time
                else:
                    self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

                last_process_time = current_time

                # Обчислення FPS самого алгоритму локалізації (YOLO + XFeat + DINO + Transform)
                process_duration = time.time() - start_process
                current_fps = 1.0 / process_duration if process_duration > 0 else 0
                self.fps_updated.emit(current_fps)

        cap.release()
        logger.info("Tracking worker thread finished cleanly.")

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._is_running = False
        self.wait()