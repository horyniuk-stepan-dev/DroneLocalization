import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealtimeTrackingWorker(QThread):
    """Real-time localization worker thread"""

    frame_ready = pyqtSignal(QPixmap)
    location_found = pyqtSignal(float, float, float, int)
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    fov_found = pyqtSignal(list)

    def __init__(self, video_source: str, localizer, model_manager=None, config=None):
        """
        Args:
            video_source:  шлях до відеофайлу або індекс камери
            localizer:     ініціалізований Localizer
            model_manager: вже існуючий ModelManager (щоб не завантажувати YOLO вдруге).
                           Якщо None — YOLO-маскування буде вимкнено.
            config:        словник конфігурації
        """
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.model_manager = model_manager
        self.config = config or {}
        self._is_running = True
        self.target_fps = self.config.get('gui', {}).get('video_fps', 30)

        logger.info(f"RealtimeTrackingWorker initialized")
        logger.info(f"Video source: {video_source}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"YOLO masking: {'enabled' if model_manager else 'disabled (no model_manager)'}")

    def run(self):
        logger.info("RealtimeTrackingWorker thread started")

        # --- Ініціалізуємо YOLO через переданий model_manager ---
        # Не створюємо новий ModelManager — це завантажило б YOLO вдруге
        # і могло б вичерпати VRAM поруч із вже завантаженими моделями.
        from src.models.wrappers.yolo_wrapper import YOLOWrapper
        yolo_wrapper = None
        if self.model_manager is not None:
            try:
                yolo_model = self.model_manager.load_yolo()  # повертає кешований екземпляр
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
                logger.success("YOLO reused from model_manager for realtime tracking")
            except Exception as e:
                logger.warning(f"Failed to get YOLO from model_manager, running without masking: {e}")
        else:
            logger.warning("No model_manager provided — running without dynamic object masking")
        # ---------------------------------------------------------

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            error_msg = f"Failed to open video source: {self.video_source}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            return

        frame_time = 1.0 / self.target_fps
        frame_count = 0
        last_frame_time = time.time()

        while self._is_running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached")
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Отримуємо маску і рахуємо точний час dt для Калман-фільтра
                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                current_time = time.time()
                dt = current_time - last_frame_time
                last_frame_time = current_time

                # Передаємо маску і dt у локалізатор
                loc_result = self.localizer.localize_frame(frame_rgb, static_mask=static_mask, dt=dt)

                if loc_result.get("success"):
                    self.location_found.emit(
                        loc_result["lat"],
                        loc_result["lon"],
                        loc_result["confidence"],
                        loc_result["inliers"]
                    )

                    if "fov_polygon" in loc_result:
                        self.fov_found.emit(loc_result["fov_polygon"])

                    self.status_update.emit(
                        f"Знайдено (Inliers: {loc_result['inliers']}, "
                        f"Кадр: {loc_result['matched_frame']})"
                    )
                else:
                    self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                frame_contiguous = np.ascontiguousarray(frame_rgb)
                q_img = QImage(
                    frame_contiguous.data, w, h,
                    bytes_per_line, QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(q_img.copy())
                self.frame_ready.emit(pixmap)

            except Exception as e:
                error_msg = f"Frame {frame_count} processing error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.error.emit(error_msg)

            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            self.fps_updated.emit(current_fps)

            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, current FPS: {current_fps:.2f}")

            sleep_time = frame_time - process_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        logger.info(f"Tracking worker stopped after processing {frame_count} frames")

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._is_running = False