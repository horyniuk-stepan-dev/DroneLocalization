import cv2
import time
import numpy as np
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealtimeTrackingWorker(QThread):
    """Real-time localization worker thread."""

    frame_ready     = pyqtSignal(np.ndarray)
    location_found  = pyqtSignal(float, float, float, int)
    fps_updated     = pyqtSignal(float)
    error           = pyqtSignal(str)
    status_update   = pyqtSignal(str)
    fov_found       = pyqtSignal(list)

    def __init__(self, video_source: str, localizer, model_manager=None, config=None):
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.model_manager = model_manager
        self.config = config or {}
        self._is_running = False

        self.target_fps  = self.config.get('gui', {}).get('video_fps', 30)
        self.process_fps = self.config.get('tracking', {}).get('process_fps', 1.0)

        self._fps_buffer: deque[float] = deque(maxlen=10)

    def run(self):
        self._is_running = True

        # Reset stateful components for new session
        if hasattr(self.localizer, 'kalman_filter'):
            self.localizer.kalman_filter.reset()
        if hasattr(self.localizer, 'outlier_detector'):
            self.localizer.outlier_detector.reset()

        # Load YOLO (cached by ModelManager)
        yolo_wrapper = None
        if self.model_manager is not None:
            try:
                from src.models.wrappers.yolo_wrapper import YOLOWrapper
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
            except Exception as e:
                logger.warning(f"YOLO unavailable: {e}")

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.video_source}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 30.0

        frame_step    = max(1, int(video_fps / self.process_fps))
        calculated_dt = frame_step / video_fps
        frame_time    = 1.0 / self.target_fps

        logger.info(
            f"Tracking started | video_fps={video_fps:.1f}, "
            f"process_fps={self.process_fps}, frame_step={frame_step}, "
            f"kalman_dt={calculated_dt:.3f}s"
        )

        try:
            while self._is_running:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    self.status_update.emit("Відео завершено")
                    break

                # Skip frames without decoding
                for _ in range(frame_step - 1):
                    cap.grab()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)

                try:
                    static_mask = None
                    if yolo_wrapper:
                        static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                    loc_result = self.localizer.localize_frame(
                        frame_rgb, static_mask=static_mask, dt=calculated_dt
                    )

                    if loc_result.get("success"):
                        self.location_found.emit(
                            loc_result["lat"], loc_result["lon"],
                            loc_result["confidence"], loc_result["inliers"],
                        )
                        if loc_result.get("fov_polygon"):
                            self.fov_found.emit(loc_result["fov_polygon"])
                        self.status_update.emit(
                            f"Знайдено (Inliers: {loc_result['inliers']}, "
                            f"Кадр: {loc_result['matched_frame']})"
                        )
                    else:
                        self.status_update.emit(
                            f"Втрата: {loc_result.get('error', 'Невідома помилка')}"
                        )

                except Exception as e:
                    logger.error(f"Localization error: {e}", exc_info=True)
                    self.error.emit(str(e))

                # FPS measured before sleep, after all processing
                process_time = time.time() - start_time
                if process_time > 0:
                    self._fps_buffer.append(1.0 / process_time)
                if len(self._fps_buffer) == self._fps_buffer.maxlen:
                    self.fps_updated.emit(sum(self._fps_buffer) / len(self._fps_buffer))

                sleep_time = frame_time - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()  # ← єдине місце звільнення
            logger.info("VideoCapture released")

    def stop(self):
        self._is_running = False
        self.wait()  # дочекатись завершення потоку перед знищенням об'єктів
