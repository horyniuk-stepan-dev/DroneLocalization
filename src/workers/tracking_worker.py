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

    def __init__(self, video_source: str, localizer, config=None):
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.config = config or {}
        self._is_running = True
        self.target_fps = self.config.get('gui', {}).get('video_fps', 30)

        logger.info(f"RealtimeTrackingWorker initialized")
        logger.info(f"Video source: {video_source}")
        logger.info(f"Target FPS: {self.target_fps}")

    def run(self):
        logger.info("RealtimeTrackingWorker thread started")

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            error_msg = f"Failed to open video source: {self.video_source}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video opened: {video_fps:.2f} FPS, {total_frames} frames")

        self.status_update.emit("Відстеження розпочато...")
        frame_time = 1.0 / self.target_fps
        frame_count = 0

        logger.info("Starting main tracking loop...")

        while self._is_running:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached")
                self.status_update.emit("Відеопотік завершився.")
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Perform localization
                logger.debug(f"Processing frame {frame_count}...")
                result = self.localizer.localize_frame(frame_rgb)

                if result.get("success"):
                    logger.debug(
                        f"Frame {frame_count}: Localization successful - "
                        f"({result['lat']:.6f}, {result['lon']:.6f})"
                    )
                    self.location_found.emit(
                        result["lat"],
                        result["lon"],
                        result["confidence"],
                        result.get("inliers", 0)
                    )
                    if "fov_polygon" in result and len(result["fov_polygon"]) == 4:
                        self.fov_found.emit(result["fov_polygon"])
                else:
                    logger.debug(
                        f"Frame {frame_count}: Localization failed - "
                        f"{result.get('error', 'Unknown error')}"
                    )

                # ВИПРАВЛЕНО: frame_rgb має бути неперервним у пам'яті,
                # а QImage.copy() гарантує що QPixmap не залежить від
                # буфера numpy, який може бути перезаписаний на наступній
                # ітерації циклу → UB / артефакти зображення
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

            # FPS control
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