import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PanoramaWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, frame_step: int = 30):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.frame_step = frame_step
        self._is_running = True

    def run(self):
        logger.info(f"Starting panorama generation from: {self.video_path}")
        try:
            self.progress.emit(0, "Opening video...")

            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    raise ValueError(
                        f"Failed to open video file: {self.video_path}. "
                        f"Ensure file exists and has a supported codec (MP4/H.264)."
                    )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_stitch = []
            frame_count = 0

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_step == 0:
                    h, w = frame.shape[:2]
                    if w > 1920:
                        scale = 1920.0 / w
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    frames_to_stitch.append(frame)

                frame_count += 1

                if frame_count % 30 == 0:
                    prog = int((frame_count / total_frames) * 50)
                    self.progress.emit(prog, f"Collecting frames: {len(frames_to_stitch)}")

            cap.release()

            if not self._is_running:
                return

            self.progress.emit(50, "Stitching panorama (this may take a while)...")
            logger.info(f"Stitching {len(frames_to_stitch)} frames...")

            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status, panorama = stitcher.stitch(frames_to_stitch)

            if status == cv2.Stitcher_OK:
                cv2.imwrite(self.output_path, panorama)
                self.progress.emit(100, "Panorama saved successfully!")
                self.completed.emit(self.output_path)
            else:
                status_names = {
                    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS (insufficient overlapping frames)",
                    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL (failed to estimate homography)",
                    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL (camera parameter adjustment failed)",
                }
                status_name = status_names.get(status, f"UNKNOWN_CODE_{status}")
                raise ValueError(
                    f"Panorama stitching failed: {status_name}. "
                    f"Collected {len(frames_to_stitch)} frames with step={self.frame_step}. "
                    f"Try reducing frame step or ensuring sufficient visual overlap."
                )

        except Exception as e:
            logger.error(
                f"Panorama generation failed: {e} | "
                f"video={self.video_path}, output={self.output_path}, "
                f"frame_step={self.frame_step}",
                exc_info=True,
            )
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False
