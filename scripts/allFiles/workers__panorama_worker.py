import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MAX_STITCH_FRAMES = 200  # cv2.Stitcher degrades beyond ~300; RAM guard

_STITCH_ERRORS = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS:
        "Недостатньо кадрів для зшивання — зменшіть крок кадрів",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        "Не вдалося знайти спільні точки між кадрами — збільшіть крок або використайте інше відео",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        "Помилка калібрування камери — спробуйте інший набір кадрів",
}


class PanoramaWorker(QThread):
    progress  = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    error     = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, frame_step: int = 30):
        super().__init__()
        self.video_path  = video_path
        self.output_path = output_path
        self.frame_step  = frame_step
        self._is_running = False

    def run(self):
        self._is_running = True
        logger.info(f"Panorama generation: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit("Не вдалося відкрити відеофайл")
            return

        try:
            self.progress.emit(0, "Відкриття відео...")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = None  # unknown length (streaming or unsupported codec)

            frames_to_stitch = []
            frame_count = 0

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_step == 0:
                    # Resize is mandatory — without it, 1000 frames × 6MB = 6GB RAM
                    small = cv2.resize(frame, (1280, 720))
                    frames_to_stitch.append(small)

                frame_count += 1

                if frame_count % 30 == 0 and total_frames:
                    prog = int((frame_count / total_frames) * 50)
                    self.progress.emit(prog, f"Збирання кадрів: {len(frames_to_stitch)} шт.")


        finally:
            cap.release()

        if not self._is_running:
            return

        if len(frames_to_stitch) < 2:
            self.error.emit("Недостатньо кадрів для зшивання (мінімум 2)")
            return

        logger.info(f"Stitching {len(frames_to_stitch)} frames...")
        self.progress.emit(50, "Зшивання панорами... (може зайняти 1–2 хвилини)")

        try:
            # PANORAMA mode: supports rotation + scale — correct for drone footage
            # SCANS mode is only for flat document scans without rotation
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, panorama = stitcher.stitch(frames_to_stitch)
        except cv2.error as e:
            logger.error(f"OpenCV stitcher crashed: {e}")
            self.error.emit(f"OpenCV помилка зшивання: {e}")
            return

        if status != cv2.Stitcher_OK:
            msg = _STITCH_ERRORS.get(status, f"Невідома помилка зшивання (код {status})")
            logger.error(f"Stitcher failed: {msg}")
            self.error.emit(msg)
            return

        self.progress.emit(90, "Збереження панорами...")
        if not cv2.imwrite(self.output_path, panorama):
            self.error.emit(f"Не вдалося зберегти файл: {self.output_path}")
            return

        logger.success(f"Panorama saved: {self.output_path}")
        self.progress.emit(100, "Панораму збережено!")
        self.completed.emit(self.output_path)

    def stop(self):
        self._is_running = False
        self.wait()
