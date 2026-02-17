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
            self.progress.emit(0, "Відкриття відео...")

            # Використовуємо FFmpeg для надійності
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    raise ValueError("Не вдалося відкрити відеофайл")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_stitch = []
            frame_count = 0

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_step == 0:
                    # Зменшуємо кадр для швидкості та економії пам'яті (опціонально)
                    # frame = cv2.resize(frame, (1280, 720))
                    frames_to_stitch.append(frame)

                frame_count += 1

                if frame_count % 30 == 0:
                    prog = int((frame_count / total_frames) * 50)  # Перші 50% прогресу - зчитування
                    self.progress.emit(prog, f"Збирання кадрів: {len(frames_to_stitch)} шт.")

            cap.release()

            if not self._is_running:
                return

            self.progress.emit(50, "Зшивання панорами (це може зайняти час)...")
            logger.info(f"Stitching {len(frames_to_stitch)} frames...")

            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status, panorama = stitcher.stitch(frames_to_stitch)

            if status == cv2.Stitcher_OK:
                cv2.imwrite(self.output_path, panorama)
                self.progress.emit(100, "Панораму збережено!")
                self.completed.emit(self.output_path)
            else:
                raise ValueError(f"Помилка зшивання (Код OpenCV: {status}). Спробуйте змінити крок кадрів.")

        except Exception as e:
            logger.error(f"Panorama generation failed: {e}")
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False