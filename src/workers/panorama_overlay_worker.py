import cv2
import numpy as np
import base64
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PanoramaOverlayWorker(QThread):
    """Фоновий потік для локалізації та підготовки панорами до відображення на карті"""

    success = pyqtSignal(str, float, float, float, float, float, float, float, float)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, localizer):
        super().__init__()
        self.image_path = image_path
        self.localizer = localizer

    def run(self):
        try:
            logger.info(f"Starting background panorama overlay for {self.image_path}")
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("Не вдалося прочитати файл зображення панорами")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loc_result = self.localizer.localize_frame(img_rgb)

            if not loc_result.get("success"):
                raise RuntimeError(loc_result.get("error", "Не вдалося локалізувати панораму"))

            fov = loc_result.get("fov_polygon")
            if not fov or len(fov) != 4:
                raise RuntimeError("Локалізатор не повернув коректні кути (FOV) для панорами")

            h, w = img.shape[:2]
            scale = 1.0
            if w > 4000:
                scale = 4000.0 / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_string = base64.b64encode(buffer).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{b64_string}"

            self.success.emit(
                data_url,
                fov[0][0], fov[0][1],
                fov[1][0], fov[1][1],
                fov[2][0], fov[2][1],
                fov[3][0], fov[3][1]
            )

            logger.success("Panorama successfully processed in background")

        except Exception as e:
            logger.error(f"Panorama overlay worker failed: {e}")
            self.error.emit(str(e))