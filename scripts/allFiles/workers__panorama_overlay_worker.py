import cv2
import json
import numpy as np
import base64
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MAX_DISPLAY_WIDTH = 4000  # px — UI display cap (localization uses full resolution)
_JPEG_QUALITY = 80


class PanoramaOverlayWorker(QThread):
    """
    Background thread: localizes panorama image and prepares it for map overlay.
    Emits base64-encoded JPEG + GPS corner coordinates on success.
    """

    # data_url (base64 JPEG), fov_json ([[lat,lon], ...] as JSON string)
    success = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, localizer):
        super().__init__()
        self.image_path = image_path
        self.localizer = localizer
        self._is_running = False

    def run(self):
        self._is_running = True
        logger.info(f"Panorama overlay: {self.image_path}")

        try:
            # БЕЗПЕЧНЕ ЧИТАННЯ: обходить баг OpenCV з кирилицею у шляхах на Windows
            img_array = np.fromfile(self.image_path, dtype=np.uint8)
            if img_array.size == 0:
                raise ValueError("Файл порожній або не знайдений")

            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Не вдалося розкодувати зображення панорами")

            # Localize at full resolution for accuracy
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loc_result = self.localizer.localize_frame(img_rgb)
            del img_rgb  # free ~23MB for 4K image immediately after use

            if not self._is_running:
                return

            if not loc_result.get("success"):
                raise RuntimeError(loc_result.get("error", "Не вдалося локалізувати панораму"))

            fov = loc_result.get("fov_polygon")
            if not fov or len(fov) != 4:
                raise RuntimeError("Локалізатор не повернув коректні кути (FOV) для панорами")

            # Validate and normalize corner coordinates to plain Python floats
            try:
                fov_normalized = [[float(pt[0]), float(pt[1])] for pt in fov]
            except (TypeError, IndexError) as e:
                raise RuntimeError(f"Некоректний формат fov_polygon: {e}")

            # Downscale only for UI display — localization already done
            h, w = img.shape[:2]
            if w > _MAX_DISPLAY_WIDTH:
                scale = _MAX_DISPLAY_WIDTH / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)

            encode_ok, buffer = cv2.imencode(
                '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
            )
            if not encode_ok:
                raise RuntimeError("Не вдалося закодувати зображення у JPEG")

            data_url = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            self.success.emit(data_url, json.dumps(fov_normalized))
            logger.success(f"Panorama overlay ready | display={img.shape[1]}×{img.shape[0]}")

        except Exception as e:
            logger.error(f"PanoramaOverlayWorker failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False
        self.wait()