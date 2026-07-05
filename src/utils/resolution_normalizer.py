"""
Нормалізація роздільної здатності вхідного кадру до еталонної роздільної здатності бази даних.

Якщо ref_width/ref_height = 0 — нормалізація вимкнена (зворотна сумісність).
Масштабує пропорційно, зберігаючи aspect ratio.
"""

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResolutionNormalizer:
    """Масштабує вхідний кадр до еталонної роздільної здатності бази даних."""

    def __init__(self, ref_width: int = 0, ref_height: int = 0):
        self.ref_width = ref_width
        self.ref_height = ref_height
        self._logged_once = False

    @property
    def is_enabled(self) -> bool:
        return self.ref_width > 0 and self.ref_height > 0

    def normalize(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Повертає (normalized_frame, scale_factor).

        scale_factor — коефіцієнт масштабування (query → ref), потрібен для
        зворотного перерахунку координат.
        Якщо нормалізація вимкнена або розміри збігаються, повертає (frame, 1.0).
        """
        if not self.is_enabled:
            return frame, 1.0

        h, w = frame.shape[:2]
        if w == self.ref_width and h == self.ref_height:
            return frame, 1.0

        scale_x = self.ref_width / w
        scale_y = self.ref_height / h
        # Однорідне масштабування (зберігаємо aspect ratio)
        scale = min(scale_x, scale_y)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if not self._logged_once:
            logger.info(
                f"ResolutionNormalizer: {w}x{h} -> {new_w}x{new_h} "
                f"(scale={scale:.4f}, ref={self.ref_width}x{self.ref_height})"
            )
            self._logged_once = True

        # A9: CUBIC замість LANCZOS4 для upscale — у рази швидше, різниця
        # для фіч-екстракторів невідчутна
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        return resized, scale

    def normalize_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """Масштабує YOLO-маску синхронно з кадром."""
        if not self.is_enabled or mask is None:
            return mask

        h, w = mask.shape[:2]
        if w == self.ref_width and h == self.ref_height:
            return mask

        scale = min(self.ref_width / w, self.ref_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
