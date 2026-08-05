"""
Normalises the input frame resolution to the database reference resolution.

If ref_width/ref_height = 0 — normalisation is disabled (backward compatibility).
Scales proportionally, preserving aspect ratio.
"""

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResolutionNormalizer:
    """Scales the input frame to the database reference resolution."""

    def __init__(self, ref_width: int = 0, ref_height: int = 0):
        self.ref_width = ref_width
        self.ref_height = ref_height
        self._logged_once = False

    @property
    def is_enabled(self) -> bool:
        return self.ref_width > 0 and self.ref_height > 0

    def normalize(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (normalized_frame, scale_factor).

        scale_factor — the scaling ratio (query → ref), required to back-project
        coordinates into the original frame.
        Returns (frame, 1.0) if normalisation is disabled or sizes already match.
        """
        if not self.is_enabled:
            return frame, 1.0

        h, w = frame.shape[:2]
        if w == self.ref_width and h == self.ref_height:
            return frame, 1.0

        scale_x = self.ref_width / w
        scale_y = self.ref_height / h
        # Uniform scaling (preserves aspect ratio)
        scale = min(scale_x, scale_y)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if not self._logged_once:
            logger.info(
                f"ResolutionNormalizer: {w}x{h} -> {new_w}x{new_h} "
                f"(scale={scale:.4f}, ref={self.ref_width}x{self.ref_height})"
            )
            self._logged_once = True

        # CUBIC instead of LANCZOS4 for upscale — much faster; the difference
        # is imperceptible to feature extractors.
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        return resized, scale

    def normalize_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """Scales a YOLO mask in sync with the frame."""
        if not self.is_enabled or mask is None:
            return mask

        h, w = mask.shape[:2]
        if w == self.ref_width and h == self.ref_height:
            return mask

        scale = min(self.ref_width / w, self.ref_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
