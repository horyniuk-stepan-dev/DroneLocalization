import cv2
import numpy as np
from src.utils.logging_utils import get_logger
import os

logger = get_logger(__name__)


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.apply_matching = self.config.get('preprocessing', {}).get('histogram_matching', True)

        reference_path = self.config.get('preprocessing', {}).get('reference_image_path', '')
        self.reference_image = None

        if self.apply_matching and os.path.exists(reference_path):
            ref_bgr = cv2.imread(reference_path)
            if ref_bgr is not None:
                self.reference_image = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                logger.info(f"ImagePreprocessor initialized with Histogram Matching. Reference: {reference_path}")
            else:
                logger.warning(f"Failed to load reference image at {reference_path}")
        else:
            logger.warning("Histogram Matching is enabled, but reference image path is invalid or missing.")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        if not self.apply_matching or self.reference_image is None:
            return image

        matched_image = np.empty_like(image)

        for d in range(image.shape[2]):
            s_val, bin_idx, s_counts = np.unique(image[:, :, d], return_inverse=True, return_counts=True)
            t_val, t_counts = np.unique(self.reference_image[:, :, d], return_counts=True)

            s_quantiles = np.cumsum(s_counts).astype(np.float64) / image[:, :, d].size
            t_quantiles = np.cumsum(t_counts).astype(np.float64) / self.reference_image[:, :, d].size

            interp_t_values = np.interp(s_quantiles, t_quantiles, t_val)
            matched_image[:, :, d] = interp_t_values[bin_idx].reshape(image.shape[:2])

        return matched_image.astype(np.uint8)