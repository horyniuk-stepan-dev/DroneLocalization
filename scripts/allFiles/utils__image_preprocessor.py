import cv2
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Preprocesses drone frames before feature extraction.

    Pipeline: CLAHE (local contrast) → optional histogram matching to reference style.
    All processing is in RGB colorspace.
    """

    def __init__(self, config=None):
        self.config = config or {}
        prep_cfg = self.config.get('preprocessing', {})

        self.clip_limit: float = prep_cfg.get('clahe_clip_limit', 2.0)
        self.tile_grid: tuple = tuple(prep_cfg.get('clahe_tile_grid', [8, 8]))

        self.use_histogram_matching: bool = prep_cfg.get('histogram_matching', False)
        self._reference_cdf: list[np.ndarray] | None = None

        ref_path = prep_cfg.get('reference_image_path')
        if self.use_histogram_matching and ref_path:
            self._load_reference(ref_path)

        logger.info(
            f"ImagePreprocessor ready | CLAHE clip={self.clip_limit} grid={self.tile_grid}"
            f" | histogram_matching={self.use_histogram_matching}"
        )

    def _load_reference(self, path: str) -> None:
        ref = cv2.imread(path)
        if ref is None:
            logger.warning(f"Reference image not found: {path} — histogram matching disabled")
            self.use_histogram_matching = False
            return
        ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        self._reference_cdf = [self._compute_cdf(ref_rgb[:, :, c]) for c in range(3)]
        logger.success(f"Reference image loaded for histogram matching: {path}")

    @staticmethod
    def _compute_cdf(channel: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum().astype(np.float32)
        return cdf / (cdf[-1] + 1e-8)

    def _histogram_match(self, image: np.ndarray) -> np.ndarray:
        """Match image histogram to reference per channel."""
        if self._reference_cdf is None:
            return image
        result = image.copy()
        for c in range(3):
            src_cdf = self._compute_cdf(image[:, :, c])
            # Map src values to reference via inverse CDF lookup
            mapping = np.interp(src_cdf, self._reference_cdf[c], np.arange(256))
            result[:, :, c] = mapping[image[:, :, c]]
        return result.astype(np.uint8)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        # Ensure uint8 — CLAHE requires it
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # CLAHE on L channel only (preserves hue/saturation)
        # Create per-call to avoid thread-safety issues with shared cv2.CLAHE object
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

        # Optional histogram matching to reference style (weather normalization)
        if self.use_histogram_matching and self._reference_cdf is not None:
            enhanced = self._histogram_match(enhanced)

        return enhanced
