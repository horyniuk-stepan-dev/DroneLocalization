import cv2
import numpy as np

from config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}
        # Initialise the CLAHE local contrast algorithm
        # clipLimit=3.0 gives strong shadow recovery; tileGridSize=(8,8) is the tile size
        clip = get_cfg(self.config, "preprocessing.clahe_clip_limit", 3.0)
        tile_cfg = get_cfg(self.config, "preprocessing.clahe_tile_grid", [8, 8])
        # Accept both list [8, 8] and scalar 8
        tile = tuple(tile_cfg) if isinstance(tile_cfg, list) else (tile_cfg, tile_cfg)
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        logger.info("ImagePreprocessor initialized with CLAHE (Local Contrast Enhancement)")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        # 1. Convert RGB to the LAB colour space to separate luminance from colour
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # 2. Apply CLAHE only to the luminance channel (L)
        l_clahe = self.clahe.apply(l_channel)

        # 3. Merge channels back and convert to RGB
        merged_lab = cv2.merge((l_clahe, a, b))
        enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

        return enhanced_rgb
