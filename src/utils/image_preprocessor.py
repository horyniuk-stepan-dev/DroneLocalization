import cv2
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}

        self.clip_limit = self.config.get('preprocessing', {}).get('clahe_clip_limit', 2.0)
        self.tile_grid_size = self.config.get('preprocessing', {}).get('clahe_tile_size', (8, 8))

        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        logger.info(f"ImagePreprocessor initialized with CLAHE clipLimit={self.clip_limit}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            logger.warning("Empty image provided for preprocessing")
            return image

        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        cl = self.clahe.apply(l_channel)

        merged_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

        return enhanced_image