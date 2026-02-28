import cv2
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}
        # Ініціалізуємо алгоритм локального контрасту CLAHE
        # clipLimit=3.0 дає сильне витягування тіней, tileGridSize=(8,8) - розмір блоку
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        logger.info("ImagePreprocessor initialized with CLAHE (Local Contrast Enhancement)")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        # 1. Переводимо RGB в колірний простір LAB, щоб відділити яскравість від кольору
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # 2. Застосовуємо CLAHE виключно до каналу яскравості (L)
        l_clahe = self.clahe.apply(l_channel)

        # 3. Збираємо канали назад і повертаємо в RGB
        merged_lab = cv2.merge((l_clahe, a, b))
        enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

        return enhanced_rgb