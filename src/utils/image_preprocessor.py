import cv2
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}

        # Цільова ідеальна яскравість (0 - повністю чорне, 255 - повністю біле)
        self.target_brightness = self.config.get('preprocessing', {}).get('target_brightness', 120.0)
        logger.info(f"ImagePreprocessor initialized with Auto-Gamma (target brightness={self.target_brightness})")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            logger.warning("Empty image provided for preprocessing")
            return image

        # Зчитуємо чорно-білу версію лише для швидкого розрахунку поточної яскравості
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)

        # Захист від ділення на нуль для абсолютно чорних кадрів
        if mean_brightness < 1.0:
            mean_brightness = 1.0

        # Математичний розрахунок необхідного показника гамми для досягнення цілі
        gamma = np.log(self.target_brightness / 255.0) / np.log(mean_brightness / 255.0)

        # Обмежуємо гамму безпечними межами, щоб не зробити кадр "вицвілим" або неприродним
        gamma = np.clip(gamma, 0.4, 2.5)

        # Створюємо швидку таблицю підстановки (LUT) для миттєвої обробки
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Застосовуємо розраховану гамму до оригінального кольорового зображення
        enhanced_image = cv2.LUT(image, table)

        return enhanced_image