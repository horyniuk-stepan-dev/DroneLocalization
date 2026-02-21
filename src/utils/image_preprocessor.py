import cv2
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}

        # Налаштування алгоритмів
        self.apply_white_balance = self.config.get('preprocessing', {}).get('white_balance', True)
        self.apply_gamma = self.config.get('preprocessing', {}).get('auto_gamma', True)
        self.target_brightness = self.config.get('preprocessing', {}).get('target_brightness', 120.0)

        logger.info(
            f"ImagePreprocessor initialized: WhiteBalance={self.apply_white_balance}, AutoGamma={self.apply_gamma}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            logger.warning("Empty image provided for preprocessing")
            return image

        # Робимо копію, щоб не змінювати оригінальний масив у пам'яті напряму
        processed_img = image.copy()

        # ЕТАП 1: Автоматичний баланс білого (Gray World)
        if self.apply_white_balance:
            img_float = processed_img.astype(np.float32)
            avg_r = np.mean(img_float[:, :, 0])
            avg_g = np.mean(img_float[:, :, 1])
            avg_b = np.mean(img_float[:, :, 2])

            avg_gray = (avg_r + avg_g + avg_b) / 3.0

            # Захист від повністю чорних кадрів
            if avg_r > 1.0 and avg_g > 1.0 and avg_b > 1.0:
                img_float[:, :, 0] *= (avg_gray / avg_r)
                img_float[:, :, 1] *= (avg_gray / avg_g)
                img_float[:, :, 2] *= (avg_gray / avg_b)

                processed_img = np.clip(img_float, 0, 255).astype(np.uint8)

        # ЕТАП 2: Автоматична гамма-корекція (вирівнювання яскравості)
        if self.apply_gamma:
            # Знаходимо поточну яскравість вже виправленого по кольорах кадру
            gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)

            if mean_brightness > 1.0:
                # Математичний розрахунок ідеальної кривої
                gamma = np.log(self.target_brightness / 255.0) / np.log(mean_brightness / 255.0)
                gamma = np.clip(gamma, 0.4, 2.5)

                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

                # Швидке застосування таблиці підстановки
                processed_img = cv2.LUT(processed_img, table)

        return processed_img