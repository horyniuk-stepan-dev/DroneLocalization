# src/models/wrappers/masking_strategy.py
#
# Поліморфний інтерфейс маскування динамічних об'єктів (Strategy Pattern).
# Дозволяє підміняти реалізацію (YOLO, EfficientViT-SAM, none) через конфіг.

from abc import ABC, abstractmethod

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MaskingStrategy(ABC):
    """Абстрактний інтерфейс для стратегій маскування динамічних об'єктів."""

    @abstractmethod
    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Повертає бінарну маску: 255 = статичний фон, 0 = динамічний об'єкт.

        Args:
            frame_rgb: RGB зображення (H, W, 3), uint8

        Returns:
            Бінарна маска (H, W), uint8: 255 = статика, 0 = динаміка
        """

    @abstractmethod
    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Батчева обробка кадрів.

        Args:
            frames_rgb: список RGB зображень

        Returns:
            Список бінарних масок (одна на кадр)
        """


class YOLOMaskingStrategy(MaskingStrategy):
    """Стратегія маскування через YOLO сегментацію.

    Делегує обробку існуючому YOLOWrapper, зберігаючи всю логіку
    micro-batching, over-masking та фільтрації за класами.
    """

    def __init__(self, yolo_wrapper):
        """
        Args:
            yolo_wrapper: екземпляр YOLOWrapper (вже ініціалізований)
        """
        self._wrapper = yolo_wrapper
        logger.info("YOLOMaskingStrategy initialized")

    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        static_mask, _detections = self._wrapper.detect_and_mask(frame_rgb)
        return static_mask

    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        results = self._wrapper.detect_and_mask_batch(frames_rgb)
        return [static_mask for static_mask, _detections in results]


class NoMaskingStrategy(MaskingStrategy):
    """Заглушка без маскування — повертає повністю білу маску.

    Використовується для тестів та режиму без YOLO.
    """

    def __init__(self):
        logger.info("NoMaskingStrategy initialized (masking disabled)")

    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        return [self.get_mask(f) for f in frames_rgb]


def create_masking_strategy(
    strategy_name: str,
    model_manager=None,
    device: str = "cuda",
) -> MaskingStrategy:
    """Фабрика стратегій маскування.

    Args:
        strategy_name: назва стратегії з конфігу ("yolo" | "none")
        model_manager: ModelManager для завантаження моделей
        device: пристрій для інференсу ("cuda" | "cpu")

    Returns:
        Екземпляр MaskingStrategy
    """
    if strategy_name == "yolo":
        if model_manager is None:
            raise ValueError("model_manager is required for YOLO masking strategy")
        from src.models.wrappers.yolo_wrapper import YOLOWrapper

        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, device)
        return YOLOMaskingStrategy(yolo_wrapper)

    if strategy_name == "none":
        return NoMaskingStrategy()

    raise ValueError(f"Unknown masking strategy: '{strategy_name}'. Supported: 'yolo', 'none'")
