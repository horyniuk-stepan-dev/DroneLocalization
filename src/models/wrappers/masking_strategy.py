"""Dynamic object masking strategy interface (Strategy Pattern).

Allows swapping masking implementations (YOLO, none) via config.
"""

from abc import ABC, abstractmethod

import numpy as np

from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MaskingStrategy(ABC):
    """Abstract interface for dynamic object masking strategies."""

    @abstractmethod
    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Returns binary mask: 255 = static background, 0 = dynamic object."""

    @abstractmethod
    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Batch processing of frames."""


class YOLOMaskingStrategy(MaskingStrategy):
    """Masking strategy using YOLO segmentation."""

    def __init__(self, yolo_wrapper):
        self._wrapper = yolo_wrapper
        logger.info("YOLOMaskingStrategy initialized")

    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        static_mask, _detections = self._wrapper.detect_and_mask(frame_rgb)
        return static_mask

    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        results = self._wrapper.detect_and_mask_batch(frames_rgb)
        return [static_mask for static_mask, _detections in results]


class NoMaskingStrategy(MaskingStrategy):
    """Fallback strategy without masking — returns fully static (white) mask."""

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
    """Factory for creating masking strategies."""
    if strategy_name == "yolo":
        if model_manager is None:
            raise ValueError("model_manager is required for YOLO masking strategy")

        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, device)
        return YOLOMaskingStrategy(yolo_wrapper)

    if strategy_name == "none":
        return NoMaskingStrategy()

    raise ValueError(f"Unknown masking strategy: '{strategy_name}'. Supported: 'yolo', 'none'")
