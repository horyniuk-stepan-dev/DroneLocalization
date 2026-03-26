"""Тести для MaskingStrategy (Strategy Pattern для маскування динамічних об'єктів)."""

import numpy as np
import pytest

from src.models.wrappers.masking_strategy import (
    MaskingStrategy,
    NoMaskingStrategy,
    YOLOMaskingStrategy,
    create_masking_strategy,
)


class TestNoMaskingStrategy:
    """Тести для NoMaskingStrategy (заглушка без маскування)."""

    def test_returns_white_mask(self):
        """NoMaskingStrategy повинна повертати повністю білу маску."""
        strategy = NoMaskingStrategy()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = strategy.get_mask(frame)

        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        assert np.all(mask == 255)

    def test_batch_returns_white_masks(self):
        """get_mask_batch повинен повертати білі маски для всіх кадрів."""
        strategy = NoMaskingStrategy()
        frames = [np.random.randint(0, 255, (h, 640, 3), dtype=np.uint8) for h in [480, 720]]
        masks = strategy.get_mask_batch(frames)

        assert len(masks) == 2
        assert masks[0].shape == (480, 640)
        assert masks[1].shape == (720, 640)
        assert np.all(masks[0] == 255)
        assert np.all(masks[1] == 255)

    def test_empty_batch(self):
        """Порожній список повертає порожній результат."""
        strategy = NoMaskingStrategy()
        assert strategy.get_mask_batch([]) == []


class TestYOLOMaskingStrategy:
    """Тести для YOLOMaskingStrategy (делегація до YOLOWrapper)."""

    def test_delegates_to_wrapper(self):
        """YOLOMaskingStrategy має делегувати виклики до YOLOWrapper."""

        class MockYOLOWrapper:
            def detect_and_mask(self, image):
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
                mask[100:200, 100:200] = 0  # динамічний об'єкт
                return mask, [{"class_id": 0, "confidence": 0.9, "bbox": [100, 100, 200, 200]}]

            def detect_and_mask_batch(self, images):
                return [self.detect_and_mask(img) for img in images]

        strategy = YOLOMaskingStrategy(MockYOLOWrapper())
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        mask = strategy.get_mask(frame)
        assert mask.shape == (480, 640)
        # Перевіряємо що є динамічна зона (0)
        assert np.any(mask == 0)
        # Перевіряємо що є статичний фон (255)
        assert np.any(mask == 255)

    def test_batch_delegates_to_wrapper(self):
        """get_mask_batch має повертати тільки маски (без detections)."""

        class MockYOLOWrapper:
            def detect_and_mask_batch(self, images):
                results = []
                for img in images:
                    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                    results.append((mask, []))
                return results

        strategy = YOLOMaskingStrategy(MockYOLOWrapper())
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]

        masks = strategy.get_mask_batch(frames)
        assert len(masks) == 3
        for m in masks:
            assert isinstance(m, np.ndarray)
            assert m.shape == (480, 640)


class TestMaskingStrategyFactory:
    """Тести для фабричної функції create_masking_strategy."""

    def test_none_strategy(self):
        """Фабрика повертає NoMaskingStrategy для strategy='none'."""
        strategy = create_masking_strategy("none")
        assert isinstance(strategy, NoMaskingStrategy)

    def test_unknown_strategy_raises(self):
        """Фабрика має кидати ValueError для невідомої стратегії."""
        with pytest.raises(ValueError, match="Unknown masking strategy"):
            create_masking_strategy("unknown_strategy")

    def test_yolo_requires_model_manager(self):
        """YOLO стратегія вимагає model_manager."""
        with pytest.raises(ValueError, match="model_manager is required"):
            create_masking_strategy("yolo", model_manager=None)

    def test_is_abstract(self):
        """MaskingStrategy — абстрактний клас, не можна інстанціювати."""
        with pytest.raises(TypeError):
            MaskingStrategy()
