from collections import deque

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history"""

    def __init__(self, window_size=10, threshold_std=3.0, max_speed_mps=1000.0, max_consecutive=5):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self._consecutive_outliers = 0
        self._max_consecutive = max_consecutive

        logger.info("Initializing OutlierDetector")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}"
        )

    def add_position(self, position: tuple):
        self.window.append(np.array(position, dtype=np.float32))
        self._consecutive_outliers = 0  # Скидаємо лічильник — позиція прийнята

    def is_outlier(self, new_position: tuple, dt: float = 1.0) -> bool:
        if len(self.window) < 3:
            return False

        new_pos_np = np.array(new_position, dtype=np.float32)
        last_pos = self.window[-1]

        # 1. Перевірка максимально допустимої швидкості
        distance = float(np.linalg.norm(new_pos_np - last_pos))
        instantaneous_speed = distance / max(dt, 0.01)

        is_speed_outlier = instantaneous_speed > self.max_speed_mps

        # 2. Статистичний Z-score тест
        history = list(self.window)
        distances = [np.linalg.norm(history[i] - history[i - 1]) for i in range(1, len(history))]

        mean_dist = np.mean(distances)
        std_dist = max(np.std(distances), 1.0)

        z_score = abs(distance - mean_dist) / std_dist
        is_zscore_outlier = z_score > self.threshold_std and abs(distance - mean_dist) > 50.0

        if is_speed_outlier or is_zscore_outlier:
            self._consecutive_outliers += 1

            # Якщо забагато підряд — дрон реально перемістився, скидаємо вікно
            if self._consecutive_outliers >= self._max_consecutive:
                logger.warning(
                    f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                    f"accepting new position as legitimate movement. "
                    f"Position: ({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"last known: ({last_pos[0]:.1f}, {last_pos[1]:.1f}), "
                    f"distance={distance:.1f}m"
                )
                self.window.clear()
                self._consecutive_outliers = 0
                return False  # Приймаємо нову позицію

            if is_speed_outlier:
                logger.warning(
                    f"OUTLIER DETECTED (speed): {instantaneous_speed:.1f} m/s > {self.max_speed_mps} m/s | "
                    f"distance={distance:.1f}m, dt={dt:.3f}s, "
                    f"position=({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            else:
                logger.warning(
                    f"OUTLIER DETECTED (z-score): z={z_score:.2f} > {self.threshold_std} | "
                    f"distance={distance:.1f}m, mean_dist={mean_dist:.1f}m, std={std_dist:.1f}m, "
                    f"position=({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            return True

        self._consecutive_outliers = 0
        return False
