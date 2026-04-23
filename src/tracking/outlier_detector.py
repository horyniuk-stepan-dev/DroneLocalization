from collections import deque

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history using speeds"""

    def __init__(self, window_size=10, threshold_std=3.0, max_speed_mps=1000.0, max_consecutive=5):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self._consecutive_outliers = 0
        self._max_consecutive = max_consecutive

        logger.info("Initializing OutlierDetector (Speed-based Z-score)")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}"
        )

    def add_position(self, position: tuple, dt: float = 1.0, reset_consecutive: bool = True):
        # Тепер зберігаємо і позицію, і dt (час, за який ця позиція була досягнута)
        self.window.append((np.array(position, dtype=np.float64), max(dt, 0.01)))
        if reset_consecutive:
            self._consecutive_outliers = 0

    def is_outlier(self, new_position: tuple, dt: float = 1.0) -> bool:
        if len(self.window) < 3:
            return False

        new_pos_np = np.array(new_position, dtype=np.float64)
        last_pos, _ = self.window[-1]
        safe_dt = max(dt, 0.01)

        # 1. Перевірка максимально допустимої швидкості
        distance = float(np.linalg.norm(new_pos_np - last_pos))
        instantaneous_speed = distance / safe_dt

        is_speed_outlier = instantaneous_speed > self.max_speed_mps

        # 2. Статистичний Z-score тест (тепер за ШВИДКІСТЮ, а не за відстанню!)
        history = list(self.window)
        speeds = []
        for i in range(1, len(history)):
            p1, _ = history[i - 1]
            p2, hist_dt = history[i]
            dist = float(np.linalg.norm(p2 - p1))
            speeds.append(dist / hist_dt)

        mean_speed = np.mean(speeds)
        std_speed = max(np.std(speeds), 1.0)

        z_score = abs(instantaneous_speed - mean_speed) / std_speed

        # 15.0 m/s - мінімальна дельта швидкості, при якій Z-score має сенс
        is_zscore_outlier = (
            z_score > self.threshold_std and abs(instantaneous_speed - mean_speed) > 15.0
        )

        if is_speed_outlier or is_zscore_outlier:
            self._consecutive_outliers += 1

            # Якщо забагато підряд — дрон реально перемістився, скидаємо вікно
            if self._consecutive_outliers >= self._max_consecutive:
                logger.warning(
                    f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                    f"accepting new position. "
                    f"Position: ({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"speed={instantaneous_speed:.1f}m/s"
                )
                self.window.clear()
                self._consecutive_outliers = 0
                return False  # Приймаємо нову позицію

            if is_speed_outlier:
                logger.warning(
                    f"OUTLIER DETECTED (speed): {instantaneous_speed:.1f} m/s > {self.max_speed_mps} m/s | "
                    f"distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            else:
                logger.warning(
                    f"OUTLIER DETECTED (z-score): z={z_score:.2f} > {self.threshold_std} | "
                    f"speed={instantaneous_speed:.1f}m/s, mean_speed={mean_speed:.1f}m/s, std={std_speed:.1f}m/s, "
                    f"distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            return True

        self._consecutive_outliers = 0
        return False
