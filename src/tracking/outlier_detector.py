import numpy as np
from collections import deque

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history"""

    def __init__(self, window_size=10, threshold_std=3.0, max_speed_mps=30.0):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps

        logger.info("Initializing OutlierDetector")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}")

    def add_position(self, position: tuple):
        self.window.append(np.array(position, dtype=np.float32))
        logger.debug(
            f"Added position to history: ({position[0]:.2f}, {position[1]:.2f}), window size: {len(self.window)}")

    def is_outlier(self, new_position: tuple, dt: float = 1.0) -> bool:
        if len(self.window) < 3:
            logger.debug("Insufficient history for outlier detection, accepting measurement")
            return False

        new_pos_np = np.array(new_position, dtype=np.float32)
        last_pos = self.window[-1]

        # Check 1: Maximum speed constraint
        distance = float(np.linalg.norm(new_pos_np - last_pos))
        instantaneous_speed = distance / dt

        if instantaneous_speed > self.max_speed_mps:
            logger.warning(
                f"OUTLIER DETECTED: Speed too high ({instantaneous_speed:.2f} m/s > {self.max_speed_mps} m/s)")
            logger.warning(f"Distance: {distance:.2f} m in {dt:.2f} s")
            return True

        # Check 2: Statistical Z-score test
        history = list(self.window)
        distances = [np.linalg.norm(history[i] - history[i - 1]) for i in range(1, len(history))]

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        if std_dist < 1e-3:
            std_dist = 1.0

        z_score = abs(distance - mean_dist) / std_dist

        if z_score > self.threshold_std:
            logger.warning(f"OUTLIER DETECTED: Z-score too high ({z_score:.2f} > {self.threshold_std})")
            logger.warning(f"Distance: {distance:.2f} m, mean: {mean_dist:.2f} m, std: {std_dist:.2f} m")
            return True

        logger.debug(
            f"Outlier check passed: distance={distance:.2f} m, speed={instantaneous_speed:.2f} m/s, z-score={z_score:.2f}")
        return False
