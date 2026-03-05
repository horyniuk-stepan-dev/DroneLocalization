import numpy as np
from collections import deque
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """
    Detects anomalous localization measurements.
    Includes freeze-recovery: auto-resets after too many consecutive rejections.
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold_std: float = 3.0,
        max_speed_mps: float = 80.0,
        max_consecutive_rejections: int = 5,
    ):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self.max_consecutive_rejections = max_consecutive_rejections

        self.window: deque[np.ndarray] = deque(maxlen=window_size)
        self._last_seen_pos: np.ndarray | None = None  # оновлюється ЗАВЖДИ
        self._consecutive_rejections: int = 0

        logger.info(
            f"OutlierDetector ready | window={window_size}, "
            f"threshold_std={threshold_std}, max_speed={max_speed_mps} m/s, "
            f"max_consecutive_rejections={max_consecutive_rejections}"
        )

    def _to_array(self, pos) -> np.ndarray:
        return np.asarray(pos, dtype=np.float64)

    def add_position(self, position) -> None:
        arr = self._to_array(position)
        self.window.append(arr)
        self._last_seen_pos = arr          # також оновлюємо last_seen
        self._consecutive_rejections = 0   # скидаємо лічильник при прийнятті
        logger.debug(f"Position accepted | window={len(self.window)}/{self.window_size}")

    def is_outlier(self, new_position, dt: float = 1.0) -> bool:
        new_pos = self._to_array(new_position)

        # Авто-відновлення після серії відхилень
        # (дрон реально перемістився, скидаємо вікно і приймаємо нову позицію)
        if self._consecutive_rejections >= self.max_consecutive_rejections:
            logger.warning(
                f"Auto-reset after {self._consecutive_rejections} consecutive rejections "
                f"— accepting new anchor position"
            )
            self.reset()
            # Одразу зберігаємо нову позицію як точку відліку
            self._last_seen_pos = new_pos
            return False

        if len(self.window) < 3:
            self._last_seen_pos = new_pos
            logger.debug("Insufficient history — accepting measurement")
            return False

        # Порівнюємо з _last_seen_pos (не window[-1]!)
        # Це не дає відстані рости при заморозці вікна
        ref_pos = self._last_seen_pos if self._last_seen_pos is not None else self.window[-1]
        distance = float(np.linalg.norm(new_pos - ref_pos))
        self._last_seen_pos = new_pos  # ← оновлюємо ЗАВЖДИ, навіть при відхиленні

        # Check 1: Physics — max plausible drone speed
        speed = distance / max(dt, 1e-3)
        if speed > self.max_speed_mps:
            self._consecutive_rejections += 1
            logger.warning(
                f"OUTLIER [{self._consecutive_rejections}/{self.max_consecutive_rejections}]: "
                f"speed {speed:.1f} m/s > {self.max_speed_mps} m/s"
            )
            return True

        # Check 2: Z-score vs historical step sizes
        hist = np.array(self.window)
        step_dists = np.linalg.norm(np.diff(hist, axis=0), axis=1)

        mean_d = float(np.mean(step_dists))
        std_d = float(np.std(step_dists))

        if std_d < 1e-3:
            logger.debug("Stationary history — skipping Z-score test")
            return False

        z_score = abs(distance - mean_d) / std_d
        if z_score > self.threshold_std:
            self._consecutive_rejections += 1
            logger.warning(
                f"OUTLIER [{self._consecutive_rejections}/{self.max_consecutive_rejections}]: "
                f"Z={z_score:.2f} > {self.threshold_std} "
                f"(dist={distance:.1f} m, mean={mean_d:.1f}, std={std_d:.1f})"
            )
            return True

        self._consecutive_rejections = 0
        logger.debug(f"OK | dist={distance:.1f} m, speed={speed:.1f} m/s, Z={z_score:.2f}")
        return False

    def reset(self) -> None:
        self.window.clear()
        self._last_seen_pos = None
        self._consecutive_rejections = 0
        logger.info("OutlierDetector reset")
