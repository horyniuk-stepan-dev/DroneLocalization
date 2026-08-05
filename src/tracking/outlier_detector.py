from collections import deque

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history using speeds"""

    def __init__(
        self,
        window_size=10,
        threshold_std=3.0,
        max_speed_mps=1000.0,
        max_consecutive=5,
        ground_scale=1.0,
        zscore_enabled=True,
        mahalanobis_enabled=False,
        chi2_threshold=13.816,
    ):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self._consecutive_outliers = 0
        self._max_consecutive = max_consecutive
        # Projection-to-ground-metres multiplier (ground_scale). 1.0 = legacy
        # Without this correction WebMercator distances at 48° latitude are
        # inflated by 1/cos(lat) ≈ 1.49×, so max_speed_mps=120 effectively gates
        # at 80.6 m/s and ~39% of rejections on a live run were clean false positives.
        self._ground_scale = float(ground_scale)
        # Z-score branch. Measured on mission top 2026-07-31 (of_stride=1,
        # of_local_speed=ON): of 70 rejections 47 came from this branch, and ALL
        # 23 rejections on CORRECT measurements (distance 2.3-4.5 m at the 3.0 m
        # nominal) — also from this branch. Not a single real jump (>12 m) was
        # caught here: those are all caught by the physical max_speed guard.
        # Root cause: a self-sustaining loop — the gate rejects everything above
        # mean_speed, so only slow measurements enter the window, mean drops
        # (11.9-78 m/s vs. the true 89.2 m/s) and even more gets rejected.
        # Also std hits the 1.0 floor and z reaches 159.
        self._zscore_enabled = bool(zscore_enabled)
        # Mahalanobis gate (flag, default off): uses the χ² distance from the KF
        # innovation covariance instead of the Z-score on speed. Default threshold
        # is χ²(2 dof, p=0.999) = 13.816, i.e. ~0.1% false rejections on a
        # correct model. Branch is independent from Z-score: both can be enabled,
        # but the point of switching is to disable Z-score.
        self._mahalanobis_enabled = bool(mahalanobis_enabled)
        self._chi2_threshold = float(chi2_threshold)

        logger.info("Initializing OutlierDetector (Speed-based Z-score)")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}"
        )

    def set_ground_scale(self, scale: float) -> None:
        """Updates the multiplier projection→ground meters (cos(lat) for WebMercator).

        Called from the localizer when a fresh latitude is available. Values <= 0
        are ignored — it is better to keep the previous one than zero out all speeds.
        """
        s = float(scale)
        if s > 0.0:
            self._ground_scale = s

    def reset(self) -> None:
        """Full state reset (new tracking session)."""
        self.window.clear()
        self._consecutive_outliers = 0

    def add_position(self, position: tuple, dt: float = 1.0, reset_consecutive: bool = True):
        # Now we store both position and dt (time it took to reach this position)
        self.window.append((np.array(position, dtype=np.float64), max(dt, 0.01)))
        if reset_consecutive:
            self._consecutive_outliers = 0

    def is_outlier(
        self,
        new_position: tuple,
        dt: float = 1.0,
        ref_position: tuple | None = None,
        maha_d2: float | None = None,
    ) -> bool:
        """Anomaly detection for measurements.

        ``ref_position`` — reference point for INSTANTANEOUS speed. By default
        it takes the last ACCEPTED position from the window, but on the OF-path
        this is incorrect: LK always tracks from a keyframe (tracking_worker.py:486),
        so the shift grows linearly from the keyframe, while dt remains a step of
        one OF-frame. Bases of the numerator and denominator differ -> speed is
        inflated exactly by N times, where N is the number of the OF-frame after
        the keyframe. Measured on mission top 2026-07-31: logged 450-1271 m/s fall
        onto a grid N * 89.2 m/s at N=5..14, and exactly 10 OF-frames fit between
        keyframe-s.

        Passing the previous RAW OF-position as ref_position makes the calculation
        local (frame relative to frame) and removes accumulation.
        """
        # The Mahalanobis branch does not need a speed window: the filter
        # covariance already encodes the full history. It is therefore checked
        # BEFORE the early exit on window length — otherwise the first frames
        # after a reset would have no gate except the physical speed limit.
        is_maha_outlier = (
            self._mahalanobis_enabled and maha_d2 is not None and maha_d2 > self._chi2_threshold
        )

        if len(self.window) < 3:
            if is_maha_outlier:
                self._consecutive_outliers += 1
                if self._consecutive_outliers >= self._max_consecutive:
                    logger.warning(
                        f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                        f"accepting new position (mahalanobis d2={maha_d2:.1f})"
                    )
                    self.window.clear()
                    self._consecutive_outliers = 0
                    return False
                logger.warning(
                    f"OUTLIER DETECTED (mahalanobis): d2={maha_d2:.2f} > "
                    f"{self._chi2_threshold:.2f} | consecutive="
                    f"{self._consecutive_outliers}/{self._max_consecutive}"
                )
                return True
            return False

        new_pos_np = np.array(new_position, dtype=np.float64)
        if ref_position is not None:
            last_pos = np.array(ref_position, dtype=np.float64)
        else:
            last_pos, _ = self.window[-1]
        safe_dt = max(dt, 0.01)

        # 1. Maximum-speed check
        # Distances are converted to TRUE ground metres before comparison with
        # the threshold; otherwise the threshold silently depends on mission latitude.
        distance = float(np.linalg.norm(new_pos_np - last_pos)) * self._ground_scale
        instantaneous_speed = distance / safe_dt

        is_speed_outlier = instantaneous_speed > self.max_speed_mps

        # 2. Statistical Z-score test (now based on SPEED, not distance)
        history = list(self.window)
        speeds = []
        for i in range(1, len(history)):
            p1, _ = history[i - 1]
            p2, hist_dt = history[i]
            dist = float(np.linalg.norm(p2 - p1)) * self._ground_scale
            speeds.append(dist / hist_dt)

        mean_speed = np.mean(speeds)
        std_speed = max(np.std(speeds), 1.0)

        z_score = abs(instantaneous_speed - mean_speed) / std_speed

        # 15.0 m/s - minimum speed delta for which Z-score is meaningful
        is_zscore_outlier = self._zscore_enabled and (
            z_score > self.threshold_std and abs(instantaneous_speed - mean_speed) > 15.0
        )

        if is_speed_outlier or is_zscore_outlier or is_maha_outlier:
            self._consecutive_outliers += 1

            # Too many consecutive outliers — drone actually moved, reset window
            if self._consecutive_outliers >= self._max_consecutive:
                logger.warning(
                    f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                    f"accepting new position. "
                    f"Position: ({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"speed={instantaneous_speed:.1f}m/s"
                )
                self.window.clear()
                self._consecutive_outliers = 0
                return False  # Accept the new position

            if is_maha_outlier and not is_speed_outlier:
                logger.warning(
                    f"OUTLIER DETECTED (mahalanobis): d2={maha_d2:.2f} > {self._chi2_threshold:.2f} | "
                    f"speed={instantaneous_speed:.1f}m/s, distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            elif is_speed_outlier:
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
