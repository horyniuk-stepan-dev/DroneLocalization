import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrajectoryFilter:
    """Kalman filter for GPS trajectory smoothing optimized for high speeds"""

    def __init__(self, process_noise=2.0, measurement_noise=5.0, dt=1.0):
        # Filter state: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Збільшений шум процесу та зменшений шум вимірювання
        # дозволяють фільтру швидше реагувати на зміни курсу на високих швидкостях
        self.process_noise = process_noise
        self.is_initialized = False

        logger.info("Initializing Kalman filter for high-speed trajectory smoothing")
        logger.info(f"Parameters: process_noise={process_noise}, measurement_noise={measurement_noise}, dt={dt}")

        self.kf.P *= 1000.0

        self.kf.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        self.kf.R = np.array([
            [measurement_noise, 0.0],
            [0.0, measurement_noise]
        ])

        self._update_matrices_for_dt(dt)

    def _update_matrices_for_dt(self, dt: float):
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        q_var = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        self.kf.Q = np.zeros((4, 4))
        self.kf.Q[0:2, 0:2] = q_var
        self.kf.Q[2:4, 2:4] = q_var

    def update(self, measurement: tuple, dt: float = 1.0) -> tuple:
        z = np.array([[measurement[0]], [measurement[1]]])

        if not self.is_initialized:
            self.kf.x = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])
            self.is_initialized = True
            logger.info(f"Kalman filter initialized: ({measurement[0]:.2f}, {measurement[1]:.2f})")
            return measurement

        dt = max(0.01, min(dt, 5.0))
        self._update_matrices_for_dt(dt)

        self.kf.predict()
        self.kf.update(z)

        filtered_x = float(self.kf.x[0, 0])
        filtered_y = float(self.kf.x[1, 0])

        return filtered_x, filtered_y