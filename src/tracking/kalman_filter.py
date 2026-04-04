import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

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
        logger.info(
            f"Parameters: process_noise={process_noise}, measurement_noise={measurement_noise}, dt={dt}"
        )

        self.kf.P *= 1000.0

        self.kf.F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        self.kf.R = np.array([[measurement_noise, 0.0], [0.0, measurement_noise]])

        self._update_matrices_for_dt(dt)

    def _update_matrices_for_dt(self, dt: float):
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        q_var = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        self.kf.Q = np.zeros((4, 4))

        # Блок осі X (позиція X та швидкість VX)
        self.kf.Q[0, 0] = q_var[0, 0]  # Дисперсія позиції X
        self.kf.Q[0, 2] = q_var[0, 1]  # Коваріація X та VX
        self.kf.Q[2, 0] = q_var[1, 0]  # Коваріація VX та X
        self.kf.Q[2, 2] = q_var[1, 1]  # Дисперсія швидкості VX

        # Блок осі Y (позиція Y та швидкість VY)
        self.kf.Q[1, 1] = q_var[0, 0]  # Дисперсія позиції Y
        self.kf.Q[1, 3] = q_var[0, 1]  # Коваріація Y та VY
        self.kf.Q[3, 1] = q_var[1, 0]  # Коваріація VY та Y
        self.kf.Q[3, 3] = q_var[1, 1]  # Дисперсія швидкості VY

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

    def reset(self) -> None:
        """
        Скидає фільтр до початкового стану.
        Викликати при кожному новому старті трекінгу, щоб уникнути
        хибних передбачень на основі швидкості попередньої сесії.
        """
        self.is_initialized = False
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000.0
        logger.info("Kalman filter reset to initial state")
