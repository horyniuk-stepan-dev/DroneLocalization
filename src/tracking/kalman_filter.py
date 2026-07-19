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
        # Two-point velocity seed (живий інцидент 2026-07-18): перше update()
        # після ініціалізації задає vx/vy з різниці перших двох сирих точок
        # замість v=0 — без цього на траєкторіях зі сталою високою швидкістю
        # (виміряно ~150 м/с на симуляторному польоті) filtered-позиція кілька
        # кроків відстає від сирих фіксів, поки KF "вивчає" швидкість із нуля.
        self._prev_raw: tuple[float, float] | None = None

        logger.info("Initializing Kalman filter for high-speed trajectory smoothing")
        logger.info(
            f"Parameters: process_noise={process_noise}, measurement_noise={measurement_noise}, dt={dt}"
        )

        self.kf.P *= 1000.0

        self.kf.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        self.kf.R = np.array([[measurement_noise, 0.0], [0.0, measurement_noise]])
        # Базовий R для адаптивного масштабування за confidence локалізації
        self._base_R = self.kf.R.copy()

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

    def update(self, measurement: tuple, dt: float = 1.0, noise_scale: float = 1.0) -> tuple:
        """noise_scale — адаптивний множник шуму вимірювання (B2):
        > 1 для слабких/відносних вимірювань (низький confidence, optical flow),
        1.0 для впевнених. Дозволяє фільтру менше довіряти поганим вимірюванням.
        """
        z = np.array([[measurement[0]], [measurement[1]]])

        if not self.is_initialized:
            self.kf.x = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])
            self.is_initialized = True
            self._prev_raw = (float(measurement[0]), float(measurement[1]))
            logger.info(f"Kalman filter initialized: ({measurement[0]:.2f}, {measurement[1]:.2f})")
            return measurement

        if self._prev_raw is not None:
            # Two-point seed: рахуємо швидкість з ПЕРШОЇ пари сирих точок і
            # підставляємо в стан ДО predict/update цього кроку. Лише один раз
            # (одразу після ініціалізації) — далі фільтр веде швидкість сам.
            safe_seed_dt = max(dt, 0.01)
            vx = (measurement[0] - self._prev_raw[0]) / safe_seed_dt
            vy = (measurement[1] - self._prev_raw[1]) / safe_seed_dt
            self.kf.x[2, 0] = vx
            self.kf.x[3, 0] = vy
            self._prev_raw = None
            logger.debug(f"Kalman two-point velocity seed: ({vx:.2f}, {vy:.2f}) m/s")

        ns = float(np.clip(noise_scale, 0.25, 25.0))
        self.kf.R = self._base_R * ns

        dt = max(0.01, min(dt, 5.0))
        self._update_matrices_for_dt(dt)

        self.kf.predict()
        self.kf.update(z)

        filtered_x = float(self.kf.x[0, 0])
        filtered_y = float(self.kf.x[1, 0])

        return filtered_x, filtered_y

    def shift(self, dx: float, dy: float) -> None:
        """Зсув позиційної частини стану (корекція від back-end smoother'а,
        RESEARCH 3.1). Швидкості та коваріація не чіпаються: корекція — це
        зсув системи відліку оцінки, а не нове вимірювання.
        """
        if not self.is_initialized:
            return
        self.kf.x[0, 0] += dx
        self.kf.x[1, 0] += dy

    def reset(self) -> None:
        """
        Скидає фільтр до початкового стану.
        Викликати при кожному новому старті трекінгу, щоб уникнути
        хибних передбачень на основі швидкості попередньої сесії.
        """
        self.is_initialized = False
        self._prev_raw = None
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000.0
        logger.info("Kalman filter reset to initial state")
