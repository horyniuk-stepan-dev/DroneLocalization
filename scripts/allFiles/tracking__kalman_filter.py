import numpy as np
from filterpy.kalman import KalmanFilter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrajectoryFilter:
    """
    Kalman filter for GPS/metric trajectory smoothing.
    State: [x, y, vx, vy] — constant velocity kinematic model.
    """

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 10.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.is_initialized = False

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.P *= 1000.0  # high initial uncertainty

        # Measurement matrix: observe x and y only
        self.kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        self.kf.R = np.eye(2) * measurement_noise

        # Initialize F and Q via shared method (single source of truth)
        self._update_matrices_for_dt(1.0)

        logger.info(
            f"Kalman filter ready | "
            f"process_noise={process_noise}, measurement_noise={measurement_noise}"
        )

    def _update_matrices_for_dt(self, dt: float) -> None:
        """Recompute F and Q for given time step (Wiener process model)."""
        pn = self.process_noise
        self.kf.F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.kf.Q = np.array([
            [dt**4/4,       0, dt**3/2,       0],
            [      0, dt**4/4,       0, dt**3/2],
            [dt**3/2,       0,   dt**2,       0],
            [      0, dt**3/2,       0,   dt**2],
        ]) * pn

    def update(self, measurement) -> tuple[float, float]:
        """Update with new (x, y) measurement using default dt=1.0."""
        return self.update_with_dt(measurement, dt=1.0)

    def update_with_dt(self, measurement, dt: float) -> tuple[float, float]:
        """
        Update filter with real inter-frame time dt (seconds).
        On first call: initializes state without filtering.
        """
        if not self.is_initialized:
            self.kf.x = np.array([[float(measurement[0])],
                                   [float(measurement[1])],
                                   [0.0], [0.0]])
            self.is_initialized = True
            logger.info(f"Kalman initialized at ({measurement[0]:.2f}, {measurement[1]:.2f})")
            return float(measurement[0]), float(measurement[1])

        dt = max(0.01, min(dt, 5.0))
        self._update_matrices_for_dt(dt)

        z = np.array([[float(measurement[0])], [float(measurement[1])]])
        self.kf.predict()
        self.kf.update(z)

        fx, fy = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
        logger.debug(f"Kalman update dt={dt:.3f}s → ({fx:.2f}, {fy:.2f})")
        return fx, fy

    def get_velocity(self) -> tuple[float, float, float]:
        """Return (vx, vy, speed) in metric units. Returns zeros if not initialized."""
        if not self.is_initialized:
            return 0.0, 0.0, 0.0
        vx = float(self.kf.x[2, 0])
        vy = float(self.kf.x[3, 0])
        speed = float(np.hypot(vx, vy))
        logger.debug(f"Velocity: vx={vx:.2f}, vy={vy:.2f}, speed={speed:.2f} m/s")
        return vx, vy, speed

    def reset(self) -> None:
        """Reset filter state — call before starting a new tracking session."""
        self.is_initialized = False
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000.0
        self._update_matrices_for_dt(1.0)
        logger.info("Kalman filter reset")
