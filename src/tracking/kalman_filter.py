import numpy as np
from filterpy.kalman import KalmanFilter

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrajectoryFilter:
    """Kalman filter for GPS trajectory smoothing"""

    def __init__(self, process_noise=0.1, measurement_noise=10.0, dt=1.0):
        # Filter state: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        logger.info("Initializing Kalman filter for trajectory smoothing")
        logger.info(f"Parameters: process_noise={process_noise}, measurement_noise={measurement_noise}, dt={dt}")

        # Initial state covariance (high uncertainty at start)
        self.kf.P *= 1000.0

        # State transition matrix (constant velocity kinematic model)
        self.kf.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Measurement matrix (we only measure x and y)
        self.kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # Process noise covariance (unpredictable drone maneuvers)
        self.kf.Q = np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2]
        ]) * process_noise

        # Measurement noise covariance (neural network localization error)
        self.kf.R = np.eye(2) * measurement_noise

        self.is_initialized = False
        logger.success("Kalman filter initialized successfully")

    def update(self, measurement: tuple) -> tuple:
        """Update filter state with new coordinates (x, y)"""
        z = np.array([[measurement[0]], [measurement[1]]])

        if not self.is_initialized:
            self.kf.x = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])
            self.is_initialized = True
            logger.info(f"Kalman filter initialized with measurement: ({measurement[0]:.2f}, {measurement[1]:.2f})")
            return measurement

        logger.debug(f"Kalman predict-update cycle for measurement: ({measurement[0]:.2f}, {measurement[1]:.2f})")

        self.kf.predict()
        self.kf.update(z)

        filtered_x = float(self.kf.x[0, 0])
        filtered_y = float(self.kf.x[1, 0])

        logger.debug(f"Filtered position: ({filtered_x:.2f}, {filtered_y:.2f})")

        return filtered_x, filtered_y

    def get_velocity(self) -> tuple:
        """Get current drone velocity estimate"""
        if not self.is_initialized:
            return 0.0, 0.0

        vx = float(self.kf.x[2, 0])
        vy = float(self.kf.x[3, 0])
        speed = np.sqrt(vx ** 2 + vy ** 2)

        logger.debug(f"Current velocity: vx={vx:.2f}, vy={vy:.2f}, speed={speed:.2f} m/s")

        return vx, vy
