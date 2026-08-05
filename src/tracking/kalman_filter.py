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

        # Elevated process noise and reduced measurement noise
        # let the filter react faster to heading changes at high speeds
        self.process_noise = process_noise
        self.is_initialized = False
        # Two-point velocity seed: the first update() after initialisation
        # seeds vx/vy from the difference of the first two raw fixes instead of
        # v=0. Without this, trajectories with sustained high speed (~150 m/s
        # measured on simulator flight) cause the filtered position to lag
        # behind raw fixes for several steps while the KF learns velocity.
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
        # Base R for adaptive scaling by localisation confidence
        self._base_R = self.kf.R.copy()

        self._update_matrices_for_dt(dt)

    def _update_matrices_for_dt(self, dt: float):
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        q_var = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        self.kf.Q = np.zeros((4, 4))

        # X-axis block (X position and VX velocity)
        self.kf.Q[0, 0] = q_var[0, 0]  # X position variance
        self.kf.Q[0, 2] = q_var[0, 1]  # X / VX covariance
        self.kf.Q[2, 0] = q_var[1, 0]  # VX / X covariance
        self.kf.Q[2, 2] = q_var[1, 1]  # VX velocity variance

        # Y-axis block (Y position and VY velocity)
        self.kf.Q[1, 1] = q_var[0, 0]  # Y position variance
        self.kf.Q[1, 3] = q_var[0, 1]  # Y / VY covariance
        self.kf.Q[3, 1] = q_var[1, 0]  # VY / Y covariance
        self.kf.Q[3, 3] = q_var[1, 1]  # VY velocity variance

    def mahalanobis_sq(
        self, measurement: tuple, dt: float = 1.0, noise_scale: float = 1.0
    ) -> float | None:
        """d^2 = y^T S^-1 y for measurement ``measurement`` — WITHOUT modifying filter state.

        y — innovation (measurement minus prediction), S = H*P_pred*H^T + R — its covariance.
        Unlike speed-based Z-score, this distance is normalized by the filter's own
        uncertainty: after a long series of consistent fixes P is small and
        gate is strict; after a loss/reset P is large and gate relaxes itself.
        That is why it has no self-sustaining loop present in Z-score (where speed
        window is both filtered and forms the threshold).

        Returns None if filter is not yet initialized (nothing to gate against)
        or S is degenerate — caller should skip measurement, not discard it.

        Calculation duplicates predict-step on LOCAL copies of F/Q/P: filterpy
        ``kf.predict()`` mutates state, whereas gate must be side-effect-free — otherwise
        a discarded measurement would still shift the filter.
        """
        if not self.is_initialized:
            return None

        dt = max(0.01, min(dt, 5.0))
        ns = float(np.clip(noise_scale, 0.25, 25.0))

        F = self.kf.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt

        q_var = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        Q = np.zeros((4, 4))
        Q[0, 0] = Q[1, 1] = q_var[0, 0]
        Q[0, 2] = Q[1, 3] = q_var[0, 1]
        Q[2, 0] = Q[3, 1] = q_var[1, 0]
        Q[2, 2] = Q[3, 3] = q_var[1, 1]

        x_pred = F @ self.kf.x
        P_pred = F @ self.kf.P @ F.T + Q

        H = self.kf.H
        R = self._base_R * ns
        S = H @ P_pred @ H.T + R

        z = np.array([[float(measurement[0])], [float(measurement[1])]])
        y = z - H @ x_pred

        try:
            # [0, 0]: result is a 1×1 matrix; float() on it is deprecated in numpy
            d2 = float((y.T @ np.linalg.inv(S) @ y)[0, 0])
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(d2) or d2 < 0.0:
            return None
        return d2

    def update(self, measurement: tuple, dt: float = 1.0, noise_scale: float = 1.0) -> tuple:
        """noise_scale — adaptive measurement noise multiplier:
        > 1 for weak/relative measurements (low confidence, optical flow),
        1.0 for confident measurements. Allows the filter to trust poor
        measurements less.
        """
        z = np.array([[measurement[0]], [measurement[1]]])

        if not self.is_initialized:
            self.kf.x = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])
            self.is_initialized = True
            self._prev_raw = (float(measurement[0]), float(measurement[1]))
            logger.info(f"Kalman filter initialized: ({measurement[0]:.2f}, {measurement[1]:.2f})")
            return measurement

        if self._prev_raw is not None:
            # Two-point seed: compute velocity from the FIRST pair of raw fixes
            # and set it in the state BEFORE predict/update of this step. Applied
            # only once (immediately after initialisation) — afterwards the filter
            # tracks velocity on its own.
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
        """Shift positional component of state (correction from back-end smoother).
        Velocities and covariance are untouched: correction is a reference-frame shift,
        not a new measurement.
        """
        if not self.is_initialized:
            return
        self.kf.x[0, 0] += dx
        self.kf.x[1, 0] += dy

    def reset(self) -> None:
        """Resets filter to initial state.
        Call on every new tracking start to avoid false predictions
        based on previous session's velocity.
        """
        self.is_initialized = False
        self._prev_raw = None
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000.0
        logger.info("Kalman filter reset to initial state")
