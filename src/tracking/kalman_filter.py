"""
Kalman filter for trajectory smoothing
"""

from filterpy.kalman import KalmanFilter
import numpy as np


class TrajectoryFilter:
    """Kalman filter for GPS trajectory"""
    
    def __init__(self, process_noise=0.1, measurement_noise=10.0):
        # TODO: Initialize KalmanFilter
        # TODO: Set state transition matrix
        # TODO: Set measurement matrix
        # TODO: Set noise covariances
        pass
    
    def update(self, measurement):
        """Update filter with new measurement"""
        # TODO: Predict
        # TODO: Update
        # TODO: Return filtered position
        pass
    
    def get_velocity(self):
        """Get estimated velocity"""
        # TODO: Return velocity from state
        pass
