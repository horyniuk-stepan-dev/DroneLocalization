"""
Outlier detection for localization
"""

from collections import deque
import numpy as np


class OutlierDetector:
    """Detect outlier measurements"""
    
    def __init__(self, window_size=10, threshold=3.0):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold
    
    def is_outlier(self, new_position):
        """Check if position is outlier"""
        # TODO: Calculate distances from history
        # TODO: Compute z-score
        # TODO: Return True if outlier
        pass
