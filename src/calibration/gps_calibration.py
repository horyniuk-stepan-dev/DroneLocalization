"""
GPS calibration
"""

import cv2
import numpy as np


class GPSCalibration:
    """GPS calibration manager"""
    
    def __init__(self):
        self.affine_matrix = None
        self.is_calibrated = False
    
    def calibrate(self, points_2d, points_gps):
        """
        Calibrate using reference points
        
        Args:
            points_2d: [(x1, y1), (x2, y2), ...] in topometric coords
            points_gps: [(lat1, lon1), (lat2, lon2), ...] in GPS
        """
        # TODO: Convert GPS to metric
        # TODO: Estimate affine transformation
        # TODO: Calculate RMSE
        # TODO: Save calibration
        pass
    
    def transform_to_gps(self, x_2d, y_2d):
        """Transform 2D coords to GPS"""
        # TODO: Apply affine transform
        # TODO: Convert to GPS
        # TODO: Return lat, lon
        pass
    
    def save(self, path):
        """Save calibration to file"""
        # TODO: Save affine matrix
        pass
    
    def load(self, path):
        """Load calibration from file"""
        # TODO: Load affine matrix
        pass
