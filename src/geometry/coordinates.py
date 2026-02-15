"""
Coordinate system conversions
"""

from pyproj import Transformer
import numpy as np


class CoordinateConverter:
    """Convert between coordinate systems"""
    
    def __init__(self):
        # TODO: Initialize transformers
        pass
    
    @staticmethod
    def gps_to_metric(lat, lon):
        """Convert GPS to metric (Web Mercator)"""
        # TODO: Use pyproj Transformer
        # TODO: Return x, y in meters
        pass
    
    @staticmethod
    def metric_to_gps(x, y):
        """Convert metric to GPS"""
        # TODO: Use pyproj Transformer
        # TODO: Return lat, lon
        pass
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between GPS coordinates"""
        # TODO: Implement haversine formula
        # TODO: Return distance in meters
        pass
