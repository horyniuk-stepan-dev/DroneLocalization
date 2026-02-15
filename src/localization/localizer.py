"""
Main localization pipeline
"""

import numpy as np


class Localizer:
    """Localizes query frames using database"""
    
    def __init__(self, database, models, calibration, config):
        self.database = database
        self.models = models
        self.calibration = calibration
        self.config = config
    
    def localize_frame(self, image: np.ndarray) -> dict:
        """
        Localize single frame
        
        Returns:
            dict with keys:
                - success: bool
                - lat: float
                - lon: float
                - confidence: float
                - matched_frame: int
        """
        # TODO: Extract features from query
        # TODO: Find similar frames
        # TODO: Match keypoints with LightGlue
        # TODO: Compute homography with RANSAC
        # TODO: Transform to 2D coordinates
        # TODO: Transform to GPS
        # TODO: Return results
        pass
