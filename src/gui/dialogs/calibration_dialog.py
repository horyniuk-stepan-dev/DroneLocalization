"""
GPS calibration dialog
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal


class CalibrationDialog(QDialog):
    """Dialog for GPS calibration"""
    
    calibration_complete = pyqtSignal(object)  # calibration_matrix
    
    def __init__(self, database_path, parent=None):
        super().__init__(parent)
        self.database_path = database_path
        self.reference_points_2d = []
        self.reference_points_gps = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        # TODO: Create split layout
        # TODO: Add video view on left
        # TODO: Add map view on right
        # TODO: Add instructions label
        # TODO: Add point list
        # TODO: Add Calculate/Reset/Close buttons
        pass
    
    def on_video_click(self, x, y):
        """Handle click on video"""
        # TODO: Store 2D point
        # TODO: Update UI
        # TODO: Prompt for GPS click
        pass
    
    def on_map_click(self, lat, lon):
        """Handle click on map"""
        # TODO: Store GPS point
        # TODO: Update UI
        # TODO: Check if enough points
        pass
    
    def calculate_calibration(self):
        """Calculate affine transformation"""
        # TODO: Call calibration module
        # TODO: Display RMSE
        # TODO: Emit signal with result
        pass
