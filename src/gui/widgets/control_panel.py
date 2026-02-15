"""
Control panel widget
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QProgressBar
from PyQt6.QtCore import pyqtSignal


class ControlPanel(QWidget):
    """Control panel for mission management"""
    
    new_mission_clicked = pyqtSignal()
    load_database_clicked = pyqtSignal()
    start_tracking_clicked = pyqtSignal()
    stop_tracking_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        # TODO: Create layout
        # TODO: Add mission section
        # TODO: Add database section
        # TODO: Add tracking section
        # TODO: Add status section
        # TODO: Connect signals
        pass
    
    def update_status(self, message: str):
        """Update status message"""
        # TODO: Update status label
        pass
    
    def update_progress(self, value: int):
        """Update progress bar"""
        # TODO: Update progress bar value
        pass
    
    def set_tracking_enabled(self, enabled: bool):
        """Enable/disable tracking controls"""
        # TODO: Enable/disable relevant buttons
        pass
