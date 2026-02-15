"""
New mission creation dialog
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QPushButton, QLabel


class NewMissionDialog(QDialog):
    """Dialog for creating new mission"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        # TODO: Create layout
        # TODO: Add mission name field
        # TODO: Add video file selector
        # TODO: Add altitude input
        # TODO: Add camera parameters
        # TODO: Add OK/Cancel buttons
        pass
    
    def get_mission_data(self) -> dict:
        """Get mission configuration data"""
        # TODO: Return dict with all inputs
        return {}
    
    def browse_video(self):
        """Open file dialog for video selection"""
        # TODO: Show file dialog
        # TODO: Update path field
        pass
