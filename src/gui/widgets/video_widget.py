"""
Video display widget using QGraphicsView
"""

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap


class VideoWidget(QGraphicsView):
    """Widget for displaying video with overlays"""
    
    frame_clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # TODO: Setup scene properties
        # TODO: Enable mouse tracking
        # TODO: Setup transformation modes
    
    def display_frame(self, pixmap: QPixmap):
        """Display a frame"""
        # TODO: Clear scene
        # TODO: Add pixmap to scene
        # TODO: Fit in view
        pass
    
    def draw_bounding_box(self, x, y, w, h, label="", confidence=0.0):
        """Draw bounding box on frame"""
        # TODO: Draw rectangle
        # TODO: Add label text
        pass
    
    def draw_trajectory(self, points: list):
        """Draw trajectory line"""
        # TODO: Draw polyline through points
        pass
    
    def clear_overlays(self):
        """Clear all overlays"""
        # TODO: Remove all graphics items except background
        pass
    
    def mousePressEvent(self, event):
        """Handle mouse click"""
        # TODO: Get scene position
        # TODO: Emit signal with coordinates
        super().mousePressEvent(event)
