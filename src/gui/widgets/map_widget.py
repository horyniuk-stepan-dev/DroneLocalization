"""
Interactive map widget using Leaflet.js through QWebEngineView
"""

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl


class MapBridge(QObject):
    """Bridge between Python and JavaScript"""
    
    updateMarkerSignal = pyqtSignal(float, float)
    addTrajectoryPointSignal = pyqtSignal(float, float)
    clearTrajectorySignal = pyqtSignal()
    
    def __init__(self):
        super().__init__()


class MapWidget(QWebEngineView):
    """Interactive map widget"""
    
    marker_clicked = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = MapBridge()
        self.setup_web_channel()
        self.load_map()
    
    def setup_web_channel(self):
        """Setup QWebChannel for Python-JS communication"""
        # TODO: Create QWebChannel
        # TODO: Register bridge object
        # TODO: Set channel to page
        pass
    
    def load_map(self):
        """Load Leaflet.js map"""
        # TODO: Load HTML with Leaflet.js
        # TODO: Setup JavaScript listeners
        # TODO: Connect to QWebChannel
        pass
    
    @pyqtSlot(float, float)
    def update_marker(self, lat, lon):
        """Update marker position"""
        self.bridge.updateMarkerSignal.emit(lat, lon)
    
    @pyqtSlot(float, float)
    def add_trajectory_point(self, lat, lon):
        """Add point to trajectory"""
        self.bridge.addTrajectoryPointSignal.emit(lat, lon)
    
    @pyqtSlot()
    def clear_trajectory(self):
        """Clear trajectory"""
        self.bridge.clearTrajectorySignal.emit()
