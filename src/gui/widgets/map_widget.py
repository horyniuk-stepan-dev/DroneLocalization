from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl
import os

from utils.logging_utils import get_logger


class MapBridge(QObject):
    updateMarkerSignal = pyqtSignal(float, float)
    addTrajectoryPointSignal = pyqtSignal(float, float)
    clearTrajectorySignal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger('MapBridge')

class MapWidget(QWebEngineView):
    marker_clicked = pyqtSignal(float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = MapBridge()
        self.setup_web_channel()
        self.load_map()
    
    def setup_web_channel(self):
        self.channel = QWebChannel()
        self.channel.registerObject("mapBridge", self.bridge)
        self.page().setWebChannel(self.channel)
    
    def load_map(self):
        map_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/maps/map_template.html"))
        if os.path.exists(map_path):
            self.setUrl(QUrl.fromLocalFile(map_path))
        else:
            self.setHtml("<html><body><h1>Карта не знайдена</h1></body></html>")
    
    @pyqtSlot(float, float)
    def update_marker(self, lat, lon):
        self.bridge.updateMarkerSignal.emit(lat, lon)
    
    @pyqtSlot(float, float)
    def add_trajectory_point(self, lat, lon):
        self.bridge.addTrajectoryPointSignal.emit(lat, lon)
    
    @pyqtSlot()
    def clear_trajectory(self):
        self.bridge.clearTrajectorySignal.emit()