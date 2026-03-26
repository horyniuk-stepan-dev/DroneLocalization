import json
from pathlib import Path

from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Resolved once at import time — safe for both dev and frozen builds
_MAP_PATH = Path(__file__).resolve().parent.parent / "resources" / "maps" / "map_template.html"


class MapBridge(QObject):
    """
    Qt↔JavaScript signal bus via QWebChannel.
    Signals here are consumed by map_template.html — not connected to Python slots.
    """

    updateMarkerSignal = pyqtSignal(float, float)
    addTrajectorySignal = pyqtSignal(float, float)
    clearTrajectorySignal = pyqtSignal()

    # 8 floats: TL, TR, BR, BL corners (lat, lon each)
    updateFOVSignal = pyqtSignal(float, float, float, float, float, float, float, float)

    # data_url (base64 JPEG) + 8 corner coords
    setPanoramaSignal = pyqtSignal(str, float, float, float, float, float, float, float, float)

    # Verification markers (JSON string of points)
    showVerificationMarkersSignal = pyqtSignal(str)
    clearVerificationMarkersSignal = pyqtSignal()

    # JS -> Python: Map Click
    mapClickedSignal = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def mapClicked(self, lat: float, lon: float):
        """Called from JavaScript when the map is clicked."""
        self.mapClickedSignal.emit(lat, lon)


class MapWidget(QWebEngineView):
    """Interactive map widget backed by Leaflet via QWebChannel."""

    mapClicked = pyqtSignal(float, float)  # Public signal

    def __init__(self, parent=None):
        super().__init__(parent)

        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        self.bridge = MapBridge()
        self.bridge.mapClickedSignal.connect(self.mapClicked)  # Re-emit for convenience

        self._channel = QWebChannel()
        self._channel.registerObject("mapBridge", self.bridge)
        self.page().setWebChannel(self._channel)

        self._load_map()

    # ── Map loading ──────────────────────────────────────────────────────────

    def _load_map(self):
        if _MAP_PATH.exists():
            self.setUrl(QUrl.fromLocalFile(str(_MAP_PATH)))
            logger.info(f"Map loaded: {_MAP_PATH}")
        else:
            logger.error(f"Map template not found: {_MAP_PATH}")
            self.setHtml(
                f"""
                <html><body style='font-family:Arial;padding:20px'>
                <h2 style='color:red'>Помилка: Файл карти не знайдено!</h2>
                <p>Очікуваний шлях:</p>
                <code style='background:#eee;padding:5px'>{_MAP_PATH}</code>
                </body></html>
            """
            )

    # ── Public API ───────────────────────────────────────────────────────────

    @pyqtSlot(float, float)
    def update_marker(self, lat: float, lon: float):
        self.bridge.updateMarkerSignal.emit(lat, lon)

    @pyqtSlot(float, float)
    def add_trajectory_point(self, lat: float, lon: float):
        self.bridge.addTrajectorySignal.emit(lat, lon)

    @pyqtSlot()
    def clear_trajectory(self):
        self.bridge.clearTrajectorySignal.emit()

    @pyqtSlot(list)
    def update_fov(self, fov: list):
        """
        Accepts FOV as a Python list: [(lat0,lon0), (lat1,lon1), (lat2,lon2), (lat3,lon3)]
        """
        if not fov or len(fov) != 4:
            logger.warning(f"update_fov: expected 4 points, got {len(fov) if fov else 0}")
            return

        try:
            self.bridge.updateFOVSignal.emit(
                float(fov[0][0]),
                float(fov[0][1]),
                float(fov[1][0]),
                float(fov[1][1]),
                float(fov[2][0]),
                float(fov[2][1]),
                float(fov[3][0]),
                float(fov[3][1]),
            )
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"update_fov: malformed point data: {e}")

    @pyqtSlot(str, float, float, float, float, float, float, float, float)
    def set_panorama_overlay(
        self,
        data_url: str,
        lat_tl: float,
        lon_tl: float,
        lat_tr: float,
        lon_tr: float,
        lat_br: float,
        lon_br: float,
        lat_bl: float,
        lon_bl: float,
    ):
        self.bridge.setPanoramaSignal.emit(
            data_url,
            lat_tl,
            lon_tl,
            lat_tr,
            lon_tr,
            lat_br,
            lon_br,
            lat_bl,
            lon_bl,
        )

    @pyqtSlot(list)
    def show_verification_markers(self, points: list):
        """
        Accepts points as list of dicts [{'lat': float, 'lon': float, 'label': str}]
        """
        self.bridge.showVerificationMarkersSignal.emit(json.dumps(points))

    @pyqtSlot()
    def clear_verification_markers(self):
        self.bridge.clearVerificationMarkersSignal.emit()
