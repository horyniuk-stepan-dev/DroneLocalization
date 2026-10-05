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

    # Object tracking markers (JSON string of points)
    updateObjectMarkersSignal = pyqtSignal(str)
    toggleObjectMarkersSignal = pyqtSignal(bool)

    # JS -> Python: Map Click
    mapClickedSignal = pyqtSignal(float, float)
    readySignal = pyqtSignal(str)

    @pyqtSlot(str)
    def ready(self, mode: str):
        """Flush updates only after the page has connected its signal handlers."""
        self.readySignal.emit(mode)

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
        self.bridge.readySignal.connect(self._on_map_ready)
        self._coordinate_mode = "GEOGRAPHIC"
        self._map_ready = False
        self._pending_updates = []

        self._channel = QWebChannel()
        self._channel.registerObject("mapBridge", self.bridge)
        self.page().setWebChannel(self._channel)

        self._load_map()

    # ── Map loading ──────────────────────────────────────────────────────────

    def _load_map(self):
        map_path = (
            _MAP_PATH.with_name("local_map.html") if self._coordinate_mode == "LOCAL" else _MAP_PATH
        )
        self._map_ready = False
        if map_path.exists():
            self.setUrl(QUrl.fromLocalFile(str(map_path)))
            logger.info(f"Map loaded: {map_path}")
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

    def set_coordinate_mode(self, mode: str, *, reset: bool = False):
        mode = "LOCAL" if mode == "LOCAL" else "GEOGRAPHIC"
        if mode != self._coordinate_mode or reset:
            self._coordinate_mode = mode
            self._pending_updates.clear()
            self._load_map()

    def _emit_update(self, signal: str, *args):
        if self._map_ready:
            getattr(self.bridge, signal).emit(*args)
        else:
            self._pending_updates.append((signal, args))

    def _on_map_ready(self, mode: str):
        if mode != self._coordinate_mode:
            return
        self._map_ready = True
        pending, self._pending_updates = self._pending_updates, []
        for signal, args in pending:
            getattr(self.bridge, signal).emit(*args)

    @pyqtSlot(float, float)
    def update_marker(self, lat: float, lon: float):
        self._emit_update("updateMarkerSignal", lat, lon)

    @pyqtSlot(float, float)
    def add_trajectory_point(self, lat: float, lon: float):
        self._emit_update("addTrajectorySignal", lat, lon)

    @pyqtSlot()
    def clear_trajectory(self):
        self._emit_update("clearTrajectorySignal")

    @pyqtSlot(list)
    def update_fov(self, fov: list):
        """
        Accepts FOV as a Python list: [(lat0,lon0), (lat1,lon1), (lat2,lon2), (lat3,lon3)]
        """
        if not fov or len(fov) != 4:
            logger.warning(f"update_fov: expected 4 points, got {len(fov) if fov else 0}")
            return

        try:
            self._emit_update(
                "updateFOVSignal",
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
        self._emit_update(
            "setPanoramaSignal",
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
        self._emit_update("showVerificationMarkersSignal", json.dumps(points))

    @pyqtSlot()
    def clear_verification_markers(self):
        self._emit_update("clearVerificationMarkersSignal")

    @pyqtSlot(list)
    def update_object_markers(self, points: list):
        """
        Accepts points as list of dicts [{'lat': float, 'lon': float, 'label': str, 'class_name': str}]
        """
        self._emit_update("updateObjectMarkersSignal", json.dumps(points))

    @pyqtSlot(bool)
    def set_objects_visible(self, visible: bool):
        self._emit_update("toggleObjectMarkersSignal", visible)
