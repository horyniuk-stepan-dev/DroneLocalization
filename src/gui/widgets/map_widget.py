import json
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from config.paths import user_data_dir
from src.gui.map_features import MapFeature, MapFeatureStore, path_length, ruler_label
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
    # Operator-selected targets with trails (JSON list)
    updateManualTargetsSignal = pyqtSignal(str)

    # JS -> Python: Map Click
    mapClickedSignal = pyqtSignal(float, float)
    readySignal = pyqtSignal(str)

    # ── User map tools ───────────────────────────────────────────────────
    # Python -> JS
    # JSON list of the current mode's features, each with a "measure" label
    loadFeaturesSignal = pyqtSignal(str)
    rulerMeasuredSignal = pyqtSignal(str)  # ruler label text
    # JS -> Python (re-emitted for MapWidget)
    featureUpsertRequested = pyqtSignal(str)
    featureDeleteRequested = pyqtSignal(str)
    rulerMeasureRequested = pyqtSignal(str)
    exportRequested = pyqtSignal()
    importRequested = pyqtSignal()
    clearFeaturesRequested = pyqtSignal()

    @pyqtSlot(str)
    def copyToClipboard(self, text: str):
        QApplication.clipboard().setText(text)

    @pyqtSlot(str)
    def featureUpsert(self, feature_json: str):
        self.featureUpsertRequested.emit(feature_json)

    @pyqtSlot(str)
    def featureDelete(self, feature_id: str):
        self.featureDeleteRequested.emit(feature_id)

    @pyqtSlot(str)
    def measureRuler(self, coords_json: str):
        self.rulerMeasureRequested.emit(coords_json)

    @pyqtSlot()
    def requestExport(self):
        self.exportRequested.emit()

    @pyqtSlot()
    def requestImport(self):
        self.importRequested.emit()

    @pyqtSlot()
    def clearFeatures(self):
        self.clearFeaturesRequested.emit()

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
    featuresChanged = pyqtSignal()  # user-drawn features were added/edited/removed

    def __init__(self, parent=None):
        super().__init__(parent)

        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        # The page draws its own right-click menu ("Копіювати координати");
        # hide Qt's browser menu (Back / Reload / View source) so they don't stack.
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

        self.bridge = MapBridge()
        self.bridge.mapClickedSignal.connect(self.mapClicked)  # Re-emit for convenience
        self.bridge.readySignal.connect(self._on_map_ready)
        self._coordinate_mode = "GEOGRAPHIC"
        self._map_ready = False
        self._pending_updates = []

        # User-drawn features: Python is the source of truth, the page is a view.
        self.features = MapFeatureStore()
        self._features_path = user_data_dir() / "map_features.geojson"
        self._load_autosave()
        b = self.bridge
        b.featureUpsertRequested.connect(self._on_feature_upsert)
        b.featureDeleteRequested.connect(self._on_feature_delete)
        b.rulerMeasureRequested.connect(self._on_ruler_measure)
        b.exportRequested.connect(self.export_features)
        b.importRequested.connect(self.import_features)
        b.clearFeaturesRequested.connect(self._on_clear_features)

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
        self._push_features()  # page reload drops drawn layers — restore them

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

    def update_manual_targets(self, targets: list):
        """targets: [{id, label, lat, lon, lost, trail: [[lat, lon], ...]}]."""
        self._emit_update("updateManualTargetsSignal", json.dumps(targets))

    # ── User map tools ───────────────────────────────────────────────────────

    def _push_features(self):
        payload = json.dumps(self.features.view_payload(self._coordinate_mode), ensure_ascii=False)
        self._emit_update("loadFeaturesSignal", payload)

    def _features_updated(self):
        self._push_features()
        self._autosave()
        self.featuresChanged.emit()

    def _load_autosave(self):
        if not self._features_path.exists():
            return
        try:
            n = self.features.load(self._features_path)
            logger.info(f"Map features loaded: {n} from {self._features_path}")
        except (OSError, ValueError) as e:
            logger.warning(f"Map features autosave unreadable ({self._features_path}): {e}")

    def _autosave(self):
        try:
            self.features.save(self._features_path)
        except OSError as e:
            logger.warning(f"Map features autosave failed: {e}")

    def _on_feature_upsert(self, feature_json: str):
        try:
            d = json.loads(feature_json)
            d["mode"] = self._coordinate_mode
            if d.get("type") == "circle" and d.get("edge") is not None:
                d["radius"] = _distance(d["coords"][0], d["edge"], self._coordinate_mode)
            if d.get("id") and self.features.get(d["id"]) is None:
                d.pop("id")  # stale id from a page that was reloaded meanwhile
            feature = self.features.upsert(MapFeature.from_dict(d))
        except (ValueError, KeyError, TypeError, IndexError) as e:
            logger.warning(f"Map feature rejected: {e}")
            return
        logger.info(f"Map feature saved: {feature.type} '{feature.name}'")
        self._features_updated()

    def _on_feature_delete(self, feature_id: str):
        if self.features.remove(feature_id):
            self._features_updated()

    def _on_clear_features(self):
        if not self.features.features(self._coordinate_mode):
            return
        answer = QMessageBox.question(self, "Очистити карту", "Видалити всі об'єкти на цій карті?")
        if answer == QMessageBox.StandardButton.Yes:
            self.features.clear(self._coordinate_mode)
            self._features_updated()

    def _on_ruler_measure(self, coords_json: str):
        try:
            coords = [[float(a), float(b)] for a, b in json.loads(coords_json)]
        except (ValueError, TypeError):
            return
        self._emit_update("rulerMeasuredSignal", ruler_label(coords, self._coordinate_mode))

    @pyqtSlot()
    def export_features(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Експорт об'єктів карти", "map_features.geojson", "GeoJSON (*.geojson *.json)"
        )
        if not path:
            return
        try:
            self.features.save(Path(path))
            logger.info(f"Map features exported: {path}")
        except OSError as e:
            QMessageBox.warning(self, "Експорт", f"Не вдалося зберегти файл:\n{e}")

    @pyqtSlot()
    def import_features(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Імпорт об'єктів карти", "", "GeoJSON (*.geojson *.json)"
        )
        if not path:
            return
        try:
            n = self.features.load(Path(path))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Імпорт", f"Не вдалося прочитати файл:\n{e}")
            return
        logger.info(f"Map features imported: {n} from {path}")
        self._features_updated()


def _distance(a, b, mode: str) -> float:
    """Distance between two transport-convention points (metres or local units)."""
    return path_length([[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]], mode)
