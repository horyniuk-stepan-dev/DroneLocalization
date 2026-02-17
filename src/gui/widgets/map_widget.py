from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl
from PyQt6.QtWebEngineCore import QWebEngineSettings
from src.utils.logging_utils import get_logger

class MapBridge(QObject):
    updateMarkerSignal = pyqtSignal(float, float)
    addTrajectoryPointSignal = pyqtSignal(float, float)
    clearTrajectorySignal = pyqtSignal()
    updateFOVSignal = pyqtSignal(float, float, float, float, float, float, float, float)
    setPanoramaSignal = pyqtSignal(str, float, float, float, float, float, float, float, float)
    def __init__(self):
        super().__init__()
        self.logger = get_logger('MapBridge')


class MapWidget(QWebEngineView):
    marker_clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.bridge = MapBridge()
        self.setup_web_channel()
        self.load_map()

    def setup_web_channel(self):
        self.channel = QWebChannel()
        self.channel.registerObject("mapBridge", self.bridge)
        self.page().setWebChannel(self.channel)

    def load_map(self):
        from pathlib import Path

        # Піднімаємось на 2 рівні вгору: map_widget.py -> widgets -> gui
        gui_folder = Path(__file__).resolve().parent.parent

        # Будуємо шлях: src/gui/resources/maps/map_template.html
        map_path = gui_folder / "resources" / "maps" / "map_template.html"

        if map_path.exists():
            # Завантажуємо чистий шлях без жодних ?nocache параметрів!
            url = QUrl.fromLocalFile(str(map_path))
            self.setUrl(url)
            self.bridge.logger.info(f"Карту завантажено з: {map_path}")
        else:
            error_html = f"""
            <html>
            <body style='font-family: Arial; padding: 20px;'>
                <h2 style='color: red;'>Помилка: Файл карти не знайдено!</h2>
                <p>Програма шукає HTML-файл за такою адресою:</p>
                <b style='background: #eee; padding: 5px;'>{map_path}</b>
            </body>
            </html>
            """
            self.setHtml(error_html)
            self.bridge.logger.error(f"Файл карти не знайдено за шляхом: {map_path}")


    @pyqtSlot(float, float)
    def update_marker(self, lat, lon):
        self.bridge.updateMarkerSignal.emit(lat, lon)

    @pyqtSlot(float, float)
    def add_trajectory_point(self, lat, lon):
        self.bridge.addTrajectoryPointSignal.emit(lat, lon)

    @pyqtSlot()
    def clear_trajectory(self):
        self.bridge.clearTrajectorySignal.emit()

    # НОВИЙ МЕТОД ДЛЯ ПРЯМОКУТНИКА
    @pyqtSlot(list)
    def update_fov(self, fov_points):
        if len(fov_points) == 4:
            self.bridge.updateFOVSignal.emit(
                fov_points[0][0], fov_points[0][1],
                fov_points[1][0], fov_points[1][1],
                fov_points[2][0], fov_points[2][1],
                fov_points[3][0], fov_points[3][1]
            )

    @pyqtSlot(str, float, float, float, float, float, float, float, float)
    def set_panorama_overlay(self, data_url, lat_tl, lon_tl, lat_tr, lon_tr, lat_br, lon_br, lat_bl, lon_bl):
        self.bridge.setPanoramaSignal.emit(
            data_url,
            float(lat_tl), float(lon_tl),
            float(lat_tr), float(lon_tr),
            float(lat_br), float(lon_br),
            float(lat_bl), float(lon_bl)
        )