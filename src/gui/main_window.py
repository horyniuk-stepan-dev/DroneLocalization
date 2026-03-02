from PyQt6.QtWidgets import QMainWindow, QDockWidget, QStatusBar
from PyQt6.QtCore import Qt

from src.gui.widgets.video_widget import VideoWidget
from src.gui.widgets.map_widget import MapWidget
from src.gui.widgets.control_panel import ControlPanel
from src.models.model_manager import ModelManager
from src.database.database_loader import DatabaseLoader
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.utils.logging_utils import get_logger
from src.gui.mixins import CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin
from config.config import APP_CONFIG


class MainWindow(CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin, QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Topometric Localizer")
        self.setGeometry(100, 100, 1600, 900)
        self.logger = get_logger('MainWindow')

        self.config                = APP_CONFIG
        self.model_manager         = ModelManager(config=APP_CONFIG)
        self.database: DatabaseLoader | None = None
        self.calibration           = MultiAnchorCalibration()
        self.current_database_path = None

        # Workers
        self.db_worker           = None
        self.tracking_worker     = None
        self.propagation_worker  = None
        self.pano_worker         = None
        self._propagation_dialog = None

        self._init_ui()

    def _init_ui(self):
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        self.control_dock = QDockWidget("Панель управління", self)
        self.control_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.control_panel = ControlPanel(self.control_dock)
        self.control_dock.setWidget(self.control_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)

        self.map_dock = QDockWidget("Інтерактивна карта", self)
        self.map_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.map_widget = MapWidget(self.map_dock)
        self.map_dock.setWidget(self.map_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.map_dock)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._create_menu_bar()
        self._connect_signals()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('Файл')
        file_menu.addAction('Вихід', self.close)

        calib_menu = menubar.addMenu('Калібрування')
        calib_menu.addAction('Додати якір...', self.on_calibrate)
        calib_menu.addAction('Завантажити калібрування...', self.on_load_calibration)
        calib_menu.addAction('Зберегти калібрування...', self.on_save_calibration)
        calib_menu.addSeparator()
        calib_menu.addAction('Запустити пропагацію вручну', self.on_run_propagation)

        view_menu = menubar.addMenu('Вигляд')
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())

    def _connect_signals(self):
        cp = self.control_panel
        cp.new_mission_clicked.connect(self.on_new_mission)
        cp.load_database_clicked.connect(self.on_load_database)
        cp.start_tracking_clicked.connect(self.on_start_tracking)
        cp.stop_tracking_clicked.connect(self.on_stop_tracking)
        cp.calibrate_clicked.connect(self.on_calibrate)
        cp.load_calibration_clicked.connect(self.on_load_calibration)
        cp.generate_panorama_clicked.connect(self.on_generate_panorama)
        cp.show_panorama_clicked.connect(self.on_show_panorama)
        cp.localize_image_clicked.connect(self.on_localize_image)
