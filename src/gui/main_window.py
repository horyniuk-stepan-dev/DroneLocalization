import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QStatusBar

from config import APP_CONFIG, APP_SETTINGS, get_cfg
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.core.project import ProjectManager
from src.database.database_loader import DatabaseLoader
from src.gui.mixins import CalibrationMixin, DatabaseMixin, PanoramaMixin, TrackingMixin
from src.gui.widgets.control_panel import ControlPanel
from src.gui.widgets.map_widget import MapWidget
from src.gui.widgets.video_widget import VideoWidget
from src.models.model_manager import ModelManager
from src.network.coordinates_broker import CoordinatesBroker
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MainWindow(CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Topometric Localizer")
        self.setGeometry(100, 100, 1600, 900)

        self.config = APP_CONFIG
        self.model_manager = ModelManager(config=APP_CONFIG)
        self.project_manager = ProjectManager()
        self.database: DatabaseLoader | None = None
        self.calibration = MultiAnchorCalibration()

        self.coordinates_broker = CoordinatesBroker(config=APP_SETTINGS.network_api)

        # Workers
        self.db_worker = None
        self.tracking_worker = None
        self.propagation_worker = None
        self.pano_worker = None
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

        # ── Debug-вікна «очима моделей» (приховані за замовчуванням) ─────────
        from src.gui.widgets.debug_view import DebugViewDock

        self.debug_docks: dict = {}
        debug_specs = [
            ("yolo", "YOLO — детекції / маска"),
            ("depth", "Depth Anything"),
            ("dino", "DINO — PCA / retrieval"),
            ("matches", "Точки / матчі (ALIKED)"),
        ]
        prev_dock = None
        for channel, title in debug_specs:
            dock = DebugViewDock(channel, title, self)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            if prev_dock is not None:
                self.tabifyDockWidget(prev_dock, dock)
            dock.hide()
            dock.visibilityChanged.connect(self._update_debug_channels)
            self.debug_docks[channel] = dock
            prev_dock = dock
        self._restore_debug_visibility()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._create_menu_bar()
        self._connect_signals()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")
        file_menu.addAction("Налаштування...", self.on_open_config)
        file_menu.addSeparator()
        file_menu.addAction("Вихід", self.close)

        calib_menu = menubar.addMenu("Калібрування")
        calib_menu.addAction("Додати якір...", self.on_calibrate)
        calib_menu.addAction("Завантажити калібрування...", self.on_load_calibration)
        calib_menu.addAction("Зберегти калібрування...", self.on_save_calibration)
        calib_menu.addSeparator()
        calib_menu.addAction("Запустити пропагацію вручну", self.on_run_propagation)
        calib_menu.addAction("Перевірити пропагацію на карті", self.on_verify_propagation)

        view_menu = menubar.addMenu("Вигляд")
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())
        view_menu.addSeparator()

        debug_menu = view_menu.addMenu("Вікна моделей")
        for _channel in ("yolo", "depth", "dino", "matches"):
            debug_menu.addAction(self.debug_docks[_channel].toggleViewAction())

        sections_menu = view_menu.addMenu("Секції панелі управління")
        cp = self.control_panel
        sections = [
            ("Управління проєктом", cp.db_group),
            ("Калібрування GPS", cp.calib_group),
            ("Локалізація", cp.track_group),
            ("Результати", cp.export_group),
            ("Інформація про проєкт", cp.info_group),
            ("Статус системи", cp.status_group),
            ("Відеоджерела", cp.sources_group),
        ]

        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QWidget, QWidgetAction

        for name, widget in sections:
            action = QWidgetAction(sections_menu)
            container = QWidget()
            # Make background transparent to match menu styling
            container.setStyleSheet("background: transparent;")
            layout = QHBoxLayout(container)
            layout.setContentsMargins(30, 4, 10, 4)

            checkbox = QCheckBox(name)
            checkbox.setChecked(not widget.isHidden())
            # When checkbox is toggled, update widget visibility
            checkbox.toggled.connect(widget.setVisible)

            layout.addWidget(checkbox)
            action.setDefaultWidget(container)
            sections_menu.addAction(action)

    def _connect_signals(self):
        cp = self.control_panel
        cp.new_mission_clicked.connect(self.on_new_mission)
        cp.load_database_clicked.connect(self.on_load_database)
        cp.rebuild_database_clicked.connect(self.on_rebuild_database)
        cp.start_tracking_clicked.connect(self.on_start_tracking)
        cp.start_live_tracking_clicked.connect(self.on_start_live_tracking)
        cp.stop_tracking_clicked.connect(self.on_stop_tracking)
        cp.calibrate_clicked.connect(self.on_calibrate)
        cp.load_calibration_clicked.connect(self.on_load_calibration)
        cp.generate_panorama_clicked.connect(self.on_generate_panorama)
        cp.show_panorama_clicked.connect(self.on_show_panorama)
        cp.localize_image_clicked.connect(self.on_localize_image)
        cp.verify_propagation_clicked.connect(self.on_verify_propagation)
        cp.clear_map_clicked.connect(self.map_widget.clear_trajectory)
        cp.export_results_clicked.connect(self.on_export_results)
        cp.toggle_objects_clicked.connect(self.on_toggle_objects)
        cp.add_source_clicked.connect(self.on_add_video_source)
        cp.source_action.connect(self.on_source_action)
        cp.active_source_changed.connect(self.on_active_source_changed)
        self.map_widget.mapClicked.connect(self._on_map_clicked)

    def _on_map_clicked(self, lat: float, lon: float):
        """Handle map click by showing coordinates in the status bar."""
        msg = f"Координати на карті: Lat {lat:.6f}, Lon {lon:.6f}"
        self.status_bar.showMessage(msg, 5000)  # Show for 5 seconds
        logger.info(f"Map click: {lat=}, {lon=}")

    def on_open_config(self):
        """Open the configuration editor dialog."""
        from src.gui.dialogs.config_dialog import ConfigDialog
        dialog = ConfigDialog(self)
        dialog.exec()

    # ── Debug-вікна «очима моделей» ─────────────────────────────────────────
    def _restore_debug_visibility(self):
        """Відновлює видимість debug-вікон із user_config.json (секція debug_views)."""
        mapping = {
            "yolo": "debug_views.show_yolo",
            "depth": "debug_views.show_depth",
            "dino": "debug_views.show_dino",
            "matches": "debug_views.show_matches",
        }
        for channel, path in mapping.items():
            if get_cfg(self.config, path, False):
                self.debug_docks[channel].show()

    def _active_debug_channels(self) -> set:
        """Набір каналів, чиї вікна зараз реально видимі користувачу."""
        # «Активний» = вікно відкрите (навіть якщо зараз за іншою вкладкою в tab-групі),
        # а не лише фронтальна вкладка. isVisible() дає False для tabbed-behind дока —
        # через це depth-вікно «замерзало», коли фронтальною ставала інша вкладка.
        return {ch for ch, dock in self.debug_docks.items() if not dock.isHidden()}

    def _update_debug_channels(self, *args):
        """visibilityChanged будь-якого дока → оновити активний набір у worker-і."""
        worker = getattr(self, "tracking_worker", None)
        if worker is not None and worker.isRunning():
            worker.set_debug_channels(self._active_debug_channels())

    @pyqtSlot(str, np.ndarray)
    def _on_debug_view_ready(self, channel: str, frame_bgr: np.ndarray):
        """Маршрутизація готового BGR-кадру у відповідне debug-вікно.

        Після показу підтверджуємо worker-у, що кадр спожито (backpressure
        drop-замість-черги): знімає in-flight, дозволяючи наступний кадр.
        """
        try:
            dock = self.debug_docks.get(channel)
            if dock is not None:
                dock.update_frame(frame_bgr)
        finally:
            worker = getattr(self, "tracking_worker", None)
            if worker is not None:
                worker.mark_debug_channel_free(channel)

    def closeEvent(self, event):
        """Зберігає стан видимості debug-вікон, не чіпаючи інші налаштування."""
        try:
            from config import load_user_config, save_user_config

            cfg = load_user_config()
            cfg.debug_views.show_yolo = not self.debug_docks["yolo"].isHidden()
            cfg.debug_views.show_depth = not self.debug_docks["depth"].isHidden()
            cfg.debug_views.show_dino = not self.debug_docks["dino"].isHidden()
            cfg.debug_views.show_matches = not self.debug_docks["matches"].isHidden()
            save_user_config(cfg)
        except Exception as e:
            logger.debug(f"Failed to persist debug view visibility: {e}")
        super().closeEvent(event)
