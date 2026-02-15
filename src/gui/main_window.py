from PyQt6.QtWidgets import QMainWindow, QDockWidget, QStatusBar
from PyQt6.QtCore import Qt, pyqtSlot

from src.gui.widgets.video_widget import VideoWidget
from src.gui.widgets.map_widget import MapWidget
from src.gui.widgets.control_panel import ControlPanel
from src.gui.dialogs.new_mission_dialog import NewMissionDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Topometric Localizer")
        self.setGeometry(100, 100, 1600, 900)
        self.init_ui()

    def init_ui(self):
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

        self.create_menu_bar()
        self.connect_signals()

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('Файл')
        file_menu.addAction('Вихід', self.close)

        mission_menu = menubar.addMenu('Місія')

        view_menu = menubar.addMenu('Вигляд')
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())

    def connect_signals(self):
        self.control_panel.new_mission_clicked.connect(self.on_new_mission)
        self.control_panel.load_database_clicked.connect(self.on_load_database)
        self.control_panel.start_tracking_clicked.connect(self.on_start_tracking)
        self.control_panel.stop_tracking_clicked.connect(self.on_stop_tracking)
        self.control_panel.calibrate_clicked.connect(self.on_calibrate)

    @pyqtSlot()
    def on_new_mission(self):
        dialog = NewMissionDialog(self)
        if dialog.exec():
            mission_data = dialog.get_mission_data()
            self.status_bar.showMessage(f"Створено місію: {mission_data['mission_name']}")
        else:
            self.status_bar.showMessage("Створення місії скасовано")

    @pyqtSlot()
    def on_load_database(self):
        self.status_bar.showMessage("Завантаження бази даних...")

    @pyqtSlot()
    def on_start_tracking(self):
        self.status_bar.showMessage("Відстеження розпочато")

    @pyqtSlot()
    def on_stop_tracking(self):
        self.status_bar.showMessage("Відстеження зупинено")

    @pyqtSlot()
    def on_calibrate(self):
        self.status_bar.showMessage("Відкрито діалог калібрування")