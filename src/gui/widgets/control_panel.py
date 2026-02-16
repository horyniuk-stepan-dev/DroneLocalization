from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QProgressBar, QGroupBox)
from PyQt6.QtCore import pyqtSignal, Qt

from utils.logging_utils import get_logger


class ControlPanel(QWidget):
    """Панель управління місією та відстеженням"""

    new_mission_clicked = pyqtSignal()
    load_database_clicked = pyqtSignal()
    start_tracking_clicked = pyqtSignal()
    stop_tracking_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    localize_image_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger('ControlPanel')
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        mission_group = QGroupBox("Управління базою даних")
        mission_layout = QVBoxLayout()

        self.btn_new_mission = QPushButton("Створити нову базу (Video -> HDF5)")
        self.btn_new_mission.clicked.connect(self.new_mission_clicked.emit)

        self.btn_load_db = QPushButton("Завантажити існуючу базу (HDF5)")
        self.btn_load_db.clicked.connect(self.load_database_clicked.emit)

        mission_layout.addWidget(self.btn_new_mission)
        mission_layout.addWidget(self.btn_load_db)
        mission_group.setLayout(mission_layout)
        layout.addWidget(mission_group)

        calib_group = QGroupBox("Калібрування GPS")
        calib_layout = QVBoxLayout()

        self.btn_calibrate = QPushButton("Виконати калібрування (Video -> Map)")
        self.btn_calibrate.clicked.connect(self.calibrate_clicked.emit)

        calib_layout.addWidget(self.btn_calibrate)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        track_group = QGroupBox("Локалізація")
        track_layout = QVBoxLayout()

        self.btn_start_tracking = QPushButton("Почати відстеження")
        self.btn_start_tracking.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;")
        self.btn_start_tracking.clicked.connect(self.start_tracking_clicked.emit)

        self.btn_stop_tracking = QPushButton("Зупинити відстеження")
        self.btn_stop_tracking.setStyleSheet(
            "background-color: #c62828; color: white; font-weight: bold; padding: 8px;")
        self.btn_stop_tracking.clicked.connect(self.stop_tracking_clicked.emit)

        self.btn_localize_image = QPushButton("Локалізувати одне фото")
        self.btn_localize_image.clicked.connect(self.localize_image_clicked.emit)

        # 2. ДОДАЄМО всі три кнопки у макет
        track_layout.addWidget(self.btn_start_tracking)
        track_layout.addWidget(self.btn_stop_tracking)
        track_layout.addWidget(self.btn_localize_image)  # <--- Цього рядка не вистачало!

        # 3. Застосовуємо макет до групи
        track_group.setLayout(track_layout)
        layout.addWidget(track_group)


        status_group = QGroupBox("Статус системи")
        status_layout = QVBoxLayout()

        self.lbl_status = QLabel("Очікування команди...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-style: italic; color: #a0a0a0; margin-bottom: 10px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)

        layout.addWidget(status_group)
        self.setLayout(layout)

    def update_status(self, message: str):
        self.lbl_status.setText(message)

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_tracking_enabled(self, enabled: bool):
        self.btn_start_tracking.setEnabled(enabled)
        self.btn_stop_tracking.setEnabled(not enabled)
        self.btn_new_mission.setEnabled(enabled)
        self.btn_load_db.setEnabled(enabled)
        self.btn_calibrate.setEnabled(enabled)