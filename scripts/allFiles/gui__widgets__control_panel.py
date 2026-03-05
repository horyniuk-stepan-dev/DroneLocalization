from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QProgressBar, QGroupBox,
)
from PyQt6.QtCore import pyqtSignal, Qt

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ControlPanel(QWidget):
    """Mission control sidebar — emits signals, holds no business logic."""

    new_mission_clicked      = pyqtSignal()
    load_database_clicked    = pyqtSignal()
    start_tracking_clicked   = pyqtSignal()
    stop_tracking_clicked    = pyqtSignal()
    calibrate_clicked        = pyqtSignal()
    load_calibration_clicked = pyqtSignal()
    localize_image_clicked   = pyqtSignal()
    generate_panorama_clicked = pyqtSignal()
    show_panorama_clicked    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.set_tracking_enabled(True)   # correct initial state on startup

    # ── UI ───────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Database group
        db_group  = QGroupBox("Управління базою даних")
        db_layout = QVBoxLayout(db_group)

        self.btn_new_mission = QPushButton("Створити нову базу (Video → HDF5)")
        self.btn_load_db     = QPushButton("Завантажити існуючу базу (HDF5)")
        self.btn_gen_pano    = QPushButton("Згенерувати панораму з відео")
        self.btn_show_pano   = QPushButton("Накласти панораму на карту")

        self.btn_new_mission.clicked.connect(self.new_mission_clicked)
        self.btn_load_db.clicked.connect(self.load_database_clicked)
        self.btn_gen_pano.clicked.connect(self.generate_panorama_clicked)
        self.btn_show_pano.clicked.connect(self.show_panorama_clicked)

        for btn in [self.btn_new_mission, self.btn_load_db,
                    self.btn_gen_pano, self.btn_show_pano]:
            db_layout.addWidget(btn)

        # Calibration group
        calib_group  = QGroupBox("Калібрування GPS")
        calib_layout = QVBoxLayout(calib_group)

        self.btn_calibrate       = QPushButton("Виконати калібрування (Video → Map)")
        self.btn_load_calibrate  = QPushButton("Завантажити калібрування (JSON)")

        self.btn_calibrate.clicked.connect(self.calibrate_clicked)
        self.btn_load_calibrate.clicked.connect(self.load_calibration_clicked)

        calib_layout.addWidget(self.btn_calibrate)
        calib_layout.addWidget(self.btn_load_calibrate)

        # Localization group
        track_group  = QGroupBox("Локалізація")
        track_layout = QVBoxLayout(track_group)

        self.btn_start_tracking = QPushButton("▶  Почати відстеження")
        self.btn_start_tracking.setStyleSheet(
            "background:#2e7d32; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_stop_tracking = QPushButton("■  Зупинити відстеження")
        self.btn_stop_tracking.setStyleSheet(
            "background:#c62828; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_localize_image = QPushButton("🔍  Локалізувати одне фото")

        self.btn_start_tracking.clicked.connect(self.start_tracking_clicked)
        self.btn_stop_tracking.clicked.connect(self.stop_tracking_clicked)
        self.btn_localize_image.clicked.connect(self.localize_image_clicked)

        track_layout.addWidget(self.btn_start_tracking)
        track_layout.addWidget(self.btn_stop_tracking)
        track_layout.addWidget(self.btn_localize_image)

        # Status group
        status_group  = QGroupBox("Статус системи")
        status_layout = QVBoxLayout(status_group)

        self.lbl_status = QLabel("Очікування команди...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-style:italic; color:#a0a0a0; margin-bottom:6px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)

        for group in [db_group, calib_group, track_group, status_group]:
            layout.addWidget(group)

    # ── Public API ───────────────────────────────────────────────────────────

    def update_status(self, message: str):
        self.lbl_status.setText(message)
        logger.debug(f"Status: {message}")

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_tracking_enabled(self, enabled: bool):
        """
        enabled=True  → idle state   (Start active, Stop disabled)
        enabled=False → running state (Start disabled, Stop active)
        """
        self.btn_start_tracking.setEnabled(enabled)
        self.btn_stop_tracking.setEnabled(not enabled)

        # Disable DB/calibration ops during tracking to prevent GPU OOM
        for btn in [self.btn_new_mission, self.btn_load_db,
                    self.btn_calibrate, self.btn_load_calibrate,
                    self.btn_localize_image, self.btn_gen_pano]:
            btn.setEnabled(enabled)
