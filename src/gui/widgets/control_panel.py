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
    rebuild_database_clicked = pyqtSignal()
    start_tracking_clicked   = pyqtSignal()
    stop_tracking_clicked    = pyqtSignal()
    calibrate_clicked        = pyqtSignal()
    load_calibration_clicked = pyqtSignal()
    localize_image_clicked   = pyqtSignal()
    generate_panorama_clicked = pyqtSignal()
    show_panorama_clicked    = pyqtSignal()
    export_results_clicked   = pyqtSignal()
    verify_propagation_clicked = pyqtSignal()
    clear_map_clicked        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.set_tracking_enabled(True)   # correct initial state on startup

    # ── UI ───────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Project group
        db_group  = QGroupBox("Управління проєктом")
        db_layout = QVBoxLayout(db_group)

        self.btn_new_mission = QPushButton("Створити новий проєкт")
        self.btn_load_db     = QPushButton("Відкрити проєкт")
        self.btn_rebuild_db  = QPushButton("🔄 Перегенерувати базу")
        self.btn_rebuild_db.setToolTip("Перебудовує базу даних з оригінального відео проєкту")
        self.btn_rebuild_db.setEnabled(False)
        self.btn_gen_pano    = QPushButton("Згенерувати панораму з відео")
        self.btn_show_pano   = QPushButton("Накласти панораму на карту")

        self.btn_new_mission.clicked.connect(self.new_mission_clicked)
        self.btn_load_db.clicked.connect(self.load_database_clicked)
        self.btn_rebuild_db.clicked.connect(self.rebuild_database_clicked)
        self.btn_gen_pano.clicked.connect(self.generate_panorama_clicked)
        self.btn_show_pano.clicked.connect(self.show_panorama_clicked)

        for btn in [self.btn_new_mission, self.btn_load_db, self.btn_rebuild_db,
                    self.btn_gen_pano, self.btn_show_pano]:
            db_layout.addWidget(btn)

        # Calibration group
        calib_group  = QGroupBox("Калібрування GPS")
        calib_layout = QVBoxLayout(calib_group)

        self.btn_calibrate       = QPushButton("Виконати калібрування (Video → Map)")
        self.btn_load_calibrate  = QPushButton("Завантажити калібрування (JSON)")
        self.btn_verify_propagation = QPushButton("🔍 Перевірити пропагацію на карті")
        self.btn_verify_propagation.setToolTip("Відображає центри всіх кадрів з обчисленими координатами на карті")
        self.btn_clear_map = QPushButton("🗑 Очистити карту")
        self.btn_clear_map.setToolTip("Видалити траєкторію, панораму та маркери з карти")

        self.btn_calibrate.clicked.connect(self.calibrate_clicked)
        self.btn_load_calibrate.clicked.connect(self.load_calibration_clicked)
        self.btn_verify_propagation.clicked.connect(self.verify_propagation_clicked)
        self.btn_clear_map.clicked.connect(self.clear_map_clicked)

        calib_layout.addWidget(self.btn_calibrate)
        calib_layout.addWidget(self.btn_load_calibrate)
        calib_layout.addWidget(self.btn_verify_propagation)
        calib_layout.addWidget(self.btn_clear_map)

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

        # Export group
        export_group = QGroupBox("Результати")
        export_layout = QVBoxLayout(export_group)
        self.btn_export = QPushButton("📊 Експорт результатів")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results_clicked)
        export_layout.addWidget(self.btn_export)

        # Project info group
        self.info_group = QGroupBox("Інформація про проєкт")
        info_layout = QVBoxLayout(self.info_group)
        self.lbl_project_info = QLabel("Проєкт не завантажено")
        self.lbl_project_info.setWordWrap(True)
        self.lbl_project_info.setStyleSheet("font-size: 11px; color: #888;")
        info_layout.addWidget(self.lbl_project_info)

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

        for group in [db_group, calib_group, track_group, export_group, self.info_group, status_group]:
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
        for btn in [self.btn_new_mission, self.btn_load_db, self.btn_rebuild_db,
                    self.btn_calibrate, self.btn_load_calibrate, self.btn_verify_propagation,
                    self.btn_clear_map,
                    self.btn_localize_image, self.btn_gen_pano, self.btn_export]:
            btn.setEnabled(enabled)

    def update_project_info(self, project_name: str = None, video_path: str = None,
                             num_frames: int = None, num_anchors: int = None,
                             num_propagated: int = None, db_size_mb: float = None):
        """Оновити інформаційну панель проєкту."""
        if project_name is None:
            self.lbl_project_info.setText("Проєкт не завантажено")
            self.lbl_project_info.setStyleSheet("font-size: 11px; color: #888;")
            self.btn_rebuild_db.setEnabled(False)
            return

        from pathlib import Path
        lines = [f"▶ <b>{project_name}</b>"]
        if video_path:
            lines.append(f"🎥 {Path(video_path).name}")
        if num_frames is not None:
            db_info = f"🗃 Кадрів: {num_frames}"
            if db_size_mb is not None:
                db_info += f" ({db_size_mb:.1f} MB)"
            lines.append(db_info)
        if num_anchors is not None:
            lines.append(f"⚓ Якорів: {num_anchors}")
        if num_propagated is not None and num_frames is not None:
            lines.append(f"📍 GPS: {num_propagated}/{num_frames} кадрів")

        self.lbl_project_info.setText("<br>".join(lines))
        self.lbl_project_info.setStyleSheet("font-size: 11px; color: #ddd;")
        self.btn_rebuild_db.setEnabled(True)
