from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ControlPanel(QWidget):
    """Mission control sidebar — emits signals, holds no business logic."""

    new_mission_clicked = pyqtSignal()
    load_database_clicked = pyqtSignal()
    rebuild_database_clicked = pyqtSignal()
    start_tracking_clicked = pyqtSignal()
    start_live_tracking_clicked = pyqtSignal()
    stop_tracking_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    load_calibration_clicked = pyqtSignal()
    localize_image_clicked = pyqtSignal()
    generate_panorama_clicked = pyqtSignal()
    show_panorama_clicked = pyqtSignal()
    export_results_clicked = pyqtSignal()
    verify_propagation_clicked = pyqtSignal()
    clear_map_clicked = pyqtSignal()
    stop_db_generation_clicked = pyqtSignal()
    toggle_objects_clicked = pyqtSignal(bool)
    add_source_clicked = pyqtSignal()
    source_action = pyqtSignal(
        str, str
    )  # (source_id, action: "build_db"/"calibrate"/"toggle"/"remove")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.set_tracking_enabled(True)  # correct initial state on startup

    # ── UI ───────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Project group
        self.db_group = QGroupBox("Управління проєктом")
        db_layout = QVBoxLayout(self.db_group)

        self.btn_new_mission = QPushButton("Створити новий проєкт")
        self.btn_load_db = QPushButton("Відкрити проєкт")
        self.btn_rebuild_db = QPushButton("🔄 Перегенерувати базу")
        self.btn_rebuild_db.setToolTip("Перебудовує базу даних з оригінального відео проєкту")
        self.btn_rebuild_db.setEnabled(False)
        self.btn_gen_pano = QPushButton("Згенерувати панораму з відео")
        self.btn_show_pano = QPushButton("Накласти панораму на карту")

        self.btn_new_mission.clicked.connect(self.new_mission_clicked)
        self.btn_load_db.clicked.connect(self.load_database_clicked)
        self.btn_rebuild_db.clicked.connect(self.rebuild_database_clicked)
        self.btn_gen_pano.clicked.connect(self.generate_panorama_clicked)
        self.btn_show_pano.clicked.connect(self.show_panorama_clicked)

        self.btn_stop_db = QPushButton("⏹  Зупинити генерацію БД")
        self.btn_stop_db.setStyleSheet(
            "background:#c62828; color:white; font-weight:bold; padding:7px;"
        )
        self.btn_stop_db.setVisible(False)
        self.btn_stop_db.clicked.connect(self.stop_db_generation_clicked)

        for btn in [
            self.btn_new_mission,
            self.btn_load_db,
            self.btn_rebuild_db,
            self.btn_gen_pano,
            self.btn_show_pano,
            self.btn_stop_db,
        ]:
            db_layout.addWidget(btn)

        # Calibration group
        self.calib_group = QGroupBox("Калібрування GPS")
        calib_layout = QVBoxLayout(self.calib_group)

        self.btn_calibrate = QPushButton("Виконати калібрування (Video → Map)")
        self.btn_load_calibrate = QPushButton("Завантажити калібрування (JSON)")
        self.btn_verify_propagation = QPushButton("🔍 Перевірити пропагацію на карті")
        self.btn_verify_propagation.setToolTip(
            "Відображає центри всіх кадрів з обчисленими координатами на карті"
        )
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
        self.track_group = QGroupBox("Локалізація")
        track_layout = QVBoxLayout(self.track_group)

        self.btn_start_tracking = QPushButton("▶  Почати відстеження (Файл)")
        self.btn_start_tracking.setStyleSheet(
            "background:#2e7d32; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_start_live = QPushButton("📡  Живий потік (RTSP/USB)")
        self.btn_start_live.setStyleSheet(
            "background:#0277bd; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_stop_tracking = QPushButton("■  Зупинити відстеження")
        self.btn_stop_tracking.setStyleSheet(
            "background:#c62828; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_localize_image = QPushButton("🔍  Локалізувати одне фото")

        self.btn_toggle_objects = QPushButton("👀 Показувати об'єкти")
        self.btn_toggle_objects.setCheckable(True)
        self.btn_toggle_objects.setChecked(True)

        self.btn_start_tracking.clicked.connect(self.start_tracking_clicked)
        self.btn_start_live.clicked.connect(self.start_live_tracking_clicked)
        self.btn_stop_tracking.clicked.connect(self.stop_tracking_clicked)
        self.btn_localize_image.clicked.connect(self.localize_image_clicked)
        self.btn_toggle_objects.toggled.connect(self.toggle_objects_clicked)

        track_layout.addWidget(self.btn_start_tracking)
        track_layout.addWidget(self.btn_start_live)
        track_layout.addWidget(self.btn_stop_tracking)
        track_layout.addWidget(self.btn_localize_image)
        track_layout.addWidget(self.btn_toggle_objects)

        # Export group
        self.export_group = QGroupBox("Результати")
        export_layout = QVBoxLayout(self.export_group)
        self.btn_export = QPushButton("📊 Експорт результатів")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results_clicked)
        export_layout.addWidget(self.btn_export)

        # Project info group
        self.info_group = QGroupBox("Інформація про проєкт")
        info_layout = QVBoxLayout(self.info_group)
        self.lbl_project_info = QLabel("Проєкт не завантажено")
        self.lbl_project_info.setWordWrap(True)
        self.lbl_project_info.setStyleSheet("font-size: 11px; color: #333;")
        info_layout.addWidget(self.lbl_project_info)

        # Status group
        self.status_group = QGroupBox("Статус системи")
        status_layout = QVBoxLayout(self.status_group)

        self.lbl_status = QLabel("Очікування команди...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-style:italic; color:#333; margin-bottom:6px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)

        # Video Sources group (мультиджерельна підтримка)
        self.sources_group = QGroupBox("Відеоджерела")
        sources_layout = QVBoxLayout(self.sources_group)

        self.sources_table = QTableWidget()
        self.sources_table.setColumnCount(3)
        self.sources_table.setHorizontalHeaderLabels(["Source ID", "Area", "Статус"])
        self.sources_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.sources_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.sources_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.sources_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sources_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sources_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sources_table.setMaximumHeight(120)
        self.sources_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources_table.customContextMenuRequested.connect(self._on_sources_context_menu)
        self.sources_table.setVisible(False)  # Прихований до відкриття мульти-проєкту
        sources_layout.addWidget(self.sources_table)

        btn_row = QHBoxLayout()
        self.btn_add_source = QPushButton("➕ Додати джерело")
        self.btn_add_source.setToolTip("Додати нове відеоджерело до проєкту")
        self.btn_add_source.clicked.connect(self.add_source_clicked)
        self.btn_add_source.setVisible(False)
        btn_row.addWidget(self.btn_add_source)
        sources_layout.addLayout(btn_row)

        self.sources_group.setVisible(False)  # Показується тільки для мульти-проєктів

        for group in [
            self.db_group,
            self.calib_group,
            self.track_group,
            self.export_group,
            self.sources_group,
            self.info_group,
            self.status_group,
        ]:
            layout.addWidget(group)

    # ── Public API ───────────────────────────────────────────────────────────

    def update_status(self, message: str):
        self.lbl_status.setText(message)
        logger.debug(f"Status: {message}")

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_db_generation_running(self, is_running: bool):
        """is_running=True: показати кнопку Stop, заблокувати решту кнопок проєкту."""
        self.btn_stop_db.setVisible(is_running)
        self.btn_new_mission.setEnabled(not is_running)
        self.btn_load_db.setEnabled(not is_running)
        self.btn_rebuild_db.setEnabled(not is_running)

    def set_tracking_enabled(self, enabled: bool):
        """
        enabled=True  → idle state   (Start active, Stop disabled)
        enabled=False → running state (Start disabled, Stop active)
        """
        self.btn_start_tracking.setEnabled(enabled)
        self.btn_start_live.setEnabled(enabled)
        self.btn_stop_tracking.setEnabled(not enabled)

        # Disable DB/calibration ops during tracking to prevent GPU OOM
        for btn in [
            self.btn_new_mission,
            self.btn_load_db,
            self.btn_rebuild_db,
            self.btn_calibrate,
            self.btn_load_calibrate,
            self.btn_verify_propagation,
            self.btn_clear_map,
            self.btn_localize_image,
            self.btn_gen_pano,
            self.btn_export,
        ]:
            btn.setEnabled(enabled)

    def update_project_info(
        self,
        project_name: str = None,
        video_path: str = None,
        num_frames: int = None,
        num_anchors: int = None,
        num_propagated: int = None,
        db_size_mb: float = None,
    ):
        """Оновити інформаційну панель проєкту."""
        if project_name is None:
            self.lbl_project_info.setText("Проєкт не завантажено")
            self.lbl_project_info.setStyleSheet("font-size: 11px; color: #222;")
            self.btn_rebuild_db.setEnabled(False)
            return

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
        self.lbl_project_info.setStyleSheet("font-size: 11px; color: #000;")
        self.btn_rebuild_db.setEnabled(True)

    # ── Video Sources Panel ──────────────────────────────────────────────────

    def update_sources_list(
        self,
        sources: list[dict],
        project_dir: str = "",
    ):
        """Оновлює таблицю відеоджерел.

        Args:
            sources: Список dict з полями source_id, area_id, enabled,
                     database_file, calibration_file.
            project_dir: Шлях до кореня проєкту для перевірки файлів.
        """
        is_multi = len(sources) > 1 or any(s.get("source_id") != "main" for s in sources)
        # Група завжди видима при відкритому проєкті, таблиця — тільки для multi
        self.sources_group.setVisible(True)
        self.sources_table.setVisible(is_multi)
        self.btn_add_source.setVisible(True)

        if not is_multi:
            return

        self.sources_table.setRowCount(len(sources))
        for row, src in enumerate(sources):
            sid = src.get("source_id", "?")
            area = src.get("area_id", "?")
            enabled = src.get("enabled", True)

            # Визначаємо статус
            status = "⏳ Очікує"
            status_color = QColor("#888")
            if project_dir:
                db_exists = (Path(project_dir) / src.get("database_file", "")).exists()
                cal_exists = (Path(project_dir) / src.get("calibration_file", "")).exists()
                if db_exists and cal_exists:
                    status = "✅ Готово"
                    status_color = QColor("#2e7d32")
                elif db_exists:
                    status = "⚠ Без калібр."
                    status_color = QColor("#e65100")
                else:
                    status = "❌ Без БД"
                    status_color = QColor("#c62828")

            if not enabled:
                status = "🔇 Вимкнено"
                status_color = QColor("#999")

            item_sid = QTableWidgetItem(sid)
            item_area = QTableWidgetItem(area)
            item_status = QTableWidgetItem(status)
            item_status.setForeground(status_color)

            # Зберігаємо source_id як data для context menu
            item_sid.setData(Qt.ItemDataRole.UserRole, sid)

            self.sources_table.setItem(row, 0, item_sid)
            self.sources_table.setItem(row, 1, item_area)
            self.sources_table.setItem(row, 2, item_status)

    @pyqtSlot("QPoint")
    def _on_sources_context_menu(self, pos):
        """Контекстне меню для таблиці джерел."""
        row = self.sources_table.rowAt(pos.y())
        if row < 0:
            return

        item = self.sources_table.item(row, 0)
        if item is None:
            return
        source_id = item.data(Qt.ItemDataRole.UserRole)
        if not source_id:
            return

        menu = QMenu(self)
        act_build = menu.addAction("🔨 Побудувати базу даних")
        act_calib = menu.addAction("📐 Калібрувати")
        menu.addSeparator()
        act_toggle = menu.addAction("🔇 Увімкнути/Вимкнути")
        menu.addSeparator()
        act_remove = menu.addAction("🗑 Видалити")

        action = menu.exec(self.sources_table.viewport().mapToGlobal(pos))
        if action == act_build:
            self.source_action.emit(source_id, "build_db")
        elif action == act_calib:
            self.source_action.emit(source_id, "calibrate")
        elif action == act_toggle:
            self.source_action.emit(source_id, "toggle")
        elif action == act_remove:
            self.source_action.emit(source_id, "remove")
