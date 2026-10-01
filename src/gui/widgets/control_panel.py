from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
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

# Status colour per LayerState value (see src/core/layer_status.py).
_LAYER_STATE_COLORS = {
    "ready": "#2e7d32",
    "not_propagated": "#9e6a00",
    "stale": "#9e6a00",
    "no_calibration": "#e65100",
    "no_db": "#c62828",
    "db_not_loaded": "#c62828",
    "disabled": "#999999",
}


def _short(text: str, limit: int = 18) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


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
    active_source_changed = pyqtSignal(str)
    # (source_id, action): "activate" / "calibrate" / "load_calibration" /
    # "propagate" / "build_db" / "toggle" / "remove"
    source_action = pyqtSignal(str, str)

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

        # Layers group: one layer = one video source with its own DB + calibration
        self.sources_group = QGroupBox("Шари (відеоджерела)")
        sources_layout = QVBoxLayout(self.sources_group)
        sources_layout.setSpacing(6)

        # ── Active source badge (always visible when a project is open) ──
        self._active_badge = QFrame()
        self._active_badge.setFrameShape(QFrame.Shape.StyledPanel)
        self._active_badge.setStyleSheet(
            "QFrame { background: #f5f5f5; border: 1px solid #ddd; border-radius: 5px; }"
        )
        badge_layout = QHBoxLayout(self._active_badge)
        badge_layout.setContentsMargins(8, 5, 8, 5)
        badge_layout.setSpacing(6)

        self._lbl_source_dot = QLabel("●")
        self._lbl_source_dot.setStyleSheet("color: #bbb; font-size: 13px;")
        self._lbl_source_dot.setFixedWidth(16)

        self._lbl_source_id = QLabel("Не завантажено")
        bold = QFont()
        bold.setBold(True)
        bold.setPointSize(10)
        self._lbl_source_id.setFont(bold)
        self._lbl_source_id.setStyleSheet("color: #333;")

        self._lbl_source_type = QLabel()
        self._lbl_source_type.setStyleSheet(
            "color: #777; font-size: 9px; background: #e0e0e0; "
            "border-radius: 3px; padding: 1px 4px;"
        )

        badge_layout.addWidget(self._lbl_source_dot)
        badge_layout.addWidget(self._lbl_source_id, 1)
        badge_layout.addWidget(self._lbl_source_type)
        sources_layout.addWidget(self._active_badge)

        self._lbl_source_video = QLabel()
        self._lbl_source_video.setStyleSheet("font-size: 10px; color: #555; padding-left: 4px;")
        self._lbl_source_video.setWordWrap(True)
        sources_layout.addWidget(self._lbl_source_video)

        # ── Layer table (always shown while a project is open) ──
        self.sources_table = QTableWidget()
        self.sources_table.setColumnCount(5)
        self.sources_table.setHorizontalHeaderLabels(["Шар", "Зона", "Якорі", "GPS", "Статус"])
        header = self.sources_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.sources_table.verticalHeader().setVisible(False)
        self.sources_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sources_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sources_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sources_table.setMinimumHeight(90)
        self.sources_table.setMaximumHeight(200)
        self.sources_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources_table.customContextMenuRequested.connect(self._on_sources_context_menu)
        self.sources_table.itemSelectionChanged.connect(self._on_sources_selection_changed)
        self.sources_table.itemDoubleClicked.connect(self._on_sources_double_clicked)
        self.sources_table.setVisible(False)
        sources_layout.addWidget(self.sources_table)

        self._lbl_sources_hint = QLabel(
            "Клік — зробити шар активним · подвійний клік — калібрувати · ПКМ — дії"
        )
        self._lbl_sources_hint.setWordWrap(True)
        self._lbl_sources_hint.setStyleSheet("font-size: 10px; color: #777; padding-left: 2px;")
        self._lbl_sources_hint.setVisible(False)
        sources_layout.addWidget(self._lbl_sources_hint)

        btn_row = QHBoxLayout()
        self.btn_add_source = QPushButton("➕ Додати шар")
        self.btn_add_source.setToolTip(
            "Додати новий шар (відео з іншої висоти/часу або нова зона) до проєкту"
        )
        self.btn_add_source.clicked.connect(self.add_source_clicked)
        self.btn_add_source.setVisible(False)
        btn_row.addWidget(self.btn_add_source)
        sources_layout.addLayout(btn_row)

        self.sources_group.setVisible(False)  # Показується тільки при відкритому проєкті

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
        # The layer being built must stay the active one until the build ends.
        self.sources_table.setEnabled(not is_running)
        self.btn_add_source.setEnabled(not is_running)

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
            self.sources_table,
            self.btn_add_source,
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
        layer_id: str | None = None,
    ):
        """Оновити інформаційну панель проєкту (дані — активного шару)."""
        if project_name is None:
            self.lbl_project_info.setText("Проєкт не завантажено")
            self.lbl_project_info.setStyleSheet("font-size: 11px; color: #222;")
            self.btn_rebuild_db.setEnabled(False)
            self.sources_group.setVisible(False)
            self.set_active_layer_context(None)
            return

        lines = [f"▶ <b>{project_name}</b>"]
        if layer_id:
            lines.append(f"🧭 Активний шар: <b>{layer_id}</b>")
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

    def update_sources_list(self, rows: list[dict]):
        """Перемальовує таблицю шарів.

        Args:
            rows: один dict на шар (готує DatabaseMixin._layer_rows): source_id,
                area_id, anchors, gps, label, state, tooltip, enabled, db_loaded,
                has_db_file, num_anchors, is_active, conflict.
        """
        self.sources_group.setVisible(True)
        self.sources_table.setVisible(True)
        self._lbl_sources_hint.setVisible(True)
        self.btn_add_source.setVisible(True)

        bold = QFont()
        bold.setBold(True)
        active_bg = QColor("#e8f5e9")

        # Programmatic re-selection of the active row must not look like a click.
        self.sources_table.blockSignals(True)
        try:
            self.sources_table.clearSelection()
            self.sources_table.setRowCount(len(rows))
            active_row = -1
            for r, row in enumerate(rows):
                sid = row.get("source_id", "?")
                is_active = bool(row.get("is_active"))
                texts = [
                    ("▶ " if is_active else "") + sid,
                    row.get("area_id", ""),
                    row.get("anchors", "—"),
                    row.get("gps", "—"),
                    row.get("label", ""),
                ]
                for col, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    item.setToolTip(row.get("tooltip", ""))
                    if col == 0:
                        item.setData(Qt.ItemDataRole.UserRole, sid)
                        if row.get("conflict"):
                            item.setForeground(QColor("#c62828"))
                    if col == 4:
                        # Row state for the context menu (sid lives in column 0).
                        item.setData(
                            Qt.ItemDataRole.UserRole,
                            {
                                "enabled": bool(row.get("enabled", True)),
                                "db_loaded": bool(row.get("db_loaded")),
                                "has_db_file": bool(row.get("has_db_file")),
                                "num_anchors": int(row.get("num_anchors") or 0),
                                "is_active": is_active,
                            },
                        )
                        item.setForeground(
                            QColor(_LAYER_STATE_COLORS.get(row.get("state", ""), "#333333"))
                        )
                    if is_active:
                        item.setFont(bold)
                        item.setBackground(active_bg)
                    self.sources_table.setItem(r, col, item)
                if is_active:
                    active_row = r
            if active_row >= 0:
                self.sources_table.selectRow(active_row)
        finally:
            self.sources_table.blockSignals(False)

    def set_active_layer_context(self, source_id: str | None, num_layers: int = 1):
        """Підписує кнопки, що діють на активний шар, його назвою."""
        if not source_id or num_layers <= 1:
            self.calib_group.setTitle("Калібрування GPS")
            self.btn_calibrate.setText("Виконати калібрування (Video → Map)")
            self.btn_load_calibrate.setText("Завантажити калібрування (JSON)")
            self.btn_rebuild_db.setText("🔄 Перегенерувати базу")
            return
        name = _short(source_id)
        self.calib_group.setTitle(f"Калібрування GPS — шар «{name}»")
        self.btn_calibrate.setText(f"Калібрувати шар «{name}» (Video → Map)")
        self.btn_load_calibrate.setText(f"Завантажити калібрування шару «{name}»")
        self.btn_rebuild_db.setText(f"🔄 Перегенерувати базу шару «{name}»")

    def set_active_source(
        self,
        source_id: str,
        video_path: str = "",
        source_type: str = "file",
    ):
        """Відображає активне відеоджерело у бейджі секції 'Відеоджерела'.

        Args:
            source_id: ID активного джерела (напр. 'main', 'area_north').
            video_path: Повний шлях або URL відео.
            source_type: 'file' | 'rtsp' | 'usb'
        """
        self.sources_group.setVisible(True)

        # Filename label
        if video_path:
            if source_type == "rtsp":
                short = video_path
            elif source_type == "usb":
                short = f"USB камера ({video_path})"
            else:
                short = Path(video_path).name
        else:
            short = ""

        type_labels = {"rtsp": "📡 RTSP", "usb": "🔌 USB", "file": "📁 Файл"}
        self._lbl_source_dot.setStyleSheet("color: #4caf50; font-size: 13px;")
        self._lbl_source_id.setText(source_id)
        self._lbl_source_type.setText(type_labels.get(source_type, "📁 Файл"))
        self._lbl_source_video.setText(short)
        self._active_badge.setStyleSheet(
            "QFrame { background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 5px; }"
        )

    def mark_source_tracking(self, video_source: str = ""):
        """Оновлює бейдж: показує що відбувається активне трекінг-відео.

        Args:
            video_source: Шлях до файлу або RTSP URL що зараз відстежується.
                          None/пусто — скинути стан трекінгу.
        """
        if not video_source:
            # Reset to project state (green, no REC marker)
            self._lbl_source_dot.setStyleSheet("color: #4caf50; font-size: 13px;")
            self._active_badge.setStyleSheet(
                "QFrame { background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 5px; }"
            )
            return

        is_rtsp = str(video_source).lower().startswith("rtsp")
        is_usb = str(video_source).isdigit() or str(video_source).startswith("usb")
        if is_rtsp:
            track_name = video_source
            type_tag = "📡 RTSP"
        elif is_usb:
            track_name = f"USB камера ({video_source})"
            type_tag = "🔌 USB"
        else:
            track_name = Path(str(video_source)).name
            type_tag = "📁 Файл"

        self._lbl_source_dot.setStyleSheet("color: #f44336; font-size: 13px;")
        self._lbl_source_type.setText(f"🔴 REC  {type_tag}")
        self._lbl_source_video.setText(track_name)
        self._active_badge.setStyleSheet(
            "QFrame { background: #fff3e0; border: 1px solid #ffcc02; border-radius: 5px; }"
        )

    @pyqtSlot("QPoint")
    def _on_sources_context_menu(self, pos):
        """Контекстне меню шару: усі дії виконуються саме для цього шару."""
        row = self.sources_table.rowAt(pos.y())
        if row < 0:
            return

        item = self.sources_table.item(row, 0)
        if item is None:
            return
        source_id = item.data(Qt.ItemDataRole.UserRole)
        if not source_id:
            return
        state_item = self.sources_table.item(row, 4)
        state = (state_item.data(Qt.ItemDataRole.UserRole) if state_item else None) or {}
        enabled = state.get("enabled", True)
        db_loaded = state.get("db_loaded", False)

        menu = QMenu(self)
        title = menu.addAction(f"Шар «{source_id}»")
        title.setEnabled(False)
        menu.addSeparator()

        def add(text: str, action: str, allowed: bool = True):
            act = menu.addAction(text)
            act.setData(action)
            act.setEnabled(bool(allowed))
            return act

        add("▶ Зробити активним", "activate", not state.get("is_active"))
        add("📐 Калібрувати…", "calibrate", enabled and db_loaded)
        add("📂 Завантажити калібрування (JSON)…", "load_calibration", enabled)
        add(
            "🧭 Запустити пропагацію GPS",
            "propagate",
            enabled and db_loaded and state.get("num_anchors", 0) > 0,
        )
        menu.addSeparator()
        add(
            "🔨 Перебудувати базу даних…"
            if state.get("has_db_file")
            else "🔨 Побудувати базу даних",
            "build_db",
            enabled,
        )
        menu.addSeparator()
        add("🔇 Вимкнути" if enabled else "🔈 Увімкнути", "toggle")
        add("🗑 Видалити з проєкту…", "remove")

        chosen = menu.exec(self.sources_table.viewport().mapToGlobal(pos))
        if chosen is not None and chosen.data():
            self.source_action.emit(source_id, str(chosen.data()))

    @pyqtSlot()
    def _on_sources_selection_changed(self):
        """Клік по рядку робить шар активним."""
        selected_items = self.sources_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        item = self.sources_table.item(row, 0)
        if item:
            source_id = item.data(Qt.ItemDataRole.UserRole)
            if source_id:
                self.active_source_changed.emit(source_id)

    @pyqtSlot(QTableWidgetItem)
    def _on_sources_double_clicked(self, clicked: QTableWidgetItem):
        """Подвійний клік — відкрити калібрування саме цього шару."""
        item = self.sources_table.item(clicked.row(), 0)
        if item is None:
            return
        source_id = item.data(Qt.ItemDataRole.UserRole)
        if source_id:
            self.source_action.emit(source_id, "calibrate")
