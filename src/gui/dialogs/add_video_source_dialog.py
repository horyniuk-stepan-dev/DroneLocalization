"""
add_video_source_dialog.py — Діалог додавання нового відеоджерела до проєкту.

Дозволяє вибрати відео, вказати source_id, area_id, режим (шар/зона),
та опціонально geo_bounds.
"""
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.core.project_video_source import ProjectVideoSource
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AddVideoSourceDialog(QDialog):
    """Діалог для додавання нового відеоджерела до мультиджерельного проєкту."""

    def __init__(self, existing_area_ids: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Додати відеоджерело")
        self.setMinimumWidth(500)
        self._existing_areas = existing_area_ids or []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ── Основні параметри ────────────────────────────────────────────────
        basic_group = QGroupBox("Основні параметри")
        form = QFormLayout(basic_group)

        self.source_id_edit = QLineEdit()
        self.source_id_edit.setPlaceholderText("Унікальний ідентифікатор (напр. winter_2025)")
        form.addRow("Source ID:", self.source_id_edit)

        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Опис (зима 2025, ранковий політ, тощо)")
        form.addRow("Опис:", self.description_edit)

        # Відео
        video_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Шлях до відео не вибрано")
        btn_browse = QPushButton("Огляд...")
        btn_browse.clicked.connect(self._browse_video)
        video_row.addWidget(self.video_path_edit)
        video_row.addWidget(btn_browse)
        form.addRow("Відеофайл:", video_row)

        # Режим
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("📊 Новий шар (overlay поверх існуючої зони)", "layer")
        self.mode_combo.addItem("🗺 Нова географічна зона", "zone")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Режим:", self.mode_combo)

        # Area ID
        self.area_combo = QComboBox()
        self.area_combo.setEditable(True)
        if self._existing_areas:
            for area in self._existing_areas:
                self.area_combo.addItem(area)
        self.area_combo.setCurrentText("")
        self.area_combo.lineEdit().setPlaceholderText("Виберіть або введіть area_id")
        form.addRow("Area ID:", self.area_combo)

        # Пріоритет
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 100)
        self.priority_spin.setValue(0)
        self.priority_spin.setToolTip("0 = найвищий пріоритет. При рівних cosine — вибирається джерело з нижчим priority.")
        form.addRow("Пріоритет:", self.priority_spin)

        layout.addWidget(basic_group)

        # ── Параметри камери ───────────────────────────────────────────────────
        camera_group = QGroupBox("Параметри камери")
        cam_form = QFormLayout(camera_group)

        self.altitude_spinbox = QDoubleSpinBox()
        self.altitude_spinbox.setRange(10.0, 5000.0)
        self.altitude_spinbox.setValue(100.0)
        self.altitude_spinbox.setSuffix(" м")
        cam_form.addRow("Висота польоту:", self.altitude_spinbox)

        self.focal_length_spinbox = QDoubleSpinBox()
        self.focal_length_spinbox.setRange(1.0, 100.0)
        self.focal_length_spinbox.setValue(13.2)
        self.focal_length_spinbox.setSuffix(" мм")
        cam_form.addRow("Фокусна відстань:", self.focal_length_spinbox)

        self.sensor_width_spinbox = QDoubleSpinBox()
        self.sensor_width_spinbox.setRange(1.0, 50.0)
        self.sensor_width_spinbox.setValue(8.8)
        self.sensor_width_spinbox.setSuffix(" мм")
        cam_form.addRow("Ширина сенсора:", self.sensor_width_spinbox)

        self.image_width_spinbox = QSpinBox()
        self.image_width_spinbox.setRange(640, 8000)
        self.image_width_spinbox.setValue(4000)
        self.image_width_spinbox.setSuffix(" px")
        cam_form.addRow("Ширина зображення:", self.image_width_spinbox)

        layout.addWidget(camera_group)

        # ── Geo Bounds (опційно) ─────────────────────────────────────────────
        self.geo_group = QGroupBox("Географічні межі (опційно)")
        self.geo_group.setCheckable(True)
        self.geo_group.setChecked(False)
        geo_layout = QFormLayout(self.geo_group)

        self.lat_min_spin = QDoubleSpinBox()
        self.lat_min_spin.setRange(-90.0, 90.0)
        self.lat_min_spin.setDecimals(6)
        self.lat_min_spin.setSuffix("°")
        geo_layout.addRow("Lat min:", self.lat_min_spin)

        self.lon_min_spin = QDoubleSpinBox()
        self.lon_min_spin.setRange(-180.0, 180.0)
        self.lon_min_spin.setDecimals(6)
        self.lon_min_spin.setSuffix("°")
        geo_layout.addRow("Lon min:", self.lon_min_spin)

        self.lat_max_spin = QDoubleSpinBox()
        self.lat_max_spin.setRange(-90.0, 90.0)
        self.lat_max_spin.setDecimals(6)
        self.lat_max_spin.setValue(90.0)
        self.lat_max_spin.setSuffix("°")
        geo_layout.addRow("Lat max:", self.lat_max_spin)

        self.lon_max_spin = QDoubleSpinBox()
        self.lon_max_spin.setRange(-180.0, 180.0)
        self.lon_max_spin.setDecimals(6)
        self.lon_max_spin.setValue(180.0)
        self.lon_max_spin.setSuffix("°")
        geo_layout.addRow("Lon max:", self.lon_max_spin)

        layout.addWidget(self.geo_group)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ── Slots ────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть відеофайл",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    @pyqtSlot(int)
    def _on_mode_changed(self, idx):
        mode = self.mode_combo.currentData()
        if mode == "layer" and self._existing_areas:
            # Для шару — пропонуємо вибрати існуючу area
            self.area_combo.setCurrentText(self._existing_areas[0])
        else:
            # Для зони — пропонуємо нову area
            self.area_combo.setCurrentText("")

    @pyqtSlot()
    def _on_accept(self):
        source_id = self.source_id_edit.text().strip()
        if not source_id:
            QMessageBox.warning(self, "Помилка", "Введіть Source ID!")
            self.source_id_edit.setFocus()
            return
        if not source_id.replace("_", "").replace("-", "").isalnum():
            QMessageBox.warning(
                self, "Помилка",
                "Source ID може містити тільки латинські літери, цифри, _ та -"
            )
            return
        if not self.video_path_edit.text():
            QMessageBox.warning(self, "Помилка", "Виберіть відеофайл!")
            return

        area_id = self.area_combo.currentText().strip()
        if not area_id:
            QMessageBox.warning(self, "Помилка", "Введіть або виберіть Area ID!")
            self.area_combo.setFocus()
            return

        if self.geo_group.isChecked():
            if self.lat_min_spin.value() >= self.lat_max_spin.value():
                QMessageBox.warning(self, "Помилка", "Lat min має бути менше Lat max!")
                return
            if self.lon_min_spin.value() >= self.lon_max_spin.value():
                QMessageBox.warning(self, "Помилка", "Lon min має бути менше Lon max!")
                return

        self.accept()

    # ── Data ─────────────────────────────────────────────────────────────────

    def get_source_config(self) -> ProjectVideoSource:
        """Повертає заповнений ProjectVideoSource."""
        source_id = self.source_id_edit.text().strip()

        geo_bounds = None
        if self.geo_group.isChecked():
            geo_bounds = (
                self.lat_min_spin.value(),
                self.lon_min_spin.value(),
                self.lat_max_spin.value(),
                self.lon_max_spin.value(),
            )

        return ProjectVideoSource(
            source_id=source_id,
            area_id=self.area_combo.currentText().strip(),
            video_path=self.video_path_edit.text(),
            database_file=f"sources/{source_id}/database.h5",
            calibration_file=f"sources/{source_id}/calibration.json",
            description=self.description_edit.text().strip(),
            enabled=True,
            priority=self.priority_spin.value(),
            geo_bounds=geo_bounds,
            camera_params={
                "altitude_m": self.altitude_spinbox.value(),
                "focal_length_mm": self.focal_length_spinbox.value(),
                "sensor_width_mm": self.sensor_width_spinbox.value(),
                "image_width_px": self.image_width_spinbox.value(),
            },
        )
