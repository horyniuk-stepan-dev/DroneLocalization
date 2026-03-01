from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox,
    QFileDialog, QDialogButtonBox, QMessageBox,
)
from PyQt6.QtCore import pyqtSlot
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NewMissionDialog(QDialog):
    """Dialog for creating a new localization mission."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Створення нової місії")
        self.setMinimumWidth(420)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.mission_name_edit = QLineEdit()
        self.mission_name_edit.setPlaceholderText("Введіть назву місії")
        form.addRow("Назва місії:", self.mission_name_edit)

        video_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Шлях до відео не вибрано")
        btn_browse = QPushButton("Огляд...")
        btn_browse.clicked.connect(self._browse_video)
        video_row.addWidget(self.video_path_edit)
        video_row.addWidget(btn_browse)
        form.addRow("Еталонне відео:", video_row)

        # Camera parameters — used for GSD / FOV calculations in DatabaseBuilder
        self.altitude_spinbox = QDoubleSpinBox()
        self.altitude_spinbox.setRange(10.0, 5000.0)
        self.altitude_spinbox.setValue(100.0)
        self.altitude_spinbox.setSuffix(" м")
        form.addRow("Висота польоту:", self.altitude_spinbox)

        self.focal_length_spinbox = QDoubleSpinBox()
        self.focal_length_spinbox.setRange(1.0, 100.0)
        self.focal_length_spinbox.setValue(13.2)
        self.focal_length_spinbox.setSuffix(" мм")
        form.addRow("Фокусна відстань:", self.focal_length_spinbox)

        self.sensor_width_spinbox = QDoubleSpinBox()
        self.sensor_width_spinbox.setRange(1.0, 50.0)
        self.sensor_width_spinbox.setValue(8.8)
        self.sensor_width_spinbox.setSuffix(" мм")
        form.addRow("Ширина сенсора:", self.sensor_width_spinbox)

        self.image_width_spinbox = QSpinBox()
        self.image_width_spinbox.setRange(640, 8000)
        self.image_width_spinbox.setValue(4000)
        self.image_width_spinbox.setSuffix(" px")
        form.addRow("Ширина зображення:", self.image_width_spinbox)

        layout.addLayout(form)

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
            self, "Виберіть еталонне відео", "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    @pyqtSlot()
    def _on_accept(self):
        if not self.mission_name_edit.text().strip():
            QMessageBox.warning(self, "Помилка", "Введіть назву місії!")
            self.mission_name_edit.setFocus()
            return
        if not self.video_path_edit.text():
            QMessageBox.warning(self, "Помилка", "Виберіть еталонне відео!")
            return
        self.accept()

    # ── Data ─────────────────────────────────────────────────────────────────

    def get_mission_data(self) -> dict:
        data = {
            "mission_name":    self.mission_name_edit.text().strip(),
            "video_path":      self.video_path_edit.text(),
            "altitude_m":      self.altitude_spinbox.value(),
            "focal_length_mm": self.focal_length_spinbox.value(),
            "sensor_width_mm": self.sensor_width_spinbox.value(),
            "image_width_px":  self.image_width_spinbox.value(),
        }
        logger.info(
            f"Mission: '{data['mission_name']}' | "
            f"video={data['video_path']} | alt={data['altitude_m']}m"
        )
        return data
