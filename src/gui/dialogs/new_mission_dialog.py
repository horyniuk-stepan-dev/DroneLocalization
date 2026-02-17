from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLineEdit, QPushButton, QDoubleSpinBox, QSpinBox,
                             QFileDialog, QDialogButtonBox)
from PyQt6.QtCore import pyqtSlot

from src.utils.logging_utils import get_logger  # ВИПРАВЛЕНО: було 'from utils...'


class NewMissionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Створення нової місії")
        self.setMinimumWidth(400)
        self.logger = get_logger('NewMissionDialog')
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.mission_name_edit = QLineEdit()
        self.mission_name_edit.setPlaceholderText("Введіть назву місії")
        form_layout.addRow("Назва місії:", self.mission_name_edit)

        video_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.btn_browse = QPushButton("Огляд...")
        self.btn_browse.clicked.connect(self.browse_video)
        video_layout.addWidget(self.video_path_edit)
        video_layout.addWidget(self.btn_browse)
        form_layout.addRow("Еталонне відео:", video_layout)

        self.altitude_spinbox = QDoubleSpinBox()
        self.altitude_spinbox.setRange(10.0, 5000.0)
        self.altitude_spinbox.setValue(100.0)
        self.altitude_spinbox.setSuffix(" м")
        form_layout.addRow("Висота польоту:", self.altitude_spinbox)

        self.focal_length_spinbox = QDoubleSpinBox()
        self.focal_length_spinbox.setRange(1.0, 100.0)
        self.focal_length_spinbox.setValue(13.2)
        self.focal_length_spinbox.setSuffix(" мм")
        form_layout.addRow("Фокусна відстань:", self.focal_length_spinbox)

        self.sensor_width_spinbox = QDoubleSpinBox()
        self.sensor_width_spinbox.setRange(1.0, 50.0)
        self.sensor_width_spinbox.setValue(8.8)
        self.sensor_width_spinbox.setSuffix(" мм")
        form_layout.addRow("Ширина сенсора:", self.sensor_width_spinbox)

        self.image_width_spinbox = QSpinBox()
        self.image_width_spinbox.setRange(640, 8000)
        self.image_width_spinbox.setValue(4000)
        self.image_width_spinbox.setSuffix(" px")
        form_layout.addRow("Ширина зображення:", self.image_width_spinbox)

        layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    @pyqtSlot()
    def browse_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть еталонне відео",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if file_name:
            self.video_path_edit.setText(file_name)

    def get_mission_data(self) -> dict:
        return {
            "mission_name": self.mission_name_edit.text(),
            "video_path": self.video_path_edit.text(),
            "altitude_m": self.altitude_spinbox.value(),
            "focal_length_mm": self.focal_length_spinbox.value(),
            "sensor_width_mm": self.sensor_width_spinbox.value(),
            "image_width_px": self.image_width_spinbox.value()
        }