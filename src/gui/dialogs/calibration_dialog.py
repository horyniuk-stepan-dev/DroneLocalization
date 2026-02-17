import cv2
import numpy as np  # ВИПРАВЛЕНО: перенесено з середини методу на верхній рівень
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QLineEdit, QListWidget, QMessageBox,
                             QSlider)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QColor

from src.gui.widgets.video_widget import VideoWidget
from src.utils.image_utils import opencv_to_qpixmap


class CalibrationDialog(QDialog):
    """Діалогове вікно для GPS калібрування"""

    calibration_complete = pyqtSignal(object)

    def __init__(self, database_path, parent=None):
        super().__init__(parent)
        self.database_path = database_path
        self.points_2d = []
        self.points_gps = []
        self.current_2d_point = None
        self.cap = None
        self.last_slider_value = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self.is_playing = False

        self.setWindowTitle("GPS Калібрування місцевості")
        self.resize(1100, 700)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        self.btn_load_frame = QPushButton("1. Завантажити еталонне зображення / відео")
        self.btn_load_frame.clicked.connect(self.load_frame)

        self.video_widget = VideoWidget()
        self.video_widget.frame_clicked.connect(self.on_video_clicked)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)

        player_controls = QHBoxLayout()
        self.btn_play = QPushButton("▶ Відтворити")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_step = QPushButton("⏭ Кадр вперед")
        self.btn_step.clicked.connect(self.step_forward)

        self.btn_play.setEnabled(False)
        self.btn_step.setEnabled(False)

        player_controls.addWidget(self.btn_play)
        player_controls.addWidget(self.btn_step)

        self.lbl_frame_info = QLabel("Кадр: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_info.setStyleSheet("color: #666; font-size: 12px;")

        left_layout.addWidget(self.btn_load_frame)
        left_layout.addWidget(self.video_widget)
        left_layout.addWidget(self.slider)
        left_layout.addLayout(player_controls)
        left_layout.addWidget(self.lbl_frame_info)
        left_layout.setStretch(1, 1)

        right_layout = QVBoxLayout()

        self.lbl_selected_px = QLabel("Вибраний піксель (X, Y): Немає")
        self.lbl_selected_px.setStyleSheet("font-weight: bold; color: #1976D2;")

        self.input_lat = QLineEdit()
        self.input_lat.setPlaceholderText("Широта (напр. 50.4501)")
        self.input_lon = QLineEdit()
        self.input_lon.setPlaceholderText("Довгота (напр. 30.5234)")

        self.btn_add_point = QPushButton("2. Додати пару координат")
        self.btn_add_point.clicked.connect(self.add_point_pair)

        self.points_list = QListWidget()

        self.btn_calculate = QPushButton("3. Розрахувати афінну матрицю")
        self.btn_calculate.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;")
        self.btn_calculate.clicked.connect(self.calculate_calibration)

        right_layout.addWidget(self.lbl_selected_px)
        right_layout.addWidget(QLabel("Введіть реальні координати (з Google Maps):"))
        right_layout.addWidget(self.input_lat)
        right_layout.addWidget(self.input_lon)
        right_layout.addWidget(self.btn_add_point)
        right_layout.addWidget(QLabel("Додані маркери (всі точки мають бути на одному кадрі):"))
        right_layout.addWidget(self.points_list)
        right_layout.addWidget(self.btn_calculate)

        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)

    def load_frame(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть еталонне зображення або відео",
            "",
            "Media Files (*.png *.jpg *.jpeg *.mp4 *.avi);;Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi)"
        )

        if not file_path:
            return

        self.clear_all_points()

        if file_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(file_path)

            if not self.cap.isOpened():
                QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити відео:\n{file_path}")
                return

            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.slider.blockSignals(True)
            self.slider.setEnabled(True)
            self.slider.setRange(0, total_frames - 1)
            self.slider.setValue(0)
            self.slider.blockSignals(False)

            self.btn_play.setEnabled(True)
            self.btn_step.setEnabled(True)

            self.on_slider_changed(0)

        else:
            if self.cap:
                self.cap.release()
                self.cap = None

            self.slider.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_step.setEnabled(False)
            self.lbl_frame_info.setText("Статичне зображення")

            # ВИПРАВЛЕНО: 'bytes' — вбудований тип Python, використання його
            # як імені змінної приховує стандартну функцію bytes() у цьому scope.
            # Перейменовано на 'raw_data'. Також прибрано 'import numpy as np'
            # з середини методу — тепер імпорт на верхньому рівні файлу.
            with open(file_path, "rb") as stream:
                raw_data = bytearray(stream.read())
                numpyarray = np.asarray(raw_data, dtype=np.uint8)
                frame = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

            if frame is not None:
                pixmap = opencv_to_qpixmap(frame)
                self.video_widget.display_frame(pixmap)
            else:
                QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")

    def toggle_playback(self):
        if not self.cap or not self.cap.isOpened():
            return

        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶ Відтворити")
            self.is_playing = False
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps) if fps > 0 else 33
            self.timer.start(delay)
            self.btn_play.setText("⏸ Пауза")
            self.is_playing = True

    def play_next_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.last_slider_value = current_frame

                self.slider.blockSignals(True)
                self.slider.setValue(current_frame)
                self.slider.blockSignals(False)

                pixmap = opencv_to_qpixmap(frame)
                self.video_widget.display_frame(pixmap)
                self.lbl_frame_info.setText(f"Кадр: {current_frame} / {self.slider.maximum()}")
            else:
                self.toggle_playback()

    def step_forward(self):
        if self.is_playing:
            self.toggle_playback()
        self.play_next_frame()

    def on_slider_changed(self, value):
        if self.is_playing:
            self.toggle_playback()

        if self.cap and self.cap.isOpened():
            if len(self.points_2d) > 0 or self.current_2d_point:
                reply = QMessageBox.question(
                    self, "Увага",
                    "Зміна кадру видалить вже встановлені точки. Продовжити?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    self.slider.blockSignals(True)
                    self.slider.setValue(self.last_slider_value)
                    self.slider.blockSignals(False)
                    return
                else:
                    self.clear_all_points()

            self.last_slider_value = value
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = self.cap.read()

            if not ret:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, (value / fps) * 1000.0)
                    ret, frame = self.cap.read()

            if ret and frame is not None:
                pixmap = opencv_to_qpixmap(frame)
                self.video_widget.display_frame(pixmap)
                self.lbl_frame_info.setText(f"Кадр: {value} / {self.slider.maximum()}")

    def clear_all_points(self):
        self.points_2d.clear()
        self.points_gps.clear()
        self.current_2d_point = None
        self.points_list.clear()
        self.video_widget.clear_overlays()
        self.lbl_selected_px.setText("Вибраний піксель (X, Y): Немає")

    def redraw_points(self):
        self.video_widget.clear_overlays()

        for i, pt in enumerate(self.points_2d):
            self.video_widget.draw_numbered_point(pt[0], pt[1], str(i + 1), QColor(0, 200, 0))

        if self.current_2d_point:
            self.video_widget.draw_numbered_point(
                self.current_2d_point[0],
                self.current_2d_point[1],
                "?",
                QColor(255, 0, 0)
            )

    def on_video_clicked(self, x, y):
        self.current_2d_point = (x, y)
        self.lbl_selected_px.setText(f"Вибраний піксель (X, Y): {x}, {y}")
        self.redraw_points()

    def add_point_pair(self):
        if not self.current_2d_point:
            QMessageBox.warning(self, "Помилка", "Спочатку клікніть мишкою на орієнтир на відео!")
            return

        try:
            lat = float(self.input_lat.text().strip())
            lon = float(self.input_lon.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Помилка", "Будь ласка, введіть числові координати.")
            return

        self.points_2d.append(self.current_2d_point)
        self.points_gps.append((lat, lon))

        point_number = len(self.points_2d)
        item_text = f"Точка {point_number}: Піксель {self.current_2d_point} ➔ GPS: {lat:.5f}, {lon:.5f}"
        self.points_list.addItem(item_text)

        self.current_2d_point = None
        self.lbl_selected_px.setText("Вибраний піксель (X, Y): Немає")
        self.input_lat.clear()
        self.input_lon.clear()
        self.redraw_points()

    def calculate_calibration(self):
        if len(self.points_2d) < 3:
            QMessageBox.warning(self, "Увага", "Потрібно мінімум 3 точки для калібрування!")
            return

        calibration_data = {
            'points_2d': self.points_2d,
            'points_gps': self.points_gps
        }
        self.calibration_complete.emit(calibration_data)
        self.accept()

    def closeEvent(self, event):
        if self.is_playing:
            self.timer.stop()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)