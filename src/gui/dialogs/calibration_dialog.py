import cv2
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QLineEdit, QListWidget, QMessageBox)
from PyQt6.QtCore import pyqtSignal
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

        self.setWindowTitle("GPS Калібрування місцевості")
        self.resize(1100, 700)
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Ліва частина: Відображення кадру з відео
        left_layout = QVBoxLayout()
        self.btn_load_frame = QPushButton("1. Завантажити кадр з еталонного відео")
        self.btn_load_frame.clicked.connect(self.load_frame)

        self.video_widget = VideoWidget()
        self.video_widget.frame_clicked.connect(self.on_video_clicked)

        left_layout.addWidget(self.btn_load_frame)
        left_layout.addWidget(self.video_widget)
        left_layout.setStretch(1, 1)

        # Права частина: Управління координатами
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
        self.btn_calculate.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;")
        self.btn_calculate.clicked.connect(self.calculate_calibration)

        right_layout.addWidget(self.lbl_selected_px)
        right_layout.addWidget(QLabel("Введіть реальні координати (з Google Maps):"))
        right_layout.addWidget(self.input_lat)
        right_layout.addWidget(self.input_lon)
        right_layout.addWidget(self.btn_add_point)
        right_layout.addWidget(QLabel("Додані маркери (потрібно мінімум 3):"))
        right_layout.addWidget(self.points_list)
        right_layout.addWidget(self.btn_calculate)

        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)

    def load_frame(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Виберіть відео", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if ret:
                pixmap = opencv_to_qpixmap(frame)
                self.video_widget.display_frame(pixmap)
            cap.release()

    def redraw_points(self):
        """Очищає екран і перемальовує всі збережені та поточну точки"""
        self.video_widget.clear_overlays()

        # 1. Малюємо вже підтверджені точки зеленим кольором
        for i, pt in enumerate(self.points_2d):
            self.video_widget.draw_numbered_point(pt[0], pt[1], str(i + 1), QColor(0, 200, 0))

        # 2. Малюємо поточну тимчасову точку червоним кольором (якщо вона є)
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

        # Додаємо у список з правильним номером
        point_number = len(self.points_2d)
        item_text = f"Точка {point_number}: Піксель {self.current_2d_point} ➔ GPS: {lat:.5f}, {lon:.5f}"
        self.points_list.addItem(item_text)

        # Скидання для наступної точки
        self.current_2d_point = None
        self.lbl_selected_px.setText("Вибраний піксель (X, Y): Немає")
        self.input_lat.clear()
        self.input_lon.clear()

        # Перемальовуємо, щоб червона точка стала зеленою з номером
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