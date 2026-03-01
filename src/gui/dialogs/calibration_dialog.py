import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QListWidget, QMessageBox, QSlider,
    QSpinBox, QGroupBox, QListWidgetItem, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QColor

from src.gui.widgets.video_widget import VideoWidget
from src.utils.image_utils import opencv_to_qpixmap

_UNKNOWN_FRAME_COUNT = 99999   # fallback when codec doesn't report frame count


class CalibrationDialog(QDialog):
    """
    Multi-anchor GPS calibration dialog.

    Workflow:
      1. Load video / image from database
      2. Navigate to target frame (slider)
      3. Click landmarks → enter GPS coordinates
      4. "Add anchor" — anchor saved, points cleared
      5. Repeat for other frames (first / middle / last)
      6. "Done" — triggers full-database propagation
    """

    anchor_added         = pyqtSignal(object)  # dict: {points_2d, points_gps, calib_frame_id}
    anchor_confirmed     = pyqtSignal(int)     # frame_id actually saved (from MainWindow)
    calibration_complete = pyqtSignal()

    def __init__(self, database_path: str, existing_anchors=None, parent=None):
        super().__init__(parent)
        self.database_path   = database_path
        self.existing_anchors = list(existing_anchors or [])

        self.points_2d        = []
        self.points_gps       = []
        self.current_2d_point = None
        self.cap              = None
        self.last_slider_value = 0
        self._is_video        = False

        self.timer     = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self.is_playing = False

        self.setWindowTitle("GPS Калібрування — Мульти-якірний режим")
        self.resize(1200, 800)
        self._init_ui()
        self._refresh_anchors_list()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left panel — video
        left = QVBoxLayout()

        self.btn_load_frame = QPushButton("📂  Завантажити відео / зображення")
        self.btn_load_frame.setStyleSheet("padding: 8px; font-weight: bold;")
        self.btn_load_frame.clicked.connect(self.load_frame)

        self.video_widget = VideoWidget()
        self.video_widget.frame_clicked.connect(self.on_video_clicked)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)

        player_row = QHBoxLayout()
        self.btn_step_back = QPushButton("◀◀")
        self.btn_play      = QPushButton("▶")
        self.btn_step      = QPushButton("▶▶")
        self.lbl_frame_info = QLabel("Кадр: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_info.setStyleSheet("color: #555; font-size: 12px;")

        for btn in [self.btn_step_back, self.btn_play, self.btn_step]:
            btn.setFixedWidth(48)
            btn.setEnabled(False)
            player_row.addWidget(btn)
        player_row.addWidget(self.lbl_frame_info, stretch=1)

        self.btn_step_back.clicked.connect(self.step_backward)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_step.clicked.connect(self.step_forward)

        left.addWidget(self.btn_load_frame)
        left.addWidget(self.video_widget, stretch=1)
        left.addWidget(self.slider)
        left.addLayout(player_row)

        # Right panel — controls
        right = QVBoxLayout()
        right.setSpacing(6)

        anchors_group = QGroupBox("Додані якорі")
        anchors_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        ag = QVBoxLayout(anchors_group)
        self.anchors_list = QListWidget()
        self.anchors_list.setMaximumHeight(90)
        self.anchors_list.setStyleSheet("font-size: 12px;")
        hint = QLabel("💡 Рекомендовано: перший кадр → середина → останній")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        ag.addWidget(self.anchors_list)
        ag.addWidget(hint)

        frame_group = QGroupBox("ID кадру в базі даних")
        frame_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        fg = QHBoxLayout(frame_group)
        self.spinbox_frame_id = QSpinBox()
        self.spinbox_frame_id.setRange(0, _UNKNOWN_FRAME_COUNT)
        self.spinbox_frame_id.setValue(0)
        self.spinbox_frame_id.setToolTip(
            "При роботі з відео — заповнюється автоматично зі слайдера."
        )
        fg.addWidget(QLabel("Кадр №:"))
        fg.addWidget(self.spinbox_frame_id)

        self.lbl_frame_id_warning = QLabel("")
        self.lbl_frame_id_warning.setStyleSheet("color: #e65100; font-size: 11px;")
        self.lbl_frame_id_warning.setWordWrap(True)

        pts_group = QGroupBox("Точки прив'язки (для поточного якоря)")
        pts_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pts = QVBoxLayout(pts_group)

        self.lbl_selected_px = QLabel("Клікніть на орієнтир у відео ↖")
        self.lbl_selected_px.setStyleSheet(
            "font-weight: bold; color: #1565C0; padding: 4px;"
            "background: #E3F2FD; border-radius: 4px;"
        )

        coords_row = QHBoxLayout()
        self.input_lat = QLineEdit()
        self.input_lat.setPlaceholderText("Широта")
        self.input_lon = QLineEdit()
        self.input_lon.setPlaceholderText("Довгота")
        coords_row.addWidget(QLabel("Lat:"))
        coords_row.addWidget(self.input_lat)
        coords_row.addWidget(QLabel("Lon:"))
        coords_row.addWidget(self.input_lon)

        self.btn_add_point = QPushButton("➕  Додати точку")
        self.btn_add_point.clicked.connect(self.add_point_pair)

        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(110)
        self.points_list.setStyleSheet("font-size: 11px;")

        self.btn_clear_points = QPushButton("🗑  Очистити поточні точки")
        self.btn_clear_points.setStyleSheet("color: #b71c1c; font-size: 11px;")
        self.btn_clear_points.clicked.connect(self.clear_current_points)

        pts.addWidget(self.lbl_selected_px)
        pts.addLayout(coords_row)
        pts.addWidget(self.btn_add_point)
        pts.addWidget(self.points_list)
        pts.addWidget(self.btn_clear_points)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;")

        self.btn_add_anchor = QPushButton("⚓  Додати якір для цього кадру")
        self.btn_add_anchor.setStyleSheet(
            "background:#1565C0; color:white; font-weight:bold; padding:11px; font-size:13px;"
        )
        self.btn_add_anchor.clicked.connect(self.add_anchor)

        self.lbl_status = QLabel("Додайте мінімум 1 якір щоб продовжити")
        self.lbl_status.setStyleSheet("color:#666; font-size:11px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setWordWrap(True)

        self.btn_done = QPushButton("✅  Готово — запустити пропагацію по всій базі")
        self.btn_done.setStyleSheet(
            "background:#2e7d32; color:white; font-weight:bold; padding:11px; font-size:13px;"
        )
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self.finish_calibration)

        self.btn_cancel = QPushButton("Скасувати")
        self.btn_cancel.setStyleSheet("color:#555; padding:7px;")
        self.btn_cancel.clicked.connect(self.reject)

        right.addWidget(anchors_group)
        right.addWidget(frame_group)
        right.addWidget(self.lbl_frame_id_warning)
        right.addWidget(pts_group, stretch=1)
        right.addWidget(sep)
        right.addWidget(self.btn_add_anchor)
        right.addWidget(self.lbl_status)
        right.addWidget(self.btn_done)
        right.addWidget(self.btn_cancel)

        main_layout.addLayout(left, stretch=2)
        main_layout.addLayout(right, stretch=1)

    # ── Anchor list ──────────────────────────────────────────────────────────

    def _refresh_anchors_list(self):
        self.anchors_list.clear()
        if not self.existing_anchors:
            item = QListWidgetItem("  (поки немає якорів)")
            item.setForeground(QColor("#aaa"))
            self.anchors_list.addItem(item)
        else:
            for i, fid in enumerate(sorted(self.existing_anchors)):
                item = QListWidgetItem(f"  ⚓ Якір {i+1}: кадр {fid}")
                item.setForeground(QColor("#1565C0"))
                self.anchors_list.addItem(item)

        has = bool(self.existing_anchors)
        self.btn_done.setEnabled(has)
        if has:
            self.lbl_status.setText(
                f"Додано якорів: {len(self.existing_anchors)}. "
                "Додайте ще або натисніть «Готово»."
            )
            self.lbl_status.setStyleSheet("color:#2e7d32; font-size:11px;")
        else:
            self.lbl_status.setText("Додайте мінімум 1 якір щоб продовжити")
            self.lbl_status.setStyleSheet("color:#666; font-size:11px;")

    def on_anchor_confirmed(self, frame_id: int):
        """Called by MainWindow after affine matrix is successfully computed."""
        if frame_id not in self.existing_anchors:
            self.existing_anchors.append(frame_id)
        self.existing_anchors.sort()
        self._refresh_anchors_list()
        self.clear_current_points()
        QMessageBox.information(
            self, "⚓ Якір додано",
            f"Якір для кадру {frame_id} успішно збережено!\n\n"
            f"Всього якорів: {len(self.existing_anchors)}\n\n"
            f"Перейдіть на інший кадр і додайте наступний якір,\n"
            f"або натисніть «Готово — запустити пропагацію».",
        )

    # ── Video loading ────────────────────────────────────────────────────────

    def load_frame(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Відео або зображення", "",
            "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mkv *.mov);;"
            "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mkv *.mov)",
        )
        if not path:
            return

        self.clear_current_points()

        if path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            self._load_video(path)
        else:
            self._load_image(path)

    def _load_video(self, path: str):
        self._is_video = True
        if self.cap:
            self.cap.release()

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити:\n{path}")
            return

        self.cap = cap
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = _UNKNOWN_FRAME_COUNT   # unknown length codec

        self.slider.blockSignals(True)
        self.slider.setEnabled(True)
        self.slider.setRange(0, total - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        for btn in [self.btn_play, self.btn_step_back, self.btn_step]:
            btn.setEnabled(True)

        self.spinbox_frame_id.setMaximum(total - 1)
        self.spinbox_frame_id.setValue(0)
        self.lbl_frame_id_warning.setText("")
        self.last_slider_value = 0
        self.on_slider_changed(0)

    def _load_image(self, path: str):
        self._is_video = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.slider.setEnabled(False)
        for btn in [self.btn_play, self.btn_step_back, self.btn_step]:
            btn.setEnabled(False)

        self.lbl_frame_info.setText("Статичне зображення")
        self.lbl_frame_id_warning.setText(
            "⚠ Статичне зображення: вкажіть вручну ID кадру з відео бази даних."
        )

        # cv2.imread fails on non-ASCII (Cyrillic) paths on Windows
        with open(path, "rb") as f:
            raw = bytearray(f.read())
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self.video_widget.display_frame(opencv_to_qpixmap(frame))
        else:
            QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")

    # ── Playback ─────────────────────────────────────────────────────────────

    def toggle_playback(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(1000 / fps) if fps > 0 else 33)
            self.btn_play.setText("⏸")
            self.is_playing = True

    def play_next_frame(self):
        if not (self.cap and self.cap.isOpened()):
            return
        ret, frame = self.cap.read()
        if not ret:
            self.toggle_playback()
            return
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.last_slider_value = cur
        self.slider.blockSignals(True)
        self.slider.setValue(cur)
        self.slider.blockSignals(False)
        self.spinbox_frame_id.setValue(cur)
        self.video_widget.display_frame(opencv_to_qpixmap(frame))
        self.lbl_frame_info.setText(f"Кадр: {cur} / {self.slider.maximum()}")

    def step_forward(self):
        if self.is_playing:
            self.toggle_playback()
        self.play_next_frame()

    def step_backward(self):
        if self.is_playing:
            self.toggle_playback()
        if not (self.cap and self.cap.isOpened()):
            return
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # POS_FRAMES points to next-to-read frame after read(), so -2 = previous
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
        self.play_next_frame()

    def on_slider_changed(self, value: int):
        if self.is_playing:
            return   # slider updated by play_next_frame — no dialog during playback

        if not (self.cap and self.cap.isOpened()):
            return

        if value == self.last_slider_value:
            return

        # Roll back slider immediately BEFORE showing dialog (prevents double signal)
        if self.points_2d or self.current_2d_point:
            self.slider.blockSignals(True)
            self.slider.setValue(self.last_slider_value)
            self.slider.blockSignals(False)

            reply = QMessageBox.question(
                self, "Увага",
                "Зміна кадру очистить незбережені точки. Продовжити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

            self.clear_current_points()

            # User confirmed — advance to requested value
            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)

        self.last_slider_value = value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        ret, frame = self.cap.read()

        # Fallback for codecs that don't support POS_FRAMES seek
        if not ret:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, (value / fps) * 1000.0)
                ret, frame = self.cap.read()

        if ret and frame is not None:
            self.spinbox_frame_id.setValue(value)
            self.video_widget.display_frame(opencv_to_qpixmap(frame))
            self.lbl_frame_info.setText(f"Кадр: {value} / {self.slider.maximum()}")

    # ── Points ───────────────────────────────────────────────────────────────

    def on_video_clicked(self, x: int, y: int):
        self.current_2d_point = (x, y)
        self.lbl_selected_px.setText(f"✔ Обрано піксель: ({x}, {y})")
        self._redraw_points()

    def add_point_pair(self):
        if not self.current_2d_point:
            QMessageBox.warning(self, "Помилка", "Спочатку клікніть на орієнтир у відео!")
            return
        try:
            lat = float(self.input_lat.text().strip())
            lon = float(self.input_lon.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Помилка", "Введіть числові координати.")
            return

        self.points_2d.append(self.current_2d_point)
        self.points_gps.append((lat, lon))
        n = len(self.points_2d)
        self.points_list.addItem(
            f"  {n}. ({self.current_2d_point[0]}, {self.current_2d_point[1]})"
            f"  →  {lat:.5f}, {lon:.5f}"
        )
        self.current_2d_point = None
        self.lbl_selected_px.setText("Клікніть на наступний орієнтир")
        self.input_lat.clear()
        self.input_lon.clear()
        self._redraw_points()

    def clear_current_points(self):
        self.points_2d.clear()
        self.points_gps.clear()
        self.current_2d_point = None
        self.points_list.clear()
        self.video_widget.clear_overlays()
        self.lbl_selected_px.setText("Клікніть на орієнтир у відео ↖")

    def _redraw_points(self):
        self.video_widget.clear_overlays()
        for i, pt in enumerate(self.points_2d):
            self.video_widget.draw_numbered_point(pt[0], pt[1], str(i + 1), QColor(0, 200, 0))
        if self.current_2d_point:
            self.video_widget.draw_numbered_point(
                self.current_2d_point[0], self.current_2d_point[1], "?", QColor(255, 80, 0)
            )

    # ── Add anchor ───────────────────────────────────────────────────────────

    def add_anchor(self):
        if len(self.points_2d) < 3:
            QMessageBox.warning(self, "Увага", "Потрібно мінімум 3 точки для якоря!")
            return

        frame_id = self.spinbox_frame_id.value()

        if frame_id in self.existing_anchors:
            reply = QMessageBox.question(
                self, "Якір існує",
                f"Якір для кадру {frame_id} вже є. Замінити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Emit to MainWindow — it computes affine and calls on_anchor_confirmed() on success
        self.anchor_added.emit({
            'points_2d':      list(self.points_2d),
            'points_gps':     list(self.points_gps),
            'calib_frame_id': frame_id,
        })
        # List update happens in on_anchor_confirmed() after MainWindow validates

    # ── Finish ───────────────────────────────────────────────────────────────

    def finish_calibration(self):
        if not self.existing_anchors:
            QMessageBox.warning(self, "Увага", "Додайте хоча б один якір!")
            return

        if self.points_2d:
            reply = QMessageBox.question(
                self, "Незбережені точки",
                f"У вас {len(self.points_2d)} незбережених точок для кадру "
                f"{self.spinbox_frame_id.value()}.\n"
                f"Додати їх як якір перед завершенням?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.add_anchor()
                return   # on_anchor_confirmed will not auto-call finish — user re-clicks Done
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.calibration_complete.emit()
        self.accept()

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.is_playing:
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)
