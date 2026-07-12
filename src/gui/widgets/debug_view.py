"""Вікно візуалізації одного debug-каналу («очима моделі»).

QDockWidget + QLabel зі scaled pixmap. Кадри приходять від
RealtimeTrackingWorker.debug_view_ready (готове BGR-зображення). Приховане за
замовчуванням; overhead нульовий, поки вікно не відкрите.
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDockWidget, QLabel, QSizePolicy

from src.utils.image_utils import opencv_to_qpixmap


class DebugViewDock(QDockWidget):
    """Один канал (yolo / depth / dino / matches)."""

    def __init__(self, channel_name: str, title: str, parent=None):
        super().__init__(title, parent)
        self.channel_name = channel_name
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setObjectName(f"debug_dock_{channel_name}")

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(160, 120)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._label.setStyleSheet("background: #101010; color: #808080;")
        self._label.setText(f"{title}\n(очікування кадру…)")
        self.setWidget(self._label)

        self._pixmap: QPixmap | None = None

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Приймає готове BGR-зображення від worker-а і показує scaled."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        self._pixmap = opencv_to_qpixmap(frame_bgr)
        self._apply_scaled()

    def _apply_scaled(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        self._label.setPixmap(
            self._pixmap.scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled()
