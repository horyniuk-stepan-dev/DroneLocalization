from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QPainter

from src.utils.logging_utils import get_logger  # ВИПРАВЛЕНО: було 'from utils...'


class VideoWidget(QGraphicsView):
    """Віджет для відображення відео з накладеннями"""

    frame_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.logger = get_logger('VideoWidget')

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.video_item = QGraphicsPixmapItem()
        self.scene.addItem(self.video_item)

        self.overlay_items = []

    def display_frame(self, pixmap: QPixmap):
        self.video_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.video_item.boundingRect())
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def draw_numbered_point(self, x: int, y: int, text: str, color: QColor):
        """Малює велику точку з номером поруч"""
        pen = QPen(color)
        pen.setWidth(2)
        brush = QBrush(color)
        radius = 8

        ellipse_item = self.scene.addEllipse(x - radius, y - radius, radius * 2, radius * 2, pen, brush)
        self.overlay_items.append(ellipse_item)

        text_item = self.scene.addText(text)
        text_item.setDefaultTextColor(QColor(255, 255, 255))

        font = text_item.font()
        font.setBold(True)
        font.setPointSize(16)
        text_item.setFont(font)

        text_item.setPos(x + radius + 2, y - radius - 20)
        self.overlay_items.append(text_item)

    def clear_overlays(self):
        for item in self.overlay_items:
            self.scene.removeItem(item)
        self.overlay_items.clear()

    def mousePressEvent(self, event):
        if self.video_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        if 0 <= x <= self.video_item.pixmap().width() and 0 <= y <= self.video_item.pixmap().height():
            self.frame_clicked.emit(x, y)

        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.video_item.pixmap().isNull():
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)