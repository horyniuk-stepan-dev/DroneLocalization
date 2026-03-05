from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QPainter

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoWidget(QGraphicsView):
    """Displays video frames with optional overlay annotations (calibration points)."""

    frame_clicked = pyqtSignal(int, int)   # pixel coords in image space

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._video_item    = QGraphicsPixmapItem()
        self._overlay_items: list = []
        self._scene.addItem(self._video_item)

    # ── Display ──────────────────────────────────────────────────────────────

    def display_frame(self, pixmap: QPixmap):
        self._video_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._video_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        logger.debug(f"Frame: {pixmap.width()}×{pixmap.height()}")

    # ── Overlays ─────────────────────────────────────────────────────────────

    def draw_numbered_point(self, x: int, y: int, label: str, color: QColor):
        """Draw a filled circle with a label at (x, y) in image pixel coordinates."""
        pen    = QPen(color, 2)
        brush  = QBrush(color)
        radius = 8

        ellipse = self._scene.addEllipse(
            x - radius, y - radius, radius * 2, radius * 2, pen, brush
        )
        self._overlay_items.append(ellipse)

        text = self._scene.addText(label)
        text.setDefaultTextColor(QColor(255, 255, 255))

        font = text.font()
        font.setBold(True)
        # Scale text relative to image width, not screen pt — stays readable at any zoom
        img_width = self._video_item.pixmap().width()
        font.setPixelSize(max(12, img_width // 80))
        text.setFont(font)
        text.setPos(x + radius + 2, y - radius - font.pixelSize())

        self._overlay_items.append(text)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._scene.removeItem(item)
            item.setParentItem(None)   # break Qt ownership before Python GC
        self._overlay_items.clear()

    # ── Events ───────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._video_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        # Map viewport → scene → item (image) coordinates
        item_pos = self._video_item.mapFromScene(
            self.mapToScene(event.pos())
        )
        if self._video_item.contains(item_pos):
            self.frame_clicked.emit(int(item_pos.x()), int(item_pos.y()))

        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._video_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
