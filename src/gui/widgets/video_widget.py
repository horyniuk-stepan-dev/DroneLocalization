from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoWidget(QGraphicsView):
    """Displays video frames with optional overlay annotations (calibration points)."""

    frame_clicked = pyqtSignal(int, int)  # pixel coords in image space

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._video_item = QGraphicsPixmapItem()
        self._overlay_items: list = []
        self._scene.addItem(self._video_item)

    # ── Display ──────────────────────────────────────────────────────────────

    def display_frame(self, pixmap: QPixmap):
        self._video_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._video_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        logger.debug(f"Frame: {pixmap.width()}×{pixmap.height()}")

    # ── Overlays ─────────────────────────────────────────────────────────────

    def _dpr(self) -> float:
        """Device pixel ratio поточного pixmap (1.0 на 100% DPI, 2.0 на 200%)."""
        pm = self._video_item.pixmap()
        return pm.devicePixelRatio() if pm and not pm.isNull() else 1.0

    def draw_numbered_point(self, x: int, y: int, label: str, color: QColor):
        """Draw a filled circle with a label at (x, y) in ACTUAL image pixel coordinates."""
        # Конвертуємо з фактичних пікселів у логічні координати сцени
        dpr = self._dpr()
        lx, ly = x / dpr, y / dpr

        pen = QPen(color, 2)
        brush = QBrush(color)
        radius = 8

        ellipse = self._scene.addEllipse(
            lx - radius, ly - radius, radius * 2, radius * 2, pen, brush
        )
        self._overlay_items.append(ellipse)

        text = self._scene.addText(label)
        text.setDefaultTextColor(QColor(255, 255, 255))

        font = text.font()
        font.setBold(True)
        # Scale text relative to logical image width
        img_width = self._video_item.boundingRect().width()
        font.setPixelSize(max(12, int(img_width) // 80))
        text.setFont(font)
        text.setPos(lx + radius + 2, ly - radius - font.pixelSize())

        self._overlay_items.append(text)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._scene.removeItem(item)
            item.setParentItem(None)  # break Qt ownership before Python GC
        self._overlay_items.clear()

    # ── Events ───────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._video_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        # Map viewport → scene → item
        scene_pos = self.mapToScene(event.pos())
        item_pos = self._video_item.mapFromScene(scene_pos)

        if self._video_item.contains(item_pos):
            pm = self._video_item.pixmap()
            pm_dpr = self._dpr()
            br = self._video_item.boundingRect()

            # Screen devicePixelRatio — може відрізнятися від pixmap dpr
            screen = self.screen()
            screen_dpr = screen.devicePixelRatio() if screen else 1.0

            # Якщо boundingRect ≠ pixmap size, масштабуємо вручну
            if br.width() > 0 and br.height() > 0:
                scale_x = pm.width() / br.width()
                scale_y = pm.height() / br.height()
            else:
                scale_x = scale_y = 1.0

            # Множимо на pm_dpr (Device Pixel Ratio), оскільки Qt на High-DPI
            # повертає "логічні" координати (напр. 1280 замість 1920).
            # Нам потрібні ФІЗИЧНІ пікселі зображення для метчингу бази даних.
            actual_x = int(item_pos.x() * scale_x * pm_dpr)
            actual_y = int(item_pos.y() * scale_y * pm_dpr)

            logger.debug(
                f"CLICK DIAG: "
                f"event=({event.pos().x()},{event.pos().y()}) "
                f"scene=({scene_pos.x():.0f},{scene_pos.y():.0f}) "
                f"item=({item_pos.x():.0f},{item_pos.y():.0f}) "
                f"actual=({actual_x},{actual_y}) "
                f"pm_dpr={pm_dpr} screen_dpr={screen_dpr} "
                f"pixmap={pm.width()}x{pm.height()} "
                f"bRect={br.width():.0f}x{br.height():.0f} "
                f"viewport={self.viewport().width()}x{self.viewport().height()} "
                f"sceneRect={self.sceneRect().width():.0f}x{self.sceneRect().height():.0f}"
            )
            self.frame_clicked.emit(actual_x, actual_y)

        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._video_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
