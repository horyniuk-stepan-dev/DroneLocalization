from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPen, QColor, QFont

class VideoWidget(QGraphicsView):
    frame_clicked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    def display_frame(self, pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def draw_bounding_box(self, x, y, w, h, label="", confidence=0.0):
        pen = QPen(QColor(0, 255, 0), 2)
        self.scene.addRect(x, y, w, h, pen)
        
        if label:
            text = self.scene.addText(f"{label} {confidence:.2f}")
            text.setDefaultTextColor(QColor(0, 255, 0))
            text.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            text.setPos(x, y - 20)
    
    def clear_overlays(self):
        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.frame_clicked.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mousePressEvent(event)