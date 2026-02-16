import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

def opencv_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """Перетворення зображення OpenCV (BGR) у QPixmap (RGB) для PyQt6"""
    if cv_image is None or cv_image.size == 0:
        return QPixmap()
        
    if len(cv_image.shape) == 3:
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    elif len(cv_image.shape) == 2:
        height, width = cv_image.shape
        bytes_per_line = width
        q_img = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    else:
        return QPixmap()
        
    return QPixmap.fromImage(q_img)

def qpixmap_to_opencv(pixmap: QPixmap) -> np.ndarray:
    """Перетворення QPixmap (RGB) у масив OpenCV (BGR)"""
    q_img = pixmap.toImage()
    q_img = q_img.convertToFormat(QImage.Format.Format_RGB888)
    
    width = q_img.width()
    height = q_img.height()
    
    ptr = q_img.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)