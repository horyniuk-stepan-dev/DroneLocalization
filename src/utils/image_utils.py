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

        # A7: Format_BGR888 (Qt ≥ 5.14) читає BGR напряму — прибирає повний
        # cvtColor(BGR2RGB) кадру на кожен виклик (30 разів/с на GUI-потоці).
        buf = np.ascontiguousarray(cv_image)
        q_img = QImage(buf.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

        # QPixmap.fromImage робить глибоку копію у власне сховище, поки buf
        # живий у цьому scope — додатковий q_img.copy() був зайвою копією кадру.
        return QPixmap.fromImage(q_img)

    elif len(cv_image.shape) == 2:
        height, width = cv_image.shape
        bytes_per_line = width

        gray = np.ascontiguousarray(cv_image)
        q_img = QImage(gray.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(q_img)

    return QPixmap()


def qpixmap_to_opencv(pixmap: QPixmap) -> np.ndarray:
    """Перетворення QPixmap (RGB) у масив OpenCV (BGR)"""
    q_img = pixmap.toImage()
    q_img = q_img.convertToFormat(QImage.Format.Format_RGB888)

    width = q_img.width()
    height = q_img.height()

    ptr = q_img.bits()
    ptr.setsize(height * width * 3)

    # ВИПРАВЛЕНО: робимо copy() щоб масив numpy не залежав від буфера
    # QImage, який може бути знищений після виходу з функції
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)).copy()

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
