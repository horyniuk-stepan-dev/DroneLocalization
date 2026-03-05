import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def rgb_to_qpixmap(rgb_image: np.ndarray) -> QPixmap:
    """Convert RGB numpy array (project-internal format) to QPixmap.

    Args:
        rgb_image: HxWx3 uint8 array in RGB order (as used throughout the project).
    """
    if rgb_image is None or rgb_image.size == 0:
        return QPixmap()

    if rgb_image.ndim == 3:
        height, width, channels = rgb_image.shape
        img = np.ascontiguousarray(rgb_image)
        bytes_per_line = channels * width
        fmt = QImage.Format.Format_RGB888
        q_img = QImage(img.data, width, height, bytes_per_line, fmt)
        return QPixmap.fromImage(q_img.copy())

    if rgb_image.ndim == 2:
        height, width = rgb_image.shape
        img = np.ascontiguousarray(rgb_image)
        q_img = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_img.copy())

    return QPixmap()


def bgr_to_qpixmap(bgr_image: np.ndarray) -> QPixmap:
    """Convert BGR numpy array (cv2.VideoCapture/imread output) to QPixmap."""
    if bgr_image is None or bgr_image.size == 0:
        return QPixmap()
    if bgr_image.ndim == 3:
        return rgb_to_qpixmap(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    return rgb_to_qpixmap(bgr_image)


# Backward-compatible alias — callers that passed BGR can keep using this
opencv_to_qpixmap = bgr_to_qpixmap


def qpixmap_to_opencv(pixmap: QPixmap) -> np.ndarray:
    """Convert QPixmap to BGR numpy array (OpenCV format).

    Returns an empty array if pixmap is null.
    """
    if pixmap is None or pixmap.isNull():
        return np.zeros((0, 0, 3), dtype=np.uint8)

    q_img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    width, height = q_img.width(), q_img.height()

    ptr = q_img.bits()
    # bytesPerLine accounts for stride/alignment padding (avoids reshape error on odd widths)
    bytes_per_line = q_img.bytesPerLine()
    ptr.setsize(height * bytes_per_line)

    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
    arr = arr[:, :width * 3].reshape((height, width, 3)).copy()

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
