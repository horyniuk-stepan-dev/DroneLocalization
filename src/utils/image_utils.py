import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def opencv_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """Converts OpenCV (BGR) image to PyQt6 QPixmap (RGB)"""
    if cv_image is None or cv_image.size == 0:
        return QPixmap()

    if len(cv_image.shape) == 3:
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width

        # Format_BGR888 (Qt ≥ 5.14) reads BGR directly — eliminates a full
        # cvtColor(BGR2RGB) per frame (called ~30 times/s on the GUI thread).
        buf = np.ascontiguousarray(cv_image)
        q_img = QImage(buf.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

        # QPixmap.fromImage makes a deep copy into its own storage while buf
        # is alive in this scope — the extra q_img.copy() was a redundant frame copy.
        return QPixmap.fromImage(q_img)

    elif len(cv_image.shape) == 2:
        height, width = cv_image.shape
        bytes_per_line = width

        gray = np.ascontiguousarray(cv_image)
        q_img = QImage(gray.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(q_img)

    return QPixmap()


def qpixmap_to_opencv(pixmap: QPixmap) -> np.ndarray:
    """Converts QPixmap (RGB) to an OpenCV array (BGR)."""
    q_img = pixmap.toImage()
    q_img = q_img.convertToFormat(QImage.Format.Format_RGB888)

    width = q_img.width()
    height = q_img.height()

    ptr = q_img.bits()
    ptr.setsize(height * width * 3)

    # Creates an independent numpy array copy to prevent loss of the QImage buffer.
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)).copy()

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
