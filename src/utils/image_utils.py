"""
Image processing utilities
"""

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def opencv_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """Convert OpenCV image to QPixmap"""
    # TODO: Convert BGR to RGB
    # TODO: Create QImage
    # TODO: Convert to QPixmap
    # TODO: Return QPixmap
    pass


def qpixmap_to_opencv(pixmap: QPixmap) -> np.ndarray:
    """Convert QPixmap to OpenCV image"""
    # TODO: Convert QPixmap to QImage
    # TODO: Convert to numpy array
    # TODO: Convert RGB to BGR
    # TODO: Return numpy array
    pass
