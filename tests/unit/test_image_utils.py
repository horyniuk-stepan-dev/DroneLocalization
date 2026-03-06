import numpy as np
import cv2
import pytest
from PyQt6.QtGui import QPixmap, QImage, QColor
from src.utils.image_utils import opencv_to_qpixmap, qpixmap_to_opencv

def test_opencv_to_qpixmap_rgb(qapp):
    """Test generating a QPixmap from a BGR OpenCV image."""
    # Create a 100x100 BGR image (red color, B=0, G=0, R=255)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)
    
    pixmap = opencv_to_qpixmap(img)
    assert not pixmap.isNull()
    assert pixmap.width() == 100
    assert pixmap.height() == 100
    
    # Check pixels: it should be converted to RGB for PyQt
    qimage = pixmap.toImage()
    assert qimage.pixelColor(50, 50).red() == 255
    assert qimage.pixelColor(50, 50).blue() == 0

def test_opencv_to_qpixmap_gray(qapp):
    """Test QPixmap generation from grayscale OpenCV array."""
    # Create a 100x100 grayscale image
    img = np.ones((100, 100), dtype=np.uint8) * 128
    
    pixmap = opencv_to_qpixmap(img)
    assert not pixmap.isNull()
    assert pixmap.width() == 100
    assert pixmap.height() == 100
    
def test_opencv_to_qpixmap_empty(qapp):
    """Test empty/None arrays."""
    img = np.array([])
    pixmap = opencv_to_qpixmap(img)
    assert pixmap.isNull()
    
    pixmap_none = opencv_to_qpixmap(None)
    assert pixmap_none.isNull()

def test_qpixmap_to_opencv(qapp):
    """Test returning OpenCV numpy array from QPixmap."""
    qimage = QImage(100, 100, QImage.Format.Format_RGB888)
    qimage.fill(QColor(255, 0, 0)) # Red in PyQt
    
    pixmap = QPixmap.fromImage(qimage)
    cv_img = qpixmap_to_opencv(pixmap)
    
    assert cv_img is not None
    assert cv_img.shape == (100, 100, 3)
    
    # Expected output in OpenCV should be BGR, so Red is (0, 0, 255)
    assert np.array_equal(cv_img[50, 50], [0, 0, 255])
