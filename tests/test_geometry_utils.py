import numpy as np
import pytest
import cv2
from src.geometry.transformations import GeometryTransforms

def test_apply_homography_identity():
    points = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    transformed = GeometryTransforms.apply_homography(points, H)
    assert np.allclose(points, transformed)

def test_apply_homography_translation():
    points = np.array([[10, 20]], dtype=np.float32)
    # Translate by x=5, y=10
    H = np.array([
        [1, 0, 5],
        [0, 1, 10],
        [0, 0, 1]
    ], dtype=np.float32)
    expected = np.array([[15, 30]], dtype=np.float32)
    transformed = GeometryTransforms.apply_homography(points, H)
    assert np.allclose(transformed, expected)

def test_apply_affine_identity():
    points = np.array([[10, 20], [30, 40]], dtype=np.float32)
    M = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    transformed = GeometryTransforms.apply_affine(points, M)
    assert np.allclose(points, transformed)

def test_apply_affine_translation():
    points = np.array([[10, 20]], dtype=np.float32)
    # Translate by x=5, y=10
    M = np.array([
        [1, 0, 5],
        [0, 1, 10]
    ], dtype=np.float32)
    expected = np.array([[15, 30]], dtype=np.float32)
    transformed = GeometryTransforms.apply_affine(points, M)
    assert np.allclose(transformed, expected)

def test_edge_cases():
    # Empty points
    empty_pts = np.array([], dtype=np.float32).reshape(0, 2)
    H = np.eye(3)
    assert GeometryTransforms.apply_homography(empty_pts, H).shape == (0, 2)
    
    # None matrix
    points = np.array([[10, 20]])
    assert np.array_equal(GeometryTransforms.apply_homography(points, None), points)
    assert np.array_equal(GeometryTransforms.apply_affine(points, None), points)

def test_estimate_affine():
    src = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
    dst = np.array([[10, 10], [110, 10], [10, 110]], dtype=np.float32)
    
    M, mask = GeometryTransforms.estimate_affine(src, dst)
    assert M is not None
    assert M.shape == (2, 3)
    # Should be identity rotation + [10, 10] translation
    expected_M = np.array([[1, 0, 10], [0, 1, 10]], dtype=np.float32)
    assert np.allclose(M, expected_M, atol=1e-5)
