"""
Geometric transformations
"""

import cv2
import numpy as np


class GeometryTransforms:
    """Geometric transformation utilities"""
    
    @staticmethod
    def estimate_homography(src_pts, dst_pts, method=cv2.RANSAC):
        """Estimate homography matrix"""
        # TODO: Call cv2.findHomography
        # TODO: Return H matrix and mask
        pass
    
    @staticmethod
    def apply_homography(points, H):
        """Apply homography to points"""
        # TODO: Use cv2.perspectiveTransform
        # TODO: Return transformed points
        pass
    
    @staticmethod
    def estimate_affine(src_pts, dst_pts):
        """Estimate affine transformation"""
        # TODO: Call cv2.estimateAffine2D
        # TODO: Return affine matrix
        pass
    
    @staticmethod
    def apply_affine(points, M):
        """Apply affine transformation"""
        # TODO: Apply transformation
        # TODO: Return transformed points
        pass
