import cv2
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class GeometryTransforms:
    """Geometric transformations for localization"""

    @staticmethod
    def estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray,
                            ransac_threshold: float = 3.0,
                            max_iters: int = 5000,
                            confidence: float = 0.999):
        logger.debug(f"Estimating homography from {len(src_pts)} point pairs")
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts_cv, dst_pts_cv,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence
        )
        return H, mask

    @staticmethod
    def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        if H is None or len(points) == 0:
            return points
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.perspectiveTransform(points_cv, H)
        return transformed_pts_cv.reshape(-1, 2)

    @staticmethod
    def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray, ransac_threshold: float = 3.0):
        """Compute full Affine transformation (6 DoF: Rotation + Translation + Scale + Shear)"""
        if len(src_pts) < 3 or len(dst_pts) < 3:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffine2D(
            src_pts_cv, dst_pts_cv,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold
        )
        return M, mask

    @staticmethod
    def estimate_affine_partial(src_pts: np.ndarray, dst_pts: np.ndarray, ransac_threshold: float = 3.0):
        """Compute STRICT Affine transformation (Rotation + Translation + Uniform Scale ONLY)"""
        if len(src_pts) < 3 or len(dst_pts) < 3:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        # Використовуємо Partial2D для заборони деформацій (shear) та віддзеркалень
        M, mask = cv2.estimateAffinePartial2D(
            src_pts_cv, dst_pts_cv,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold
        )
        return M, mask

    @staticmethod
    def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
        if M is None or len(points) == 0:
            return points
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.transform(points_cv, M)
        return transformed_pts_cv.reshape(-1, 2)