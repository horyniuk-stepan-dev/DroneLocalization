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
        """Compute homography matrix between two point sets using MAGSAC++"""
        logger.debug(f"Estimating homography from {len(src_pts)} point pairs")
        logger.debug(f"MAGSAC++ params: threshold={ransac_threshold}, max_iters={max_iters}, confidence={confidence}")

        if len(src_pts) < 4 or len(dst_pts) < 4:
            logger.warning(f"Insufficient points for homography: {len(src_pts)} source, {len(dst_pts)} destination")
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2)

        # Використовуємо надточний алгоритм USAC_MAGSAC замість звичайного RANSAC
        H, mask = cv2.findHomography(
            src_pts_cv, dst_pts_cv,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence
        )

        if H is not None:
            inliers = int(np.sum(mask))
            logger.debug(f"Homography computed: {inliers}/{len(src_pts)} inliers")
        else:
            logger.warning("Failed to compute homography")

        return H, mask

    @staticmethod
    def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply homography to point array"""
        if H is None or len(points) == 0:
            logger.warning("Cannot apply homography: H is None or no points")
            return points

        logger.debug(f"Applying homography to {len(points)} points")

        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.perspectiveTransform(points_cv, H)

        result = transformed_pts_cv.reshape(-1, 2)
        logger.debug(f"Homography applied successfully")

        return result

    @staticmethod
    def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray):
        """Compute affine transformation (2D -> 2D) using MAGSAC++"""
        logger.debug(f"Estimating affine transformation from {len(src_pts)} point pairs")

        if len(src_pts) < 3 or len(dst_pts) < 3:
            logger.warning(f"Insufficient points for affine: {len(src_pts)} source, {len(dst_pts)} destination")
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        # Використовуємо LMEDS або RANSAC для афінних перетворень,
        # оскільки estimateAffine2D не підтримує MAGSAC безпосередньо, але ми оптимізуємо параметри
        M, mask = cv2.estimateAffine2D(
            src_pts_cv, dst_pts_cv,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.999
        )

        if M is not None:
            inliers = int(np.sum(mask))
            logger.debug(f"Affine transformation computed: {inliers}/{len(src_pts)} inliers")
        else:
            logger.warning("Failed to compute affine transformation")

        return M, mask

    @staticmethod
    def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply affine matrix (2x3) to points"""
        if M is None or len(points) == 0:
            logger.warning("Cannot apply affine: M is None or no points")
            return points

        logger.debug(f"Applying affine transformation to {len(points)} points")

        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.transform(points_cv, M)

        result = transformed_pts_cv.reshape(-1, 2)
        logger.debug(f"Affine transformation applied successfully")

        return result