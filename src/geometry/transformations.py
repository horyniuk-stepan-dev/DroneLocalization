import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GeometryTransforms:
    """Geometric transformations for localization with robust estimation (MAGSAC++)"""

    @staticmethod
    def is_matrix_valid(
        M: np.ndarray,
        is_homography: bool = False,
        min_scale: float = 0.001,
        max_scale: float = 100.0,
        max_shear: float = 0.8,
    ) -> bool:
        """
        Check if the transformation matrix is physically realistic for drone imagery.

        Args:
            M: Transformation matrix (2x3 for Affine or 3x3 for Homography)
            is_homography: True if M is a 3x3 Homography matrix
            min_scale: Minimum allowed scale factor
            max_scale: Maximum allowed scale factor
            max_shear: Maximum allowed shear (dot product of normalized basis vectors)
        """
        if M is None:
            return False

        try:
            if is_homography:
                # For Homography, we care about the affine part for stability checks
                if M.shape != (3, 3):
                    return False
                # Normalize by M[2,2] if possible
                if abs(M[2, 2]) < 1e-9:
                    return False
                M = M / M[2, 2]

                # Check perspective components (should be very small for top-down drone imagery)
                # If these are large, the corners will fly off to infinity
                if abs(M[2, 0]) > 0.005 or abs(M[2, 1]) > 0.005:
                    logger.debug(
                        f"Matrix invalid: Extreme perspective warp ({M[2, 0]:.5f}, {M[2, 1]:.5f})"
                    )
                    return False

                A = M[:2, :2]
                det = np.linalg.det(A)
            else:
                if M.shape != (2, 3):
                    return False
                A = M[:2, :2]
                det = np.linalg.det(A)

            # 1. Determinant must be non-zero (prevent degenerate matrices)
            if abs(det) < 1e-9:
                logger.debug(
                    f"Matrix invalid: Degenerate matrix with determinant near zero ({det})"
                )
                return False
            # Allow negative determinant since mapping Image Y (down) to Map Y (up) requires reflection!

            # 2. Extract scale and shear from basis vectors
            u = A[:, 0]
            v = A[:, 1]
            scale_u = np.linalg.norm(u)
            scale_v = np.linalg.norm(v)

            # Check scale bounds (drone altitude/zoom sanity)
            if not (min_scale < scale_u < max_scale and min_scale < scale_v < max_scale):
                logger.debug(f"Matrix invalid: Scale out of bounds ({scale_u:.2f}, {scale_v:.2f})")
                return False

            # 3. Check Aspect Ratio (should be close to 1.0 for drone imagery)
            aspect_ratio = scale_u / (scale_v + 1e-9)
            if not (0.5 < aspect_ratio < 2.0):
                logger.debug(
                    f"Matrix invalid: Extreme aspect ratio distortion ({aspect_ratio:.2f})"
                )
                return False

            # 4. Check Shear (cos of angle between basis vectors)
            shear = abs(np.dot(u, v) / (scale_u * scale_v + 1e-9))
            if shear > max_shear:
                logger.debug(f"Matrix invalid: Extreme shear detected ({shear:.2f} > {max_shear})")
                return False

            return True

        except Exception as e:
            logger.error(f"Error during matrix validation: {e}")
            return False

    @staticmethod
    def estimate_homography(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 3.0,
        max_iters: int = 2000,
        confidence: float = 0.99,
        fallback_to_affine: bool = True,
    ):
        """
        Estimate Homography using MAGSAC++ with validation and optional fallback.
        """
        if len(src_pts) < 4:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        # Use standard RANSAC instead of USAC_MAGSAC for stability in OpenCV
        H, mask = cv2.findHomography(
            src_pts_cv,
            dst_pts_cv,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )

        # Validate Homography
        if not GeometryTransforms.is_matrix_valid(H, is_homography=True):
            if fallback_to_affine:
                logger.warning("Homography invalid/degenerate, falling back to Partial Affine")
                return GeometryTransforms.estimate_affine_partial(
                    src_pts, dst_pts, ransac_threshold
                )
            return None, None

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
        """Compute full Affine transformation (6 DoF) using MAGSAC++"""
        if len(src_pts) < 3:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffine2D(
            src_pts_cv, dst_pts_cv, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

        if not GeometryTransforms.is_matrix_valid(M, is_homography=False):
            return None, None

        return M, mask

    @staticmethod
    def estimate_affine_partial(
        src_pts: np.ndarray, dst_pts: np.ndarray, ransac_threshold: float = 3.0
    ):
        """Compute STRICT Affine transformation (4 DoF: R+T+S only) using MAGSAC++"""
        if len(src_pts) < 2:  # Partial needs only 2 points minimum
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffinePartial2D(
            src_pts_cv, dst_pts_cv, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

        if not GeometryTransforms.is_matrix_valid(M, is_homography=False):
            return None, None

        return M, mask

    @staticmethod
    def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
        if M is None or len(points) == 0:
            return points
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.transform(points_cv, M)
        return transformed_pts_cv.reshape(-1, 2)
