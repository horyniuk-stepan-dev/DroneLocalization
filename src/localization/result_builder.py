"""Result assembly for localization (IMPROVEMENT_PLAN 1.1): confidence score,
FOV -> GPS polygon (with exploded-homography guard), and the retrieval-only
fallback.

Stateless: database / calibration / converter are passed in explicitly because
they are switched per-source in multi-database mode. Formulas and log messages
are copied verbatim from Localizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import get_cfg
from src.geometry.point_spread import spread_confidence_factor
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResultBuilder:
    """Confidence, field-of-view polygon and retrieval-only fallback."""

    def __init__(self, config: Any, ransac_thresh: float) -> None:
        self.config = config
        self.ransac_thresh = ransac_thresh

    def compute_confidence(
        self,
        best_candidate_id: int,
        best_inliers: int,
        total_matches: int,
        rmse_val: float,
        database: Any,
        spread: float | None = None,
    ) -> float:
        """Confidence from DB QA (rmse/disagreement) + inliers + match ratio/RMSE.

        ``spread`` (ADDENDUM 1.1) — просторовий розкид інлаєрів у кадрі,
        ``src.geometry.point_spread.inlier_spread``. ``None`` = сигнал
        недоступний → множник 1.0. Застосовується лише за прапорцем
        ``localization.spread_confidence_enabled``.
        """
        max_inliers = get_cfg(self.config, "localization.confidence.confidence_max_inliers", 80)
        rmse_norm = get_cfg(self.config, "localization.confidence.rmse_norm_m", 10.0)
        diag_norm = get_cfg(self.config, "localization.confidence.disagreement_norm_m", 5.0)
        w_inlier = get_cfg(self.config, "localization.confidence.inlier_weight", 0.7)
        w_stability = get_cfg(self.config, "localization.confidence.stability_weight", 0.3)

        inlier_score = min(1.0, best_inliers / max_inliers)

        rmse = (
            database.frame_rmse[best_candidate_id]
            if database.frame_rmse is not None
            else 0.0
        )
        disagreement = (
            database.frame_disagreement[best_candidate_id]
            if database.frame_disagreement is not None
            else 0.0
        )

        stability_score = 1.0 - (
            min(rmse, rmse_norm) / rmse_norm * 0.5
            + min(disagreement, diag_norm) / diag_norm * 0.5
        )
        stability_score = float(np.clip(stability_score, 0.0, 1.0))

        ratio_score = float(best_inliers / (total_matches + 1e-6))
        rmse_score_val = 1.0 / (1.0 + (rmse_val / (self.ransac_thresh + 1e-6)))
        match_score = ratio_score * 0.5 + rmse_score_val * 0.5

        final_conf = stability_score * 0.3 + inlier_score * 0.4 + match_score * 0.3

        # ADDENDUM 1.1: скупчені інлаєри → ill-conditioned H. Множник, а не
        # відкидання: далі confidence керує R у Kalman (B2), тож слабкий фікс
        # просто важить менше. На межі покриття скупчення легітимне.
        if get_cfg(self.config, "localization.spread_confidence_enabled", False):
            factor = spread_confidence_factor(
                spread,
                spread_ref=get_cfg(self.config, "localization.spread_ref", 0.15),
                floor=get_cfg(self.config, "localization.spread_floor", 0.35),
            )
            if factor < 1.0:
                logger.debug(f"Spatial collapse: spread={spread:.4f} → confidence ×{factor:.2f}")
            final_conf *= factor

        return float(np.clip(final_conf, 0.05, 1.0))

    def fallback(self, frame_id: int, score: float, database: Any, calibration: Any) -> dict | None:
        """Approximate localization at the reference-frame centre (retrieval-only)."""
        if frame_id == -1:
            return None

        threshold = get_cfg(self.config, "localization.retrieval_only_min_score", 0.90)
        if score < threshold:
            logger.debug(
                f"Retrieval-only fallback rejected: score {score:.3f} < threshold {threshold:.3f} | "
                f"frame={frame_id}"
            )
            return None

        affine_ref = database.get_frame_affine(frame_id)
        if affine_ref is None:
            logger.debug(
                f"Retrieval-only fallback failed: no affine matrix for frame {frame_id}. "
                f"Frame not reached during calibration propagation."
            )
            return None

        ref_h, ref_w = database.get_frame_size(frame_id)
        center_ref = np.array([[ref_w / 2, ref_h / 2]], dtype=np.float64)
        metric_pt = GeometryTransforms.apply_affine(center_ref, affine_ref)[0]

        lat, lon = calibration.converter.metric_to_gps(metric_pt[0], metric_pt[1])

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": 0.3,
            "inliers": 0,
            "matched_frame": frame_id,
            "fallback_mode": "retrieval_only",
            "global_score": score,
            "fov_polygon": None,
        }

    def build_fov(
        self, M_query_to_ref: Any, affine_ref: Any, rot_width: int, rot_height: int,
        mkpts_q_inliers: Any, converter: Any, dx: float, dy: float,
        mx: float, my: float, filtered_pt: Any, candidate_id: int,
    ) -> list:
        """Project the frame FOV to a GPS polygon, guarding against exploded homographies."""
        corners = np.array(
            [[0, 0], [rot_width, 0], [rot_width, rot_height], [0, rot_height]],
            dtype=np.float32,
        )
        ref_corners = GeometryTransforms.apply_homography(corners, M_query_to_ref)

        is_exploded = False
        if ref_corners is not None:
            max_coord = np.max(np.abs(ref_corners))
            if max_coord > 50000:
                is_exploded = True

        if is_exploded and mkpts_q_inliers is not None and len(mkpts_q_inliers) > 0:
            logger.warning(
                f"Homography exploded the FOV (max_coord={max_coord:.0f}px > 50000px threshold). "
                f"Cause: perspective distortion from locally-clustered ALIKED matches. "
                f"Falling back to inliers bounding box for safe FOV estimation."
            )
            pts = mkpts_q_inliers
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            pad_x, pad_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
            safe_corners = np.array(
                [
                    [max(0, min_x - pad_x), max(0, min_y - pad_y)],
                    [min(rot_width, max_x + pad_x), max(0, min_y - pad_y)],
                    [min(rot_width, max_x + pad_x), min(rot_height, max_y + pad_y)],
                    [max(0, min_x - pad_x), min(rot_height, max_y + pad_y)],
                ],
                dtype=np.float32,
            )
            ref_corners = GeometryTransforms.apply_homography(safe_corners, M_query_to_ref)
            original_poly_px = safe_corners
        else:
            original_poly_px = corners
            logger.debug("FOV projected using full frame Homography matrix.")

        logger.debug(f"--- FOV DIAGNOSTICS FOR FRAME {candidate_id} ---")
        w_px = np.linalg.norm(original_poly_px[0] - original_poly_px[1])
        h_px = np.linalg.norm(original_poly_px[1] - original_poly_px[2])
        logger.debug(f"[1] Original FOV in Query image: {w_px:.1f} x {h_px:.1f} pixels")

        if ref_corners is not None:
            w_ref = np.linalg.norm(ref_corners[0] - ref_corners[1])
            h_ref = np.linalg.norm(ref_corners[1] - ref_corners[2])
            logger.debug(
                f"[2] FOV mapped to Reference via Homography: {w_ref:.1f} x {h_ref:.1f} pixels"
            )

        gps_corners = []
        if ref_corners is not None:
            metric_corners = GeometryTransforms.apply_affine(ref_corners, affine_ref)
            if metric_corners is not None:
                fov_w = np.linalg.norm(metric_corners[1] - metric_corners[0])
                fov_h = np.linalg.norm(metric_corners[3] - metric_corners[0])
                logger.debug(
                    f"[3] FOV mapped to metric space: {fov_w:.1f}m x {fov_h:.1f}m"
                )
                logger.debug(
                    f"FOV dimensions: {fov_w:.1f}m x {fov_h:.1f}m | "
                    f"Center metric: ({mx:.1f}, {my:.1f}) | "
                    f"Filtered: ({filtered_pt[0]:.1f}, {filtered_pt[1]:.1f})"
                )
                for cx, cy in metric_corners:
                    try:
                        clat, clon = converter.metric_to_gps(
                            float(cx + dx), float(cy + dy)
                        )
                        gps_corners.append((clat, clon))
                    except Exception:
                        pass

        return gps_corners
