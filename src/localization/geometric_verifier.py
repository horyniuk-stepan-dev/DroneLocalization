"""Geometric verification: candidate matching + RANSAC homography (IMPROVEMENT_PLAN 1.1).

Extracted verbatim from Localizer.localize_frame's candidate loop. Stateless: the
frame database is passed to ``verify`` explicitly because it is switched per-source
in multi-database mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    candidate_id: int
    H_query_to_ref: np.ndarray
    inliers: int
    mkpts_q_in: np.ndarray
    mkpts_r_in: np.ndarray
    total_matches: int
    rmse: float


class GeometricVerifier:
    """Match each candidate, estimate a homography, keep the best (with early stop)."""

    def __init__(self, matcher: Any, min_matches: int, ransac_thresh: float,
                 homography_backend: str, use_mad_ransac: bool, mad_k_factor: float,
                 early_stop_inliers: int) -> None:
        self.matcher = matcher
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self.homography_backend = homography_backend
        self.use_mad_ransac = use_mad_ransac
        self.mad_k_factor = mad_k_factor
        self.early_stop_inliers = early_stop_inliers

    def verify(self, query_features: dict, candidates: list, database: Any
               ) -> VerificationResult | None:
        best_inliers = 0
        best_candidate_id = -1
        best_H_query_to_ref = None
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0
        best_rmse = 999.0

        early_stop = self.early_stop_inliers

        for candidate_id, score in candidates:
            logger.debug(f"  → Trying candidate {candidate_id} (global_score={score:.3f})")
            ref_features = database.get_local_features(candidate_id)

            with Telemetry.profile("match"):
                mkpts_q, mkpts_r = self.matcher.match(query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                with Telemetry.profile("ransac_homography"):
                    H_eval, mask = GeometryTransforms.estimate_homography(
                        mkpts_q,
                        mkpts_r,
                        ransac_threshold=self.ransac_thresh,
                        backend=self.homography_backend,
                        use_mad_ransac=self.use_mad_ransac,
                        mad_k_factor=self.mad_k_factor,
                    )

                if H_eval is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(np.sum(inlier_mask))
                    pts_q_in = mkpts_q[inlier_mask]
                    pts_r_in = mkpts_r[inlier_mask]

                    pts_q_transformed = GeometryTransforms.apply_homography(pts_q_in, H_eval)
                    rmse = float(
                        np.sqrt(np.mean(np.sum((pts_q_transformed - pts_r_in) ** 2, axis=1)))
                    )

                    if inliers > best_inliers and inliers >= self.min_matches:
                        best_inliers = inliers
                        best_candidate_id = candidate_id
                        best_H_query_to_ref = H_eval
                        best_mkpts_q_inliers = pts_q_in
                        best_mkpts_r_inliers = pts_r_in
                        best_total_matches = len(mkpts_q)
                        best_rmse = rmse
                        logger.debug(
                            f"Homography for {candidate_id}: {inliers} inliers, RMSE: {rmse:.2f}"
                        )

            if best_inliers >= early_stop:
                logger.debug(
                    f"Early stop triggered with {best_inliers} inliers on candidate {best_candidate_id}"
                )
                break

        if (
            best_inliers < self.min_matches
            or best_mkpts_r_inliers is None
            or best_H_query_to_ref is None
        ):
            return None

        return VerificationResult(
            candidate_id=best_candidate_id,
            H_query_to_ref=best_H_query_to_ref,
            inliers=best_inliers,
            mkpts_q_in=best_mkpts_q_inliers,
            mkpts_r_in=best_mkpts_r_inliers,
            total_matches=best_total_matches,
            rmse=best_rmse,
        )
