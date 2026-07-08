"""Rotation + scale selection for localization (IMPROVEMENT_PLAN 1.1 + SCALE_INVARIANCE).

A3 temporal-angle prior (1 global-descriptor forward pass reusing the last good
angle) with fallback to the A2 batched 4-angle scan. Now extended with ScaleManager:
when scale_manager provides multiple candidates, the selector generates (angle × scale)
combinations in a single batched forward pass.

In steady-state (prior angle + prior scale): 1 forward pass — same as before.
Bootstrap / recovery: up to 20 variants (4 rotations × 5 scales), batched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import get_cfg
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


@dataclass
class RotationResult:
    angle: int
    score: float
    candidates: list
    source_id: str | None
    best_scale: float = 1.0


class RotationSelector:
    """Pick the best frame rotation (and scale) via global-descriptor retrieval."""

    def __init__(self, feature_extractor: Any, candidate_retriever: Any, config: Any) -> None:
        self.feature_extractor = feature_extractor
        self._candidate_retriever = candidate_retriever
        self.config = config

    def select(
        self, query_frame: Any, prior_angle: int | None, use_prior: bool,
        angles_to_try: list[int], top_k: int,
        scale_manager: Any = None,
    ) -> RotationResult | None:
        best_global_score = -1.0
        best_global_angle = 0
        best_global_candidates = []
        best_source_id_per_angle: str | None = None
        best_scale: float = 1.0

        rescan_min = get_cfg(self.config, "localization.rotation_rescan_min_score", 0.70)

        # Determine scale candidates
        if scale_manager is not None:
            scale_candidates = scale_manager.candidates()
            scale_rescan_min = scale_manager.rescan_min_score
        else:
            scale_candidates = [1.0]
            scale_rescan_min = rescan_min

        # A3: reuse the last good angle (+ prior scale) first; full rescan only
        # if its retrieval score dips below the threshold.
        if use_prior and prior_angle is not None:
            # In steady state, scale_candidates is [prior] — 1 element.
            # We try the prior angle with each scale candidate (usually 1).
            prior_scales = scale_candidates if len(scale_candidates) == 1 else [scale_candidates[0]]
            for sc in prior_scales:
                rotated_frame = np.ascontiguousarray(np.rot90(query_frame, k=prior_angle // 90))
                if scale_manager is not None and abs(sc - 1.0) > 0.15:
                    rotated_frame, _ = scale_manager.normalize(rotated_frame, sc)
                global_desc = self.feature_extractor.extract_global_descriptor(rotated_frame)
                with Telemetry.profile("retrieval"):
                    src_id, candidates = self._candidate_retriever.retrieve(global_desc, top_k)
                if candidates and candidates[0][1] >= min(rescan_min, scale_rescan_min):
                    if candidates[0][1] > best_global_score:
                        best_global_score = candidates[0][1]
                        best_global_angle = prior_angle
                        best_global_candidates = candidates
                        best_source_id_per_angle = src_id
                        best_scale = sc
                else:
                    logger.debug(
                        f"Prior angle {prior_angle}° scale {sc:.2f} score too low "
                        f"({candidates[0][1] if candidates else -1:.3f} < {rescan_min}) — full rescan"
                    )

        if not best_global_candidates:
            # A2: all rotations × all scales in ONE batched forward pass.
            combos = []
            frames = []
            for a in angles_to_try:
                rotated = np.ascontiguousarray(np.rot90(query_frame, k=a // 90))
                for sc in scale_candidates:
                    if scale_manager is not None and abs(sc - 1.0) > 0.15:
                        scaled, _ = scale_manager.normalize(rotated, sc)
                    else:
                        scaled = rotated
                    combos.append((a, sc))
                    frames.append(scaled)

            # Batch extraction
            if len(frames) > 1 and hasattr(
                self.feature_extractor, "extract_global_descriptors_multi"
            ):
                descs = self.feature_extractor.extract_global_descriptors_multi(frames)
            else:
                descs = [
                    self.feature_extractor.extract_global_descriptor(f)
                    for f in frames
                ]

            for (angle, sc), global_desc in zip(combos, descs):
                with Telemetry.profile("retrieval"):
                    src_id, candidates = self._candidate_retriever.retrieve(global_desc, top_k)

                if candidates:
                    top_score = candidates[0][1]
                    if top_score > best_global_score:
                        best_global_score = top_score
                        best_global_angle = angle
                        best_global_candidates = candidates
                        best_source_id_per_angle = src_id
                        best_scale = sc

        if not best_global_candidates:
            return None

        return RotationResult(
            angle=best_global_angle,
            score=best_global_score,
            candidates=best_global_candidates,
            source_id=best_source_id_per_angle,
            best_scale=best_scale,
        )
