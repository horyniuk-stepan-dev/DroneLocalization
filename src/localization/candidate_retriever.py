"""Candidate retrieval + patchify expansion for localization (IMPROVEMENT_PLAN 1.1).

Extracted from Localizer as a stateless collaborator: single- or multi-database
top-k retrieval, plus optional patchify merge for the chosen rotation. Logic and
log messages are unchanged.
"""

from __future__ import annotations

from config import get_cfg
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


class CandidateRetriever:
    """Top-k retrieval (single/multi DB) with optional patch-descriptor merge."""

    def __init__(self, db_manager, retriever, patchify_retrieval, config) -> None:
        self.db_manager = db_manager
        self.retriever = retriever
        self.patchify_retrieval = patchify_retrieval
        self.config = config

    def retrieve(self, global_desc, top_k: int):
        """Retrieve candidates: (source_id | None, [(frame_id, score), ...])."""
        if self.db_manager is not None:
            return self.db_manager.get_best_match(global_desc, top_k=top_k)
        return None, self.retriever.find_similar_frames(global_desc, top_k=top_k)

    def expand(self, rotated_frame, base_candidates, top_k: int):
        """Patchify-expand candidates for the chosen rotation (no-op if disabled)."""
        if self.patchify_retrieval is None:
            return base_candidates
        try:
            with Telemetry.profile("patchify_retrieval"):
                patch_descs = self.patchify_retrieval.compute_patch_descriptors(rotated_frame)
                patch_candidates = self.patchify_retrieval.search(patch_descs, top_k=top_k)
            if patch_candidates:
                merged = self.merge(base_candidates, patch_candidates, max_results=top_k * 2)
                logger.debug(
                    f"Patchify expanded candidates: "
                    f"{len(base_candidates)} → {len(merged)} "
                    f"(top patchify score: {patch_candidates[0][1]:.3f})"
                )
                return merged
        except Exception as e:
            logger.warning(f"Patchify retrieval failed, using standard candidates: {e}")
        return base_candidates

    def merge(
        self,
        standard: list[tuple[int, float]],
        patches: list[tuple[int, float]],
        max_results: int | None = None,
    ) -> list[tuple[int, float]]:
        """Weighted-sum merge of standard + patch retrieval.

        Weight from localization.patchify_merge_weight. Frame in both:
        score = w_std*s + w_patch*p; in one only: the score as-is.
        """
        w_patch = get_cfg(self.config, "localization.patchify_merge_weight", 0.4)
        w_standard = 1.0 - w_patch

        standard_dict = dict(standard)
        patch_dict = dict(patches)
        all_fids = set(standard_dict.keys()) | set(patch_dict.keys())

        merged = {}
        for fid in all_fids:
            s = standard_dict.get(fid, 0.0)
            p = patch_dict.get(fid, 0.0)

            if fid in standard_dict and fid in patch_dict:
                merged[fid] = w_standard * s + w_patch * p
            elif fid in standard_dict:
                merged[fid] = s
            else:
                merged[fid] = p

        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        if max_results is not None:
            sorted_results = sorted_results[:max_results]

        return sorted_results
