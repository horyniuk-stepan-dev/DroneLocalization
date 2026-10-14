"""Candidate retrieval + patchify expansion for localization (IMPROVEMENT_PLAN 1.1).

Extracted from Localizer as a stateless collaborator: single- or multi-database
top-k retrieval, plus optional patchify merge for the chosen rotation. Logic and
log messages are unchanged.

Patchify with several layers: frame ids are per database, so the patch hits merged
into a candidate list must come from the same layer as the list. ``patchify_factory``
builds one patch index per layer on the layer's first use and again after the layer's
database object changes (reload).
"""

from __future__ import annotations

from config import get_cfg
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


class CandidateRetriever:
    """Top-k retrieval (single/multi DB) with optional patch-descriptor merge."""

    def __init__(
        self, db_manager, retriever, patchify_retrieval, config, patchify_factory=None
    ) -> None:
        self.db_manager = db_manager
        self.retriever = retriever
        # Single-database mode: the index of that database.
        self.patchify_retrieval = patchify_retrieval
        # Multi-database mode: database -> PatchifyRetrieval | None, called once per layer.
        self.patchify_factory = patchify_factory
        self._layer_patchify: dict[str, tuple[object, object]] = {}
        self.last_by_source: dict | None = None
        self.config = config

    def retrieve(self, global_desc, top_k: int):
        """Retrieve candidates: (source_id | None, [(frame_id, score), ...]).

        Multi-layer: ``last_by_source`` then holds every layer's candidates of this
        call (None in single-database mode).
        """
        if self.db_manager is not None:
            result = self.db_manager.get_best_match(global_desc, top_k=top_k)
            self.last_by_source = getattr(self.db_manager, "last_matches_by_source", None)
            return result
        self.last_by_source = None
        return None, self.retriever.find_similar_frames(global_desc, top_k=top_k)

    def patchify_for(self, source_id=None):
        """Patch index whose frame ids belong to the candidates of ``source_id``."""
        if self.db_manager is None or self.patchify_factory is None:
            return self.patchify_retrieval
        if source_id is None:
            return None
        database = self.db_manager.get_database(source_id)
        if database is None:
            return None
        cached = self._layer_patchify.get(source_id)
        if cached is None or cached[0] is not database:
            cached = (database, self.patchify_factory(database))
            self._layer_patchify[source_id] = cached
        return cached[1]

    def patch_candidates(self, rotated_frame, top_k: int, source_id=None, desc_cache=None):
        """Patchify hits of one layer, ``[(frame_id, score)]`` (joint candidate ranking).

        ``desc_cache`` (one dict per localized frame) keeps the query's crop
        descriptors per prepared view, so layers that share the view's angle and
        scale run the 14 crop forward passes once.
        """
        patchify = self.patchify_for(source_id)
        if patchify is None or rotated_frame is None:
            return []
        try:
            with Telemetry.profile("patchify_retrieval"):
                key = (
                    id(rotated_frame),
                    tuple(getattr(patchify, "grids", ()) or ()),
                    getattr(patchify, "descriptor_dim", None),
                )
                descs = desc_cache.get(key) if desc_cache is not None else None
                if descs is None:
                    descs = patchify.compute_patch_descriptors(rotated_frame)
                    if desc_cache is not None:
                        desc_cache[key] = descs
                return list(patchify.search(descs, top_k=top_k))
        except Exception as e:
            logger.warning(f"Patchify retrieval failed for layer '{source_id}': {e}")
            return []

    def expand(self, rotated_frame, base_candidates, top_k: int, source_id=None):
        """Patchify-expand candidates for the chosen rotation (no-op if disabled)."""
        patchify = self.patchify_for(source_id)
        if patchify is None:
            return base_candidates
        try:
            with Telemetry.profile("patchify_retrieval"):
                patch_descs = patchify.compute_patch_descriptors(rotated_frame)
                patch_candidates = patchify.search(patch_descs, top_k=top_k)
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
