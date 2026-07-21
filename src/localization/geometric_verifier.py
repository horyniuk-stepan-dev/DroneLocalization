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

    def __init__(
        self,
        matcher: Any,
        min_matches: int,
        ransac_thresh: float,
        homography_backend: str,
        use_mad_ransac: bool,
        mad_k_factor: float,
        early_stop_inliers: int,
        prefilter_enabled: bool = False,
        prefilter_keep: int = 2,
    ) -> None:
        self.matcher = matcher
        self.min_matches = min_matches
        self.ransac_thresh = ransac_thresh
        self.homography_backend = homography_backend
        self.use_mad_ransac = use_mad_ransac
        self.mad_k_factor = mad_k_factor
        self.early_stop_inliers = early_stop_inliers
        self.prefilter_enabled = prefilter_enabled
        self.prefilter_keep = prefilter_keep

    def verify(
        self,
        query_features: dict,
        candidates: list,
        database: Any,
        ref_cache: dict | None = None,
    ) -> VerificationResult | None:
        best_inliers = 0
        best_candidate_id = -1
        best_H_query_to_ref = None
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0
        best_rmse = 999.0

        early_stop = self.early_stop_inliers

        # ADDENDUM §1: дешевий MNN-скоринг усіх кандидатів, LightGlue — лише
        # на найкращих. ref_cache уникає повторного читання фіч із БД.
        ref_cache = {} if ref_cache is None else ref_cache
        if self.prefilter_enabled and len(candidates) > self.prefilter_keep:
            candidates = self._prefilter(query_features, candidates, database, ref_cache)

        for candidate_id, score in candidates:
            logger.debug(f"  → Trying candidate {candidate_id} (global_score={score:.3f})")
            ref_features = ref_cache.get(candidate_id)
            if ref_features is None:
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

    # ── ADDENDUM §1: MNN-передфільтр кандидатів ──────────────────────────

    def mnn_counts(
        self,
        query_features: dict,
        candidates: list,
        database: Any,
        ref_cache: dict | None = None,
    ) -> list[tuple[int, int, float]] | None:
        """Кількість mutual-NN пар (з Lowe ratio) між query і кожним кандидатом.

        Повертає ``[(пар, candidate_id, score), ...]`` або ``None``, якщо
        дескриптори query вироджені. Два споживачі:
        передфільтр нижче (ADDENDUM §1) і темпоральний prior
        (PIPELINE_OPTIMIZATION_PLAN §A1), для якого це дешева проба гіпотези —
        один матмул замість повного LightGlue.

        Lowe ratio обовʼязковий: голий mutual-NN рахує ~50% випадкових пар
        між будь-якими наборами (перевірено юніт-тестом), ratio їх зрізає.
        Той самий поріг, що і в ``_fast_numpy_match`` (L2-нормовані
        дескриптори: d = sqrt(2-2s)). GPU-шлях (один матмул на кандидата)
        з фолбеком на numpy.
        """
        q = np.asarray(query_features.get("descriptors")) if query_features else None
        if q is None or q.ndim != 2 or len(q) == 0:
            return None
        q32 = np.ascontiguousarray(q, dtype=np.float32)

        use_torch = False
        q_t = None
        try:
            import torch

            if torch.cuda.is_available():
                q_t = torch.from_numpy(q32).cuda()
                use_torch = True
        except Exception:  # noqa: BLE001 — GPU-шлях опційний, фолбек numpy
            use_torch = False

        ratio = 0.75

        def _mnn_count(r: np.ndarray) -> int:
            r32 = np.ascontiguousarray(r, dtype=np.float32)
            if r32.shape[0] < 2:
                return 0
            if use_torch:
                try:
                    r_t = torch.from_numpy(r32).cuda()
                    sim = q_t @ r_t.T
                    top2 = torch.topk(sim, 2, dim=1).values
                    d1 = torch.sqrt(torch.clamp(2.0 - 2.0 * top2[:, 0], min=0.0))
                    d2 = torch.sqrt(torch.clamp(2.0 - 2.0 * top2[:, 1], min=0.0))
                    ratio_ok = d1 < ratio * d2
                    nn12 = sim.argmax(dim=1)
                    nn21 = sim.argmax(dim=0)
                    idx = torch.arange(sim.shape[0], device=sim.device)
                    return int(((nn21[nn12] == idx) & ratio_ok).sum().item())
                except Exception:  # noqa: BLE001 — OOM тощо → numpy
                    pass
            sim = q32 @ r32.T
            top2 = -np.partition(-sim, 1, axis=1)[:, :2]
            d1 = np.sqrt(np.clip(2.0 - 2.0 * top2[:, 0], 0.0, None))
            d2 = np.sqrt(np.clip(2.0 - 2.0 * top2[:, 1], 0.0, None))
            ratio_ok = d1 < ratio * d2
            nn12 = sim.argmax(axis=1)
            nn21 = sim.argmax(axis=0)
            mutual = nn21[nn12] == np.arange(sim.shape[0])
            return int(np.sum(mutual & ratio_ok))

        scored: list[tuple[int, int, float]] = []
        with Telemetry.profile("prefilter"):
            for candidate_id, score in candidates:
                ref = ref_cache.get(candidate_id) if ref_cache is not None else None
                if ref is None:
                    try:
                        ref = database.get_local_features(candidate_id)
                    except (ValueError, KeyError) as e:
                        # Темпоральний prior пропонує сусідів за індексом —
                        # частина з них може бути відсутня у БД. Це не помилка.
                        logger.debug(f"MNN probe: candidate {candidate_id} unreadable ({e})")
                        scored.append((0, candidate_id, score))
                        continue
                    if ref_cache is not None:
                        ref_cache[candidate_id] = ref
                r = np.asarray(ref.get("descriptors"))
                if r is None or r.ndim != 2 or len(r) == 0 or r.shape[1] != q.shape[1]:
                    scored.append((0, candidate_id, score))
                    continue
                scored.append((_mnn_count(r), candidate_id, score))
        return scored

    def _prefilter(
        self, query_features: dict, candidates: list, database: Any, ref_cache: dict
    ) -> list:
        """Ранжує кандидатів за кількістю MNN-пар і лишає ``prefilter_keep``
        найкращих (LightGlue далі йде лише по них).

        Консервативність: якщо жоден кандидат не набрав жодної MNN-пари
        (виродження: порожні/несумісні дескриптори) — список без змін.
        """
        scored = self.mnn_counts(query_features, candidates, database, ref_cache)
        if scored is None:
            return candidates
        if all(s[0] == 0 for s in scored):
            logger.debug("Prefilter: no MNN pairs on any candidate — keeping full list")
            return candidates
        scored.sort(key=lambda t: (-t[0], -t[2]))
        kept = [(cid, sc) for _, cid, sc in scored[: self.prefilter_keep]]
        logger.debug(
            f"Prefilter kept {len(kept)}/{len(candidates)} candidates "
            f"(mnn top: {[(c, m) for m, c, _ in scored[: self.prefilter_keep]]})"
        )
        return kept
