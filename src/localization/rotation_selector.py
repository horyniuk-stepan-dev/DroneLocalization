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
        use_cascade = get_cfg(self.config, "localization.recovery_cascade", False)

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
            # ADDENDUM 2.1 (recovery_cascade): у два етапи — спершу лише кути
            # на одному масштабі, повна піраміда лише за потреби.
            stages = self._plan_stages(angles_to_try, scale_candidates, scale_manager, use_cascade)

            for stage_combos in stages:
                if not stage_combos:
                    continue
                # rot90 кешується per-angle: інакше кадр копіювався б на
                # кожну (кут, масштаб) пару замість одного разу на кут
                # (на 4K це десятки МБ memcpy на кожну зайву копію).
                rot_cache: dict[int, Any] = {}
                frames = [
                    self._prepare_frame(query_frame, a, sc, scale_manager, rot_cache)
                    for a, sc in stage_combos
                ]

                # Batch extraction
                if len(frames) > 1 and hasattr(
                    self.feature_extractor, "extract_global_descriptors_multi"
                ):
                    descs = self.feature_extractor.extract_global_descriptors_multi(frames)
                else:
                    descs = [self.feature_extractor.extract_global_descriptor(f) for f in frames]

                for (angle, sc), global_desc in zip(stage_combos, descs):
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

                # Етап 1 дав достатньо впевнений збіг — решту піраміди
                # (16 із 20 форвардів у типовій конфігурації) не рахуємо.
                if best_global_score >= rescan_min:
                    break

        if not best_global_candidates:
            return None

        return RotationResult(
            angle=best_global_angle,
            score=best_global_score,
            candidates=best_global_candidates,
            source_id=best_source_id_per_angle,
            best_scale=best_scale,
        )

    # ── ADDENDUM 2.1: планування етапів recovery ─────────────────────────────

    @staticmethod
    def _plan_stages(
        angles_to_try: list[int],
        scale_candidates: list[float],
        scale_manager: Any,
        use_cascade: bool,
    ) -> list[list[tuple[int, float]]]:
        """Комбінації (кут, масштаб), розбиті на етапи.

        ``use_cascade=False`` → один етап із повним декартовим добутком
        (ПОТОЧНА поведінка, побітово та сама послідовність).

        ``use_cascade=True`` → етап 1: усі кути × ОДИН масштаб; етап 2: усі
        ІНШІ комбінації. Ключова властивість — етап 2 не повторює вже
        пораховане, тож найгірший випадок (етап 1 провалився) лишається рівно
        стільки ж форвардів, скільки й зараз. Каскад не може бути повільнішим
        за поточну поведінку — лише швидшим.

        Опорний масштаб етапу 1: prior ScaleManager-а, якщо він є; інакше
        перший елемент ``scale_candidates`` (``ScaleManager.candidates()`` уже
        сортує піраміду за близькістю до depth-hint); інакше 1.0.
        """
        combos = [(a, sc) for a in angles_to_try for sc in scale_candidates]
        if not use_cascade or len(scale_candidates) <= 1:
            return [combos]

        primary = None
        prior = getattr(scale_manager, "prior", None) if scale_manager is not None else None
        if prior is not None and prior in scale_candidates:
            primary = prior
        elif 1.0 in scale_candidates:
            # Масштаб 1.0 — «як у БД»; найімовірніший, коли prior відсутній.
            primary = 1.0
        else:
            primary = scale_candidates[0]

        stage1 = [c for c in combos if c[1] == primary]
        stage2 = [c for c in combos if c[1] != primary]
        return [stage1, stage2]

    @staticmethod
    def _prepare_frame(
        query_frame: Any,
        angle: int,
        sc: float,
        scale_manager: Any,
        rot_cache: dict[int, Any] | None = None,
    ) -> Any:
        """Кадр, повернутий на ``angle`` і нормалізований до масштабу ``sc``.

        ``rot_cache`` — спільний на етап словник {кут: повернутий кадр}:
        повороти дорогі (копія повного кадру), а масштабів на кут кілька.
        """
        if rot_cache is not None and angle in rot_cache:
            rotated = rot_cache[angle]
        else:
            rotated = np.ascontiguousarray(np.rot90(query_frame, k=angle // 90))
            if rot_cache is not None:
                rot_cache[angle] = rotated
        if scale_manager is not None and abs(sc - 1.0) > 0.15:
            scaled, _ = scale_manager.normalize(rotated, sc)
            return scaled
        return rotated
