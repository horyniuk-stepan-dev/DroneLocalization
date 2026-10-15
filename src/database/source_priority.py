"""Which layer's candidates a legacy multi-layer retrieval hands to verification.

Without a margin the layer with the best top score wins (``MultiDatabaseManager``'s
behaviour so far; ``priority`` only orders equal scores). With
``localization.source_priority_rank_margin = N`` the candidates of all layers are
ranked together by score, and a layer of higher priority (lower ``priority``
number, the main layer first) still wins when its best candidate is within the
first ``N + 1`` ranks. N = 3: the main layer's best candidate at rank 2 beats
another layer's rank 1; at rank 10 the rank-1 layer is used.

``localization.joint_candidate_ranking`` goes one step further: one verification
list from the standard candidates of every layer and the patchify-only candidates
of every layer (``joint_order``). Each layer has two priorities: ``priority`` for its
standard candidates and ``patchify_priority`` (default: ``priority``) for its patch
candidates. Layers that overlap need distinct priorities, otherwise no rule can
order them; layers of different terrain may share one (``priority_conflicts``).

``localization.layer_fineness_weight`` adds the layer's map accuracy to the project
priority: effective step = priority step + weight x log2(ground pixel size / the
finest layer's), the pixel size measured from the layer's calibration
(``layer_gsd``). ``patchify_layers`` decides where patchify can help: only where no
real layer is close in scale to the query.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


def choose_source(
    per_source: dict[str, list[tuple[int, float]]],
    priorities: dict[str, int],
    rank_margin: int,
) -> str | None:
    """Source id whose candidates are verified, or None without candidates."""
    ranked = sorted(
        (
            (float(score), sid)
            for sid, candidates in per_source.items()
            for _frame, score in candidates
        ),
        key=lambda item: (-item[0], priorities.get(item[1], 0), item[1]),
    )
    if not ranked:
        return None
    if rank_margin <= 0:
        return ranked[0][1]
    best_rank: dict[str, int] = {}
    for rank, (_score, sid) in enumerate(ranked, start=1):
        best_rank.setdefault(sid, rank)
    for sid in sorted(best_rank, key=lambda s: (priorities.get(s, 0), s)):
        if best_rank[sid] <= rank_margin + 1:
            return sid
    return ranked[0][1]


def main_before_patchify(
    expanded: list[tuple[int, float]],
    standard_ids: set[int],
    main_candidates: list[tuple[int, float]],
    rank_margin: int,
) -> bool:
    """Verify the main layer's candidates before this layer's patchify-only ones?

    ``expanded``: the chosen layer's candidates after patchify (standard and
    patchify-only, merged scores); ``standard_ids``: the frames of its standard
    retrieval; ``main_candidates``: the main layer's standard retrieval. All are
    ranked together by score: the patchify-only candidates go first only when the
    best of them is more than ``rank_margin`` ranks above the main layer's best
    (patchify 1st, main 2nd -> main; main 10th -> patchify with a margin of 3).
    ``rank_margin`` 0 never puts the main layer first (old behaviour).
    """
    patch_only = [score for frame, score in expanded if int(frame) not in standard_ids]
    if rank_margin <= 0 or not patch_only or not main_candidates:
        return False
    ranked = sorted(
        [(float(score), 0) for _frame, score in main_candidates]
        + [(float(score), 1) for score in patch_only]
        + [(float(score), 2) for frame, score in expanded if int(frame) in standard_ids],
        key=lambda item: (-item[0], item[1]),
    )
    kinds = [kind for _score, kind in ranked]
    main_rank = kinds.index(0) + 1
    patch_rank = kinds.index(1) + 1
    return main_rank - patch_rank <= rank_margin


# ── Joint candidate list (localization.joint_candidate_ranking) ──────────────────


@dataclass(frozen=True)
class JointCandidate:
    """One candidate of the joint list: a frame of one layer, standard or patchify."""

    source_id: str
    frame_id: int
    score: float
    patch: bool = False


def effective_patchify_priority(source: Any) -> int:
    """``patchify_priority`` of a layer (ProjectVideoSource or dict), else its ``priority``."""
    get = source.get if isinstance(source, dict) else lambda key: getattr(source, key, None)
    priority = int(get("priority") or 0)
    patchify = get("patchify_priority")
    return priority if patchify is None else int(patchify)


def priority_steps(priorities: dict[str, int]) -> dict[str, int]:
    """Dense rank of the priority values: {main: 0, 2: 1, 3: 2}; equal values share a step.

    Gaps in the numbers (0, 2, 3) do not add penalty steps.
    """
    values = sorted({int(p) for p in priorities.values()})
    index = {value: i for i, value in enumerate(values)}
    return {sid: index[int(p)] for sid, p in priorities.items()}


def joint_order(
    candidates: Iterable[JointCandidate],
    priorities: dict[str, int],
    patch_priorities: dict[str, int],
    rank_step: int,
    patch_penalty: int,
    top_k: int | None = None,
) -> list[JointCandidate]:
    """Verification order of the joint candidate list, first ``top_k``.

    Place = rank by score within the candidate's kind + ``rank_step`` per priority
    step of its layer (``priorities`` for standard candidates, ``patch_priorities``
    for patchify ones) + ``patch_penalty`` for patchify candidates. Standard and
    patchify scores are not on one scale (cosine of the global descriptors vs
    coverage x mean similarity of the crops, which can exceed 1), so each kind is
    ranked on its own: the best patchify candidate competes with the standard
    candidate ``patch_penalty`` places below the best one. Ties: standard first,
    then the higher priority, then the better rank in its kind. Two layers, no
    patchify: the main layer's best goes first while it ranks at most ``rank_step``
    places below the other layer's best (rank_step 2: main 3rd behind the other's
    1st -> main first; 4th -> the other), the rule of ``choose_source``.
    """
    std_steps = priority_steps(priorities) if priorities else {}
    patch_steps = priority_steps(patch_priorities) if patch_priorities else {}
    pool = list(candidates)

    def ranked(kind_is_patch: bool, steps: dict[str, int]) -> list[JointCandidate]:
        return sorted(
            (c for c in pool if c.patch == kind_is_patch),
            key=lambda c: (-c.score, steps.get(c.source_id, 0), c.source_id, c.frame_id),
        )

    keyed = []
    for pos, cand in enumerate(ranked(False, std_steps), start=1):
        step = std_steps.get(cand.source_id, 0)
        keyed.append((pos + rank_step * step, 0, step, pos, cand))
    for pos, cand in enumerate(ranked(True, patch_steps), start=1):
        step = patch_steps.get(cand.source_id, 0)
        keyed.append((pos + patch_penalty + rank_step * step, 1, step, pos, cand))
    keyed.sort(key=lambda item: item[:4])
    ordered = [item[4] for item in keyed]
    return ordered if top_k is None else ordered[: max(0, int(top_k))]


def verification_runs(
    ordered: Iterable[JointCandidate],
) -> list[tuple[str, list[JointCandidate]]]:
    """Consecutive candidates of one layer, in order: one verifier call each."""
    runs: list[tuple[str, list[JointCandidate]]] = []
    for cand in ordered:
        if runs and runs[-1][0] == cand.source_id:
            runs[-1][1].append(cand)
        else:
            runs.append((cand.source_id, [cand]))
    return runs


# ── Priority validation ──────────────────────────────────────────────────────────


def next_priority(used: Iterable[int]) -> int:
    """Next number for a layer stacked on the given ones: one past the largest, 0 first."""
    values = [int(v) for v in used]
    return max(values) + 1 if values else 0


def coverage_cells(gps: Any, cell_deg: float = 0.02) -> set[tuple[int, int]]:
    """Grid cells (~2 km at cell_deg 0.02) that contain at least one map frame position."""
    if gps is None:
        return set()
    cells = set()
    for row in gps:
        try:
            lat, lon = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if lat != lat or lon != lon:  # NaN: frame without a position
            continue
        cells.add((int(lat // cell_deg), int(lon // cell_deg)))
    return cells


def priority_conflicts(
    sources: Iterable[Any],
    overlap: Callable[[Any, Any], bool],
    distinguishable: Callable[[Any, Any], bool] | None = None,
) -> list[str]:
    """Pairs of overlapping layers whose priorities cannot be told apart.

    Each message names the two layers and the shared number, for ``priority``
    (standard candidates) and ``patchify_priority`` (patchify candidates).
    Layers that do not overlap may share a number, and so may layers that
    ``distinguishable`` tells apart otherwise (e.g. by map accuracy).
    """
    items = list(sources)

    def sid(source: Any) -> str:
        return str(source.get("source_id") if isinstance(source, dict) else source.source_id)

    def prio(source: Any) -> int:
        value = source.get("priority") if isinstance(source, dict) else source.priority
        return int(value or 0)

    def explicit_patch(source: Any) -> Any:
        if isinstance(source, dict):
            return source.get("patchify_priority")
        return getattr(source, "patchify_priority", None)

    messages = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            same = prio(a) == prio(b)
            # a patchify priority that only follows the layer priority adds no new clash
            same_patch = effective_patchify_priority(a) == effective_patchify_priority(b) and not (
                same and explicit_patch(a) is None and explicit_patch(b) is None
            )
            if not (same or same_patch) or not overlap(a, b):
                continue
            if distinguishable is not None and distinguishable(a, b):
                continue
            if same:
                messages.append(
                    f"layers '{sid(a)}' and '{sid(b)}' overlap but share priority {prio(a)}"
                )
            if same_patch:
                messages.append(
                    f"layers '{sid(a)}' and '{sid(b)}' overlap but share patchify priority "
                    f"{effective_patchify_priority(a)}"
                )
    return messages


# ── Layer accuracy (localization.layer_fineness_weight, patchify_adaptive) ───────────


def layer_gsd(database: Any) -> float | None:
    """Median ground size of one map pixel of a layer, from its frame calibration.

    sqrt(|det|) of the 2x2 part of each valid frame affine (pixel -> map units);
    None without a calibration. Only ratios between layers are used.
    """
    affine = getattr(database, "frame_affine", None)
    if affine is None:
        return None
    a = np.asarray(affine, dtype=float)
    if a.ndim == 2 and a.shape[1] == 6:
        a = a.reshape(-1, 2, 3)
    if a.ndim != 3 or a.shape[1] < 2 or a.shape[2] < 2:
        return None
    valid = getattr(database, "frame_valid", None)
    if valid is not None and len(valid) == len(a):
        a = a[np.asarray(valid, dtype=bool)]
    det = np.abs(a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0])
    det = det[np.isfinite(det) & (det > 0)]
    return float(np.sqrt(np.median(det))) if det.size else None


def effective_steps(
    priorities: dict[str, int],
    gsds: dict[str, float | None],
    weight: float,
) -> dict[str, float]:
    """Priority step of each layer + weight x log2(its pixel size / the finest layer's).

    Relief project (main 1000 m = 0, 2000 m = 1, 500 m = 2; 0.72 / 1.51 / 0.33 m/px),
    weight 2: main 2.3, 2000 m 5.4, 500 m 2.0. Layers without a pixel size get no
    fineness term.
    """
    steps = {sid: float(v) for sid, v in priority_steps(priorities).items()}
    known = {sid: float(g) for sid, g in gsds.items() if g and g > 0 and sid in steps}
    if weight > 0 and known:
        finest = min(known.values())
        for sid, g in known.items():
            steps[sid] += weight * math.log2(g / finest)
    return steps


def choose_source_weighted(
    per_source: dict[str, list[tuple[int, float]]],
    steps: dict[str, float],
    rank_step: int,
) -> str | None:
    """Layer whose best candidate has the lowest rank + rank_step x effective step.

    Ranks are over the candidates of all layers by score (as ``choose_source``);
    ties go to the lower effective step. With two layers one whole step apart it
    is ``choose_source``'s rule.
    """
    ranked = sorted(
        (
            (float(score), sid)
            for sid, candidates in per_source.items()
            for _frame, score in candidates
        ),
        key=lambda item: (-item[0], steps.get(item[1], 0.0), item[1]),
    )
    if not ranked:
        return None
    best_rank: dict[str, int] = {}
    for rank, (_score, sid) in enumerate(ranked, start=1):
        best_rank.setdefault(sid, rank)
    return min(
        best_rank,
        key=lambda sid: (
            best_rank[sid] + rank_step * steps.get(sid, 0.0),
            steps.get(sid, 0.0),
            sid,
        ),
    )


def patchify_layers(
    query_gsd: float | None,
    gsds: dict[Any, float | None],
    factors: Iterable[float],
    gap_ratio: float,
) -> set | None:
    """Layers whose patchify search can help a query of ground pixel size ``query_gsd``.

    None (unknown query scale or layer sizes): every layer, the old behaviour.
    Empty: a real layer is within ``gap_ratio`` of the query's pixel size. Otherwise
    the layers with a crop factor (2x2 crop -> pixel size / 2) that comes closer to
    the query than any real layer: layers 500 m + 2000 m and a drone near 1000 m ->
    the 2000 m layer.
    """
    known = {sid: float(g) for sid, g in gsds.items() if g and g > 0}
    if query_gsd is None or query_gsd <= 0 or not known:
        return None

    def distance(size: float) -> float:
        return abs(math.log(size / query_gsd))

    nearest_real = min(distance(g) for g in known.values())
    if nearest_real <= math.log(max(gap_ratio, 1.0)):
        return set()
    crops = [float(f) for f in factors if float(f) > 1.0]
    return {sid for sid, g in known.items() if any(distance(g / f) < nearest_real for f in crops)}


def grid_factors(grids: Iterable[Any]) -> list[float]:
    """Linear crop factor of each patchify grid: (2, 2) -> 2, (2, 3) -> sqrt(6)."""
    factors = []
    for grid in grids:
        try:
            rows, cols = int(grid[0]), int(grid[1])
        except (TypeError, ValueError, IndexError):
            continue
        if rows > 0 and cols > 0:
            factors.append(math.sqrt(rows * cols))
    return factors
