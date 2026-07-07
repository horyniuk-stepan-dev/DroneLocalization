"""Spatial loop-closure pruning (two-stage prune) mixin for PoseGraphOptimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.geometry.pose_graph.model_5dof import GraphEdge


class PruningMixin:
    """Two-stage spatial-edge pruning. Pure move from PoseGraphOptimizer."""

    def _anchor_reachable(self, edges: list[GraphEdge]) -> set[int]:
        """Множина вузлів, досяжних із будь-якого якоря по заданому набору ребер."""
        adj: dict[int, list[int]] = {}
        for e in edges:
            adj.setdefault(e.from_id, []).append(e.to_id)
            adj.setdefault(e.to_id, []).append(e.from_id)
        seen = set(self._fixed_nodes.keys())
        stack = list(seen)
        while stack:
            n = stack.pop()
            for m in adj.get(n, []):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    def _prune_bad_spatial_edges(
        self, mad_k: float = 5.0, max_frac: float = 0.2
    ) -> list[GraphEdge]:
        """Викидає spatial-викиди за MAD-порогом ВСЕРЕДИНІ класу spatial.

        Захисні правила (Етап 3.3):
          - викидаємо ЛИШЕ spatial (temporal-ланцюг — хребет графа);
          - не більше max_frac від кількості spatial-ребер;
          - ніколи, якщо це роз'єднає вузол від усіх якорів.
        """
        res = self.compute_edge_residuals()
        spatial_idx = [
            k for k, e in enumerate(self._edges)
            if e.edge_type == "spatial" and not np.isnan(res[k])
        ]
        if len(spatial_idx) < 3:
            return []  # замало для оцінки MAD

        sres = np.array([res[k] for k in spatial_idx])
        med = float(np.median(sres))
        mad = float(np.median(np.abs(sres - med)))
        thresh = med + mad_k * 1.4826 * mad

        candidates = sorted(
            [k for k in spatial_idx if res[k] > thresh],
            key=lambda k: -res[k],
        )
        max_prune = int(np.floor(max_frac * len(spatial_idx)))
        candidates = candidates[:max_prune]
        if not candidates:
            return []

        base_reach = self._anchor_reachable(self._edges)
        removed_idx: list[int] = []
        for k in candidates:
            trial = [
                e for j, e in enumerate(self._edges)
                if j != k and j not in removed_idx
            ]
            if self._anchor_reachable(trial) == base_reach:
                removed_idx.append(k)  # безпечно: нічого не роз'єднали

        if not removed_idx:
            return []

        removed_set = set(removed_idx)
        pruned = [e for j, e in enumerate(self._edges) if j in removed_set]
        self._edges = [e for j, e in enumerate(self._edges) if j not in removed_set]
        self._pruned_edges.extend(pruned)
        self._last_edge_residuals = None
        return pruned

