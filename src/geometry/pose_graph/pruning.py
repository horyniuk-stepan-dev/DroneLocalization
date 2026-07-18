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
            k
            for k, e in enumerate(self._edges)
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
            trial = [e for j, e in enumerate(self._edges) if j != k and j not in removed_idx]
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

    def _run_gnc_spatial(
        self,
        rounds: int,
        mad_k: float,
        *,
        max_iterations: int,
        tolerance: float,
        progress_callback,
        use_analytic_jac: bool,
        kinematic_prior_weight: float = 0.0,
    ) -> dict[int, np.ndarray]:
        """GNC-переваження spatial-ребер (Етап 3) — плавна еволюція two-stage prune.

        Замість бінарного викидання: раунди L2 із Geman-McClure-вагами
        w' = w · σ²/(σ²+r²), де σ = µ·thr, thr = median + mad_k·1.4826·MAD резидуалів
        ВЛАСНОГО (spatial) класу — той самий поріг, що й у prune (урок soft_l1:
        temporal-ланцюг недоторканий, пороги класо-відносні). µ спадає від
        опуклого (усі ваги≈1) до 1 (справжній GM), warm start між раундами.

        ЧИСТА СЦЕНА (жоден spatial-резидуал не перевищує thr) → миттєвий вихід із
        БАЗОВИМИ вагами → розв'язок ІДЕНТИЧНИЙ чистому L2 (нуль деградації).
        Геометричний (незалежний від ваги) резидуал = ‖residual‖ / weight.
        """
        spatial_idx = [k for k, e in enumerate(self._edges) if e.edge_type == "spatial"]
        base_w = {k: float(self._edges[k].weight) for k in spatial_idx}
        if len(base_w) < 3:
            return self._export_results()

        def _geom_residuals():
            res = self.compute_edge_residuals()
            geo = {}
            for k in base_w:
                w_cur = float(self._edges[k].weight)
                if not np.isnan(res[k]) and w_cur > 1e-12:
                    geo[k] = float(res[k]) / w_cur
            return geo

        # Раунд-0 діагностика на БАЗОВИХ вагах (розв'язок уже є з головного L2).
        geo0 = _geom_residuals()
        if len(geo0) < 3:
            return self._export_results()
        vals0 = np.array(list(geo0.values()))
        med = float(np.median(vals0))
        mad = float(np.median(np.abs(vals0 - med)))
        thr = med + mad_k * 1.4826 * mad
        if float(np.max(vals0)) <= thr * (1.0 + 1e-9):
            return self._export_results()  # немає викидів → чиста сцена → no-op

        opt_kw = dict(
            max_iterations=max_iterations,
            tolerance=tolerance,
            progress_callback=progress_callback,
            use_analytic_jac=use_analytic_jac,
            gnc_spatial=False,
            two_stage_prune=False,
            kinematic_prior_weight=kinematic_prior_weight,
        )
        mu = max(1.0, float(np.max(vals0)) / max(thr, 1e-9))
        for _ in range(max(1, rounds)):
            geo = _geom_residuals()
            sigma2 = (mu * thr) ** 2
            for k, w0 in base_w.items():
                r = geo.get(k)
                if r is None:
                    continue
                self._edges[k].weight = w0 * (sigma2 / (sigma2 + r * r))
            self.optimize(**opt_kw)  # повторний L2, warm start із self._free_nodes
            if mu <= 1.0:
                break
            mu = max(1.0, mu / 1.4)

        # Відновлюємо базові ваги (розв'язок уже в self._free_nodes; ваги — лише
        # для GNC-ітерацій, звіти мають бачити оригінальні ваги ребер).
        for k, w0 in base_w.items():
            self._edges[k].weight = w0
        self._last_edge_residuals = None
        return self._export_results()
