"""Read-only residual/anchor diagnostics and GeoJSON export mixin."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from src.geometry.pose_graph.model_5dof import (
    GraphEdge,
    _predict_forward,
    _predict_inverse,
    edge_residual,
)


class DiagnosticsMixin:
    """Residual stats, anchor stress, reports and GeoJSON. Pure move from PoseGraphOptimizer."""

    def _current_states_full(self) -> dict[int, np.ndarray]:
        """Current states of all INITIALIZED nodes (fixed + reachable free)."""
        states: dict[int, np.ndarray] = dict(self._fixed_nodes)
        for fid, st in self._free_nodes.items():
            if fid in self._initialized_nodes:
                states[fid] = st
        return states

    def _single_edge_residual(self, si: np.ndarray, sj: np.ndarray, e: GraphEdge) -> np.ndarray:
        """Weighted 5-vector edge residual — SAME formula as in _residuals_vec."""
        return edge_residual(
            si,
            sj,
            e.dtx,
            e.dty,
            e.log_dsx,
            e.log_dsy,
            e.dtheta,
            e.weight,
            self.cx,
            self._sign,
        )

    def compute_edge_residuals(self) -> np.ndarray:
        """Norm of weighted residual per EACH edge (based on current states).

        result.fun already contains these numbers during optimization, but is discarded —
        here we recreate them for diagnostics. NaN if edge node is unreachable.
        """
        states = self._current_states_full()
        res = np.full(len(self._edges), np.nan, dtype=np.float64)
        for k, e in enumerate(self._edges):
            si, sj = states.get(e.from_id), states.get(e.to_id)
            if si is None or sj is None:
                continue
            res[k] = float(np.linalg.norm(self._single_edge_residual(si, sj, e)))
        self._last_edge_residuals = res
        return res

    def edge_residual_stats(self) -> dict:
        """Residual statistics SEPARATELY for temporal and spatial (different scales!)."""
        res = self.compute_edge_residuals()
        out: dict[str, dict] = {}
        for cls in ("temporal", "spatial"):
            vals = np.array(
                [
                    res[k]
                    for k, e in enumerate(self._edges)
                    if e.edge_type == cls and not np.isnan(res[k])
                ]
            )
            if vals.size:
                out[cls] = {
                    "count": int(vals.size),
                    "median": float(np.median(vals)),
                    "p95": float(np.percentile(vals, 95)),
                    "max": float(np.max(vals)),
                }
            else:
                out[cls] = {"count": 0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return out

    def compute_anchor_stress(self) -> dict[int, float]:
        """For each anchor: mean residual of incident edges / graph median.

        An anchor with stress >> 1 conflicts with the graph (bad user point).
        """
        res = self._last_edge_residuals
        if res is None:
            res = self.compute_edge_residuals()
        valid = res[~np.isnan(res)]
        med = float(np.median(valid)) if valid.size else 0.0

        incident: dict[int, list[float]] = {}
        for k, e in enumerate(self._edges):
            if np.isnan(res[k]):
                continue
            incident.setdefault(e.from_id, []).append(res[k])
            incident.setdefault(e.to_id, []).append(res[k])

        stress: dict[int, float] = {}
        # anchor_states() returns hard + soft anchors. Using only _fixed_nodes
        # when soft_anchors=True caused the anchor-stress report to disappear entirely.
        for fid in self.anchor_states():
            rs = incident.get(fid, [])
            if not rs:
                continue
            mean_r = float(np.mean(rs))
            stress[fid] = mean_r / med if med > 0 else mean_r
        return stress

    def leave_one_out_anchor_check(self, threshold_m: float = 5.0) -> dict:
        """LOO-validation of anchors (read-only).

        For each anchor: predict its state via temporal chain from the NEAREST anchor
        on each side (smaller / larger frame_id), separately.
        disagreement = MIN over available sides -> an anchor conflicting with BOTH
        neighbors (bad point) pops up, while a good neighbor of a bad point stays
        normal (since it matches its second, good side). Same forward/inverse predict
        as in BFS; anchor-stress remains post-factum.

        Does not mutate optimizer state. {fid -> {reachable, disagreement_m, flag}}.
        Works with both hard (fix_node) and soft (add_anchor) anchors.
        """
        anchors: dict[int, np.ndarray] = dict(self._fixed_nodes)
        for fid, (st, _w) in getattr(self, "_anchor_priors", {}).items():
            anchors.setdefault(fid, st)
        if len(anchors) < 2:
            return {}

        adj: dict[int, list] = {}
        for e in self._edges:
            adj.setdefault(e.from_id, []).append((e.to_id, e))
            adj.setdefault(e.to_id, []).append((e.from_id, e))

        def predict_from(seed: int, target: int):
            """Pure odometric projection of target state from single seed anchor
            along edges (BFS, first reach). None if unreachable."""
            pred = {seed: anchors[seed]}
            visited = {seed}
            queue = deque([seed])
            while queue:
                cur = queue.popleft()
                cur_state = pred[cur]
                for nb, e in adj.get(cur, []):
                    if nb in visited:
                        continue
                    predicted = (
                        _predict_forward(cur_state, e, self._sign)
                        if e.from_id == cur
                        else _predict_inverse(cur_state, e, self._sign)
                    )
                    if nb == target:
                        return predicted
                    pred[nb] = predicted
                    visited.add(nb)
                    queue.append(nb)
            return None

        ids = sorted(anchors)
        results: dict[int, dict] = {}
        for target in anchors:
            prevs = [a for a in ids if a < target]
            nexts = [a for a in ids if a > target]
            sides = ([max(prevs)] if prevs else []) + ([min(nexts)] if nexts else [])
            diffs = []
            for seed in sides:
                p = predict_from(seed, target)
                if p is not None:
                    diffs.append(float(np.linalg.norm(p[:2] - anchors[target][:2])))
            if not diffs:
                results[target] = {"reachable": False}
                continue
            d = min(diffs)
            results[target] = {
                "reachable": True,
                "disagreement_m": d,
                "flag": "warning" if d > threshold_m else "normal",
            }
        return results

    def diagnostics_report(self, top_n: int = 5, loo_threshold_m: float = 5.0) -> dict:
        """Full propagation report: edge classes, residuals, top worst,
        anchor stress, LOO-валідація якорів (1.2). Read-only — нуль впливу на розв'язок."""
        res = self.compute_edge_residuals()
        stats = self.edge_residual_stats()
        stress = self.compute_anchor_stress()
        loo = self.leave_one_out_anchor_check(loo_threshold_m)

        order = [k for k in np.argsort(res)[::-1] if not np.isnan(res[k])]
        worst = []
        for k in order[:top_n]:
            e = self._edges[k]
            worst.append(
                {
                    "from_id": e.from_id,
                    "to_id": e.to_id,
                    "type": e.edge_type,
                    "residual": float(res[k]),
                    "inliers": e.inliers,
                    "rmse": e.rmse,
                }
            )

        return {
            "num_edges": len(self._edges),
            "num_temporal": sum(1 for e in self._edges if e.edge_type == "temporal"),
            "num_spatial": sum(1 for e in self._edges if e.edge_type == "spatial"),
            "num_anchors": len(self.anchor_states()),
            "residual_stats": stats,
            "worst_edges": worst,
            "anchor_stress": {int(k): float(v) for k, v in stress.items()},
            "anchor_loo": {int(k): v for k, v in loo.items()},
        }

    def format_diagnostics(self, top_n: int = 5, loo_threshold_m: float = 5.0) -> str:
        """Text report for log/summary dialog."""
        r = self.diagnostics_report(top_n=top_n, loo_threshold_m=loo_threshold_m)
        lines = [
            f"Ребер: {r['num_edges']} ({r['num_temporal']} temporal + "
            f"{r['num_spatial']} spatial), якорів: {r['num_anchors']}",
        ]
        for cls in ("temporal", "spatial"):
            st = r["residual_stats"][cls]
            lines.append(
                f"  {cls}: медіана={st['median']:.1f}, p95={st['p95']:.1f}, "
                f"max={st['max']:.1f} (n={st['count']})"
            )
        if r["worst_edges"]:
            lines.append("  Топ-гірших ребер:")
            for w in r["worst_edges"]:
                lines.append(
                    f"    #{w['from_id']}→#{w['to_id']} [{w['type']}] резидуал={w['residual']:.1f}"
                )
        hot = {k: v for k, v in r["anchor_stress"].items() if v >= 2.0}
        if hot:
            for fid, v in sorted(hot.items(), key=lambda kv: -kv[1]):
                lines.append(f"  ⚠ Якір #{fid}: stress {v:.1f}× медіани — перевірте точки")
        loo_warn = {k: v for k, v in r["anchor_loo"].items() if v.get("flag") == "warning"}
        if loo_warn:
            for fid, v in sorted(loo_warn.items(), key=lambda kv: -kv[1]["disagreement_m"]):
                lines.append(
                    f"  ⚠ Якір #{fid}: конфліктує з ланцюгом сусідніх якорів на "
                    f"{v['disagreement_m']:.1f} м (LOO) — перевірте точки"
                )
        return "\n".join(lines)

    def export_graph_geojson(
        self, converter, frame_w: int, frame_h: int, origin_xy: tuple = (0.0, 0.0)
    ) -> dict:
        """origin_xy — Local Origin of propagation: internal graph states are local,
        без цього зсуву GeoJSON опинявся біля (0°, 0°) (баг, сесія 2026-07-12)."""
        features = []
        results = self._export_results()
        cx, cy = frame_w / 2.0, frame_h / 2.0
        ox, oy = float(origin_xy[0]), float(origin_xy[1])

        # Per-edge residual in properties → edges can be colour-coded on the map
        # by residual magnitude (bad loop closures become visually obvious).
        edge_res = self.compute_edge_residuals()
        anchor_ids = set(self.anchor_states())  # жорсткі + м'які (soft_anchors)

        for fid, affine in results.items():
            pt = np.array([[cx, cy]], dtype=np.float64)
            metric = cv2.transform(pt.reshape(-1, 1, 2), affine).reshape(-1, 2)[0]
            try:
                lat, lon = converter.metric_to_gps(float(metric[0]) + ox, float(metric[1]) + oy)
            except Exception:
                continue
            is_fixed = fid in anchor_ids
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"frame_id": fid, "type": "anchor" if is_fixed else "frame"},
                }
            )

        for e_idx, edge in enumerate(self._edges):
            affine_from = results.get(edge.from_id)
            affine_to = results.get(edge.to_id)
            if affine_from is None or affine_to is None:
                continue
            try:
                pt = np.array([[cx, cy]], dtype=np.float64).reshape(-1, 1, 2)
                m_from = cv2.transform(pt, affine_from).reshape(-1, 2)[0]
                m_to = cv2.transform(pt, affine_to).reshape(-1, 2)[0]
                lat1, lon1 = converter.metric_to_gps(float(m_from[0]) + ox, float(m_from[1]) + oy)
                lat2, lon2 = converter.metric_to_gps(float(m_to[0]) + ox, float(m_to[1]) + oy)
            except Exception:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                    "properties": {
                        "from_id": edge.from_id,
                        "to_id": edge.to_id,
                        "edge_type": edge.edge_type,
                        "residual": (None if np.isnan(edge_res[e_idx]) else float(edge_res[e_idx])),
                        "weight": float(edge.weight),
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}
