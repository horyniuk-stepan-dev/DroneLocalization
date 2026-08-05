"""5-DoF pose-graph optimizer (Levenberg-Marquardt / TRF) with 5-DoF anisotropic model."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from src.geometry.affine_utils import decompose_affine_5dof
from src.geometry.pose_graph.diagnostics import DiagnosticsMixin
from src.geometry.pose_graph.model_5dof import (
    GraphEdge,
    _affine_to_state,
    _predict_forward,
    _predict_inverse,
    _state_to_affine,
    edge_residual,
)
from src.geometry.pose_graph.pruning import PruningMixin
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PoseGraphOptimizer(DiagnosticsMixin, PruningMixin):
    """5-DoF Pose Graph Optimizer with Levenberg-Marquardt."""

    def __init__(
        self, frame_w: int = 1920, frame_h: int = 1080, isotropy_weight: float = 200.0
    ) -> None:
        # frame_id -> [center_x_metric, center_y_metric, log_sx, log_sy, theta]
        self._free_nodes: dict[int, np.ndarray] = {}
        self._fixed_nodes: dict[int, np.ndarray] = {}
        self._edges: list[GraphEdge] = []
        self._node_ids: set[int] = set()

        self._initialized_nodes: set[int] = set()
        self._sign: float = 1.0

        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

        # Isotropy regularizer weight: r = isotropy_weight * cx * (log_sx - log_sy)
        self.isotropy_weight = float(isotropy_weight)

        # Per-edge residual cache
        self._last_edge_residuals: np.ndarray | None = None

        # Edges pruned during optimization
        self._pruned_edges: list[GraphEdge] = []

        # Soft anchor priors: frame_id -> (state_anchor 5-vector, w_a)
        self._anchor_priors: dict[int, tuple[np.ndarray, float]] = {}

    @property
    def num_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def num_free(self) -> int:
        return len(self._free_nodes)

    @property
    def num_fixed(self) -> int:
        return len(self._fixed_nodes)

    @property
    def edges(self) -> list[GraphEdge]:
        return self._edges

    def add_node(self, frame_id: int, initial_state: np.ndarray | None = None) -> None:
        self._node_ids.add(frame_id)
        if initial_state is not None:
            self._free_nodes[frame_id] = np.array(initial_state, dtype=np.float64)
            self._initialized_nodes.add(frame_id)
        elif frame_id not in self._free_nodes and frame_id not in self._fixed_nodes:
            self._free_nodes[frame_id] = np.zeros(5, dtype=np.float64)

    def fix_node(self, frame_id: int, affine_2x3: np.ndarray) -> None:
        det = affine_2x3[0, 0] * affine_2x3[1, 1] - affine_2x3[0, 1] * affine_2x3[1, 0]
        if det < 0:
            self._sign = -1.0

        state = _affine_to_state(affine_2x3, self.cx, self.cy)
        self._fixed_nodes[frame_id] = state
        self._node_ids.add(frame_id)
        self._initialized_nodes.add(frame_id)
        self._free_nodes.pop(frame_id, None)

    def add_anchor(
        self,
        frame_id: int,
        affine_2x3: np.ndarray,
        sigma_m: float,
        base_w: float = 200.0,
        sigma_floor: float = 0.05,
    ) -> None:
        """Adds a soft anchor prior (unary factor) for the given frame_id."""
        affine_2x3 = np.asarray(affine_2x3, dtype=np.float64)
        det = affine_2x3[0, 0] * affine_2x3[1, 1] - affine_2x3[0, 1] * affine_2x3[1, 0]
        if det < 0:
            self._sign = -1.0

        state = _affine_to_state(affine_2x3, self.cx, self.cy)
        w_a = float(base_w) / max(float(sigma_m), float(sigma_floor))
        self._anchor_priors[frame_id] = (state, w_a)

        self._free_nodes[frame_id] = state.copy()
        self._node_ids.add(frame_id)
        self._initialized_nodes.add(frame_id)
        self._fixed_nodes.pop(frame_id, None)

    @property
    def sign(self) -> float:
        """Determinant sign (-1.0 for mirrored calibration matrices)."""
        return self._sign

    def anchor_states(self) -> dict[int, np.ndarray]:
        """Returns states of all fixed and soft anchors."""
        states: dict[int, np.ndarray] = dict(self._fixed_nodes)
        for fid, (st, _w) in self._anchor_priors.items():
            states.setdefault(fid, st)
        return states

    def _build_graph_edge(
        self,
        from_id: int,
        to_id: int,
        relative_affine_2x3: np.ndarray,
        weight: float,
        edge_type: str,
        inliers: int,
        rmse: float,
    ) -> GraphEdge:
        """Constructs a GraphEdge from a relative 2x3 affine matrix."""
        M = relative_affine_2x3
        tx, ty, sx, sy, angle = decompose_affine_5dof(M)

        c_x_local = M[0, 0] * self.cx + M[0, 1] * self.cy + tx
        c_y_local = M[1, 0] * self.cx + M[1, 1] * self.cy + ty
        return GraphEdge(
            from_id=from_id,
            to_id=to_id,
            dtx=c_x_local - self.cx,
            dty=c_y_local - self.cy,
            log_dsx=np.log(max(sx, 1e-9)),
            log_dsy=np.log(max(sy, 1e-9)),
            dtheta=angle,
            weight=weight,
            edge_type=edge_type,
            inliers=inliers,
            rmse=rmse,
        )

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        relative_affine_2x3: np.ndarray,
        weight: float,
        edge_type: str = "temporal",
        inliers: int = 0,
        rmse: float = 0.0,
    ) -> None:
        edge = self._build_graph_edge(
            from_id, to_id, relative_affine_2x3, weight, edge_type, inliers, rmse
        )
        self._edges.append(edge)
        self._node_ids.add(from_id)
        self._node_ids.add(to_id)

    def initialize_from_bfs(self) -> int:
        seeds = set(self._fixed_nodes) | set(self._anchor_priors)
        if not seeds:
            logger.warning("No fixed/anchor nodes for BFS initialization")
            return 0

        adj: dict[int, list[tuple[int, GraphEdge]]] = {}
        for edge in self._edges:
            adj.setdefault(edge.from_id, []).append((edge.to_id, edge))
            adj.setdefault(edge.to_id, []).append((edge.from_id, edge))

        queue: deque[int] = deque(seeds)
        count = 0

        while queue:
            current = queue.popleft()
            current_state = self._get_node_state(current)

            for neighbor_id, edge in adj.get(current, []):
                if neighbor_id in self._initialized_nodes:
                    continue

                if edge.from_id == current:
                    predicted = _predict_forward(current_state, edge, self._sign)
                else:
                    predicted = _predict_inverse(current_state, edge, self._sign)

                self._free_nodes[neighbor_id] = predicted
                self._initialized_nodes.add(neighbor_id)
                queue.append(neighbor_id)
                count += 1

        logger.info(f"BFS initialization: {count} nodes initialized from {len(seeds)} seeds")
        return count

    def warm_start_from_affines(self, affines: dict[int, np.ndarray]) -> int:
        """Warm initialization of states from previous solution.
        Instead of BFS from scratch: when user added/moved one anchor, x0 =
        previous solution (frame_affine from HDF5 -> states). Same model,
        faster user iteration. Fixed nodes (anchors) untouched; BFS
        remains for first run and as fallback for nodes without state.
        """
        count = 0
        for fid, affine in affines.items():
            if fid in self._fixed_nodes or fid not in self._node_ids:
                continue
            state = _affine_to_state(np.asarray(affine, dtype=np.float64), self.cx, self.cy)
            self._free_nodes[fid] = state
            self._initialized_nodes.add(fid)
            count += 1
        logger.info(f"Warm start: {count} nodes seeded from previous solution")
        return count

    def preliminary_states(self, seed_affines: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Estimate FULL node states by BFS chain ONLY along temporal edges from
        specified anchor matrices (BEFORE matching/optimization).
        Distances/poses invariant to Local Origin, so take absolute matrices
        of anchors. Does not mutate optimizer state. Returns {fid -> state[5]}.
        """
        seeds = {}
        for fid, aff in seed_affines.items():
            if fid in self._node_ids:
                seeds[fid] = _affine_to_state(np.asarray(aff, dtype=np.float64), self.cx, self.cy)
        if not seeds:
            return {}

        adj: dict[int, list] = {}
        for e in self._edges:
            if e.edge_type != "temporal":
                continue
            adj.setdefault(e.from_id, []).append((e.to_id, e))
            adj.setdefault(e.to_id, []).append((e.from_id, e))

        states: dict[int, np.ndarray] = dict(seeds)
        queue = deque(seeds)
        while queue:
            cur = queue.popleft()
            cur_state = states[cur]
            for nb, e in adj.get(cur, []):
                if nb in states:
                    continue
                states[nb] = (
                    _predict_forward(cur_state, e, self._sign)
                    if e.from_id == cur
                    else _predict_inverse(cur_state, e, self._sign)
                )
                queue.append(nb)
        return states

    def preliminary_centers(self, seed_affines: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Estimate metric centers — thin wrapper over preliminary_states."""
        return {fid: st[:2].copy() for fid, st in self.preliminary_states(seed_affines).items()}

    def odometry_consistency_factors(
        self,
        specs: list,
        prelim_states: dict[int, np.ndarray],
        frame_w: int,
        frame_h: int,
        margin: float = 1.5,
        drift_frac: float = 0.25,
        factor: float = 0.3,
    ) -> list:
        """Odometry-consistency (PCM-lite): xfactor weight for spatial edges
        inconsistent with temporal chain.
        For each candidate (i, j, similarity): predict center j from node i
        VIA EDGE (predict_forward on preliminary state i) and compare to where
        j is placed by temporal chain (prelim center). Tolerance grows with chain length
        |i-j| (drift compensation). Inconsistent edge -> weight xfactor. NO pruning.
        specs — list of dicts with keys 'i','j','similarity'. Returns list of factors.
        """
        n = len(specs)
        factors = [1.0] * n
        if not prelim_states or n == 0:
            return factors

        # Median metric displacement per slot (to compensate for drift in long chains).
        ids = sorted(prelim_states)
        consec = [
            float(np.linalg.norm(prelim_states[b][:2] - prelim_states[a][:2])) / max(b - a, 1)
            for a, b in zip(ids, ids[1:])
            if b - a <= 3
        ]
        per_slot_disp_m = float(np.median(consec)) if consec else 0.0
        frame_diag_px = float(np.hypot(frame_w, frame_h))

        for k, spec in enumerate(specs):
            i, j = int(spec["i"]), int(spec["j"])
            si = prelim_states.get(i)
            sj = prelim_states.get(j)
            if si is None or sj is None:
                continue  # cannot judge without estimate — keep full weight
            edge = self._build_graph_edge(i, j, spec["similarity"], 1.0, "spatial", 0, 0.0)
            pred = _predict_forward(si, edge, self._sign)
            inconsistency = float(np.linalg.norm(pred[:2] - sj[:2]))
            scale_i = float(np.exp(si[2]))
            frame_diag_m = frame_diag_px * scale_i
            allowed = frame_diag_m * margin + per_slot_disp_m * abs(i - j) * drift_frac
            if inconsistency > allowed:
                factors[k] = factor
        return factors

    def estimate_min_loop_gap(
        self, frame_w: int, frame_h: int, k_overlap: float = 1.0
    ) -> int | None:
        """Auto min_frame_gap for loop closure from motion geometry.
        Median frame center motion per slot in px taken from temporal edges (dtx, dty
        normalized to edge span). Minimum gap WITHOUT physical overlap:
        gap_min = ceil(k_overlap * frame_diag_px / median_disp_px). Below this
        gap two frames still overlap in time (not a loop closure); above —
        feature match means a real loop closure. None if no temporal edges
        or motion is degenerate (call then leaves explicit constant).
        """
        disps = [
            np.hypot(e.dtx, e.dty) / max(abs(e.to_id - e.from_id), 1)
            for e in self._edges
            if e.edge_type == "temporal"
        ]
        if not disps:
            return None
        median_disp = float(np.median(disps))
        if median_disp < 1e-6:
            return None
        frame_diag = float(np.hypot(frame_w, frame_h))
        return int(np.ceil(max(k_overlap, 1e-6) * frame_diag / median_disp))

    def optimize(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        progress_callback=None,
        use_analytic_jac: bool = False,
        two_stage_prune: bool = False,
        prune_mad_k: float = 5.0,
        prune_max_spatial_frac: float = 0.2,
        gnc_spatial: bool = False,
        gnc_rounds: int = 5,
        gnc_mad_k: float = 3.0,
        kinematic_prior_weight: float = 0.0,
    ) -> dict[int, np.ndarray]:

        if not self._edges:
            logger.warning("No edges — returning current states as-is")
            return self._export_results()

        free_ids = sorted(
            [fid for fid in self._free_nodes.keys() if fid in self._initialized_nodes]
        )
        id_to_var: dict[int, int] = {fid: idx for idx, fid in enumerate(free_ids)}
        n_vars = len(free_ids) * 5

        if n_vars == 0:
            logger.warning("All nodes are fixed or unreachable — nothing to optimize")
            return self._export_results()

        x0 = np.zeros(n_vars, dtype=np.float64)
        for fid, idx in id_to_var.items():
            x0[5 * idx : 5 * idx + 5] = self._free_nodes[fid]

        valid_edges = []
        for e in self._edges:
            from_ok = e.from_id in id_to_var or e.from_id in self._fixed_nodes
            to_ok = e.to_id in id_to_var or e.to_id in self._fixed_nodes
            if from_ok and to_ok:
                valid_edges.append(e)

        if not valid_edges:
            logger.warning("No valid edges after filtering — returning BFS results")
            return self._export_results()

        n_edges = len(valid_edges)

        # Soft anchors: unary priors only for free anchor nodes.
        # Empty (soft_anchors off) → n_anch=0 → all branches below are no-ops.
        anchor_var_idx: list[int] = []
        anchor_states_list: list[np.ndarray] = []
        anchor_w_list: list[float] = []
        for fid, (a_state, a_w) in self._anchor_priors.items():
            if fid in id_to_var:
                anchor_var_idx.append(id_to_var[fid])
                anchor_states_list.append(a_state)
                anchor_w_list.append(a_w)
        n_anch = len(anchor_var_idx)

        # Kinematic prior: second-difference factors on frame centres.
        # w=0 (default) → n_kin=0 → block is completely absent, residual
        # structure is unchanged (contract 5·E+N+5·A preserved byte-for-byte).
        kin_ids, kin_alpha_l, kin_w_l = self._build_kinematic_triples(
            id_to_var, kinematic_prior_weight
        )
        n_kin = len(kin_ids)

        n_residuals = n_edges * 5 + len(free_ids) + n_anch * 5 + 2 * n_kin
        jac_sp = self._build_jac_sparsity(
            valid_edges,
            id_to_var,
            n_residuals,
            n_vars,
            n_edges,
            anchor_var_idx,
            kin_free=[
                (id_to_var.get(a, -1), id_to_var.get(b, -1), id_to_var.get(c, -1))
                for a, b, c in kin_ids
            ],
        )

        logger.info(
            f"Optimization: {n_vars} variables ({len(free_ids)} free nodes), "
            f"{n_residuals} residuals, {len(self.anchor_states())} anchors"
        )

        # jac_sparsity is ignored by 'lm'. method='trf' with sparse jacobian runs 100x faster.
        import time

        self._nfev = 0
        self._start_time = time.time()
        self._last_cb_time = self._start_time

        # Prepare vectorized data
        total_nodes = len(self._node_ids)
        node_id_to_idx = {nid: i for i, nid in enumerate(self._node_ids)}

        X_fixed = np.zeros((total_nodes, 5), dtype=np.float64)
        for fid, state in self._fixed_nodes.items():
            if fid in node_id_to_idx:
                X_fixed[node_id_to_idx[fid]] = state

        free_indices_in_full = [node_id_to_idx[fid] for fid in free_ids]
        edges_from = np.array([node_id_to_idx[e.from_id] for e in valid_edges], dtype=np.int32)
        edges_to = np.array([node_id_to_idx[e.to_id] for e in valid_edges], dtype=np.int32)
        # Free-variable index (−1 = fixed node) — used in the analytic Jacobian
        edge_from_free = np.array(
            [id_to_var.get(e.from_id, -1) for e in valid_edges], dtype=np.int64
        )
        edge_to_free = np.array([id_to_var.get(e.to_id, -1) for e in valid_edges], dtype=np.int64)

        d = {
            "X_full": X_fixed,
            "free_indices": free_indices_in_full,
            "edges_from": edges_from,
            "edges_to": edges_to,
            "dtx": np.array([e.dtx for e in valid_edges], dtype=np.float64),
            "dty": np.array([e.dty for e in valid_edges], dtype=np.float64),
            "log_dsx": np.array([e.log_dsx for e in valid_edges], dtype=np.float64),
            "log_dsy": np.array([e.log_dsy for e in valid_edges], dtype=np.float64),
            "dtheta": np.array([e.dtheta for e in valid_edges], dtype=np.float64),
            "weights": np.array([e.weight for e in valid_edges], dtype=np.float64),
            "cx": self.cx,
            "w_reg": self.isotropy_weight * self.cx,
            "sign": self._sign,
            "n_edges": n_edges,
            "n_free": len(free_ids),
            "edge_from_free": edge_from_free,
            "edge_to_free": edge_to_free,
            "anchor_var_idx": np.array(anchor_var_idx, dtype=np.int64),
            "anchor_states": (
                np.array(anchor_states_list, dtype=np.float64)
                if anchor_states_list
                else np.zeros((0, 5), dtype=np.float64)
            ),
            "anchor_w": np.array(anchor_w_list, dtype=np.float64),
            "n_anch": n_anch,
            "kin_ia": np.array([node_id_to_idx[a] for a, _b, _c in kin_ids], dtype=np.int64),
            "kin_ib": np.array([node_id_to_idx[b] for _a, b, _c in kin_ids], dtype=np.int64),
            "kin_ic": np.array([node_id_to_idx[c] for _a, _b, c in kin_ids], dtype=np.int64),
            "kin_fa": np.array([id_to_var.get(a, -1) for a, _b, _c in kin_ids], dtype=np.int64),
            "kin_fb": np.array([id_to_var.get(b, -1) for _a, b, _c in kin_ids], dtype=np.int64),
            "kin_fc": np.array([id_to_var.get(c, -1) for _a, _b, c in kin_ids], dtype=np.int64),
            "kin_alpha": np.array(kin_alpha_l, dtype=np.float64),
            "kin_w": np.array(kin_w_l, dtype=np.float64),
            "n_kin": n_kin,
            "callback": progress_callback,
        }

        def residual_wrapper(x, d_dict):
            self._nfev += 1
            now = time.time()
            if d_dict["callback"] and (now - self._last_cb_time > 1.0):
                elapsed = now - self._start_time
                rate = self._nfev / elapsed if elapsed > 0 else 0
                max_evals = max_iterations * n_vars
                remaining = max_evals - self._nfev

                if remaining < 0:
                    msg = f"Global optimization: finalizing... (evals: {self._nfev}), speed {rate:.0f} it/s"
                else:
                    eta = remaining / rate if rate > 0 else 0
                    m, s = divmod(int(eta), 60)
                    eta_str = f"{m}m {s:02d}s" if m > 0 else f"{s}s"
                    msg = f"Global optimization: evaluating {self._nfev}/{max_evals}, speed {rate:.0f} it/s, ETA: {eta_str}"

                d_dict["callback"](msg)
                self._last_cb_time = now
            return self._residuals_vec(x, d_dict)

        if use_analytic_jac:
            # Analytic Jacobian: more accurate gradients, 3-10× faster.
            jac_kwargs = {"jac": self._jacobian_vec}
        else:
            jac_kwargs = {"jac": "2-point", "jac_sparsity": jac_sp}

        result = least_squares(
            fun=residual_wrapper,
            x0=x0,
            args=(d,),
            method="trf",
            max_nfev=max_iterations * n_vars,
            ftol=tolerance,
            xtol=tolerance,
            gtol=tolerance,
            **jac_kwargs,
        )

        logger.info(
            f"Optimization finished | cost={result.cost:.4f}, nfev={result.nfev}, status={result.status}, message='{result.message}'"
        )

        for fid, idx in id_to_var.items():
            self._free_nodes[fid] = result.x[5 * idx : 5 * idx + 5].copy()

        # GNC: gradual prune schedule. Flag-gated, default off.
        # Mutually exclusive with two_stage_prune; on a clean scene it is a no-op.
        if gnc_spatial:
            return self._run_gnc_spatial(
                gnc_rounds,
                gnc_mad_k,
                max_iterations=max_iterations,
                tolerance=tolerance,
                progress_callback=progress_callback,
                use_analytic_jac=use_analytic_jac,
                kinematic_prior_weight=kinematic_prior_weight,
            )

        # Two-stage L2 → prune → L2. Flag-gated, default off.
        # Threshold is computed RELATIVE to other spatial residuals (same nature),
        # not as an absolute constant — valid loop closures keep full L2 weight.
        # Warm-started from step-1 solution (self._free_nodes).
        if two_stage_prune:
            pruned = self._prune_bad_spatial_edges(prune_mad_k, prune_max_spatial_frac)
            if pruned:
                logger.info(
                    f"Two-stage prune: discarded {len(pruned)} spatial edges "
                    f"(outside median+{prune_mad_k}*MAD), re-running L2 (warm start)"
                )
                return self.optimize(
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    progress_callback=progress_callback,
                    use_analytic_jac=use_analytic_jac,
                    two_stage_prune=False,
                    kinematic_prior_weight=kinematic_prior_weight,
                )

        return self._export_results()

    def _build_kinematic_triples(
        self, id_to_var: dict[int, int], weight: float
    ) -> tuple[list[tuple[int, int, int]], list[float], list[float]]:
        """Triplets (a, b, c) of adjacent nodes for kinematic prior.
        Non-equidistant slots (keyframe selection gaps, bridges): center b
        is compared against alpha*a + (1-alpha)*c, alpha = h2/(h1+h2), h1 = b-a, h2 = c-b;
        weight scales as 2/(h1+h2) — at step 1 this equals ``weight`` and
        corresponds to (a - 2b + c)/2, further prior weakens proportionally to gap.
        Triplets with all three fixed nodes are skipped (constant).
        """
        if weight <= 0.0:
            return [], [], []
        participating = sorted(
            fid for fid in self._node_ids if fid in id_to_var or fid in self._fixed_nodes
        )
        ids: list[tuple[int, int, int]] = []
        alphas: list[float] = []
        weights: list[float] = []
        for a, b, c in zip(participating, participating[1:], participating[2:]):
            if a not in id_to_var and b not in id_to_var and c not in id_to_var:
                continue
            h1 = float(b - a)
            h2 = float(c - b)
            denom = h1 + h2
            if denom <= 0.0:
                continue
            ids.append((a, b, c))
            alphas.append(h2 / denom)
            weights.append(weight * 2.0 / denom)
        return ids, alphas, weights

    def _residuals_vec(self, x: np.ndarray, d: dict) -> np.ndarray:
        X_full = d["X_full"]
        X_full[d["free_indices"]] = x.reshape(-1, 5)

        res_edges = edge_residual(
            X_full[d["edges_from"]],
            X_full[d["edges_to"]],
            d["dtx"],
            d["dty"],
            d["log_dsx"],
            d["log_dsy"],
            d["dtheta"],
            d["weights"],
            d["cx"],
            d["sign"],
        )

        w_reg = d["w_reg"]
        x_reshaped = x.reshape(-1, 5)
        res_reg = w_reg * (x_reshaped[:, 2] - x_reshaped[:, 3])

        parts = [res_edges.flatten(), res_reg]

        if d.get("n_anch", 0) > 0:
            # Unary anchor prior: w_a·(state − state_anchor), angle is SO(2)-safe.
            ap = x_reshaped[d["anchor_var_idx"]]  # (n_anch, 5)
            diff = ap - d["anchor_states"]
            ang = np.arctan2(np.sin(diff[:, 4]), np.cos(diff[:, 4]))
            aw = d["anchor_w"][:, None]
            parts.append(
                (
                    aw * np.column_stack([diff[:, 0], diff[:, 1], diff[:, 2], diff[:, 3], ang])
                ).ravel()
            )

        if d.get("n_kin", 0) > 0:
            # Kinematic prior: r = w·(α·a + (1−α)·c − b), centres only.
            ca = X_full[d["kin_ia"]][:, :2]
            cb = X_full[d["kin_ib"]][:, :2]
            cc = X_full[d["kin_ic"]][:, :2]
            al = d["kin_alpha"][:, None]
            wk = d["kin_w"][:, None]
            parts.append((wk * (al * ca + (1.0 - al) * cc - cb)).ravel())

        return np.concatenate(parts)

    def _jacobian_vec(self, x: np.ndarray, d: dict):
        """Analytical Jacobian of _residuals_vec.
        Hand-derived partial derivatives for 5-DoF anisotropic model. Same model
        as FD variant — just exact gradients and 3-10x faster optimization.
        Verified against 2-point FD (see tests/test_pose_graph_jacobian.py).
        """
        X_full = d["X_full"]
        X_full[d["free_indices"]] = x.reshape(-1, 5)

        si = X_full[d["edges_from"]]
        sj = X_full[d["edges_to"]]
        txi, tyi, lxi, lyi, thi = si.T
        txj, tyj, lxj, lyj, thj = sj.T

        w = d["weights"]
        cx = d["cx"]
        sign = d["sign"]
        dtx = d["dtx"]
        dty = d["dty"]

        sxi = np.exp(lxi)
        syi = np.exp(lyi)
        inv_sxi = np.exp(-lxi)
        inv_syi = np.exp(-lyi)
        ci = np.cos(thi)
        s_i = np.sin(thi)
        syx = np.exp(lyi - lxi)  # syi / sxi
        sxy = np.exp(lxi - lyi)  # sxi / syi

        pred_tx = txi + ci * sxi * dtx - sign * s_i * syi * dty
        pred_ty = tyi + s_i * sxi * dtx + sign * ci * syi * dty
        res0 = (w * inv_sxi) * (txj - pred_tx)
        res1 = (w * inv_syi) * (tyj - pred_ty)

        # ── Derivatives w.r.t. node i (from) ──
        j0_txi = -w * inv_sxi
        j0_lxi = -w * ci * dtx - res0
        j0_lyi = w * sign * s_i * dty * syx
        j0_thi = w * s_i * dtx + w * sign * ci * dty * syx
        j1_tyi = -w * inv_syi
        j1_lxi = -w * s_i * dtx * sxy
        j1_lyi = -w * sign * ci * dty - res1
        j1_thi = -w * ci * dtx * sxy + w * sign * s_i * dty
        jcx = w * cx  # scale/angle weights (- for i, + for j)

        # ── Derivatives w.r.t. node j (to) ──
        j0_txj = w * inv_sxi
        j1_tyj = w * inv_syi

        n_edges = d["n_edges"]
        n_free = d["n_free"]
        n_anch = int(d.get("n_anch", 0))
        n_kin = int(d.get("n_kin", 0))
        ff = d["edge_from_free"]
        ft = d["edge_to_free"]
        base_r = 5 * np.arange(n_edges)

        rows: list = []
        cols: list = []
        data: list = []

        def add(r, c, v):
            rows.append(np.asarray(r))
            cols.append(np.asarray(c))
            data.append(np.asarray(v, dtype=np.float64))

        mf = ff >= 0
        if np.any(mf):
            bi = 5 * ff[mf]
            br = base_r[mf]
            add(br + 0, bi + 0, j0_txi[mf])
            add(br + 0, bi + 2, j0_lxi[mf])
            add(br + 0, bi + 3, j0_lyi[mf])
            add(br + 0, bi + 4, j0_thi[mf])
            add(br + 1, bi + 1, j1_tyi[mf])
            add(br + 1, bi + 2, j1_lxi[mf])
            add(br + 1, bi + 3, j1_lyi[mf])
            add(br + 1, bi + 4, j1_thi[mf])
            add(br + 2, bi + 2, -jcx[mf])
            add(br + 3, bi + 3, -jcx[mf])
            add(br + 4, bi + 4, -jcx[mf])
        mt = ft >= 0
        if np.any(mt):
            bj = 5 * ft[mt]
            br = base_r[mt]
            add(br + 0, bj + 0, j0_txj[mt])
            add(br + 1, bj + 1, j1_tyj[mt])
            add(br + 2, bj + 2, jcx[mt])
            add(br + 3, bj + 3, jcx[mt])
            add(br + 4, bj + 4, jcx[mt])

        # Isotropy regulariser for node: reg_p = 200*cx*(log_sx_p - log_sy_p)
        if n_free > 0:
            p_idx = np.arange(n_free)
            reg_row = 5 * n_edges + p_idx
            w_reg = d["w_reg"]
            add(reg_row, 5 * p_idx + 2, np.full(n_free, w_reg))
            add(reg_row, 5 * p_idx + 3, np.full(n_free, -w_reg))

        # Soft anchors: d(w_a·Δstate)/d(state) = w_a·I for all 5 components
        # (derivative of atan2(sin dθ, cos dθ) w.r.t. θ = 1). Block is diagonal.
        if n_anch > 0:
            av = d["anchor_var_idx"]
            aw = d["anchor_w"]
            base_a = 5 * n_edges + n_free
            a_rows = base_a + 5 * np.arange(n_anch)
            for comp in range(5):
                add(a_rows + comp, 5 * av + comp, aw)

        # Kinematic prior: r = w·(α·a + (1−α)·c − b), centres only —
        # linear in states, so Jacobian entries are constant.
        if n_kin > 0:
            fa = d["kin_fa"]
            fb = d["kin_fb"]
            fc = d["kin_fc"]
            al = d["kin_alpha"]
            wk = d["kin_w"]
            base_k = 5 * n_edges + n_free + 5 * n_anch + 2 * np.arange(n_kin)
            for f_idx, coef in ((fa, wk * al), (fc, wk * (1.0 - al)), (fb, -wk)):
                m = f_idx >= 0
                if np.any(m):
                    add(base_k[m] + 0, 5 * f_idx[m] + 0, coef[m])
                    add(base_k[m] + 1, 5 * f_idx[m] + 1, coef[m])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        J = coo_matrix(
            (data, (rows, cols)),
            shape=(5 * n_edges + n_free + 5 * n_anch + 2 * n_kin, 5 * n_free),
            dtype=np.float64,
        )
        return J.tocsr()

    def _build_jac_sparsity(
        self,
        valid_edges,
        id_to_var,
        n_residuals,
        n_vars,
        n_edges,
        anchor_var_idx=None,
        kin_free=None,
    ):
        # COO constructor (list-based rows/cols) is faster than element-wise lil
        # on large graphs. Sparsity pattern is IDENTICAL to the previous version.
        rows: list[int] = []
        cols: list[int] = []
        for k, edge in enumerate(valid_edges):
            base_r = 5 * k
            idx_i, idx_j = id_to_var.get(edge.from_id), id_to_var.get(edge.to_id)

            if idx_i is not None:
                bi = 5 * idx_i
                # res0 (tx): tx_i, log_sx_i, log_sy_i, theta_i
                rows += [base_r + 0] * 4
                cols += [bi + 0, bi + 2, bi + 3, bi + 4]
                # res1 (ty): ty_i, log_sx_i, log_sy_i, theta_i
                rows += [base_r + 1] * 4
                cols += [bi + 1, bi + 2, bi + 3, bi + 4]
                # res2/res3/res4 (log_sx / log_sy / theta)
                rows += [base_r + 2, base_r + 3, base_r + 4]
                cols += [bi + 2, bi + 3, bi + 4]
            if idx_j is not None:
                bj = 5 * idx_j
                rows += [base_r + 0, base_r + 1, base_r + 2, base_r + 3, base_r + 4]
                cols += [bj + 0, bj + 1, bj + 2, bj + 3, bj + 4]

        n_free = len(id_to_var)
        for idx in range(n_free):
            row = n_edges * 5 + idx
            bi = 5 * idx
            rows += [row, row]
            cols += [bi + 2, bi + 3]

        # Soft anchors: 5 diagonal entries per anchor node.
        for a, v in enumerate(anchor_var_idx or []):
            base = n_edges * 5 + n_free + 5 * a
            for comp in range(5):
                rows.append(base + comp)
                cols.append(5 * v + comp)

        # Kinematic prior: 2 rows per triple, tx/ty entries of free nodes.
        base_k = n_edges * 5 + n_free + 5 * len(anchor_var_idx or [])
        for t, (fa, fb, fc) in enumerate(kin_free or []):
            for f_idx in (fa, fb, fc):
                if f_idx >= 0:
                    rows += [base_k + 2 * t, base_k + 2 * t + 1]
                    cols += [5 * f_idx + 0, 5 * f_idx + 1]

        data = np.ones(len(rows), dtype=np.int8)
        sp = coo_matrix((data, (rows, cols)), shape=(n_residuals, n_vars), dtype=np.int8)
        return sp.tocsr()

    def _get_node_state(self, frame_id: int):
        if frame_id in self._fixed_nodes:
            return self._fixed_nodes[frame_id]
        return self._free_nodes.get(frame_id, np.zeros(5, dtype=np.float64))

    def _read_state(self, x, frame_id, id_to_var):
        if frame_id in self._fixed_nodes:
            return self._fixed_nodes[frame_id]
        idx = id_to_var[frame_id]
        return x[5 * idx : 5 * idx + 5]

    def _export_results(self) -> dict[int, np.ndarray]:
        results = {
            fid: _state_to_affine(state, self.cx, self.cy, self._sign)
            for fid, state in self._fixed_nodes.items()
        }
        for fid, state in self._free_nodes.items():
            if fid in self._initialized_nodes:
                results[fid] = _state_to_affine(state, self.cx, self.cy, self._sign)
        return results


def homography_to_affine(H: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
    """Projects homography onto 2x3 affine model via 5 control points around frame center."""
    cx, cy = frame_w / 2.0, frame_h / 2.0
    d = min(frame_w, frame_h) * 0.25
    pts = np.array(
        [[cx, cy], [cx - d, cy - d], [cx + d, cy - d], [cx + d, cy + d], [cx - d, cy + d]],
        dtype=np.float64,
    )
    transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float64))
    if transformed is None:
        return None
    transformed = transformed.reshape(-1, 2).astype(np.float64)

    T, _ = cv2.estimateAffine2D(pts, transformed, method=cv2.LMEDS)
    return T


def affine_fit_residual(H: np.ndarray, frame_w: int, frame_h: int) -> float | None:
    """RMS residual of affine approximation of homography H on 5 control points.
    homography_to_affine discards this residual. It is large when H is non-affine
    (camera tilt / terrain relief) — such temporal frames deserve lower trust.
    Returns residual in pixels or None. ~0 for pure affine H.
    """
    cx, cy = frame_w / 2.0, frame_h / 2.0
    d = min(frame_w, frame_h) * 0.25
    pts = np.array(
        [[cx, cy], [cx - d, cy - d], [cx + d, cy - d], [cx + d, cy + d], [cx - d, cy + d]],
        dtype=np.float64,
    )
    transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float64))
    if transformed is None:
        return None
    transformed = transformed.reshape(-1, 2).astype(np.float64)
    T, _ = cv2.estimateAffine2D(pts, transformed, method=cv2.LMEDS)
    if T is None:
        return None
    proj = (T[:, :2] @ pts.T).T + T[:, 2]
    return float(np.sqrt(np.mean(np.sum((proj - transformed) ** 2, axis=1))))


# Alias for backward compatibility (worker uses the name homography_to_similarity)
homography_to_similarity = homography_to_affine
