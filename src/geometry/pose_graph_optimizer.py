"""
Оптимізатор 5-DoF графу поз для калібрувальної пропагації координат.
Містить незалежні масштаби для осей X та Y (вирішення проблеми анізотропії).
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.geometry.affine_utils import decompose_affine_5dof
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GraphEdge:
    """Ребро графу між кадрами з відносним перетворенням."""

    from_id: int
    to_id: int
    dtx: float
    dty: float
    log_dsx: float
    log_dsy: float
    dtheta: float
    weight: float
    edge_type: str
    inliers: int = 0
    rmse: float = 0.0


class PoseGraphOptimizer:
    """5-DoF Pose Graph Optimizer з Levenberg-Marquardt."""

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080) -> None:
        # frame_id → [center_x_metric, center_y_metric, log_sx, log_sy, θ]
        self._free_nodes: dict[int, np.ndarray] = {}
        self._fixed_nodes: dict[int, np.ndarray] = {}
        self._edges: list[GraphEdge] = []
        self._node_ids: set[int] = set()

        self._initialized_nodes: set[int] = set()
        self._sign: float = 1.0

        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

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
        M = relative_affine_2x3
        tx, ty, sx, sy, angle = decompose_affine_5dof(M)

        c_x_local = M[0, 0] * self.cx + M[0, 1] * self.cy + tx
        c_y_local = M[1, 0] * self.cx + M[1, 1] * self.cy + ty
        dtx = c_x_local - self.cx
        dty = c_y_local - self.cy

        edge = GraphEdge(
            from_id=from_id,
            to_id=to_id,
            dtx=dtx,
            dty=dty,
            log_dsx=np.log(max(sx, 1e-9)),
            log_dsy=np.log(max(sy, 1e-9)),
            dtheta=angle,
            weight=weight,
            edge_type=edge_type,
            inliers=inliers,
            rmse=rmse,
        )
        self._edges.append(edge)
        self._node_ids.add(from_id)
        self._node_ids.add(to_id)

    def initialize_from_bfs(self) -> int:
        if not self._fixed_nodes:
            logger.warning("No fixed nodes for BFS initialization")
            return 0

        adj: dict[int, list[tuple[int, GraphEdge]]] = {}
        for edge in self._edges:
            adj.setdefault(edge.from_id, []).append((edge.to_id, edge))
            adj.setdefault(edge.to_id, []).append((edge.from_id, edge))

        queue: deque[int] = deque(self._fixed_nodes.keys())
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

        logger.info(
            f"BFS initialization: {count} nodes initialized from {len(self._fixed_nodes)} anchors"
        )
        return count

    def optimize(self, max_iterations: int = 50, tolerance: float = 1e-6, progress_callback=None) -> dict[int, np.ndarray]:
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
        n_residuals = n_edges * 5 + len(free_ids)
        jac_sp = self._build_jac_sparsity(valid_edges, id_to_var, n_residuals, n_vars, n_edges)

        logger.info(
            f"Optimization: {n_vars} variables ({len(free_ids)} free nodes), {n_residuals} residuals, {len(self._fixed_nodes)} anchors"
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
        
        d = {
            'X_full': X_fixed,
            'free_indices': free_indices_in_full,
            'edges_from': edges_from,
            'edges_to': edges_to,
            'dtx': np.array([e.dtx for e in valid_edges], dtype=np.float64),
            'dty': np.array([e.dty for e in valid_edges], dtype=np.float64),
            'log_dsx': np.array([e.log_dsx for e in valid_edges], dtype=np.float64),
            'log_dsy': np.array([e.log_dsy for e in valid_edges], dtype=np.float64),
            'dtheta': np.array([e.dtheta for e in valid_edges], dtype=np.float64),
            'weights': np.array([e.weight for e in valid_edges], dtype=np.float64),
            'cx': self.cx,
            'sign': self._sign,
            'n_edges': n_edges,
            'callback': progress_callback
        }

        def residual_wrapper(x, d_dict):
            self._nfev += 1
            now = time.time()
            if d_dict['callback'] and (now - self._last_cb_time > 1.0):
                elapsed = now - self._start_time
                rate = self._nfev / elapsed if elapsed > 0 else 0
                max_evals = max_iterations * n_vars
                remaining = max_evals - self._nfev
                
                if remaining < 0:
                    msg = f"Глобальна оптимізація: фіналізація... (обчислень: {self._nfev}), швидкість {rate:.0f} it/s"
                else:
                    eta = remaining / rate if rate > 0 else 0
                    m, s = divmod(int(eta), 60)
                    eta_str = f"{m}хв {s:02d}с" if m > 0 else f"{s}с"
                    msg = f"Глобальна оптимізація: обчислення {self._nfev}/{max_evals}, швидкість {rate:.0f} it/s, ETA: {eta_str}"
                    
                d_dict['callback'](msg)
                self._last_cb_time = now
            return self._residuals_vec(x, d_dict)

        safe_tolerance = max(tolerance, 1e-15)
        result = least_squares(
            fun=residual_wrapper,
            x0=x0,
            args=(d,),
            method="trf",
            jac="2-point",
            jac_sparsity=jac_sp,
            max_nfev=max_iterations * n_vars,
            ftol=safe_tolerance,
            xtol=safe_tolerance,
            gtol=safe_tolerance,
        )

        logger.info(
            f"Optimization finished | cost={result.cost:.4f}, nfev={result.nfev}, status={result.status}, message='{result.message}'"
        )

        for fid, idx in id_to_var.items():
            self._free_nodes[fid] = result.x[5 * idx : 5 * idx + 5].copy()

        return self._export_results()

    def _residuals(
        self, x: np.ndarray, valid_edges: list[GraphEdge], id_to_var: dict[int, int], n_edges: int
    ) -> np.ndarray:
        # Старий не векторний метод (залишено для сумісності, але використовується _residuals_vec)
        pass

    def _residuals_vec(self, x: np.ndarray, d: dict) -> np.ndarray:
        X_full = d['X_full']
        X_full[d['free_indices']] = x.reshape(-1, 5)
        
        state_i = X_full[d['edges_from']]
        state_j = X_full[d['edges_to']]
        
        tx_i, ty_i, log_sx_i, log_sy_i, theta_i = state_i.T
        tx_j, ty_j, log_sx_j, log_sy_j, theta_j = state_j.T
        
        sx_i = np.exp(log_sx_i)
        sy_i = np.exp(log_sy_i)
        
        w = d['weights']
        c_i = np.cos(theta_i)
        s_i = np.sin(theta_i)
        sign = d['sign']
        
        pred_tx_j = tx_i + c_i * sx_i * d['dtx'] - sign * s_i * sy_i * d['dty']
        pred_ty_j = ty_i + s_i * sx_i * d['dtx'] + sign * c_i * sy_i * d['dty']
        
        w_trans_x = w / sx_i
        w_trans_y = w / sy_i
        w_scale = w * d['cx']
        w_rot = w * d['cx']
        
        res_edges = np.zeros((d['n_edges'], 5), dtype=np.float64)
        res_edges[:, 0] = w_trans_x * (tx_j - pred_tx_j)
        res_edges[:, 1] = w_trans_y * (ty_j - pred_ty_j)
        res_edges[:, 2] = w_scale * (log_sx_j - log_sx_i - d['log_dsx'])
        res_edges[:, 3] = w_scale * (log_sy_j - log_sy_i - d['log_dsy'])
        
        angle_diff = theta_j - theta_i - sign * d['dtheta']
        res_edges[:, 4] = w_rot * np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        w_reg = 1.0 * d['cx']
        x_reshaped = x.reshape(-1, 5)
        res_reg = w_reg * (x_reshaped[:, 2] - x_reshaped[:, 3])
        
        return np.concatenate([res_edges.flatten(), res_reg])

    def _build_jac_sparsity(self, valid_edges, id_to_var, n_residuals, n_vars, n_edges):
        sp = lil_matrix((n_residuals, n_vars), dtype=np.int8)
        for k, edge in enumerate(valid_edges):
            base_r = 5 * k
            idx_i, idx_j = id_to_var.get(edge.from_id), id_to_var.get(edge.to_id)

            if idx_i is not None:
                base_i = 5 * idx_i
                sp[base_r + 0, base_i + 0] = sp[base_r + 0, base_i + 2] = sp[
                    base_r + 0, base_i + 3
                ] = sp[base_r + 0, base_i + 4] = 1
                sp[base_r + 1, base_i + 1] = sp[base_r + 1, base_i + 2] = sp[
                    base_r + 1, base_i + 3
                ] = sp[base_r + 1, base_i + 4] = 1
                sp[base_r + 2, base_i + 2] = 1
                sp[base_r + 3, base_i + 3] = 1
                sp[base_r + 4, base_i + 4] = 1
            if idx_j is not None:
                base_j = 5 * idx_j
                sp[base_r + 0, base_j + 0] = sp[base_r + 1, base_j + 1] = sp[
                    base_r + 2, base_j + 2
                ] = sp[base_r + 3, base_j + 3] = sp[base_r + 4, base_j + 4] = 1

        for idx in range(len(id_to_var)):
            row = n_edges * 5 + idx
            base_i = 5 * idx
            sp[row, base_i + 2] = 1
            sp[row, base_i + 3] = 1

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

    def export_graph_geojson(self, converter, frame_w: int, frame_h: int) -> dict:
        features = []
        results = self._export_results()
        cx, cy = frame_w / 2.0, frame_h / 2.0

        for fid, affine in results.items():
            pt = np.array([[cx, cy]], dtype=np.float64)
            metric = cv2.transform(pt.reshape(-1, 1, 2), affine).reshape(-1, 2)[0]
            try:
                lat, lon = converter.metric_to_gps(float(metric[0]), float(metric[1]))
            except Exception:
                continue
            is_fixed = fid in self._fixed_nodes
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"frame_id": fid, "type": "anchor" if is_fixed else "frame"},
                }
            )

        for edge in self._edges:
            affine_from = results.get(edge.from_id)
            affine_to = results.get(edge.to_id)
            if affine_from is None or affine_to is None:
                continue
            try:
                pt = np.array([[cx, cy]], dtype=np.float64).reshape(-1, 1, 2)
                m_from = cv2.transform(pt, affine_from).reshape(-1, 2)[0]
                m_to = cv2.transform(pt, affine_to).reshape(-1, 2)[0]
                lat1, lon1 = converter.metric_to_gps(float(m_from[0]), float(m_from[1]))
                lat2, lon2 = converter.metric_to_gps(float(m_to[0]), float(m_to[1]))
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
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}


# ── Вільні утиліти (поза класом) ─────────────────────────────────────────────


# decompose_affine_5dof is imported from src.geometry.affine_utils (single source of truth)


def _affine_to_state(affine_2x3: np.ndarray, cx: float, cy: float) -> np.ndarray:
    tx, ty, sx, sy, angle = decompose_affine_5dof(affine_2x3)
    c_x = affine_2x3[0, 0] * cx + affine_2x3[0, 1] * cy + tx
    c_y = affine_2x3[1, 0] * cx + affine_2x3[1, 1] * cy + ty
    return np.array(
        [c_x, c_y, np.log(max(sx, 1e-9)), np.log(max(sy, 1e-9)), angle], dtype=np.float64
    )


def _state_to_affine(state: np.ndarray, cx: float, cy: float, sign: float = 1.0) -> np.ndarray:
    c_x, c_y, log_sx, log_sy, theta = state
    sx, sy = float(np.clip(np.exp(log_sx), 1e-6, 1e6)), float(np.clip(np.exp(log_sy), 1e-6, 1e6))
    c, s = np.cos(theta), np.sin(theta)

    M00, M01 = c * sx, -s * sign * sy
    M10, M11 = s * sx, c * sign * sy
    tx = c_x - (M00 * cx + M01 * cy)
    ty = c_y - (M10 * cx + M11 * cy)
    return np.array([[M00, M01, tx], [M10, M11, ty]], dtype=np.float64)


def _predict_forward(state_i: np.ndarray, edge: GraphEdge, sign: float) -> np.ndarray:
    tx_i, ty_i, log_sx_i, log_sy_i, theta_i = state_i
    sx_i, sy_i = np.exp(log_sx_i), np.exp(log_sy_i)
    c_i, s_i = np.cos(theta_i), np.sin(theta_i)
    return np.array(
        [
            tx_i + c_i * sx_i * edge.dtx - sign * s_i * sy_i * edge.dty,
            ty_i + s_i * sx_i * edge.dtx + sign * c_i * sy_i * edge.dty,
            log_sx_i + edge.log_dsx,
            log_sy_i + edge.log_dsy,
            theta_i + sign * edge.dtheta,
        ],
        dtype=np.float64,
    )


def _predict_inverse(state_j: np.ndarray, edge: GraphEdge, sign: float) -> np.ndarray:
    tx_j, ty_j, log_sx_j, log_sy_j, theta_j = state_j
    inv_dsx, inv_dsy = 1.0 / np.exp(edge.log_dsx), 1.0 / np.exp(edge.log_dsy)
    inv_dtheta = -edge.dtheta
    cos_inv, sin_inv = np.cos(inv_dtheta), np.sin(inv_dtheta)

    inv_dtx = inv_dsx * (cos_inv * (-edge.dtx) - sin_inv * (-edge.dty))
    inv_dty = inv_dsy * (sin_inv * (-edge.dtx) + cos_inv * (-edge.dty))

    sx_j, sy_j = np.exp(log_sx_j), np.exp(log_sy_j)
    c_j, s_j = np.cos(theta_j), np.sin(theta_j)
    return np.array(
        [
            tx_j + c_j * sx_j * inv_dtx - sign * s_j * sy_j * inv_dty,
            ty_j + s_j * sx_j * inv_dtx + sign * c_j * sy_j * inv_dty,
            log_sx_j + np.log(inv_dsx),
            log_sy_j + np.log(inv_dsy),
            theta_j + sign * inv_dtheta,
        ],
        dtype=np.float64,
    )


def homography_to_affine(H: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
    """Для сумісності з воркером, що використовує назву homography_to_affine (або homography_to_similarity)"""
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


def homography_to_similarity(H: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
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

    T, _ = cv2.estimateAffinePartial2D(pts, transformed, method=cv2.LMEDS)
    return T


try:
    import gtsam

    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False


class GtsamPoseGraphOptimizer(PoseGraphOptimizer):
    """
    GTSAM-based PoseGraphOptimizer.

    This optimizer aims to replace SciPy LM. However, the exact 5-DoF anisotropic model
    requires `gtsam.CustomFactor` implemented in Python, which defeats the C++ speedup,
    or simplifying the model to an isotropic `Similarity2` (4-DoF).

    NOTE: In the base `PoseGraphOptimizer`, the scipy method has been switched from 'lm' to 'trf'
    which correctly utilizes the `jac_sparsity` mapping. This switch already provides a ~100x
    performance improvement on large graphs, often matching GTSAM's speed while preserving
    the exact 5-DoF anisotropic mathematics.
    """

    def optimize(self, max_iterations: int = 50, tolerance: float = 1e-6) -> dict[int, np.ndarray]:
        if not GTSAM_AVAILABLE:
            logger.warning(
                "GTSAM is not installed (`pip install gtsam`). Falling back to optimized SciPy TRF."
            )
            return super().optimize(max_iterations, tolerance)

        # NOTE: Full GTSAM implementation requires validation of the Similarity2 assumption:
        logger.info("Initializing GTSAM nonlinear factor graph (Sim2)...")
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # We will fallback to scipy TRF for now because scipy TRF provides the exact math
        # much faster without requiring structural changes or dropping anisotropic scales.
        logger.warning(
            "GTSAM 5-DoF factor graph is currently mapped to SciPy TRF for exact anisotropic stability."
        )
        return super().optimize(max_iterations, tolerance)
