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

        # Кеш пер-ребрових резидуалів (діагностика, Етап 1). None = ще не рахували.
        self._last_edge_residuals: np.ndarray | None = None

        # Ребра, викинуті two-stage prune (Етап 3). Порожньо = prune не спрацював.
        self._pruned_edges: list[GraphEdge] = []

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

    def warm_start_from_affines(self, affines: dict[int, np.ndarray]) -> int:
        """Тепла ініціалізація станів із попереднього розв'язку (Етап 4.2).

        Замість BFS з нуля: коли користувач додав/посунув один якір, x0 =
        попередній розв'язок (frame_affine з HDF5 → стани). Та сама модель,
        швидша ітерація користувача. Фіксовані вузли (якорі) не чіпаємо; BFS
        лишається для першого запуску та як fallback для вузлів без стану.
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

    def optimize(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        progress_callback=None,
        use_analytic_jac: bool = False,
        two_stage_prune: bool = False,
        prune_mad_k: float = 5.0,
        prune_max_spatial_frac: float = 0.2,
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
        # Індекс вільної змінної (−1 = фіксований вузол) — для аналітичного якобіана
        edge_from_free = np.array(
            [id_to_var.get(e.from_id, -1) for e in valid_edges], dtype=np.int64
        )
        edge_to_free = np.array(
            [id_to_var.get(e.to_id, -1) for e in valid_edges], dtype=np.int64
        )


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
            'n_free': len(free_ids),
            'edge_from_free': edge_from_free,
            'edge_to_free': edge_to_free,
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

        if use_analytic_jac:
            # Аналітичний якобіан (Етап 4.1): точніші градієнти, 3-10× швидше.
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

        # ── Етап 3: two-stage L2 → prune → L2 (за прапорцем, дефолт off) ──
        # Поріг рахується ВІДНОСНО інших spatial-резидуалів (та сама природа),
        # а не абсолютною константою — валідні loop closures лишаються з
        # повною L2-вагою. Warm start із розв'язку кроку 1 (self._free_nodes).
        if two_stage_prune:
            pruned = self._prune_bad_spatial_edges(prune_mad_k, prune_max_spatial_frac)
            if pruned:
                logger.info(
                    f"Two-stage prune: викинуто {len(pruned)} spatial-ребер "
                    f"(поза median+{prune_mad_k}·MAD), повторна L2 (warm start)"
                )
                return self.optimize(
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    progress_callback=progress_callback,
                    use_analytic_jac=use_analytic_jac,
                    two_stage_prune=False,
                )

        return self._export_results()

    def _residuals_vec(self, x: np.ndarray, d: dict) -> np.ndarray:
        X_full = d['X_full']
        X_full[d['free_indices']] = x.reshape(-1, 5)

        res_edges = edge_residual(
            X_full[d['edges_from']], X_full[d['edges_to']],
            d['dtx'], d['dty'], d['log_dsx'], d['log_dsy'], d['dtheta'],
            d['weights'], d['cx'], d['sign'],
        )

        w_reg = 200.0 * d['cx']
        x_reshaped = x.reshape(-1, 5)
        res_reg = w_reg * (x_reshaped[:, 2] - x_reshaped[:, 3])

        return np.concatenate([res_edges.flatten(), res_reg])

    def _jacobian_vec(self, x: np.ndarray, d: dict):
        """Аналітичний якобіан _residuals_vec (Етап 4.1).

        Похідні виписані руками для 5-DoF анізотропної моделі. Та сама модель,
        що й FD-варіант — лише точніші градієнти та 3-10× швидша оптимізація.
        Верифікується проти 2-point FD (див. tests/test_pose_graph_jacobian.py).
        """
        X_full = d['X_full']
        X_full[d['free_indices']] = x.reshape(-1, 5)

        si = X_full[d['edges_from']]
        sj = X_full[d['edges_to']]
        txi, tyi, lxi, lyi, thi = si.T
        txj, tyj, lxj, lyj, thj = sj.T

        w = d['weights']
        cx = d['cx']
        sign = d['sign']
        dtx = d['dtx']
        dty = d['dty']

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

        # ── похідні по вузлу i (from) ──
        j0_txi = -w * inv_sxi
        j0_lxi = -w * ci * dtx - res0
        j0_lyi = w * sign * s_i * dty * syx
        j0_thi = w * s_i * dtx + w * sign * ci * dty * syx
        j1_tyi = -w * inv_syi
        j1_lxi = -w * s_i * dtx * sxy
        j1_lyi = -w * sign * ci * dty - res1
        j1_thi = -w * ci * dtx * sxy + w * sign * s_i * dty
        jcx = w * cx  # ваги масштабу/кута (−для i, +для j)

        # ── похідні по вузлу j (to) ──
        j0_txj = w * inv_sxi
        j1_tyj = w * inv_syi

        n_edges = d['n_edges']
        n_free = d['n_free']
        ff = d['edge_from_free']
        ft = d['edge_to_free']
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

        # регуляризатор ізотропії вузла: reg_p = 200*cx*(log_sx_p - log_sy_p)
        if n_free > 0:
            p_idx = np.arange(n_free)
            reg_row = 5 * n_edges + p_idx
            w_reg = 200.0 * cx
            add(reg_row, 5 * p_idx + 2, np.full(n_free, w_reg))
            add(reg_row, 5 * p_idx + 3, np.full(n_free, -w_reg))

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        J = coo_matrix((data, (rows, cols)),
                       shape=(5 * n_edges + n_free, 5 * n_free), dtype=np.float64)
        return J.tocsr()

    def _build_jac_sparsity(self, valid_edges, id_to_var, n_residuals, n_vars, n_edges):
        # COO-конструктор (rows/cols списками) швидший за поелементний lil на
        # великих графах. Патерн розрідженості ІДЕНТИЧНИЙ попередньому.
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

        for idx in range(len(id_to_var)):
            row = n_edges * 5 + idx
            bi = 5 * idx
            rows += [row, row]
            cols += [bi + 2, bi + 3]

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
    """Проєктує гомографію на афінну модель через 5 опорних точок навколо центру."""
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


# Аліас для сумісності з worker-ом (використовує назву homography_to_similarity)
homography_to_similarity = homography_to_affine
