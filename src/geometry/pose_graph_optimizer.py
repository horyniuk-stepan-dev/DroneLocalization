"""
Оптимізатор 5-DoF графу поз для калібрувальної пропагації координат.
Містить незалежні масштаби для осей X та Y (вирішення проблеми анізотропії).
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

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

    # ── Етап 3: two-stage prune ────────────────────────────────────────────

    def _anchor_reachable(self, edges: list["GraphEdge"]) -> set[int]:
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
    ) -> list["GraphEdge"]:
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

    # ── Діагностика (read-only, Етап 1): нуль впливу на розв'язок ──────────

    def _current_states_full(self) -> dict[int, np.ndarray]:
        """Поточні стани всіх ІНІЦІАЛІЗОВАНИХ вузлів (fixed + досяжні free)."""
        states: dict[int, np.ndarray] = dict(self._fixed_nodes)
        for fid, st in self._free_nodes.items():
            if fid in self._initialized_nodes:
                states[fid] = st
        return states

    def _single_edge_residual(
        self, si: np.ndarray, sj: np.ndarray, e: "GraphEdge"
    ) -> np.ndarray:
        """Зважений 5-вектор резидуала ребра — ТА САМА формула, що в _residuals_vec."""
        txi, tyi, lxi, lyi, thi = si
        txj, tyj, lxj, lyj, thj = sj
        sxi, syi = np.exp(lxi), np.exp(lyi)
        ci, s_i = np.cos(thi), np.sin(thi)
        w, cx, sign = e.weight, self.cx, self._sign

        pred_tx = txi + ci * sxi * e.dtx - sign * s_i * syi * e.dty
        pred_ty = tyi + s_i * sxi * e.dtx + sign * ci * syi * e.dty
        ad = thj - thi - sign * e.dtheta
        return np.array([
            (w / sxi) * (txj - pred_tx),
            (w / syi) * (tyj - pred_ty),
            w * cx * (lxj - lxi - e.log_dsx),
            w * cx * (lyj - lyi - e.log_dsy),
            w * cx * np.arctan2(np.sin(ad), np.cos(ad)),
        ], dtype=np.float64)

    def compute_edge_residuals(self) -> np.ndarray:
        """Норма зваженого резидуала на КОЖНЕ ребро (за поточними станами).

        result.fun уже містить ці числа під час оптимізації, але викидається —
        тут відтворюємо їх для діагностики. NaN, якщо вузол ребра недосяжний.
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
        """Статистика резидуалів ОКРЕМО для temporal і spatial (різні масштаби!)."""
        res = self.compute_edge_residuals()
        out: dict[str, dict] = {}
        for cls in ("temporal", "spatial"):
            vals = np.array([
                res[k] for k, e in enumerate(self._edges)
                if e.edge_type == cls and not np.isnan(res[k])
            ])
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
        """Для кожного якоря: середній резидуал інцидентних ребер / медіана графу.

        Якір зі stress ≫ 1 конфліктує з графом (крива точка користувача).
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
        for fid in self._fixed_nodes:
            rs = incident.get(fid, [])
            if not rs:
                continue
            mean_r = float(np.mean(rs))
            stress[fid] = mean_r / med if med > 0 else mean_r
        return stress

    def diagnostics_report(self, top_n: int = 5) -> dict:
        """Повний звіт пропагації (Етап 1.3): класи ребер, резидуали, топ-гірших,
        anchor stress. Read-only — нуль впливу на розв'язок."""
        res = self.compute_edge_residuals()
        stats = self.edge_residual_stats()
        stress = self.compute_anchor_stress()

        order = [k for k in np.argsort(res)[::-1] if not np.isnan(res[k])]
        worst = []
        for k in order[:top_n]:
            e = self._edges[k]
            worst.append({
                "from_id": e.from_id, "to_id": e.to_id,
                "type": e.edge_type, "residual": float(res[k]),
                "inliers": e.inliers, "rmse": e.rmse,
            })

        return {
            "num_edges": len(self._edges),
            "num_temporal": sum(1 for e in self._edges if e.edge_type == "temporal"),
            "num_spatial": sum(1 for e in self._edges if e.edge_type == "spatial"),
            "num_anchors": len(self._fixed_nodes),
            "residual_stats": stats,
            "worst_edges": worst,
            "anchor_stress": {int(k): float(v) for k, v in stress.items()},
        }

    def format_diagnostics(self, top_n: int = 5) -> str:
        """Текстовий звіт для лога/діалогу-підсумку."""
        r = self.diagnostics_report(top_n=top_n)
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
                    f"    #{w['from_id']}→#{w['to_id']} [{w['type']}] "
                    f"резидуал={w['residual']:.1f}"
                )
        hot = {k: v for k, v in r["anchor_stress"].items() if v >= 2.0}
        if hot:
            for fid, v in sorted(hot.items(), key=lambda kv: -kv[1]):
                lines.append(f"  ⚠ Якір #{fid}: stress {v:.1f}× медіани — перевірте точки")
        return "\n".join(lines)

    def export_graph_geojson(self, converter, frame_w: int, frame_h: int) -> dict:
        features = []
        results = self._export_results()
        cx, cy = frame_w / 2.0, frame_h / 2.0

        # Пер-ребровий резидуал у properties → на карті розфарбувати ребра
        # за резидуалом (погані loop closures стає ВИДНО очима).
        edge_res = self.compute_edge_residuals()

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

        for e_idx, edge in enumerate(self._edges):
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
                        "residual": (
                            None if np.isnan(edge_res[e_idx])
                            else float(edge_res[e_idx])
                        ),
                        "weight": float(edge.weight),
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


# GtsamPoseGraphOptimizer видалено (senior review, п.9): клас був заглушкою,
# яка в усіх гілках делегувала в SciPy TRF і губила progress_callback/loss/f_scale.
