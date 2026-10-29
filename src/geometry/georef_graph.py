"""Non-rigid georeference of a local (no-GPS) map along its frame graph.

A single similarity (``georef_fit``) fixes the gauge of a local map but cannot
undo the scale/rotation drift that monocular propagation accumulates along the
flight: the map is right near the control points and wrong between them.

Here every frame gets its own similarity correction ``w = a_i * z + t_i``
(complex form: ``z`` local x+iy, ``w`` projected metres). Corrections are tied
together along the frame graph (temporal neighbours + frames that overlap in
the local map), so the field is smooth where the graph is dense and simply
interpolates between control points where it is a chain:

  * 2 control points  -> exactly the global similarity (constant field);
  * 3+ control points -> the field bends to follow them, i.e. scale/rotation
    drift is corrected piecewise along the trajectory;
  * beyond the last control point the correction stays constant (no
    extrapolation of the drift trend);
  * a graph component without any control point keeps the global similarity.

The problem is linear in (a_i, t_i), so it is one sparse least-squares solve.
Pure numpy/scipy, no Qt / h5py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from src.geometry.georef_fit import _as_points, fit_similarity

GRAPH = "graph"
MIN_GRAPH_POINTS = 3
STIFFNESS_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_STIFFNESS = 0.1
_IDW_K = 4
_REVISIT_K = 3
_PRIOR = 1e-4  # pull towards the global similarity, relative to edge weight


@dataclass
class LocalGraph:
    frame_ids: np.ndarray  # (N,) layer frame ids of the nodes
    centers: np.ndarray  # (N, 2) frame centres in local map units
    pairs: np.ndarray  # (E, 2) node indices
    footprint: float  # typical frame size, local units

    @property
    def n(self) -> int:
        return len(self.centers)


def build_local_graph(
    frame_ids, centers, footprint: float, overlap: float = 0.5, revisit_gap: int = 10
) -> LocalGraph:
    """Temporal chain (consecutive frame ids) + revisit pairs: up to
    ``_REVISIT_K`` nearest frames closer than ``overlap * footprint`` that are
    at least ``revisit_gap`` frames apart (they image the same ground on
    another pass; near-in-time neighbours are already tied by the chain)."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    centers = _as_points(centers)
    if len(frame_ids) != len(centers) or len(centers) < 2:
        raise ValueError("need >= 2 frames")
    if not np.isfinite(footprint) or footprint <= 0:
        raise ValueError("footprint must be positive")
    order = np.argsort(frame_ids, kind="stable")
    frame_ids, centers = frame_ids[order], centers[order]
    n = len(centers)
    chain = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    k = min(n, 4 * _REVISIT_K)
    d, idx = cKDTree(centers).query(centers, k=k, distance_upper_bound=overlap * footprint)
    d, idx = d.reshape(n, k), idx.reshape(n, k)
    src = np.repeat(np.arange(n), k)
    dst, dd = idx.ravel(), d.ravel()
    ok = np.isfinite(dd) & (dst < n)
    src, dst = src[ok], dst[ok]
    ok = np.abs(src - dst) >= revisit_gap
    src, dst = src[ok], dst[ok]
    # keep the first _REVISIT_K revisit neighbours per node (query is sorted by distance)
    keep = np.zeros(len(src), bool)
    if len(src):
        starts = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
        rank = np.arange(len(src)) - np.repeat(starts, np.diff(np.r_[starts, len(src)]))
        keep = rank < _REVISIT_K
    rev = np.sort(np.column_stack([src[keep], dst[keep]]), axis=1)
    pairs = np.unique(np.vstack([chain, rev]), axis=0) if len(rev) else chain
    return LocalGraph(frame_ids, centers, pairs.astype(np.int64), float(footprint))


def _idw(graph: LocalGraph, tree: cKDTree, pts: np.ndarray):
    """Nodes and weights that define the correction at arbitrary local points."""
    k = min(_IDW_K, graph.n)
    d, idx = tree.query(pts, k=k)
    d, idx = d.reshape(len(pts), k), idx.reshape(len(pts), k)
    w = 1.0 / (d + 0.05 * graph.footprint) ** 2
    return idx, w / w.sum(axis=1, keepdims=True)


@dataclass
class GraphGeorefFit:
    matrix: np.ndarray  # 2x3 global similarity (gauge reference / fallback)
    frame_ids: np.ndarray
    frame_matrices: np.ndarray  # (N, 2, 3) local -> projected metric per frame
    residuals: np.ndarray  # per control point, projected units
    loo_residuals: np.ndarray | None
    global_loo_residuals: np.ndarray | None  # same check for the global similarity
    stiffness: float
    max_correction: float  # largest frame-centre shift vs global similarity, proj. units
    scale_range: tuple[float, float]  # min/max frame scale relative to global
    kind: str = GRAPH
    warnings: list[str] = field(default_factory=list)
    _graph: LocalGraph | None = None
    _tree: cKDTree | None = None
    _a: np.ndarray | None = None  # complex, per node (centred coordinates)
    _t: np.ndarray | None = None
    _mirror: bool = False
    _cz: complex = 0j
    _cw: complex = 0j

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if len(self.residuals) else 0.0

    @property
    def scale(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:, :2]))))

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))

    @property
    def reflected(self) -> bool:
        return bool(np.linalg.det(self.matrix[:, :2]) < 0)

    def map_points(self, pts) -> np.ndarray:
        """Local map points -> projected metric through the correction field."""
        pts = _as_points(pts)
        z = _to_complex(pts, self._mirror) - self._cz
        idx, w = _idw(self._graph, self._tree, pts)
        a = (self._a[idx] * w).sum(axis=1)
        t = (self._t[idx] * w).sum(axis=1)
        out = a * z + t + self._cw
        return np.column_stack([out.real, out.imag])


def _to_complex(pts: np.ndarray, mirror: bool) -> np.ndarray:
    return pts[:, 0] + (-1j if mirror else 1j) * pts[:, 1]


def _rows(node_i, coef_a_i, coef_t_i, node_j, coef_a_j, coef_t_j, b, weight, n):
    """Sparse rows for complex equations  ca_i*a_i + ct_i*t_i [+ ca_j*a_j + ct_j*t_j] = b.

    Each equation gives a real and an imaginary row over the node unknowns
    [ar, ai, tx, ty]. All arguments are arrays of the same length m (j-terms
    may be None)."""
    m = len(node_i)
    terms = [(node_i, coef_a_i, coef_t_i)]
    if node_j is not None:
        terms.append((node_j, coef_a_j, coef_t_j))
    rr, cc, vv = [], [], []
    re_row = 2 * np.arange(m)
    im_row = re_row + 1
    for node, ca, ct in terms:
        base = 4 * node
        ca = np.asarray(ca, complex) * weight
        ct = np.asarray(ct, float) * weight
        rr += [re_row, re_row, re_row, im_row, im_row, im_row]
        cc += [base, base + 1, base + 2, base, base + 1, base + 3]
        vv += [ca.real, -ca.imag, ct, ca.imag, ca.real, ct]
    A = coo_matrix(
        (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(2 * m, 4 * n)
    ).tocsr()
    b = np.asarray(b, complex) * weight
    rhs = np.empty(2 * m)
    rhs[0::2], rhs[1::2] = b.real, b.imag
    return A, rhs


class _System:
    """Normal equations split into the fixed graph part (scaled by stiffness²)
    and the small control-point part, so LOO / stiffness search re-solve
    cheaply."""

    def __init__(self, graph, z_nodes, a0, t0):
        n = graph.n
        i, j = graph.pairs[:, 0], graph.pairs[:, 1]
        m = 0.5 * (z_nodes[i] + z_nodes[j])
        L = np.full(len(i), complex(graph.footprint))
        one = np.ones(len(i))
        A1, b1 = _rows(i, m, one, j, -m, -one, np.zeros(len(i)), 1.0, n)
        A2, b2 = _rows(i, L, 0 * one, j, -L, 0 * one, np.zeros(len(i)), 1.0, n)
        nodes = np.arange(n)
        Ln = np.full(n, complex(graph.footprint))
        A3, b3 = _rows(nodes, z_nodes, np.ones(n), None, None, None, a0 * z_nodes + t0, _PRIOR, n)
        A4, b4 = _rows(nodes, Ln, np.zeros(n), None, None, None, a0 * Ln, _PRIOR, n)
        self.n = n
        self.scale2 = float(max(len(graph.pairs), 1))  # edge weight = lam*sqrt(E)
        self.H = (A1.T @ A1 + A2.T @ A2 + A3.T @ A3 + A4.T @ A4).tocsc()
        self.g = A1.T @ b1 + A2.T @ b2 + A3.T @ b3 + A4.T @ b4

    def solve(self, lam, gz, gw, cache):
        nodes, ca, ct, b = [], [], [], []
        for k in range(len(gz)):
            idx, w = cache[k]
            nodes.append(idx)
            ca.append(w * gz[k])
            ct.append(w)
        n = self.n
        # each control point: one complex equation over its IDW nodes
        rr, cc, vv, rhs = [], [], [], np.empty(2 * len(gz))
        for k in range(len(gz)):
            base = 4 * nodes[k]
            a_, t_ = ca[k], ct[k]
            rr += [np.full(len(base), 2 * k)] * 3 + [np.full(len(base), 2 * k + 1)] * 3
            cc += [base, base + 1, base + 2, base, base + 1, base + 3]
            vv += [a_.real, -a_.imag, t_, a_.imag, a_.real, t_]
            rhs[2 * k], rhs[2 * k + 1] = gw[k].real, gw[k].imag
        G = coo_matrix(
            (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
            shape=(2 * len(gz), 4 * n),
        ).tocsr()
        s = lam * lam * self.scale2
        u = spsolve((s * self.H + G.T @ G).tocsc(), s * self.g + G.T @ rhs).reshape(n, 4)
        return u[:, 0] + 1j * u[:, 1], u[:, 2] + 1j * u[:, 3]


def fit_graph_georef(
    graph: LocalGraph,
    src,
    dst,
    rotation_sign: int = 1,
    stiffness: float | None = None,
) -> GraphGeorefFit:
    """Fit the correction field. ``stiffness=None`` picks it by leave-one-out
    when there are >= 4 control points, else uses ``DEFAULT_STIFFNESS``."""
    src, dst = _as_points(src), _as_points(dst)
    if len(src) != len(dst):
        raise ValueError("src/dst size mismatch")
    if len(src) < MIN_GRAPH_POINTS:
        raise ValueError(f"graph: need >= {MIN_GRAPH_POINTS} points, have {len(src)}")
    M0 = fit_similarity(src, dst, rotation_sign)
    mirror = bool(np.linalg.det(M0[:, :2]) < 0)
    P = M0[:, :2] @ (np.diag([1.0, -1.0]) if mirror else np.eye(2))  # proper similarity
    tree = cKDTree(graph.centers)

    cz = complex(_to_complex(graph.centers, mirror).mean())
    cw = complex(dst[:, 0].mean() + 1j * dst[:, 1].mean())
    z_nodes = _to_complex(graph.centers, mirror) - cz
    gz = _to_complex(src, mirror) - cz
    gw = dst[:, 0] + 1j * dst[:, 1] - cw
    a0 = complex(P[0, 0] + 1j * P[1, 0])
    t0 = complex(M0[0, 2] + 1j * M0[1, 2]) + a0 * cz - cw
    idx, w = _idw(graph, tree, src)
    cache = list(zip(idx, w, strict=True))
    system = _System(graph, z_nodes, a0, t0)

    def predict(a, t, k):
        i, wk = cache[k]
        return (a[i] * wk).sum() * gz[k] + (t[i] * wk).sum()

    def loo(lam):
        out = np.empty(len(src))
        for k in range(len(src)):
            keep = [j for j in range(len(src)) if j != k]
            a, t = system.solve(lam, gz[keep], gw[keep], [cache[j] for j in keep])
            out[k] = abs(predict(a, t, k) - gw[k])
        return out

    global_loo = None
    if len(src) > 2:
        global_loo = np.empty(len(src))
        for k in range(len(src)):
            keep = np.arange(len(src)) != k
            Mk = fit_similarity(src[keep], dst[keep], rotation_sign)
            global_loo[k] = float(np.linalg.norm(Mk[:, :2] @ src[k] + Mk[:, 2] - dst[k]))

    warnings: list[str] = []
    loo_res = None
    if stiffness is None:
        if len(src) >= 4:
            best = None
            for lam in STIFFNESS_GRID:
                res = loo(lam)
                score = float(np.sqrt(np.mean(res**2)))
                if best is None or score < best[0]:
                    best = (score, lam, res)
            _, stiffness, loo_res = best
        else:
            stiffness = DEFAULT_STIFFNESS
            warnings.append(
                "3 точки: нежорстка модель проходить через усі точки й не має перевірки — "
                "додайте 4-ту, щоб помилку в координатах було видно"
            )
    if loo_res is None and len(src) >= 4:
        loo_res = loo(stiffness)

    a, t = system.solve(stiffness, gz, gw, cache)
    residuals = np.array([abs(predict(a, t, k) - gw[k]) for k in range(len(src))])

    # Per-frame 2x3 in uncentred coordinates: w = a*(z - cz) + t + cw.
    flip = np.diag([1.0, -1.0]) if mirror else np.eye(2)
    mats = np.empty((graph.n, 2, 3))
    for i in range(graph.n):
        Pi = np.array([[a[i].real, -a[i].imag], [a[i].imag, a[i].real]]) @ flip
        ti = t[i] + cw - a[i] * cz
        mats[i, :, :2] = Pi
        mats[i, :, 2] = [ti.real, ti.imag]

    glob = graph.centers @ M0[:, :2].T + M0[:, 2]
    field_c = np.einsum("nij,nj->ni", mats[:, :, :2], graph.centers) + mats[:, :, 2]
    max_corr = float(np.max(np.linalg.norm(field_c - glob, axis=1)))
    rel = np.abs(a) / abs(a0)

    # Components the control points do not reach keep the global similarity.
    adj = coo_matrix(
        (np.ones(len(graph.pairs)), (graph.pairs[:, 0], graph.pairs[:, 1])),
        shape=(graph.n, graph.n),
    )
    _, labels = connected_components(adj, directed=False)
    gcp_comp = {}
    for k in range(len(src)):
        c = labels[int(cache[k][0][np.argmax(cache[k][1])])]
        gcp_comp[c] = gcp_comp.get(c, 0) + 1
    weak = sum(int(np.sum(labels == c)) for c in set(labels) if gcp_comp.get(c, 0) < 2)
    if weak:
        warnings.append(
            f"{weak} кадрів у частинах карти, не зв’язаних з ≥2 точками — для них "
            "лишається загальна прив’язка (масштаб/поворот не виправлено)"
        )

    return GraphGeorefFit(
        matrix=M0,
        frame_ids=graph.frame_ids,
        frame_matrices=mats,
        residuals=residuals,
        loo_residuals=loo_res,
        global_loo_residuals=global_loo,
        stiffness=float(stiffness),
        max_correction=max_corr,
        scale_range=(float(rel.min()), float(rel.max())),
        warnings=warnings,
        _graph=graph,
        _tree=tree,
        _a=a,
        _t=t,
        _mirror=mirror,
        _cz=cz,
        _cw=cw,
    )


def graph_from_layer(local_affines, valid, width: int, height: int) -> LocalGraph:
    """LocalGraph from a layer's local frame affines (pixels -> local units)."""
    A = np.asarray(local_affines, dtype=np.float64)
    ids = np.flatnonzero(np.asarray(valid, dtype=bool))
    Av = A[ids, :2, :]
    centers = Av @ np.array([width / 2.0, height / 2.0, 1.0])
    ok = np.isfinite(centers).all(axis=1)
    ids, Av, centers = ids[ok], Av[ok], centers[ok]
    dets = np.abs(np.linalg.det(Av[:, :, :2]))
    footprint = float(np.median(np.sqrt(dets))) * float(np.sqrt(width * height))
    return build_local_graph(ids, centers, footprint)
