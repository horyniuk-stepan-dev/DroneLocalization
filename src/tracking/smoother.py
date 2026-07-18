"""Sliding-window back-end smoother for the live trajectory (RESEARCH 3.1).

Architecture (docs/RESEARCH_INTEGRATION_PLAN.md §3.1, design record in
.agents/SMOOTHER_DESIGN.md):

- Front-end (TrajectoryFilter KF + OutlierDetector) stays unchanged and keeps
  producing the per-frame output.
- This back-end maintains a sliding window of keyframe nodes (2D metric
  positions). Every H-based keyframe fix enters the window — including fixes
  the front-end rejected as outliers. Robust (Huber) IRLS weighting arbitrates
  instead of the front-end's binary accept/reject: a wrong fix gets a
  suppressed weight, a run of consistent "outliers" (real relocation, Z-score
  false positives during maneuvers) regains influence naturally.
- Inter-node edges are relative odometry derived from the optical-flow track
  (metric OF positions are anchored at the last accepted keyframe's H, so the
  difference of two OF samples is a fix-independent relative measurement).
- The solved window periodically corrects the KF via a position shift
  (TrajectoryFilter.shift): the correction is the difference between the
  smoothed and the KF estimate at the newest node.

Deliberate deviations from the plan text (rationale in the design note):

- Dedicated 2D linear solver instead of reusing the 5-DoF PoseGraphOptimizer.
  The live problem is linear in positions; Huber-IRLS over two per-axis SPD
  systems solves a 100-node window in well under a millisecond with plain
  numpy. The 5-DoF optimizer's residual contract (5E+N+5A) is calibration
  infrastructure with no per-anchor robust loss; wiring live usage into it
  would couple two hot paths for no numerical benefit.
- Synchronous solve on the keyframe cadence instead of an async thread. The
  async design existed to hide scipy 5-DoF solve latency; a sub-millisecond
  solve does not need hiding, and staying on the worker thread removes every
  lock/queue hazard around the shared KF state.

The problem solved per window (positions p_i, i = 0..N-1):

    sum_i  w_i * ||p_i - z_i||^2          (fixes; w_i Huber-reweighted, IRLS)
  + sum_e  w_e * ||p_j - p_i - d_e||^2    (odometry edges; quadratic)
  + entry prior on the oldest node        (carries marginalized info forward)

x and y decouple into two SPD N x N systems sharing the same matrix.
"""

from dataclasses import dataclass

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# OF samples older than this (relative to the new node's time) are considered
# stale and produce no odometry edge. 0.5 s covers any realistic keyframe
# cadence (worker default: 5 frames at 30 fps = 0.17 s).
_OF_STALE_SEC = 0.5

# Corrections are only meaningful once the window carries more information
# than the raw fixes themselves. Below this node count add_fix() returns None.
_MIN_NODES_FOR_CORRECTION = 5

# Floors mirroring front-end conventions (Localizer clips confidence at 0.25
# for the KF noise scale; OF quality below 0.2 is de facto garbage).
_CONF_FLOOR = 0.25
_QUALITY_FLOOR = 0.2


@dataclass
class _Node:
    uid: int
    t: float  # window-internal time of the keyframe [s]
    z: np.ndarray  # raw metric fix (2,)
    sigma: float  # fix sigma [m]
    accepted: bool  # front-end verdict
    kf_xy: np.ndarray | None  # front-end KF output at this node (accepted only)
    # OF-track position at this node's time, recorded at ingestion (used as
    # the edge start when this node is rejected — its own fix is untrusted).
    of_boundary: np.ndarray | None = None
    prior: tuple[np.ndarray, float] | None = None  # entry prior (xy, sigma)


@dataclass
class _Edge:
    a: int  # uid of the older node
    b: int  # uid of the newer node
    delta: np.ndarray  # measured p_b - p_a (2,)
    weight: float  # 1 / sigma^2


@dataclass
class _OfTrack:
    """OF samples of the current inter-keyframe interval (metric coords,
    anchored at the last accepted keyframe's homography). Survives rejected
    keyframes — the anchor only changes on acceptance."""

    prev: tuple[float, np.ndarray] | None = None  # (t, xy) — one before last
    last: tuple[float, np.ndarray] | None = None  # (t, xy) — latest sample
    q_min: float = 1.0
    n: int = 0

    def push(self, t: float, xy: np.ndarray, quality: float) -> None:
        self.prev = self.last
        self.last = (t, xy)
        self.q_min = min(self.q_min, quality)
        self.n += 1

    def clear(self) -> None:
        self.prev = None
        self.last = None
        self.q_min = 1.0
        self.n = 0

    def value_at(self, t: float) -> np.ndarray | None:
        """OF-track position extrapolated forward to time t (linear from the
        last two samples; falls back to the last sample; None if the track is
        empty or its newest sample is stale relative to t)."""
        if self.last is None:
            return None
        t_last, xy_last = self.last
        if t - t_last > _OF_STALE_SEC:
            return None
        if self.prev is not None:
            t_prev, xy_prev = self.prev
            dt = t_last - t_prev
            if dt > 1e-6:
                v = (xy_last - xy_prev) / dt
                return xy_last + v * (t - t_last)
        return xy_last.copy()


class SlidingWindowSmoother:
    """Robust sliding-window smoother over live keyframe fixes + OF odometry.

    Numpy-only, synchronous, deterministic. All inputs and outputs are in the
    metric coordinate frame of the active calibration source; the window is
    reset whenever the source changes (metric frames of different sources are
    not commensurable).
    """

    def __init__(
        self,
        window: int = 60,
        huber_k: float = 1.2,
        fix_sigma_base_m: float = 5.0,
        odom_sigma_base_m: float = 3.0,
        max_correction_m: float = 50.0,
        entry_prior_sigma_m: float = 15.0,
        irls_iterations: int = 4,
    ) -> None:
        self.window = max(int(window), 2)
        self.huber_k = float(huber_k)
        self.fix_sigma_base_m = float(fix_sigma_base_m)
        self.odom_sigma_base_m = float(odom_sigma_base_m)
        self.max_correction_m = float(max_correction_m)
        self.entry_prior_sigma_m = float(entry_prior_sigma_m)
        self.irls_iterations = max(int(irls_iterations), 1)

        self._nodes: list[_Node] = []
        self._edges: list[_Edge] = []
        self._of = _OfTrack()
        self._uid_seq = 0
        self._source_id: str | None = None
        # Two clocks accumulating the worker's video-time deltas:
        # _t_last_node — previous keyframe (accepted or rejected; KF dt
        # semantics), _t_last_success — last successful localization event
        # (accepted keyframe or OF sample; OF dt semantics).
        self._t_last_success = 0.0
        self._t_last_node = 0.0
        self._last_solution: dict[int, np.ndarray] = {}
        self._last_fix_weights: dict[int, float] = {}

        logger.info(
            f"SlidingWindowSmoother: window={self.window}, huber_k={self.huber_k}, "
            f"fix_sigma={self.fix_sigma_base_m}m, odom_sigma={self.odom_sigma_base_m}m"
        )

    # ── ingestion ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._of.clear()
        self._source_id = None
        self._t_last_success = 0.0
        self._t_last_node = 0.0
        self._last_solution.clear()
        self._last_fix_weights.clear()

    def note_of(self, metric_xy, dt: float, quality: float | None = None) -> None:
        """Register a successful optical-flow localization (raw metric point).

        dt follows the worker semantics: time since the last successful
        localization (previous OF frame or the anchoring keyframe).
        """
        if not self._nodes:
            return  # OF before any keyframe fix has no anchor in the window
        t = self._t_last_success + max(float(dt), 1e-3)
        self._t_last_success = t
        if quality is None:
            q = 0.7  # mirrors the front-end's default OF confidence
        else:
            q = max(float(quality), _QUALITY_FLOOR)
        self._of.push(t, np.asarray(metric_xy, dtype=np.float64).reshape(2), q)

    def add_fix(
        self,
        metric_xy,
        dt: float,
        confidence: float,
        source_id: str | None = None,
        accepted: bool = True,
        kf_xy=None,
    ) -> np.ndarray | None:
        """Ingest a keyframe fix, solve the window, return a KF correction.

        dt — time since the previous keyframe (worker semantics, rejected
        keyframes included). Returns the (dx, dy) shift to apply to the
        front-end KF, or None (window too small / node not accepted / solve
        skipped).
        """
        if source_id != self._source_id:
            if self._nodes:
                logger.info(
                    f"Smoother window reset: source change "
                    f"{self._source_id!r} -> {source_id!r}"
                )
            self.reset()
            self._source_id = source_id

        t_node = self._t_last_node + max(float(dt), 1e-3)
        z = np.asarray(metric_xy, dtype=np.float64).reshape(2)
        sigma = self.fix_sigma_base_m / max(float(confidence), _CONF_FLOOR)

        # OF-track position at this node's time — recorded NOW, while the
        # track state is contemporary (extrapolating backward later from newer
        # samples would be invalid).
        of_boundary = self._of.value_at(t_node)

        node = _Node(
            uid=self._uid_seq,
            t=t_node,
            z=z,
            sigma=sigma,
            accepted=bool(accepted),
            kf_xy=None
            if kf_xy is None
            else np.asarray(kf_xy, dtype=np.float64).reshape(2),
            of_boundary=of_boundary,
        )
        self._uid_seq += 1

        # Odometry edge from the previous node: OF-track displacement between
        # the two node times. Both boundary samples are anchored at the same
        # accepted keyframe's homography, so their difference is a
        # fix-independent relative measurement. An accepted previous node IS
        # the track's origin — its raw fix is the boundary sample.
        if self._nodes and of_boundary is not None:
            prev = self._nodes[-1]
            start = prev.z if prev.accepted else prev.of_boundary
            if start is not None:
                w = 1.0 / (self.odom_sigma_base_m / self._of.q_min) ** 2
                self._edges.append(
                    _Edge(a=prev.uid, b=node.uid, delta=of_boundary - start, weight=w)
                )

        self._nodes.append(node)
        self._t_last_node = t_node
        if accepted:
            self._t_last_success = t_node
            self._of.clear()  # the OF track re-anchors at this keyframe

        self._slide()
        solution = self.solve()

        if solution is None or not accepted or node.kf_xy is None:
            return None
        if len(self._nodes) < _MIN_NODES_FOR_CORRECTION:
            return None

        corr = solution[-1] - node.kf_xy
        norm = float(np.linalg.norm(corr))
        if norm > self.max_correction_m:
            logger.warning(
                f"Smoother correction clamped: {norm:.1f} m -> {self.max_correction_m} m"
            )
            corr = corr * (self.max_correction_m / norm)
        if float(np.linalg.norm(corr)) < 1e-9:
            return None
        return corr

    # ── window maintenance ───────────────────────────────────────────────────

    def _slide(self) -> None:
        while len(self._nodes) > self.window:
            dropped = self._nodes.pop(0)
            self._edges = [
                e for e in self._edges if e.a != dropped.uid and e.b != dropped.uid
            ]
            # Entry prior: the new head keeps its last smoothed estimate as a
            # weak unary factor — cheap stand-in for proper marginalization,
            # prevents the window head from floating when old fixes leave.
            head = self._nodes[0]
            smoothed = self._last_solution.get(head.uid)
            if smoothed is not None:
                head.prior = (smoothed.copy(), self.entry_prior_sigma_m)

    # ── solver ───────────────────────────────────────────────────────────────

    def solve(self) -> np.ndarray | None:
        """Huber-IRLS solve of the current window. Returns (N, 2) smoothed
        positions in node order, or None if the window is empty/degenerate."""
        n = len(self._nodes)
        if n == 0:
            return None

        uid_to_idx = {node.uid: i for i, node in enumerate(self._nodes)}
        z = np.stack([node.z for node in self._nodes])  # (n, 2)
        sig = np.array([node.sigma for node in self._nodes])  # (n,)
        w_fix_base = 1.0 / np.square(sig)

        e_a = np.array([uid_to_idx[e.a] for e in self._edges], dtype=np.int64)
        e_b = np.array([uid_to_idx[e.b] for e in self._edges], dtype=np.int64)
        e_d = (
            np.stack([e.delta for e in self._edges])
            if self._edges
            else np.zeros((0, 2))
        )
        e_w = np.array([e.weight for e in self._edges], dtype=np.float64)

        diag = np.arange(n)
        hw = np.ones(n, dtype=np.float64)  # Huber weights on fixes
        p = z.copy()

        for _ in range(self.irls_iterations):
            w_fix = w_fix_base * hw
            A = np.zeros((n, n), dtype=np.float64)
            b = np.zeros((n, 2), dtype=np.float64)

            A[diag, diag] += w_fix
            b += w_fix[:, None] * z

            for i, node in enumerate(self._nodes):
                if node.prior is not None:
                    pw = 1.0 / node.prior[1] ** 2
                    A[i, i] += pw
                    b[i] += pw * node.prior[0]

            if len(self._edges):
                np.add.at(A, (e_a, e_a), e_w)
                np.add.at(A, (e_b, e_b), e_w)
                np.add.at(A, (e_a, e_b), -e_w)
                np.add.at(A, (e_b, e_a), -e_w)
                np.subtract.at(b, e_a, e_w[:, None] * e_d)
                np.add.at(b, e_b, e_w[:, None] * e_d)

            try:
                p = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                logger.warning("Smoother solve failed (singular system) — skipping")
                return None

            u = np.linalg.norm(p - z, axis=1) / sig
            hw = np.where(u <= self.huber_k, 1.0, self.huber_k / np.maximum(u, 1e-12))

        self._last_solution = {
            node.uid: p[i].copy() for i, node in enumerate(self._nodes)
        }
        self._last_fix_weights = {
            node.uid: float(hw[i]) for i, node in enumerate(self._nodes)
        }
        return p

    # ── introspection (tests / telemetry) ────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def last_fix_weights(self) -> dict[int, float]:
        """uid -> final Huber weight of the fix (1.0 = fully trusted)."""
        return dict(self._last_fix_weights)
