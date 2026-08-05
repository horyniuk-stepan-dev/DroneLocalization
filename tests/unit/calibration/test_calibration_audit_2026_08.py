"""Regression tests for the calibration-pipeline audit of 2026-08-01.

Each test locks one fix and would FAIL against the pre-audit code:

* soft anchors were invisible to the prune disconnect-guard and to the
  diagnostics report, because both read ``_fixed_nodes`` (empty when
  ``soft_anchors=True``) instead of ``anchor_states()``;
* ``frame_disagreement`` was ``std`` of neighbours' ``tx`` — a quantity that is
  large even when every edge in the graph agrees perfectly.

Pure numpy/scipy except where marked; runs in any environment.
"""

import numpy as np
import pytest

from src.geometry.pose_graph.optimizer import PoseGraphOptimizer

# pixel->metric anchor matrix: det < 0 (pixel Y down, map Y up).
ANCHOR_M = np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]], dtype=np.float64)


def rel(dx_px: float) -> np.ndarray:
    """Relative frame-to-frame transform: near-identity, det > 0."""
    return np.array([[1.0, 0.0, dx_px], [0.0, 1.0, 0.0]], dtype=np.float64)


def _consistent_graph(soft: bool, n: int = 8) -> PoseGraphOptimizer:
    """Chain of equal steps plus loop closures that are exact multiples of it.

    Every edge agrees with every other, so the true disagreement is zero.
    """
    o = PoseGraphOptimizer(1920, 1080)
    for i in range(n):
        o.add_node(i)
    if soft:
        o.add_anchor(0, ANCHOR_M, sigma_m=5.0)
        o.add_anchor(n - 1, ANCHOR_M.copy(), sigma_m=5.0)
    else:
        o.fix_node(0, ANCHOR_M)
    for i in range(n - 1):
        o.add_edge(i, i + 1, rel(300), 1.0, "temporal", inliers=200, rmse=0.8)
    o.add_edge(1, 5, rel(1200), 2.0, "spatial", inliers=120, rmse=1.2)
    o.add_edge(2, 6, rel(1200), 2.0, "spatial", inliers=120, rmse=1.2)
    o.initialize_from_bfs()
    o.optimize(max_iterations=30)
    return o


class TestSoftAnchorVisibility:
    """soft_anchors must not blind the guards that protect the graph."""

    def test_reachability_seeds_from_soft_anchors(self):
        o = _consistent_graph(soft=True)
        # Pre-fix: seeds came from _fixed_nodes, which is empty here.
        assert o._fixed_nodes == {}
        assert set(o.anchor_states()) == {0, 7}
        assert o._anchor_reachable(o.edges) == set(range(8))

    def test_prune_guard_detects_orphaning_with_soft_anchors(self):
        """Dropping the only link to a node must change reachability.

        Pre-fix both sides were the empty set, so the comparison in
        ``_prune_bad_spatial_edges`` was tautologically true and the guard
        allowed a prune that cut a segment loose from every anchor.
        """
        o = PoseGraphOptimizer(1920, 1080)
        for i in range(3):
            o.add_node(i)
        o.add_anchor(0, ANCHOR_M, sigma_m=5.0)
        o.add_edge(0, 1, rel(300), 1.0, "temporal")
        o.add_edge(1, 2, rel(300), 1.0, "spatial")  # sole link to node 2

        full = o._anchor_reachable(o.edges)
        without_spatial = o._anchor_reachable([e for e in o.edges if e.edge_type != "spatial"])
        assert full == {0, 1, 2}
        assert without_spatial == {0, 1}
        assert full != without_spatial

    def test_diagnostics_report_counts_soft_anchors(self):
        o = _consistent_graph(soft=True)
        report = o.diagnostics_report()
        assert report["num_anchors"] == 2
        assert set(report["anchor_stress"]) == {0, 7}

    def test_geojson_marks_soft_anchors(self):
        class Conv:
            def metric_to_gps(self, x, y):
                return 48.0 + y * 1e-7, 30.0 + x * 1e-7

        o = _consistent_graph(soft=True)
        gj = o.export_graph_geojson(Conv(), 1920, 1080)
        anchors = [
            f
            for f in gj["features"]
            if f["geometry"]["type"] == "Point" and f["properties"]["type"] == "anchor"
        ]
        assert len(anchors) == 2


def test_isotropy_weight_is_configurable_and_default_preserved():
    assert PoseGraphOptimizer(1920, 1080).isotropy_weight == 200.0
    assert PoseGraphOptimizer(1920, 1080, isotropy_weight=50.0).isotropy_weight == 50.0


class TestGroundScaleThresholds:
    """Metric constants must be readable as ground metres when asked."""

    def test_cos_lat_from_anchor_gps_then_reference(self):
        pytest.importorskip("faiss")
        pytest.importorskip("h5py")
        from src.workers.propagation_pipeline import PropagationPipeline

        class Conv:
            reference_gps = (48.0, 30.0)

            def ground_scale_factor(self, lat=None):
                import math

                return math.cos(math.radians(lat if lat is not None else 48.0))

        class Cal:
            converter = Conv()
            anchors = []

        class DB:
            metadata = {"frame_width": 1920, "frame_height": 1080}

            def get_num_frames(self):
                return 10

        class Anchor:
            def __init__(self, lat=None):
                self.points_gps = [[lat, 30.0]] if lat is not None else []

        p = PropagationPipeline(DB(), Cal(), None, config={})
        assert p._ground_scale_factor([Anchor(50.0)]) == pytest.approx(np.cos(np.radians(50)))
        # No anchor GPS -> converter reference latitude.
        assert p._ground_scale_factor([Anchor()]) == pytest.approx(np.cos(np.radians(48)))

    def test_missing_converter_is_a_noop(self):
        pytest.importorskip("faiss")
        pytest.importorskip("h5py")
        import types

        from src.workers.propagation_pipeline import PropagationPipeline

        class DB:
            metadata = {}

            def get_num_frames(self):
                return 1

        p = PropagationPipeline(DB(), types.SimpleNamespace(converter=None), None, config={})
        assert p._ground_scale_factor([]) == 1.0


class TestTrueDisagreement:
    """The metric feeds ResultBuilder.compute_confidence, so its shape matters."""

    @staticmethod
    def _run(tmp_path, flag):
        pytest.importorskip("faiss")
        h5py = pytest.importorskip("h5py")
        import threading

        from src.workers.propagation_pipeline import PropagationPipeline

        db_path = tmp_path / "db.h5"
        with h5py.File(db_path, "w"):
            pass

        class Conv:
            is_initialized = True

            def metric_to_gps(self, x, y):
                return 48.0, 30.0

            def export_metadata(self):
                return {"mode": "WEB_MERCATOR", "reference_gps": None}

        class Cal:
            converter = Conv()

        class DB:
            metadata = {"frame_width": 1920, "frame_height": 1080}
            lock = threading.Lock()

            def __init__(self):
                self.db_path = str(db_path)

            def get_num_frames(self):
                return 8

            def close(self):
                pass

            def _load_hot_data(self):
                pass

        class Anchor:
            frame_id = 0

            def to_dict(self):
                return {"frame_id": 0}

        o = _consistent_graph(soft=False)
        p = PropagationPipeline(
            DB(),
            Cal(),
            None,
            config={"graph_optimization": {"true_disagreement": flag, "export_geojson": False}},
        )
        p._save_to_hdf5(o._export_results(), [Anchor()], o)
        with h5py.File(db_path, "r") as f:
            return f["calibration"]["frame_disagreement"][:], o

    def test_zero_on_a_graph_where_every_edge_agrees(self, tmp_path):
        values, _ = self._run(tmp_path, True)
        assert float(np.max(values)) < 1e-6

    def test_legacy_form_is_unchanged_by_default(self, tmp_path):
        """Default stays bit-for-bit the historical metric (no silent behaviour change).

        It also documents the defect: on a perfectly consistent graph the legacy
        metric still reports tens of metres, saturating ``disagreement_norm_m``.
        """
        values, o = self._run(tmp_path, False)
        multi = [i for i in range(8) if sum(1 for e in o.edges if i in (e.from_id, e.to_id)) >= 2]
        assert float(np.mean([values[i] for i in multi])) > 10.0
