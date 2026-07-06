"""
Тести діагностики оптимізатора (Етап 1 плану): пер-ребровий residual-звіт,
anchor stress, звіт пропагації. Усе read-only — нуль впливу на розв'язок.
"""

import numpy as np

from src.geometry.affine_utils import compose_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer


def _sim(tx, ty, scale, angle_deg):
    return compose_affine(tx, ty, scale, np.radians(angle_deg))


def _chain_with_bad_closure():
    """Добре зафіксований ланцюг 0..4 (2 якорі) + ОДНЕ слабке хибне spatial 1->3.

    Якорі на обох кінцях + temporal-ланцюг тримають правильну форму, тож слабке
    (мала вага) хибне закриття лишається з ВЕЛИКИМ резидуалом — його й має
    підсвітити діагностика.
    """
    opt = PoseGraphOptimizer()
    opt.fix_node(0, _sim(0.0, 0.0, 1.0, 0.0))
    opt.fix_node(4, _sim(400.0, 0.0, 1.0, 0.0))
    for i in (1, 2, 3):
        opt.add_node(i)
    step = _sim(100.0, 0.0, 1.0, 0.0)
    for i in range(4):
        opt.add_edge(i, i + 1, step, weight=1.0, edge_type="temporal", inliers=50, rmse=1.0)
    # хибне закриття 1->3: стверджує нульовий зсув (реальний ~200) — слабка вага
    opt.add_edge(1, 3, _sim(0.0, 0.0, 1.0, 0.0), weight=0.3,
                 edge_type="spatial", inliers=16, rmse=4.0)
    opt.initialize_from_bfs()
    return opt


class TestEdgeResiduals:
    def test_worst_edge_is_the_false_closure(self):
        opt = _chain_with_bad_closure()
        opt.optimize(max_iterations=60, tolerance=1e-10)
        res = opt.compute_edge_residuals()
        assert res.shape[0] == opt.num_edges
        assert np.all(~np.isnan(res))
        worst = opt.edges[int(np.argmax(res))]
        assert (worst.from_id, worst.to_id) == (1, 3), "хибне закриття має бути найгіршим"
        assert worst.edge_type == "spatial"

    def test_residual_stats_split_by_class(self):
        opt = _chain_with_bad_closure()
        opt.optimize(max_iterations=60, tolerance=1e-10)
        stats = opt.edge_residual_stats()
        assert stats["temporal"]["count"] == 4
        assert stats["spatial"]["count"] == 1
        assert stats["spatial"]["max"] > stats["temporal"]["p95"]


class TestAnchorStress:
    def test_conflicting_anchor_lights_up(self):
        """Якір, що суперечить ланцюгу → високий stress (>2× медіани)."""
        opt = PoseGraphOptimizer()
        opt.fix_node(0, _sim(0.0, 0.0, 1.0, 0.0))
        for i in (1, 2, 3):
            opt.add_node(i)
        step = _sim(100.0, 0.0, 1.0, 0.0)
        for i in range(3):
            opt.add_edge(i, i + 1, step, weight=1.0, edge_type="temporal", inliers=50, rmse=1.0)
        # додаткове коротке ребро, щоб було кілька «нормальних» резидуалів для медіани
        opt.add_edge(0, 2, _sim(200.0, 0.0, 1.0, 0.0), weight=1.0,
                     edge_type="temporal", inliers=50, rmse=1.0)
        # якір на кадрі 3 «кривий»: далеко від того, куди веде ланцюг (300,0)
        opt.fix_node(3, _sim(300.0, 250.0, 1.0, 0.0))
        opt.initialize_from_bfs()
        opt.optimize(max_iterations=60, tolerance=1e-10)

        stress = opt.compute_anchor_stress()
        assert set(stress.keys()) == {0, 3}
        assert stress[3] > 2.0, f"конфліктний якір має світитись, stress={stress.get(3)}"


class TestDiagnosticsReport:
    def test_report_structure_and_text(self):
        opt = _chain_with_bad_closure()
        opt.optimize(max_iterations=60, tolerance=1e-10)
        rep = opt.diagnostics_report(top_n=3)
        assert rep["num_edges"] == 5
        assert rep["num_temporal"] == 4 and rep["num_spatial"] == 1
        assert len(rep["worst_edges"]) == 3
        assert rep["worst_edges"][0]["residual"] >= rep["worst_edges"][1]["residual"]
        text = opt.format_diagnostics()
        assert "temporal" in text and "spatial" in text


class TestDiagnosticsAreReadOnly:
    def test_diagnostics_do_not_change_solution(self):
        opt = _chain_with_bad_closure()
        r1 = opt.optimize(max_iterations=60, tolerance=1e-10)
        before = {k: v.copy() for k, v in r1.items()}
        _ = opt.diagnostics_report()
        _ = opt.format_diagnostics()
        _ = opt.compute_anchor_stress()
        after = opt._export_results()
        for k in before:
            np.testing.assert_allclose(before[k], after[k], atol=1e-12)


class TestGeoJSONHasResidual:
    def test_edge_properties_include_residual(self):
        opt = _chain_with_bad_closure()
        opt.optimize(max_iterations=60, tolerance=1e-10)

        class MockConv:
            is_initialized = True
            def metric_to_gps(self, x, y):
                return (y * 1e-5, x * 1e-5)

        gj = opt.export_graph_geojson(MockConv(), 1920, 1080)
        edges = [f for f in gj["features"] if f["geometry"]["type"] == "LineString"]
        assert edges, "немає ребер у GeoJSON"
        assert all("residual" in f["properties"] for f in edges)
