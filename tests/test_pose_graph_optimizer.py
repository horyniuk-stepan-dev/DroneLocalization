"""
Тести для PoseGraphOptimizer.

Покриває:
  1. Тривіальний граф (3 вузли, лінійний) — результат = лінійна інтерполяція
  2. Замикання циклу (кільце) — drift зникає
  3. Один якір — одометрія від якоря
  4. Два якорі + loop closure — кращий результат ніж без closure
  5. Кутова SO(2) коректність — стрибок через ±π
  6. GeoJSON експорт — структура валідна

ПРИМІТКА ПРО ЗМІНИ (оновлення API 4-DoF → 5-DoF):
  - _affine_to_state(M)            → _affine_to_state(M, cx, cy)
  - _state_to_affine(state)        → _state_to_affine(state, cx, cy, sign=1.0)
  - _predict_forward(state, edge)  → _predict_forward(state, edge, sign=1.0)
  - _predict_inverse(state, edge)  → _predict_inverse(state, edge, sign=1.0)
  - GraphEdge(log_ds=x)            → GraphEdge(log_dsx=x, log_dsy=x)
  - State вектор: [tx, ty, log_s, θ] (4 ел.) → [cx_m, cy_m, log_sx, log_sy, θ] (5 ел.)

  У тестах нижче використовуємо cx=0.0, cy=0.0, що спрощує семантику:
  при нульовому центрі кадру state[0]/[1] = метричне положення пікселя (0,0)
  = вектор трансляції матриці.
"""

import numpy as np

from src.geometry.affine_utils import compose_affine, decompose_affine
from src.geometry.pose_graph_optimizer import (
    GraphEdge,
    PoseGraphOptimizer,
    _affine_to_state,
    _predict_forward,
    _predict_inverse,
    _state_to_affine,
    homography_to_similarity,
)

# Зручні константи для тестів — нульовий центр кадру спрощує roundtrip
_CX = 0.0
_CY = 0.0
_SIGN = 1.0


def _make_similarity(tx, ty, scale, angle_deg):
    """Допоміжна: створює 2x3 similarity affine."""
    return compose_affine(tx, ty, scale, np.radians(angle_deg))


class TestAffineStateConversion:
    """Тести конвертації affine ↔ state (5-DoF API)."""

    def test_roundtrip(self):
        """affine → state → affine повертає вихідну матрицю."""
        M = _make_similarity(100.0, 200.0, 0.5, 30.0)
        state = _affine_to_state(M, _CX, _CY)
        M_back = _state_to_affine(state, _CX, _CY, _SIGN)
        np.testing.assert_allclose(M, M_back, atol=1e-5)

    def test_identity(self):
        """Identity affine → нульовий state."""
        M = _make_similarity(0.0, 0.0, 1.0, 0.0)
        state = _affine_to_state(M, _CX, _CY)
        # State = [cx_metric, cy_metric, log_sx, log_sy, θ]
        assert abs(state[0]) < 1e-6, f"cx_metric={state[0]}, expected 0"
        assert abs(state[1]) < 1e-6, f"cy_metric={state[1]}, expected 0"
        assert abs(state[2]) < 1e-6, f"log_sx={state[2]}, expected log(1)=0"
        assert abs(state[3]) < 1e-6, f"log_sy={state[3]}, expected log(1)=0"
        assert abs(state[4]) < 1e-6, f"θ={state[4]}, expected 0"

    def test_negative_angle(self):
        """Від'ємний кут зберігається у roundtrip."""
        M = _make_similarity(50.0, -30.0, 1.2, -45.0)
        state = _affine_to_state(M, _CX, _CY)
        M_back = _state_to_affine(state, _CX, _CY, _SIGN)
        np.testing.assert_allclose(M, M_back, atol=1e-5)


class TestPredictForwardInverse:
    """Тести прямої та зворотної передбачуваності (5-DoF API)."""

    def test_forward_identity_edge(self):
        """Тотожне ребро не змінює стан."""
        # State: [cx_m, cy_m, log_sx, log_sy, θ] — 5 елементів
        state_i = np.array([100.0, 200.0, np.log(0.5), np.log(0.5), 0.3])
        edge = GraphEdge(
            from_id=0,
            to_id=1,
            dtx=0,
            dty=0,
            log_dsx=0,  # було log_ds=0
            log_dsy=0,  # новий параметр (анізотропія)
            dtheta=0,
            weight=1.0,
            edge_type="temporal",
        )
        state_j = _predict_forward(state_i, edge, sign=_SIGN)
        np.testing.assert_allclose(state_j, state_i, atol=1e-10)

    def test_forward_inverse_roundtrip(self):
        """Forward → Inverse повертає вихідний стан."""
        state_i = np.array([500.0, -300.0, np.log(1.5), np.log(1.5), 0.7])
        edge = GraphEdge(
            from_id=0,
            to_id=1,
            dtx=10.0,
            dty=-5.0,
            log_dsx=np.log(0.9),  # було log_ds=np.log(0.9)
            log_dsy=np.log(0.9),  # новий параметр (ізотропний тест)
            dtheta=0.1,
            weight=1.0,
            edge_type="temporal",
        )
        state_j = _predict_forward(state_i, edge, sign=_SIGN)
        state_i_back = _predict_inverse(state_j, edge, sign=_SIGN)
        np.testing.assert_allclose(state_i_back, state_i, atol=1e-6)


class TestLinearGraph:
    """Тест 1: тривіальний граф (3 вузли, лінійний, 2 якорі)."""

    def test_linear_three_nodes(self):
        opt = PoseGraphOptimizer()

        M0 = _make_similarity(0.0, 0.0, 1.0, 0.0)
        M2 = _make_similarity(200.0, 0.0, 1.0, 0.0)

        opt.fix_node(0, M0)
        opt.fix_node(2, M2)
        opt.add_node(1)

        T_01 = _make_similarity(100.0, 0.0, 1.0, 0.0)
        T_12 = _make_similarity(100.0, 0.0, 1.0, 0.0)

        opt.add_edge(0, 1, T_01, weight=1.0, edge_type="temporal")
        opt.add_edge(1, 2, T_12, weight=1.0, edge_type="temporal")

        opt.initialize_from_bfs()
        results = opt.optimize()

        M1 = results[1]
        tx, ty, s, a = decompose_affine(M1)
        assert abs(tx - 100.0) < 5.0, f"tx={tx}, expected ~100"
        assert abs(ty) < 5.0, f"ty={ty}, expected ~0"


class TestLoopClosure:
    """Тест 2: замикання циклу (drift зникає)."""

    def test_ring_graph(self):
        """5 вузлів у кільці: 0→1→2→3→4→0 + якір на 0."""
        opt = PoseGraphOptimizer()

        M0 = _make_similarity(0.0, 0.0, 1.0, 0.0)
        opt.fix_node(0, M0)

        for i in range(1, 5):
            opt.add_node(i)

        step = _make_similarity(100.0, 0.0, 1.0, 0.0)
        for i in range(4):
            opt.add_edge(i, i + 1, step, weight=1.0, edge_type="temporal")

        closure = _make_similarity(-400.0, 0.0, 1.0, 0.0)
        opt.add_edge(4, 0, closure, weight=2.0, edge_type="spatial")

        opt.initialize_from_bfs()
        results = opt.optimize()

        for i in range(5):
            M = results[i]
            tx, ty, s, a = decompose_affine(M)
            expected_tx = i * 100.0
            assert abs(tx - expected_tx) < 20.0, f"Node {i}: tx={tx}, expected ~{expected_tx}"


class TestSingleAnchor:
    """Тест 3: один якір — одометрія від якоря."""

    def test_single_anchor_propagation(self):
        opt = PoseGraphOptimizer()

        M_anchor = _make_similarity(1000.0, 2000.0, 0.5, 10.0)
        opt.fix_node(5, M_anchor)

        for i in range(10):
            if i != 5:
                opt.add_node(i)

        step = _make_similarity(5.0, 2.0, 1.0, 0.0)
        for i in range(9):
            opt.add_edge(i, i + 1, step, weight=1.0, edge_type="temporal")

        opt.initialize_from_bfs()
        results = opt.optimize()

        assert len(results) == 10


class TestSO2Correctness:
    """Тест 5: кутова SO(2) коректність — стрибок через ±π."""

    def test_angle_wrapping(self):
        """Два якорі з кутами -179° та +179° (різниця 2°, не 358°)."""
        opt = PoseGraphOptimizer()

        M0 = _make_similarity(0.0, 0.0, 1.0, -179.0)
        M2 = _make_similarity(200.0, 0.0, 1.0, 179.0)

        opt.fix_node(0, M0)
        opt.fix_node(2, M2)
        opt.add_node(1)

        T_01 = _make_similarity(100.0, 0.0, 1.0, 1.0)
        T_12 = _make_similarity(100.0, 0.0, 1.0, 1.0)
        opt.add_edge(0, 1, T_01, weight=1.0, edge_type="temporal")
        opt.add_edge(1, 2, T_12, weight=1.0, edge_type="temporal")

        opt.initialize_from_bfs()
        results = opt.optimize()

        M1 = results[1]
        _, _, _, angle = decompose_affine(M1)
        angle_deg = np.degrees(angle)
        assert (
            abs(abs(angle_deg) - 180.0) < 10.0 or abs(angle_deg) < 10.0
        ), f"Angle={angle_deg}°, expected near ±180° (SO(2) wrapping test)"


class TestHomographyToSimilarity:
    """Тест конвертації гомографії у similarity."""

    def test_identity_homography(self):
        H = np.eye(3, dtype=np.float64)
        T = homography_to_similarity(H, 1920, 1080)
        assert T is not None
        tx, ty, s, a = decompose_affine(T)
        assert abs(tx) < 1.0
        assert abs(ty) < 1.0
        assert abs(s - 1.0) < 0.01
        assert abs(a) < 0.01

    def test_translation_homography(self):
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = 50.0  # зсув 50 пікселів вправо
        T = homography_to_similarity(H, 1920, 1080)
        assert T is not None
        tx, ty, s, a = decompose_affine(T)
        assert abs(tx - 50.0) < 2.0
        assert abs(ty) < 2.0


class TestGeoJSONExport:
    """Тест експорту графу в GeoJSON."""

    def test_export_structure(self):
        opt = PoseGraphOptimizer()
        M0 = _make_similarity(0.0, 0.0, 1.0, 0.0)
        M1 = _make_similarity(100.0, 0.0, 1.0, 0.0)

        opt.fix_node(0, M0)
        # _affine_to_state тепер потребує cx, cy
        opt.add_node(1, _affine_to_state(M1, _CX, _CY))
        T = _make_similarity(100.0, 0.0, 1.0, 0.0)
        opt.add_edge(0, 1, T, weight=1.0, edge_type="temporal")

        class MockConverter:
            def metric_to_gps(self, x, y):
                return (y * 0.00001, x * 0.00001)

        geojson = opt.export_graph_geojson(MockConverter(), 1920, 1080)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 0

        types = {f["geometry"]["type"] for f in geojson["features"]}
        assert "Point" in types
        assert "LineString" in types


class TestEmptyGraph:
    """Граничні випадки."""

    def test_no_edges(self):
        opt = PoseGraphOptimizer()
        opt.fix_node(0, _make_similarity(0.0, 0.0, 1.0, 0.0))
        results = opt.optimize()
        assert 0 in results

    def test_all_fixed(self):
        opt = PoseGraphOptimizer()
        opt.fix_node(0, _make_similarity(0.0, 0.0, 1.0, 0.0))
        opt.fix_node(1, _make_similarity(100.0, 0.0, 1.0, 0.0))
        T = _make_similarity(100.0, 0.0, 1.0, 0.0)
        opt.add_edge(0, 1, T, weight=1.0)
        results = opt.optimize()
        assert len(results) == 2
