"""GNC-переваження spatial (Етап 3) — плавна еволюція two-stage prune. Прапорець.

Гейт плану: на графі з хибними ребрами median_err(GNC) ≤ two-stage prune;
на чистій сцені — БЕЗ деградації (GNC no-op, розв'язок ідентичний L2).
Temporal-ланцюг ніколи не переважується (урок soft_l1).
"""

import numpy as np

from src.geometry.affine_utils import compose_affine, decompose_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

GT = {i: i * 100.0 for i in range(7)}


def _sim(tx, ty, scale=1.0, angle_deg=0.0):
    return compose_affine(tx, ty, scale, np.radians(angle_deg))


def _base_graph(with_false):
    opt = PoseGraphOptimizer()
    opt.fix_node(0, _sim(0.0, 0.0))
    opt.fix_node(6, _sim(600.0, 0.0))
    for i in range(1, 6):
        opt.add_node(i)
    for i in range(6):
        opt.add_edge(
            i, i + 1, _sim(100.0, 0.0), weight=1.0, edge_type="temporal", inliers=50, rmse=1.0
        )
    for a, b in [(0, 2), (2, 4), (4, 6), (1, 3)]:
        opt.add_edge(a, b, _sim(200.0, 0.0), weight=2.0, edge_type="spatial", inliers=30, rmse=2.0)
    if with_false:
        opt.add_edge(
            1, 5, _sim(0.0, 0.0), weight=0.5, edge_type="spatial", inliers=16, rmse=5.0
        )  # хибне: зсув 0 замість 400
    opt.initialize_from_bfs()
    return opt


def _med_err(res):
    return float(np.median([abs(decompose_affine(res[i])[0] - GT[i]) for i in range(7)]))


def test_gnc_flag_off_matches_plain_l2():
    r_off = _base_graph(True).optimize(max_iterations=60, tolerance=1e-10, use_analytic_jac=True)
    r_plain = _base_graph(True).optimize(
        max_iterations=60, tolerance=1e-10, use_analytic_jac=True, gnc_spatial=False
    )
    for i in range(7):
        np.testing.assert_allclose(r_off[i], r_plain[i], atol=1e-9)


def test_gnc_clean_scene_no_degradation():
    """Без хибних ребер GNC не має чіпати розв'язок (no-op)."""
    r_plain = _base_graph(False).optimize(max_iterations=60, tolerance=1e-10, use_analytic_jac=True)
    r_gnc = _base_graph(False).optimize(
        max_iterations=60,
        tolerance=1e-10,
        use_analytic_jac=True,
        gnc_spatial=True,
        gnc_rounds=5,
        gnc_mad_k=3.0,
    )
    for i in range(7):
        np.testing.assert_allclose(r_gnc[i], r_plain[i], atol=1e-6)


def test_gnc_improves_accuracy_with_false_edge():
    r_plain = _base_graph(True).optimize(max_iterations=60, tolerance=1e-10, use_analytic_jac=True)
    r_gnc = _base_graph(True).optimize(
        max_iterations=80,
        tolerance=1e-10,
        use_analytic_jac=True,
        gnc_spatial=True,
        gnc_rounds=5,
        gnc_mad_k=3.0,
    )
    e_plain, e_gnc = _med_err(r_plain), _med_err(r_gnc)
    assert e_gnc <= e_plain + 1e-6, f"GNC гірше за чистий L2: {e_gnc:.3f} vs {e_plain:.3f}"
    assert e_gnc < e_plain * 0.85, f"GNC не придушив хибне ребро: {e_gnc:.3f} vs {e_plain:.3f}"
    print(f"median_err: plain L2={e_plain:.3f}, GNC={e_gnc:.3f}")


def test_gnc_ballpark_of_two_stage_prune():
    """Ballpark-парність із prune. На ОДНОМУ грубому викиді бінарний prune ідеальний
    (0 м), а м'який GNC лишає викиду мізерну вагу → трохи гірше, але в межах допуску.
    Плановий вердикт «GNC ≤ prune, інакше лишаємо prune» вимагає РЕАЛЬНОГО
    бенчмарку місій (torch/PyQt, Windows) — тут лише перевірка того ж порядку."""
    r_prune = _base_graph(True).optimize(
        max_iterations=60, tolerance=1e-10, use_analytic_jac=True, two_stage_prune=True
    )
    r_gnc = _base_graph(True).optimize(
        max_iterations=80,
        tolerance=1e-10,
        use_analytic_jac=True,
        gnc_spatial=True,
        gnc_rounds=5,
        gnc_mad_k=3.0,
    )
    e_prune, e_gnc = _med_err(r_prune), _med_err(r_gnc)
    assert e_gnc <= e_prune + 0.5, f"GNC={e_gnc:.3f} гірше за prune={e_prune:.3f}"
    print(f"median_err: two-stage prune={e_prune:.3f}, GNC={e_gnc:.3f}")


def test_gnc_never_reweights_temporal():
    """Ваги temporal-ребер незмінні після GNC (переважуються лише spatial)."""
    opt = _base_graph(True)
    tw_before = [e.weight for e in opt.edges if e.edge_type == "temporal"]
    opt.optimize(
        max_iterations=80,
        tolerance=1e-10,
        use_analytic_jac=True,
        gnc_spatial=True,
        gnc_rounds=5,
        gnc_mad_k=3.0,
    )
    tw_after = [e.weight for e in opt.edges if e.edge_type == "temporal"]
    assert tw_before == tw_after
    # spatial-ваги теж відновлені до базових (звіти бачать оригінали)
    sw = {(e.from_id, e.to_id): e.weight for e in opt.edges if e.edge_type == "spatial"}
    assert sw[(1, 5)] == 0.5 and sw[(0, 2)] == 2.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
