"""
Тести two-stage prune L2 → prune → L2 (Етап 3 плану). Прапорець, дефолт off.

Ключова відмінність від robust loss: поріг рахується ВІДНОСНО інших spatial
(MAD усередині класу), а не абсолютною константою; валідні loop closures
лишаються з повною L2-вагою. Захист: лише spatial, ≤20%, ніколи не роз'єднати.
"""

import numpy as np

from src.geometry.affine_utils import compose_affine, decompose_affine
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer


def _sim(tx, ty, scale=1.0, angle_deg=0.0):
    return compose_affine(tx, ty, scale, np.radians(angle_deg))


def _graph_with_false_closure():
    """Ланцюг 0..6 (2 якорі) + 4 добрих spatial + 1 хибне spatial-ребро 1->5."""
    opt = PoseGraphOptimizer()
    opt.fix_node(0, _sim(0.0, 0.0))
    opt.fix_node(6, _sim(600.0, 0.0))
    for i in range(1, 6):
        opt.add_node(i)
    for i in range(6):
        opt.add_edge(i, i + 1, _sim(100.0, 0.0), weight=1.0, edge_type="temporal",
                     inliers=50, rmse=1.0)
    # добрі spatial (узгоджені з геометрією)
    for a, b in [(0, 2), (2, 4), (4, 6), (1, 3)]:
        opt.add_edge(a, b, _sim(200.0, 0.0), weight=2.0, edge_type="spatial",
                     inliers=30, rmse=2.0)
    # ХИБНЕ spatial 1->5: стверджує зсув 0 (реальний 400) — виражений викид
    opt.add_edge(1, 5, _sim(0.0, 0.0), weight=0.5, edge_type="spatial",
                 inliers=16, rmse=5.0)
    opt.initialize_from_bfs()
    return opt


class TestTwoStagePrune:
    def test_flag_off_prunes_nothing(self):
        opt = _graph_with_false_closure()
        n_before = opt.num_edges
        opt.optimize(max_iterations=60, tolerance=1e-10, two_stage_prune=False)
        assert opt.num_edges == n_before
        assert opt._pruned_edges == []

    def test_prunes_the_false_closure(self):
        opt = _graph_with_false_closure()
        opt.optimize(max_iterations=60, tolerance=1e-10,
                     use_analytic_jac=True, two_stage_prune=True)
        pruned = [(e.from_id, e.to_id) for e in opt._pruned_edges]
        assert (1, 5) in pruned, f"хибне закриття не викинуто; pruned={pruned}"
        # усе викинуте — spatial (temporal-ланцюг недоторканий)
        assert all(e.edge_type == "spatial" for e in opt._pruned_edges)

    def test_prune_improves_accuracy(self):
        gt = {i: i * 100.0 for i in range(7)}  # правильні tx центрів
        opt_no = _graph_with_false_closure()
        r_no = opt_no.optimize(max_iterations=60, tolerance=1e-10, use_analytic_jac=True)
        opt_yes = _graph_with_false_closure()
        r_yes = opt_yes.optimize(max_iterations=60, tolerance=1e-10,
                                 use_analytic_jac=True, two_stage_prune=True)

        def med_err(res):
            errs = [abs(decompose_affine(res[i])[0] - gt[i]) for i in range(7)]
            return float(np.median(errs))

        assert med_err(r_yes) <= med_err(r_no) + 1e-6, "prune не має погіршувати"

    def test_never_disconnects_node_from_anchors(self):
        """Вузол, з'єднаний ЛИШЕ хибним spatial-ребром, не має бути роз'єднаний."""
        opt = _graph_with_false_closure()
        # додаємо «висячий» вузол 7, з'єднаний тільки хибним spatial 3->7
        opt.add_node(7)
        opt.add_edge(3, 7, _sim(999.0, 999.0), weight=0.3, edge_type="spatial",
                     inliers=16, rmse=6.0)
        opt.initialize_from_bfs()
        opt.optimize(max_iterations=60, tolerance=1e-10,
                     use_analytic_jac=True, two_stage_prune=True)
        pruned = [(e.from_id, e.to_id) for e in opt._pruned_edges]
        assert (3, 7) not in pruned, "не можна викидати єдиний зв'язок вузла з графом"

    def test_prune_cap_20_percent(self):
        """Не більше 20% spatial-ребер за прохід (5 spatial -> max 1)."""
        opt = _graph_with_false_closure()  # 5 spatial
        opt.optimize(max_iterations=60, tolerance=1e-10,
                     use_analytic_jac=True, two_stage_prune=True)
        assert len(opt._pruned_edges) <= 1
