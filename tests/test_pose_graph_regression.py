"""
Регресійні тести PoseGraphOptimizer на синтетичних траєкторіях.

Покривають історичні баги калібрувальної пропагації:
  1. Втрата Y-віддзеркалення (det<0) якірних матриць.
  2. Дрейф масштабу вздовж траєкторії.

ПРИМІТКА щодо robust loss: механізм loss="soft_l1" було випробувано і СВІДОМО
прибрано. Spatial-ребра (loop closures) мають природно великі резидуали, і
robust loss пригнічував саме їх — якорі тримали свої кадри, а решта траєкторії
"пливла". Чистий L2 дає loop closures повний вплив і стягує граф у правильну
форму. Аналогічно, ваги w/sx_i (індивідуальні для вузла) правильно переводять
метричний резидуал у піксельний для КОЖНОГО вузла; фіксований s_ref порушував
баланс ваг між близькими й далекими вузлами.
"""

import numpy as np

from src.geometry.affine_utils import compose_affine_5dof
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

W, H = 1920, 1080
CX, CY = W / 2.0, H / 2.0


def _gt_affine(i: int) -> np.ndarray:
    """Ground-truth: дрон летить по дузі, матриці з det<0 (px Y↓ → metric Y↑)."""
    theta = 0.1 + 0.02 * i
    s = 0.05 * (1 + 0.005 * i)
    M = compose_affine_5dof(0, 0, s, s, theta, sign=-1.0)
    ctr = (100 + 25 * i, 50 + 15 * i)
    M[0, 2] = ctr[0] - (M[0, 0] * CX + M[0, 1] * CY)
    M[1, 2] = ctr[1] - (M[1, 0] * CX + M[1, 1] * CY)
    return M


def _relative(Ma: np.ndarray, Mb: np.ndarray) -> np.ndarray:
    """Відносна трансформація: Mb = Ma ∘ R  =>  R = Ma^-1 ∘ Mb (пікселі→пікселі)."""
    A = np.vstack([Ma, [0, 0, 1]])
    B = np.vstack([Mb, [0, 0, 1]])
    return (np.linalg.inv(A) @ B)[:2, :]


def _center_error(result_affine: np.ndarray, gt_affine: np.ndarray) -> float:
    p = np.array([CX, CY])
    est = result_affine[:, :2] @ p + result_affine[:, 2]
    gt = gt_affine[:, :2] @ p + gt_affine[:, 2]
    return float(np.linalg.norm(est - gt))


def _build_graph(n: int) -> tuple[PoseGraphOptimizer, list, np.ndarray]:
    gt = [_gt_affine(i) for i in range(n)]
    opt = PoseGraphOptimizer(W, H)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        rel = _relative(gt[i], gt[i + 1])
        assert np.linalg.det(rel[:2, :2]) > 0, "ребро (px→px) не має містити відбиття"
        opt.add_edge(i, i + 1, rel, weight=1.0, inliers=50, rmse=1.0)

    origin = gt[0][:, 2].copy()
    for aid in (0, n - 1):
        local = gt[aid].copy()
        local[:, 2] -= origin
        opt.fix_node(aid, local)
    opt.initialize_from_bfs()
    return opt, gt, origin


class TestReflectionPreservation:
    """Баг №1: якірні матриці мають det<0, оптимізатор мусить це зберігати."""

    def test_trajectory_recovery_with_reflected_anchors(self):
        n = 12
        opt, gt, origin = _build_graph(n)
        results = opt.optimize(max_iterations=60, tolerance=1e-12)

        for i in range(n):
            M = results[i].copy()
            M[:, 2] += origin
            assert np.linalg.det(M[:2, :2]) < 0, f"det>=0 у кадрі {i} — втрачено Y-flip"
            assert _center_error(M, gt[i]) < 0.05, f"кадр {i}: центр відхилився"


class TestScaleStability:
    """Баг №2: масштаб не має "тікати" вздовж траєкторії."""

    def test_scale_preserved_along_trajectory(self):
        n = 10
        opt, gt, origin = _build_graph(n)
        results = opt.optimize(max_iterations=60, tolerance=1e-12)

        for i in range(n):
            est_s = float(np.linalg.norm(results[i][:2, 0]))
            gt_s = float(np.linalg.norm(gt[i][:2, 0]))
            assert abs(est_s - gt_s) / gt_s < 0.02, (
                f"кадр {i}: масштаб {est_s:.4f} відхилився від GT {gt_s:.4f}"
            )
