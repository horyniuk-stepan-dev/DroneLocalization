"""
Реалістичний синтетичний граф-тест PoseGraphOptimizer (Етап 0.2 плану).

Мета: захисна сітка в CI (без GPU), що відтворює РЕАЛЬНУ структуру графу
пропагації, а не тривіальний ланцюг із 1 хибним ребром. Будь-яка майбутня
зміна оптимізатора мусить пройти цей тест.

Структура графу (сцена «маршрут із поверненням»):
  - дрон летить уперед (кадри 0..N_OUT-1), потім вертається паралельним
    треком зі зсувом по Y (кадри N_OUT..2*N_OUT-1);
  - temporal-ланцюг: сусідні кадри, з реалістичним піксельним шумом;
  - валідні spatial-ребра (loop closures): кадри зустрічного треку поруч,
    з БІЛЬШИМ шумом (це нормально, у них великі резидуали);
  - ~9% ХИБНИХ loop closures із малою вагою (мало inliers, високий rmse);
  - 3 якорі, один із них трохи «кривий» (імітація похибки кліків).

Assert: медіанна похибка центру < поріг. Пороги свідомо з запасом —
тест ловить ГРУБІ регресії (розвал калібрування), а не мікродрейф.
"""

import numpy as np

from src.geometry.affine_utils import compose_affine_5dof
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

W, H = 1920, 1080
CX, CY = W / 2.0, H / 2.0

N_OUT = 8                # кадрів на кожному з двох треків
N = 2 * N_OUT            # усього кадрів (16 — компактно, щоб 2-point FD
                         # чисто збігався в CI за секунди; структура реальна)
STEP_X_M = 8.0           # крок центру по X, метри
OFFSET_Y_M = 5.0         # зсув зустрічного треку по Y, метри
GSD = 0.05               # метрів на піксель (масштаб афінної матриці)
SEED = 20260705


def _gt_affine(i: int) -> np.ndarray:
    """Ground-truth афінна матриця кадру i (px->metric), з det<0 (Y-flip).

    Масштаб ізотропний (sx=sy) — реальні надирні кадри дрона близькі до
    ізотропних, і саме це очікує м'який регуляризатор ізотропії вузлів.
    """
    if i < N_OUT:
        cx_m = i * STEP_X_M
        cy_m = 0.0
        k = i
    else:
        k = i - N_OUT
        cx_m = (N_OUT - 1 - k) * STEP_X_M
        cy_m = OFFSET_Y_M

    s = GSD * (1.0 + 0.004 * k)
    theta = 0.05 + 0.01 * k

    M = compose_affine_5dof(0.0, 0.0, s, s, theta, sign=-1.0)
    M[0, 2] = cx_m - (M[0, 0] * CX + M[0, 1] * CY)
    M[1, 2] = cy_m - (M[1, 0] * CX + M[1, 1] * CY)
    return M


def _relative(Ma: np.ndarray, Mb: np.ndarray) -> np.ndarray:
    """Відносна трансформація (px->px): R = Ma^-1 . Mb."""
    A = np.vstack([Ma, [0, 0, 1]])
    B = np.vstack([Mb, [0, 0, 1]])
    return (np.linalg.inv(A) @ B)[:2, :]


def _noisy_edge(rel, rng, sigma_px, rot_sigma=0.0):
    """Додає реалістичний піксельний шум до відносного (px->px) ребра."""
    out = rel.copy()
    out[0, 2] += rng.normal(0.0, sigma_px)
    out[1, 2] += rng.normal(0.0, sigma_px)
    if rot_sigma > 0.0:
        dphi = rng.normal(0.0, rot_sigma)
        c, s = np.cos(dphi), np.sin(dphi)
        Rn = np.array([[c, -s], [s, c]])
        out[:2, :2] = Rn @ out[:2, :2]
    return out


def _center_metric(affine):
    p = np.array([CX, CY])
    return affine[:, :2] @ p + affine[:, 2]


def _center_error(result_affine, gt_affine):
    return float(np.linalg.norm(_center_metric(result_affine) - _center_metric(gt_affine)))


def build_realistic_graph(seed=SEED):
    """Будує реалістичний граф. Повертає (optimizer, gt_list, origin, meta)."""
    rng = np.random.default_rng(seed)
    gt = [_gt_affine(i) for i in range(N)]

    opt = PoseGraphOptimizer(W, H)
    for i in range(N):
        opt.add_node(i)

    n_temporal = n_spatial = n_false = 0

    # 1) temporal-ланцюг із реалістичним шумом
    for i in range(N - 1):
        rel = _noisy_edge(_relative(gt[i], gt[i + 1]), rng, sigma_px=1.0, rot_sigma=0.0015)
        opt.add_edge(i, i + 1, rel, weight=1.0, edge_type="temporal", inliers=60, rmse=1.0)
        n_temporal += 1

    # 2) валідні spatial loop closures: кадр (N_OUT+k) поруч із (N_OUT-1-k)
    for k in range(1, N_OUT - 1):
        a = N_OUT + k
        b = N_OUT - 1 - k
        rel = _noisy_edge(_relative(gt[a], gt[b]), rng, sigma_px=2.5, rot_sigma=0.004)
        opt.add_edge(a, b, rel, weight=2.0, edge_type="spatial", inliers=30, rmse=2.5)
        n_spatial += 1

    # 3) ХИБНІ loop closures: ребро «плутає» кадр b з місцем c (десятки метрів).
    #    Реалістично: мало inliers, високий rmse -> МАЛА вага. Чистий L2 має це
    #    толерувати; Етап 3 (prune) — прибрати повністю.
    false_pairs = [(2, 13), (5, 9)]
    for a, wrong in false_pairs:
        b = (a + N_OUT) % N
        rel = _noisy_edge(_relative(gt[a], gt[wrong]), rng, sigma_px=2.5)
        opt.add_edge(a, b, rel, weight=0.35, edge_type="spatial", inliers=16, rmse=4.0)
        n_false += 1

    # 4) якорі: 0, N_OUT-1, N-1. Один — трохи «кривий» (похибка кліків ~1.4 м).
    origin = gt[0][:, 2].copy()
    anchor_ids = (0, N_OUT - 1, N - 1)
    for aid in anchor_ids:
        local = gt[aid].copy()
        local[:, 2] -= origin
        if aid == N_OUT - 1:
            local[0, 2] += 1.2
            local[1, 2] -= 0.8
        opt.fix_node(aid, local)

    opt.initialize_from_bfs()
    meta = {"n_temporal": n_temporal, "n_spatial": n_spatial,
            "n_false": n_false, "anchor_ids": anchor_ids}
    return opt, gt, origin, meta


def _solve_and_measure(opt, gt, origin, **kw):
    results = opt.optimize(max_iterations=kw.get("max_iterations", 120), tolerance=1e-10)
    errs, dets_ok = [], 0
    for i in range(N):
        M = results[i].copy()
        M[:, 2] += origin
        errs.append(_center_error(M, gt[i]))
        if np.linalg.det(M[:2, :2]) < 0:
            dets_ok += 1
    errs = np.array(errs)
    return {"median": float(np.median(errs)), "p95": float(np.percentile(errs, 95)),
            "max": float(np.max(errs)), "det_sign_ok": dets_ok / N, "errs": errs}


class TestRealisticGraph:
    """Реалістичний граф: маршрут із поверненням, шум, хибні closures, якорі."""

    def test_median_error_within_threshold(self):
        opt, gt, origin, meta = build_realistic_graph()
        assert meta["n_spatial"] >= 5, "замало валідних loop closures у сцені"
        assert meta["n_false"] >= 2, "замало хибних loop closures у сцені"

        m = _solve_and_measure(opt, gt, origin)

        assert m["median"] < 2.0, f"median={m['median']:.3f} m - calibration broke"
        assert m["p95"] < 8.0, f"p95={m['p95']:.3f} m - too many distorted frames"
        assert m["det_sign_ok"] == 1.0, f"lost Y-flip on {(1 - m['det_sign_ok']) * N:.0f} frames"

    def test_anchors_pinned_near_ground_truth(self):
        """Якорі (крім навмисно кривого) мають лишитись близько до GT."""
        opt, gt, origin, meta = build_realistic_graph()
        results = opt.optimize(max_iterations=120, tolerance=1e-10)
        for aid in meta["anchor_ids"]:
            if aid == N_OUT - 1:
                continue
            M = results[aid].copy()
            M[:, 2] += origin
            err = _center_error(M, gt[aid])
            assert err < 1.0, f"anchor {aid} drifted from GT: {err:.3f} m"


if __name__ == "__main__":
    opt, gt, origin, meta = build_realistic_graph()
    print("meta:", meta)
    m = _solve_and_measure(opt, gt, origin)
    print("median={:.3f} p95={:.3f} max={:.3f} det_ok={:.2f}".format(m["median"], m["p95"], m["max"], m["det_sign_ok"]))
