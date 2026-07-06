"""
Контракт-тест оптимізатора графу поз (Етап 0.3 плану).

Заморожує СВІДОМІ інваріанти робочої 5-DoF анізотропної моделі, щоб випадкова
зміна формул ваг/резидуалів впала явним фейлом, а не тихим дрейфом калібрування.

Пінимо:
  1. Сигнатуру optimize() (max_iterations, tolerance, progress_callback).
  2. Структуру вектора резидуалів: 5*E + N (5 на ребро + регуляризатор на вузол).
  3. Формули ваг на відомому стані — порівняння з ЕТАЛОННИМИ ЧИСЛАМИ:
       - трансляція:   w/sx_i, w/sy_i (індивідуально для вузла!);
       - масштаб/кут:  w*cx;
       - регуляризатор ізотропії вузла: 200*cx*(log_sx - log_sy).

БУДЬ-ЯКА зміна цих чисел = свідоме рішення, задокументоване у PR, а не побічний
ефект. Якщо ви навмисно міняєте модель — оновіть еталони тут разом із поясненням.
"""

import inspect

import numpy as np

from src.geometry.pose_graph_optimizer import PoseGraphOptimizer


def _build_d(cx=960.0, sign=1.0):
    """Ручна збірка словника d для _residuals_vec (як усередині optimize())."""
    # 2 вільні вузли, 1 ребро 0->1
    X_full = np.zeros((2, 5), dtype=np.float64)
    return {
        "X_full": X_full,
        "free_indices": [0, 1],
        "edges_from": np.array([0], dtype=np.int32),
        "edges_to": np.array([1], dtype=np.int32),
        "dtx": np.array([8.0]),
        "dty": np.array([3.0]),
        "log_dsx": np.array([0.05]),
        "log_dsy": np.array([0.15]),
        "dtheta": np.array([0.25]),
        "weights": np.array([2.0]),
        "cx": cx,
        "sign": sign,
        "n_edges": 1,
    }


class TestOptimizerContract:

    def test_optimize_signature_frozen(self):
        params = inspect.signature(PoseGraphOptimizer.optimize).parameters
        assert "max_iterations" in params
        assert "tolerance" in params
        assert "progress_callback" in params

    def test_residual_vector_structure(self):
        """Довжина вектора резидуалів = 5*E + N."""
        opt = PoseGraphOptimizer()
        d = _build_d()
        # state0 = identity, state1 — довільний
        x = np.array([0, 0, 0, 0, 0,
                      10.0, 5.0, 0.1, 0.2, 0.3], dtype=np.float64)
        res = opt._residuals_vec(x, d)
        E, Nfree = 1, 2
        assert res.shape[0] == 5 * E + Nfree, f"структура резидуалів змінилась: {res.shape[0]}"

    def test_weight_formulas_reference_numbers(self):
        """Еталонні числа пінять формули ваг (w/sx_i, w*cx, w_reg=200*cx)."""
        opt = PoseGraphOptimizer()
        d = _build_d(cx=960.0, sign=1.0)
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0,      # вузол 0: identity
                      10.0, 5.0, 0.1, 0.2, 0.3],     # вузол 1
                     dtype=np.float64)
        res = opt._residuals_vec(x, d)

        # Еталони (виведені вручну з інваріантних формул, звірені з поточним кодом):
        #   sx_i=sy_i=1, theta_i=0 -> pred_tx=8, pred_ty=3
        #   res0 = (w/sx_i)*(tx_j-pred_tx) = 2*(10-8) = 4
        #   res1 = (w/sy_i)*(ty_j-pred_ty) = 2*(5-3)  = 4
        #   res2 = w*cx*(log_sx_j-log_sx_i-log_dsx) = 1920*(0.1-0.05) = 96
        #   res3 = w*cx*(log_sy_j-log_sy_i-log_dsy) = 1920*(0.2-0.15) = 96
        #   res4 = w*cx*wrap(theta_j-theta_i-dtheta) = 1920*wrap(0.05)
        #   reg0 = 200*cx*(log_sx0-log_sy0) = 192000*0 = 0
        #   reg1 = 200*cx*(log_sx1-log_sy1) = 192000*(0.1-0.2) = -19200
        expected = np.array([
            4.0,
            4.0,
            96.0,
            96.0,
            1920.0 * np.arctan2(np.sin(0.05), np.cos(0.05)),
            0.0,
            -19200.0,
        ])
        np.testing.assert_allclose(res, expected, rtol=0, atol=1e-9,
                                   err_msg="ФОРМУЛИ ВАГ ЗМІНИЛИСЬ — див. інваріанти плану")

    def test_translation_weight_is_per_node(self):
        """w/sx_i індивідуальна: більший sx_i -> менша вага трансляції (не фіксована)."""
        opt = PoseGraphOptimizer()
        d = _build_d()
        # робимо sx вузла 0 удвічі більшим (log_sx0 = ln 2) — вага трансляції ділиться на 2
        x = np.array([0, 0, np.log(2.0), np.log(2.0), 0,
                      10.0, 5.0, 0.1, 0.2, 0.3], dtype=np.float64)
        res = opt._residuals_vec(x, d)
        # sx_i=sy_i=2 -> pred_tx = 0 + 1*2*8 = 16 ; w_trans_x = 2/2 = 1
        # res0 = 1*(10-16) = -6
        assert abs(res[0] - (-6.0)) < 1e-9, f"вага трансляції не w/sx_i: res0={res[0]}"


if __name__ == "__main__":
    opt = PoseGraphOptimizer()
    d = _build_d()
    x = np.array([0, 0, 0, 0, 0, 10.0, 5.0, 0.1, 0.2, 0.3], dtype=np.float64)
    print("residuals:", opt._residuals_vec(x, d))
