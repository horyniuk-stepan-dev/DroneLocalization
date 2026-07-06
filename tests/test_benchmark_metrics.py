"""Тести чистих функцій бенчмарка пропагації (Етап 0.1): метрики + гейт."""


import scripts.benchmark_propagation as B
from src.geometry.affine_utils import compose_affine_5dof

W, H = 1920, 1080
CX, CY = W / 2.0, H / 2.0


def _affine(Cx, Cy, s=0.05, sign=-1.0, angle=0.0):
    M = compose_affine_5dof(0.0, 0.0, s, s, angle, sign=sign)
    M[0, 2] = Cx - (M[0, 0] * CX + M[0, 1] * CY)
    M[1, 2] = Cy - (M[1, 0] * CX + M[1, 1] * CY)
    return M


class TestComputeMetrics:
    def test_perfect_prediction_zero_error(self):
        gt = {i: _affine(i * 10.0, 0.0) for i in range(4)}
        pred = {i: gt[i].copy() for i in gt}
        m = B.compute_metrics(pred, gt, W, H)
        assert m["median_err_m"] < 1e-9
        assert m["max_err_m"] < 1e-9
        assert m["det_sign_ok"] == 1.0
        assert m["scale_drift"] < 1e-9
        assert m["coverage"] == 1.0

    def test_known_offset(self):
        gt = {i: _affine(i * 10.0, 0.0) for i in range(3)}
        pred = {i: gt[i].copy() for i in gt}
        pred[1][0, 2] += 3.0  # зсув центру на (3,4) → похибка 5 м
        pred[1][1, 2] += 4.0
        m = B.compute_metrics(pred, gt, W, H)
        assert abs(m["max_err_m"] - 5.0) < 1e-6
        assert m["median_err_m"] < 1e-6  # медіана [0,5,0] = 0

    def test_scale_drift_and_det_flip(self):
        gt = {i: _affine(i * 10.0, 0.0, s=0.05, sign=-1.0) for i in range(3)}
        # усі кадри на 10% більший масштаб + втрачене відбиття (sign +1)
        pred = {i: _affine(i * 10.0, 0.0, s=0.055, sign=1.0) for i in range(3)}
        m = B.compute_metrics(pred, gt, W, H)
        assert abs(m["scale_drift"] - 0.10) < 1e-6
        assert m["det_sign_ok"] == 0.0

    def test_coverage_partial(self):
        gt = {i: _affine(i * 10.0, 0.0) for i in range(4)}
        pred = {0: gt[0].copy(), 2: gt[2].copy()}  # лише половина
        m = B.compute_metrics(pred, gt, W, H)
        assert m["coverage"] == 0.5
        assert m["n_frames"] == 2


class TestCheckGate:
    def _baseline(self):
        return {"median_err_m": 1.0, "max_err_m": 5.0}

    def test_no_baseline_passes(self):
        ok, _ = B.check_gate({"median_err_m": 9, "max_err_m": 99}, None)
        assert ok

    def test_within_tolerance_passes(self):
        m = {"median_err_m": 1.10, "max_err_m": 6.0}  # +10%, +20% рівно на межі
        ok, _ = B.check_gate(m, self._baseline())
        assert ok

    def test_median_regression_fails(self):
        m = {"median_err_m": 1.2, "max_err_m": 5.0}
        ok, details = B.check_gate(m, self._baseline())
        assert not ok and "РЕГРЕСІЯ" in details

    def test_max_regression_fails(self):
        m = {"median_err_m": 1.0, "max_err_m": 6.5}
        ok, _ = B.check_gate(m, self._baseline())
        assert not ok
