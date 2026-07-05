"""
Регресійні тести інтерполяції якорів MultiAnchorCalibration.

Історичний баг: 4-DoF декомпозиція (tx, ty, scale, angle) не кодувала
Y-віддзеркалення якірних матриць (det<0) — між якорями координати
дзеркалилися на десятки метрів (2·s·y ≈ 54 м на центрі кадру), а точно на
якорях були правильними → різкі стрибки. Плюс extrapolate=True розлітався
за межами діапазону якорів.
"""

import numpy as np

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.geometry.affine_utils import compose_affine_5dof

W, H = 1920, 1080
CX, CY = W / 2.0, H / 2.0


def _anchor_matrix(theta: float, s: float, center_metric: tuple) -> np.ndarray:
    """Якірна матриця px→metric з det<0 і заданою метричною позицією центру."""
    M = compose_affine_5dof(0, 0, s, s, theta, sign=-1.0)
    M[0, 2] = center_metric[0] - (M[0, 0] * CX + M[0, 1] * CY)
    M[1, 2] = center_metric[1] - (M[1, 0] * CX + M[1, 1] * CY)
    return M


def _make_calibration() -> MultiAnchorCalibration:
    cal = MultiAnchorCalibration()
    cal.set_frame_size(W, H)
    cal.add_anchor(
        10,
        _anchor_matrix(0.1, 0.05, (5000.0, 3000.0)),
        {"points_2d": [[100, 100], [1800, 900]], "points_gps": [[47.8, 34.9], [47.81, 34.91]]},
    )
    cal.add_anchor(
        200,
        _anchor_matrix(0.5, 0.06, (5500.0, 3300.0)),
        {"points_2d": [[150, 120], [1700, 880]], "points_gps": [[47.82, 34.92], [47.83, 34.93]]},
    )
    return cal


class TestInterpolationReflection:
    def test_exact_on_anchor_frame(self):
        cal = _make_calibration()
        pos = cal.get_metric_position(10, CX, CY)
        assert pos is not None
        assert np.allclose(pos, (5000.0, 3000.0), atol=1e-6)

    def test_no_mirroring_between_anchors(self):
        """Між якорями позиція має лежати між якірними центрами (без дзеркалення)."""
        cal = _make_calibration()
        pos = cal.get_metric_position(105, CX, CY)
        assert pos is not None
        assert 5000.0 < pos[0] < 5500.0, f"x={pos[0]} поза інтервалом якорів"
        assert 3000.0 < pos[1] < 3300.0, f"y={pos[1]} поза інтервалом якорів (дзеркалення?)"

    def test_interpolated_matrix_keeps_negative_det(self):
        cal = _make_calibration()
        for fid in range(10, 201, 10):
            M = cal._get_interpolated_matrix(float(fid))
            assert M is not None
            assert np.linalg.det(M[:2, :2]) < 0, f"det>=0 на кадрі {fid} — втрачено Y-flip"

    def test_clamp_outside_anchor_range(self):
        """За межами діапазону — крайній якір, а не кубічна екстраполяція."""
        cal = _make_calibration()
        before = cal.get_metric_position(0, CX, CY)
        after = cal.get_metric_position(99999, CX, CY)
        assert np.allclose(before, (5000.0, 3000.0), atol=1e-6)
        assert np.allclose(after, (5500.0, 3300.0), atol=1e-6)


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_interpolation(self, tmp_path):
        cal = _make_calibration()
        mid_before = cal.get_metric_position(105, CX, CY)

        path = str(tmp_path / "calibration.json")
        cal.save(path)

        cal2 = MultiAnchorCalibration()
        cal2.load(path)
        assert cal2._frame_size == (W, H)
        mid_after = cal2.get_metric_position(105, CX, CY)
        assert np.allclose(mid_before, mid_after, atol=1e-9)

    def test_save_is_atomic_no_partial_file(self, tmp_path):
        """Після save файл — валідний JSON без сміття (атомарний запис)."""
        import json

        cal = _make_calibration()
        path = tmp_path / "calibration.json"
        cal.save(str(path))

        raw = path.read_bytes()
        assert b"\x00" not in raw, "файл містить null-байти"
        data = json.loads(raw)
        assert len(data["anchors"]) == 2
