import numpy as np

from src.calibration.multi_anchor_calibration import AnchorCalibration, MultiAnchorCalibration


def test_pixel_to_metric_round_trip():
    M = np.array([[1.0, 0, 100.0], [0, 1.0, 200.0]], dtype=np.float32)
    anchor = AnchorCalibration(frame_id=0, affine_matrix=M)
    x_m, y_m = anchor.pixel_to_metric(50.0, 75.0)
    assert abs(x_m - 150.0) < 1e-4
    assert abs(y_m - 275.0) < 1e-4


def test_interpolation_at_midpoint():
    calib = MultiAnchorCalibration()

    M1 = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0]], dtype=np.float32)
    M2 = np.array([[1.0, 0, 5.0], [0, 1.0, 5.0]], dtype=np.float32)

    calib.add_anchor(0, M1)
    calib.add_anchor(10, M2)

    result = calib.get_metric_position(5, 0.0, 0.0)

    assert result is not None
    assert abs(result[0] - 2.5) < 1e-4
    assert abs(result[1] - 2.5) < 1e-4
