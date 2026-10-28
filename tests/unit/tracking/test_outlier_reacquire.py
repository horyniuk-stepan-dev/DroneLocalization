"""Re-acquisition gate (LOCALIZER_IMPROVEMENT_PLAN 1.1).

Legacy: the max_consecutive-th rejected measurement is accepted unconditionally,
which turns a burst of random false fixes into a trajectory jump. With
``reacquire_consistent`` the streak is accepted only when two consecutive
rejected candidates share the same offset from the KF prediction (a real
relocation), or after ``reacquire_max_wait`` extra rejections.
"""

import numpy as np

from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector


def _detector(**kw):
    params = dict(
        window_size=10,
        max_speed_mps=50.0,
        max_consecutive=3,
        zscore_enabled=False,
    )
    params.update(kw)
    det = OutlierDetector(**params)
    for i in range(5):
        det.add_position((10.0 * i, 0.0), dt=1.0)
    return det


def _pred(k):
    """KF prediction during the streak: keeps flying +10 m/s along x."""
    return (40.0 + 10.0 * k, 0.0)


SCATTER = [(300.0, 250.0), (-250.0, 300.0), (310.0, -280.0), (-300.0, -260.0)]


def _streak(det, offsets, with_pred=True):
    out = []
    for k, off in enumerate(offsets, start=1):
        p = _pred(k)
        cand = (p[0] + off[0], p[1] + off[1])
        out.append(det.is_outlier(cand, dt=1.0, pred_position=p if with_pred else None))
    return out


def test_legacy_accepts_third_scattered_rejection():
    det = _detector()
    assert _streak(det, SCATTER[:3]) == [True, True, False]


def test_flag_off_ignores_pred_position_bitwise():
    a = _detector()
    b = _detector()
    ra = [a.is_outlier((500.0 + k, 400.0), dt=1.0) for k in range(4)]
    rb = [b.is_outlier((500.0 + k, 400.0), dt=1.0, pred_position=_pred(k)) for k in range(4)]
    assert ra == rb


def test_scattered_false_fixes_are_not_accepted():
    det = _detector(reacquire_consistent=True, reacquire_tol_m=15.0, reacquire_max_wait=10)
    assert _streak(det, SCATTER) == [True, True, True, True]


def test_consistent_relocation_is_accepted():
    det = _detector(reacquire_consistent=True, reacquire_tol_m=15.0)
    off = (200.0, -120.0)
    # 3rd rejection reaches max_consecutive with two agreeing offsets -> accept.
    assert _streak(det, [off, (off[0] + 5.0, off[1]), (off[0] + 8.0, off[1] - 4.0)]) == [
        True,
        True,
        False,
    ]


def test_consistency_waits_until_two_agree():
    det = _detector(reacquire_consistent=True, reacquire_tol_m=15.0)
    off = (200.0, 0.0)
    assert _streak(det, [SCATTER[0], SCATTER[1], off, off]) == [True, True, True, False]


def test_forced_accept_after_max_wait():
    det = _detector(reacquire_consistent=True, reacquire_max_wait=2)
    offs = [SCATTER[k % 4] for k in range(6)]
    # max_consecutive=3 + max_wait=2 -> the 5th rejection is forced through.
    assert _streak(det, offs)[:5] == [True, True, True, True, False]


def test_max_wait_zero_never_forces():
    det = _detector(reacquire_consistent=True, reacquire_max_wait=0)
    offs = [SCATTER[k % 4] for k in range(12)]
    assert all(_streak(det, offs))


def test_missing_prediction_falls_back_to_legacy():
    det = _detector(reacquire_consistent=True)
    assert _streak(det, SCATTER[:3], with_pred=False) == [True, True, False]


def test_accepted_measurement_clears_streak_offsets():
    det = _detector(reacquire_consistent=True, reacquire_tol_m=15.0)
    off = (200.0, 0.0)
    assert _streak(det, [off]) == [True]
    # Clean on-track measurement resets the streak (legacy contract).
    assert det.is_outlier((50.0, 0.0), dt=1.0, pred_position=(50.0, 0.0)) is False
    det.add_position((50.0, 0.0), dt=1.0)
    # A fresh streak needs its own two agreeing offsets.
    assert _streak(det, [SCATTER[0], off]) == [True, True]


def test_tolerance_is_in_ground_metres():
    # ground_scale 0.5: a 20-unit projection drift is 10 ground metres.
    det = _detector(reacquire_consistent=True, reacquire_tol_m=15.0, ground_scale=0.5)
    det.window.clear()
    for i in range(5):
        det.add_position((20.0 * i, 0.0), dt=1.0)
    p = [(80.0 + 20.0 * k, 0.0) for k in range(1, 4)]
    offs = [(600.0, 0.0), (620.0, 0.0), (640.0, 0.0)]
    res = [
        det.is_outlier((pk[0] + o[0], pk[1] + o[1]), dt=1.0, pred_position=pk)
        for pk, o in zip(p, offs)
    ]
    assert res == [True, True, False]


def test_kf_predict_position_is_pure_and_constant_velocity():
    kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=1.0)
    assert kf.predict_position(1.0) is None
    for i in range(15):
        kf.update((10.0 * i, 5.0), dt=1.0)
    x_before = kf.kf.x.copy()
    P_before = kf.kf.P.copy()
    px, py = kf.predict_position(2.0)
    assert np.array_equal(kf.kf.x, x_before)
    assert np.array_equal(kf.kf.P, P_before)
    assert abs(px - (140.0 + 2.0 * 10.0)) < 1.0
    assert abs(py - 5.0) < 0.5
