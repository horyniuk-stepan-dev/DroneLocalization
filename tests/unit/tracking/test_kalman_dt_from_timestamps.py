"""KF prediction interval (tracking.kalman_dt_from_timestamps, 2026-10-28).

The worker hands a keyframe the time since the previous KEYFRAME (~0.9 s at 1 Hz)
although optical flow updated the KF ~0.17 s earlier. Predicting over 0.9 s on
every keyframe made the constant-velocity filter lag between keyframes and snap
back on each keyframe (a 1 Hz sawtooth = "jumps"). These tests pin the mechanism
with exact measurements so the fix cannot silently regress.
"""

import numpy as np

from src.tracking.kalman_filter import TrajectoryFilter

V = 50.0  # m/s
KF_PERIOD = 1.0
OF_STEP = 1.0 / 6.0


def _schedule(seconds=20):
    """(t, path, worker_dt) like TrackingWorker with keyframe_interval=30,
    of_stride=5, of_local_speed=True at 30 fps."""
    out = []
    t_last_kf = None
    t_last_of = None
    for k in range(int(seconds / OF_STEP) + 1):
        t = round(k * OF_STEP, 9)
        if k % 6 == 0:
            dt = KF_PERIOD if t_last_kf is None else t - t_last_kf
            out.append((t, "kf", dt))
            t_last_kf = t
        else:
            out.append((t, "of", t - t_last_of))
        t_last_of = t
    return out


def _run(use_timestamps):
    kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0)
    steps = []
    prev_err = None
    for t, path, worker_dt in _schedule():
        dt = worker_dt
        if use_timestamps and kf.last_update_t is not None:
            dt = t - kf.last_update_t
        est = kf.update((V * t, 0.0), dt=dt, noise_scale=1.0, timestamp=t)
        err = est[0] - V * t
        if prev_err is not None and t > 3.0:
            steps.append(abs(err - prev_err))
        prev_err = err
    return np.array(steps)


def test_last_update_t_tracks_and_resets():
    kf = TrajectoryFilter()
    assert kf.last_update_t is None
    kf.update((0.0, 0.0), dt=1.0, timestamp=2.5)
    assert kf.last_update_t == 2.5
    kf.update((1.0, 0.0), dt=1.0)  # no timestamp: keeps the last known one
    assert kf.last_update_t == 2.5
    kf.reset()
    assert kf.last_update_t is None


def test_worker_dt_reproduces_the_sawtooth():
    steps = _run(use_timestamps=False)
    # exact measurements, yet the output jumps on every keyframe
    assert steps.max() > 10.0


def test_timestamp_dt_removes_it():
    steps = _run(use_timestamps=True)
    assert steps.max() < 1.0
