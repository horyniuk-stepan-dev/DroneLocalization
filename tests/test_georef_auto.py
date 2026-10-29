import numpy as np

from src.geometry.georef_auto import consensus_similarity, select_probe_frames
from src.geometry.georef_fit import apply


def _sim(scale, deg, t, mirror=False):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if mirror:
        R = R @ np.diag([1.0, -1.0])
    return np.hstack([scale * R, np.asarray(t, float)[:, None]])


def test_probes_spread_and_include_ends():
    s = np.linspace(0, 1, 500)
    c = np.column_stack([s * 100, np.zeros(500)])
    ids = np.arange(1000, 1500)
    p = select_probe_frames(c, ids, 5)
    assert len(p) == 5 and p[0] == 1000 and p[-1] == 1499
    gaps = np.diff(p)
    assert gaps.min() > 80


def test_probes_budget_larger_than_frames():
    c = np.random.default_rng(0).random((4, 2))
    assert list(select_probe_frames(c, [3, 1, 2, 0], 10)) == [0, 1, 2, 3]


def test_consensus_rejects_wrong_fixes():
    rng = np.random.default_rng(3)
    src = rng.random((12, 2)) * 100
    M = _sim(4.0, 40, [2e6, 5e6])
    dst = apply(M, src) + rng.normal(0, 2, (12, 2))
    bad = [2, 7, 9]
    dst[bad] += rng.uniform(300, 900, (3, 2))  # aliasing: confident, far off
    c = consensus_similarity(src, dst, tolerance=20.0)
    assert set(np.flatnonzero(~c.inliers)) == set(bad)
    np.testing.assert_allclose(c.matrix[:, :2], M[:, :2], rtol=0.02)
    assert not c.warnings


def test_consensus_tolerates_scale_drift():
    s = np.linspace(0, 1, 10)
    src = np.column_stack([s * 200, np.zeros(10)])
    # 20% scale drift along the path (integrated)
    true = np.column_stack([np.cumsum(np.r_[0, np.diff(s * 200) * (1 + 0.2 * s[1:])]), 0 * s])
    M = _sim(3.0, -20, [1e5, 1e5])
    dst = apply(M, true)
    c = consensus_similarity(src, dst, tolerance=10.0, drift_frac=0.1)
    assert c.inliers.all()


def test_consensus_mirror_and_too_few():
    src = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10]])
    M = _sim(2.0, 10, [0, 0], mirror=True)
    c = consensus_similarity(src, apply(M, src), rotation_sign=-1, tolerance=1.0)
    assert c.inliers.all() and np.linalg.det(c.matrix[:, :2]) < 0
    one = consensus_similarity(src[:1], src[:1], tolerance=1.0)
    assert one.matrix is None and one.warnings
    two = consensus_similarity(src[:2], apply(M, src[:2]), rotation_sign=-1, tolerance=1.0)
    assert two.n_inliers == 2 and two.warnings
