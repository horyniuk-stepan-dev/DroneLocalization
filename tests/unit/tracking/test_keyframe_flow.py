"""KeyframeFlow must reproduce the LK block that used to live inline in
TrackingWorker.run (pre-2026-10-27) bit for bit, so the offline tracking replay
measures exactly what the live worker does.
"""

import cv2
import numpy as np
import pytest

from src.tracking.keyframe_flow import KeyframeFlow


def _reference(prev_gray, prev_gray_half, prev_pts, curr_gray, fb_check, fb_max_px, half_res):
    """Verbatim copy of the former inline worker logic (returns None on loss)."""
    if not (prev_pts is not None and len(prev_pts) > 10):
        return None
    of_scale = 0.5 if (half_res and prev_gray_half is not None) else 1.0
    if of_scale != 1.0:
        g_prev = prev_gray_half
        g_curr = cv2.resize(curr_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        pts_prev = np.ascontiguousarray(prev_pts * of_scale, dtype=np.float32)
    else:
        g_prev, g_curr, pts_prev = prev_gray, curr_gray, prev_pts
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        g_prev, g_curr, pts_prev, None, winSize=(15, 15), maxLevel=2
    )
    keep = status.reshape(-1) == 1
    if fb_check and keep.any():
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            g_curr, g_prev, curr_pts, None, winSize=(15, 15), maxLevel=2
        )
        rt_err = np.linalg.norm(back_pts.reshape(-1, 2) - pts_prev.reshape(-1, 2), axis=1)
        fb_ok = (back_status.reshape(-1) == 1) & (rt_err <= fb_max_px)
        if int((keep & fb_ok).sum()) >= 10:
            keep = keep & fb_ok
    good_new = curr_pts.reshape(-1, 2)[keep]
    good_old = pts_prev.reshape(-1, 2)[keep]
    if of_scale != 1.0:
        good_new = good_new / of_scale
        good_old = good_old / of_scale
    if not len(good_new) > 10:
        return None
    dx, dy = np.median(good_new - good_old, axis=0)
    flow_affine = None
    flow_quality = None
    S_of, of_mask = cv2.estimateAffinePartial2D(
        good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if S_of is not None and np.all(np.isfinite(S_of)):
        flow_affine = S_of
        if of_mask is not None and len(of_mask) > 0:
            flow_quality = float(of_mask.sum()) / len(of_mask) * min(1.0, len(good_new) / 120.0)
    return dx, dy, flow_affine, flow_quality


def _scene(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h // 8, w // 8), dtype=np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _warp(img, dx, dy, angle=0.0):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("fb_check", [False, True])
@pytest.mark.parametrize("half_res", [False, True])
@pytest.mark.parametrize("motion", [(6.0, -3.0, 0.0), (20.0, 11.0, 2.0), (-35.0, 4.0, -4.0)])
def test_matches_former_inline_worker_logic(fb_check, half_res, motion):
    kf_img = _scene(1)
    cur = _warp(kf_img, *motion)
    flow = KeyframeFlow(fb_check=fb_check, fb_max_px=2.0, half_res=half_res)
    flow.anchor(kf_img)
    ref = _reference(
        flow.prev_gray, flow.prev_gray_half, flow.prev_pts.copy(), cur, fb_check, 2.0, half_res
    )
    got = flow.track(cur)
    assert ref is not None and got is not None
    assert got.dx_px == ref[0] and got.dy_px == ref[1]
    assert np.array_equal(got.flow_affine, ref[2])
    assert got.flow_quality == ref[3]
    assert flow.has_anchor  # anchor is never advanced on OF frames


def test_recovers_translation():
    kf_img = _scene(2)
    flow = KeyframeFlow()
    flow.anchor(kf_img)
    m = flow.track(_warp(kf_img, 12.0, -7.0))
    assert abs(m.dx_px - 12.0) < 0.5 and abs(m.dy_px + 7.0) < 0.5
    assert m.flow_quality is not None and m.flow_quality > 0.5


def test_lost_track_drops_anchor():
    kf_img = _scene(3)
    flow = KeyframeFlow()
    flow.anchor(kf_img)
    noise = np.random.default_rng(9).integers(0, 255, kf_img.shape, dtype=np.uint8)
    flat = np.full_like(kf_img, 128)
    for frame in (flat, noise):
        flow.anchor(kf_img)
        res = flow.track(frame)
        if res is None:
            assert not flow.has_anchor


def test_deeper_pyramid_captures_large_displacement():
    # LK tracks FROM the keyframe, so late in the interval the scene can be far
    # away (route3min, 500 m AGL, 50 m/s: ~160 px). The legacy 15/2 pyramid
    # loses such a track; 21/4 recovers it.
    kf_img = _scene(4)
    cur = _warp(kf_img, 60.0, -20.0)  # fine synthetic texture: 60 px already beats 15/2
    legacy = KeyframeFlow(fb_check=True)
    deep = KeyframeFlow(fb_check=True, win_px=21, max_level=4)
    for f in (legacy, deep):
        f.anchor(kf_img)
    m_deep = deep.track(cur)
    assert abs(m_deep.dx_px - 60.0) < 1.0 and abs(m_deep.dy_px + 20.0) < 1.0
    m_legacy = legacy.track(cur)
    assert m_legacy is None or abs(m_legacy.dx_px - 60.0) > 5.0


def test_featureless_anchor_is_not_tracked():
    flow = KeyframeFlow()
    flow.anchor(np.full((360, 640), 128, np.uint8))
    assert flow.track(np.full((360, 640), 128, np.uint8)) is None
    assert not flow.has_anchor
