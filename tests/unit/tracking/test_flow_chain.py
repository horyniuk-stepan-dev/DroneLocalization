"""FlowChain: keep optical flow going through failed keyframes (2026-10-28)."""

import numpy as np

from src.tracking.keyframe_flow import FlowChain, KeyframeFlow


class FakeLocalizer:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def rebase_optical_flow_anchor(self, affine):
        self.calls.append(np.asarray(affine))
        return self.ok


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((120, 160)) * 255).astype(np.uint8)


S = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -3.0]])


def test_disabled_is_legacy():
    chain, loc, flow = FlowChain(enabled=False), FakeLocalizer(), KeyframeFlow()
    chain.note_keyframe_success(0.0)
    chain.note_of_success(_img(), S, 0.5)
    assert chain.try_rebase(loc, flow, 1.0) is False and not loc.calls


def test_rebase_moves_anchor_to_last_good_frame():
    chain, loc, flow = FlowChain(enabled=True), FakeLocalizer(), KeyframeFlow()
    flow.anchor(_img(1), t=0.0)
    chain.note_keyframe_success(0.0)
    good = _img(2)
    chain.note_of_success(good, S, 0.83)
    assert chain.try_rebase(loc, flow, 1.0) is True
    assert np.array_equal(loc.calls[0], S)
    assert flow.anchor_t == 0.83 and flow.prev_gray is good
    assert chain.links == 1 and chain.last_good is None  # consumed: relative to old anchor


def test_no_rebase_without_fresh_good_of():
    chain, loc, flow = FlowChain(enabled=True, max_good_age_s=1.0), FakeLocalizer(), KeyframeFlow()
    chain.note_keyframe_success(0.0)
    assert chain.try_rebase(loc, flow, 1.0) is False  # nothing to chain on
    chain.note_of_success(_img(), S, 0.2)
    assert chain.try_rebase(loc, flow, 1.5) is False  # last good OF too old
    assert not loc.calls


def test_chain_stops_after_max_s_without_a_real_fix():
    chain, loc, flow = FlowChain(enabled=True, max_s=30.0), FakeLocalizer(), KeyframeFlow()
    chain.note_keyframe_success(0.0)
    chain.note_of_success(_img(), S, 30.5)
    assert chain.try_rebase(loc, flow, 31.0) is False


def test_refused_rebase_keeps_flow_untouched():
    chain, loc, flow = FlowChain(enabled=True), FakeLocalizer(ok=False), KeyframeFlow()
    flow.anchor(_img(1), t=0.0)
    chain.note_keyframe_success(0.0)
    chain.note_of_success(_img(2), S, 0.83)
    assert chain.try_rebase(loc, flow, 1.0) is False
    assert flow.anchor_t == 0.0


def test_rebase_composition_equals_direct_flow():
    """H_new = H Sc S1^-1 Sc^-1; a second flow S2 from the new anchor must land
    where the composed flow S2 S1 from the old anchor lands."""
    H = np.array([[0.9, 0.1, 30.0], [-0.1, 0.9, 50.0], [1e-5, 0.0, 1.0]])
    scale = 0.75
    Sc, Sc_inv = np.diag([scale, scale, 1.0]), np.diag([1 / scale, 1 / scale, 1.0])
    S1 = np.vstack([S, [0, 0, 1]])
    S2 = np.array([[np.cos(0.1), -np.sin(0.1), 5.0], [np.sin(0.1), np.cos(0.1), 2.0], [0, 0, 1]])
    H_new = H @ Sc @ np.linalg.inv(S1) @ Sc_inv
    c = np.array([640.0, 360.0, 1.0])
    via_chain = H_new @ Sc @ np.linalg.inv(S2) @ c
    direct = H @ Sc @ np.linalg.inv(S2 @ S1) @ c
    assert np.allclose(via_chain / via_chain[2], direct / direct[2])
