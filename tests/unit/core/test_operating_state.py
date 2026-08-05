"""HARDENING P1-10 follow-up: anchor-staleness DEGRADED signal.

Closes the content-blind gap (docs/THREAT_MODEL_AND_SECURITY.md §4a): the state
machine must be able to demote TRACKING -> DEGRADED when the pipeline has been
coasting on optical-flow propagation without a fresh keyframe anchor, even while
the fix clock stays fresh. All new behavior is gated on
`network_api.propagation_stale_sec > 0` (default 0 = off = old behavior).

Runs on the Windows side (imports PyQt6). The broker is constructed with
`enabled=False` so no WS/REST servers or event loop start.
"""

from __future__ import annotations

import time

import pytest

from config import NetworkApiConfig
from src.network.coordinates_broker import CoordinatesBroker


def _broker(**net_kwargs) -> CoordinatesBroker:
    """A tracking-active broker with no network services running."""
    cfg = NetworkApiConfig(enabled=False, **net_kwargs)
    b = CoordinatesBroker(cfg)
    b.is_tracking_active = True
    return b


def _tracking_with(broker, *, fix_age: float, anchor_age: float | None) -> None:
    """Plant the two staleness clocks at explicit ages (seconds in the past)."""
    now = time.monotonic()
    broker._last_fix_mono = now - fix_age
    broker._last_anchor_mono = None if anchor_age is None else now - anchor_age


# --- flag off: old behavior preserved ---------------------------------------


def test_flag_off_never_degraded_from_anchor():
    b = _broker(propagation_stale_sec=0.0)
    _tracking_with(b, fix_age=0.1, anchor_age=999.0)  # anchor ancient
    assert b.get_operating_state()["op_state"] == "TRACKING"


# --- flag on: anchor-staleness demotes TRACKING -> DEGRADED ------------------


def test_fresh_anchor_stays_tracking():
    b = _broker(propagation_stale_sec=5.0)
    _tracking_with(b, fix_age=0.1, anchor_age=0.5)
    assert b.get_operating_state()["op_state"] == "TRACKING"


def test_stale_anchor_degrades():
    b = _broker(propagation_stale_sec=1.0)
    _tracking_with(b, fix_age=0.1, anchor_age=2.0)  # fix fresh, anchor stale
    st = b.get_operating_state()
    assert st["op_state"] == "DEGRADED"
    assert st["anchor_age_sec"] == pytest.approx(2.0, abs=0.2)


def test_no_anchor_yet_degrades_when_enabled():
    # Emitting OF fixes but a fresh keyframe anchor has never landed.
    b = _broker(propagation_stale_sec=1.0)
    _tracking_with(b, fix_age=0.1, anchor_age=None)
    assert b.get_operating_state()["op_state"] == "DEGRADED"


# --- precedence: LOST (fix clock) outranks anchor-staleness ------------------


def test_stale_fix_is_lost_not_degraded():
    b = _broker(propagation_stale_sec=1.0, fix_stale_sec=3.0)
    _tracking_with(b, fix_age=5.0, anchor_age=6.0)  # both stale -> LOST wins
    assert b.get_operating_state()["op_state"] == "LOST"


# --- clock plumbing ----------------------------------------------------------


def test_on_anchor_fix_refreshes_clock():
    b = _broker(propagation_stale_sec=1.0)
    _tracking_with(b, fix_age=0.1, anchor_age=2.0)
    assert b.get_operating_state()["op_state"] == "DEGRADED"
    b.on_anchor_fix()  # a fresh keyframe anchor lands
    assert b.get_operating_state()["op_state"] == "TRACKING"


def test_set_tracking_active_clears_anchor():
    b = _broker(propagation_stale_sec=1.0)
    b._last_anchor_mono = time.monotonic()
    b.set_tracking_active(True)  # new session: no anchor yet
    assert b._last_anchor_mono is None


def test_idle_when_not_tracking():
    b = _broker(propagation_stale_sec=1.0)
    b.is_tracking_active = False
    assert b.get_operating_state()["op_state"] == "IDLE"
