"""Tests for the P3-15 fault injector (src/utils/fault_injection.py).

Two layers, matching the module's own split:
  * The FaultInjector / FaultProfile logic is pure numpy — always runs here.
  * FaultInjectingVideoSource subclasses VideoSource (cv2); its non-cv2 wiring
    (frame indexing, looping bookkeeping, delay sleep) is exercised with a stub
    that bypasses the real capture, so it runs without a video file or cv2 open.
"""

import numpy as np
import pytest

from src.utils.fault_injection import (
    PROFILES,
    FaultInjectionError,
    FaultInjector,
    FaultProfile,
    FaultType,
    get_profile,
)


def _frame(h=8, w=8):
    return np.full((h, w, 3), 128, dtype=np.uint8)


# --- profile validation ------------------------------------------------------


def test_probabilities_over_one_rejected():
    with pytest.raises(ValueError):
        FaultProfile(name="bad", probabilities={FaultType.CORRUPT: 0.6, FaultType.BLACK: 0.6})


def test_from_dict_coerces_string_keys():
    p = FaultProfile.from_dict(
        {
            "name": "j",
            "probabilities": {"corrupt": 0.5},
            "schedule": {"3": "black"},
            "windows": [[1, 2, "freeze"]],
        }
    )
    assert p.probabilities == {FaultType.CORRUPT: 0.5}
    assert p.schedule == {3: FaultType.BLACK}
    assert p.windows == [(1, 2, FaultType.FREEZE)]


def test_all_builtin_profiles_resolve():
    for name in PROFILES:
        assert isinstance(get_profile(name), FaultProfile)
    with pytest.raises(KeyError):
        get_profile("does-not-exist")


# --- selection precedence ----------------------------------------------------


def test_schedule_beats_window_beats_probability():
    p = FaultProfile(
        name="prec",
        schedule={5: FaultType.EOF},
        windows=[(0, 10, FaultType.FREEZE)],
        probabilities={FaultType.CORRUPT: 1.0},
    )
    inj = FaultInjector(p)
    assert inj._select(5) == FaultType.EOF  # schedule wins
    assert inj._select(3) == FaultType.FREEZE  # window wins over probability
    assert inj._select(20) == FaultType.CORRUPT  # probability outside window


def test_disabled_profile_is_passthrough():
    inj = FaultInjector(
        FaultProfile(name="off", enabled=False, probabilities={FaultType.CORRUPT: 1.0})
    )
    ret, frame, delay = inj.apply(True, _frame(), 0)
    assert ret and delay == 0.0
    assert np.array_equal(frame, _frame())


# --- determinism -------------------------------------------------------------


def test_same_seed_same_sequence():
    def prof():
        return FaultProfile(
            name="d",
            seed=42,
            probabilities={FaultType.CORRUPT: 0.3, FaultType.BLACK: 0.3},
        )

    a = FaultInjector(prof())
    b = FaultInjector(prof())
    seq_a = [a._select(i) for i in range(200)]
    seq_b = [b._select(i) for i in range(200)]
    assert seq_a == seq_b


def test_probability_distribution_is_reasonable():
    inj = FaultInjector(FaultProfile(name="d", seed=7, probabilities={FaultType.CORRUPT: 0.25}))
    hits = sum(inj._select(i) == FaultType.CORRUPT for i in range(10000))
    assert 2200 < hits < 2800  # ~25% of 10k, generous band


# --- per-fault transforms ----------------------------------------------------


def test_black_zeroes_frame():
    inj = FaultInjector(FaultProfile(name="b", schedule={0: FaultType.BLACK}))
    ret, frame, delay = inj.apply(True, _frame(), 0)
    assert ret and delay == 0.0 and not frame.any()


def test_corrupt_keeps_shape_changes_content():
    inj = FaultInjector(FaultProfile(name="c", schedule={0: FaultType.CORRUPT}, seed=1))
    src = _frame()
    ret, frame, _ = inj.apply(True, src, 0)
    assert frame.shape == src.shape and frame.dtype == src.dtype
    assert not np.array_equal(frame, src)


def test_shape_fault_changes_dimensions():
    inj = FaultInjector(FaultProfile(name="s", schedule={0: FaultType.SHAPE}))
    ret, frame, _ = inj.apply(True, _frame(8, 8), 0)
    assert frame.shape[:2] == (4, 4)


def test_freeze_repeats_last_good_frame():
    inj = FaultInjector(FaultProfile(name="f", schedule={1: FaultType.FREEZE}))
    good = _frame()
    good[0, 0] = [1, 2, 3]
    inj.apply(True, good, 0)  # frame 0: good, remembered
    _, frozen, _ = inj.apply(True, _frame(), 1)  # frame 1: freeze
    assert np.array_equal(frozen, good)


def test_delay_returns_delay_not_mutation():
    inj = FaultInjector(FaultProfile(name="d", schedule={0: FaultType.DELAY}, delay_sec=0.25))
    ret, frame, delay = inj.apply(True, _frame(), 0)
    assert ret and delay == 0.25 and np.array_equal(frame, _frame())


def test_eof_returns_false_none():
    inj = FaultInjector(FaultProfile(name="e", schedule={0: FaultType.EOF}))
    ret, frame, delay = inj.apply(True, _frame(), 0)
    assert ret is False and frame is None and delay == 0.0


def test_exception_fault_raises():
    inj = FaultInjector(FaultProfile(name="x", schedule={0: FaultType.EXCEPTION}))
    with pytest.raises(FaultInjectionError):
        inj.apply(True, _frame(), 0)


def test_pixel_fault_on_none_frame_is_noop():
    # A BLACK/CORRUPT scheduled on a frame that is already None must not crash.
    inj = FaultInjector(FaultProfile(name="n", schedule={0: FaultType.BLACK}))
    ret, frame, delay = inj.apply(True, None, 0)
    assert frame is None and delay == 0.0


# --- counters ----------------------------------------------------------------


def test_summary_counts_only_fired_faults():
    inj = FaultInjector(FaultProfile(name="w", windows=[(0, 2, FaultType.BLACK)]))
    for i in range(5):
        inj.apply(True, _frame(), i)
    summary = inj.summary()
    assert summary == {"black": 3}  # frames 0,1,2
    assert "none" not in summary  # NONE never reported


# --- VideoSource wrapper wiring (no cv2 open needed) --------------------------


class _StubSource:
    """Stands in for the VideoSource base: yields N frames then EOF."""

    def __init__(self, n):
        self._n = n
        self._i = 0
        self._cap = object()  # non-None so looping bookkeeping engages

    def read(self):
        if self._i >= self._n:
            return False, None
        f = np.full((4, 4, 3), self._i % 256, dtype=np.uint8)
        self._i += 1
        return True, f


def _make_wrapper(profile, n_frames, loop):
    """Build a FaultInjectingVideoSource without running VideoSource.__init__
    (which would open a real capture)."""
    from src.utils.fault_injection import FaultInjectingVideoSource

    w = FaultInjectingVideoSource.__new__(FaultInjectingVideoSource)
    stub = _StubSource(n_frames)
    # Route the base-class read()/_cap that the wrapper calls into the stub.
    w._stub = stub
    w._cap = stub._cap
    import types

    w._base_read = types.MethodType(lambda self: self._stub.read(), w)
    from src.utils.fault_injection import FaultInjector, get_profile

    w.injector = FaultInjector(get_profile(profile))
    w.loop = loop
    w.frame_idx = -1
    w.loops_completed = 0
    slept = []
    w._sleep = slept.append
    w._slept = slept
    return w


def test_wrapper_indexes_and_reports(monkeypatch):
    from src.utils import fault_injection as fi

    w = _make_wrapper("clean", n_frames=3, loop=False)
    # Patch super().read() by pointing _read_looping's super call at the stub.
    monkeypatch.setattr(fi.VideoSource, "read", lambda self: self._stub.read())

    frames = []
    for _ in range(3):
        ret, frame = w.read()
        assert ret
        frames.append(frame)
    ret, frame = w.read()
    assert ret is False  # EOF, looping off

    rep = w.report()
    assert rep["profile"] == "clean"
    assert rep["frames_read"] == 4
    assert rep["faults_injected"] == {}


def test_wrapper_loops_on_natural_eof(monkeypatch):
    from src.utils import fault_injection as fi

    w = _make_wrapper("clean", n_frames=2, loop=True)

    def _stub_read(self):
        return self._stub.read()

    def _reseek(idx):
        w._stub._i = 0  # cv2 CAP_PROP_POS_FRAMES=0 re-seek

    monkeypatch.setattr(fi.VideoSource, "read", _stub_read)
    w._cap = type("C", (), {"set": staticmethod(lambda *a: _reseek(0))})()

    # Read past the 2-frame clip; looping should re-seek and keep yielding.
    results = [w.read()[0] for _ in range(4)]
    assert all(results)
    assert w.loops_completed >= 1


def test_wrapper_sleeps_on_delay(monkeypatch):
    from src.utils import fault_injection as fi

    w = _make_wrapper(
        FaultProfile(name="d", schedule={0: FaultType.DELAY}, delay_sec=0.5),
        n_frames=3,
        loop=False,
    )
    monkeypatch.setattr(fi.VideoSource, "read", lambda self: self._stub.read())

    w.read()
    assert w._slept == [0.5]


# --- harness report logic (scripts/soak_test.py) -----------------------------


def _load_summarize():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("soak_test", root / "scripts" / "soak_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.summarize_transitions


def test_summarize_empty():
    summarize = _load_summarize()
    r = summarize([])
    assert r["transitions"] == [] and r["lost_episodes"] == [] and r["samples"] == 0


def test_summarize_lost_episode_with_recovery():
    summarize = _load_summarize()
    events = [
        (0.0, "ACQUIRING"),
        (1.0, "TRACKING"),
        (5.0, "LOST"),
        (8.0, "TRACKING"),  # recovered after 3s
        (10.0, "TRACKING"),
    ]
    r = summarize(events)
    assert [t["state"] for t in r["transitions"]] == ["ACQUIRING", "TRACKING", "LOST", "TRACKING"]
    assert r["lost_episodes"] == [{"at_sec": 5.0, "duration_sec": 3.0}]
    # Dwell: ACQUIRING 1s, TRACKING (1->5)+(8->10)=6s, LOST 3s.
    assert r["dwell_sec"]["LOST"] == 3.0
    assert r["dwell_sec"]["TRACKING"] == 6.0


def test_summarize_unrecovered_lost_marked():
    summarize = _load_summarize()
    events = [(0.0, "TRACKING"), (2.0, "LOST"), (5.0, "LOST")]
    r = summarize(events)
    assert r["lost_episodes"] == [{"at_sec": 2.0, "duration_sec": 3.0, "recovered": False}]
