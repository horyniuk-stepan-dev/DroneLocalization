"""HARDENING P3-15 (first slice): deterministic fault injection for the
soak / fault-injection harness.

The payload runs unattended for hours in a contested environment; the failure
modes that matter are *runtime* ones the unit suite never sees — a decoder
handing back a garbled frame, a link stall that freezes the stream, a mid-flight
end-of-stream, a decode exception. This module manufactures those on demand so
``scripts/soak_test.py`` can drive the real pipeline through them and watch the
operating-state machine (P1-9/10) and latency tracker (P1-8) react.

Two layers, deliberately split:

* ``FaultInjector`` — pure logic. Given ``(ret, frame, frame_idx)`` it returns a
  possibly-transformed ``(ret, frame)`` plus an injected ``delay_sec``. Seeded
  RNG makes a run byte-for-byte reproducible. No cv2, no I/O — unit-testable
  anywhere numpy is available.
* ``FaultInjectingVideoSource`` — a thin ``VideoSource`` subclass that reads a
  real clip (optionally looping it for a long soak) and pipes each frame through
  a ``FaultInjector``. It plugs into the existing seam: ``RealtimeTrackingWorker``
  already accepts a pre-built ``VideoSource`` object, so nothing on the
  production hot path changes.

Nothing here is wired into the application. It is a test tool, invoked only by
the harness — so it needs no config flag and carries no risk to a stock run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.video.video_source import VideoSource, VideoSourceConfig


class FaultType(str, Enum):
    """A single injectable fault. String-valued so profiles round-trip to JSON."""

    NONE = "none"  # pass the frame through unchanged
    CORRUPT = "corrupt"  # same shape, garbled pixels (decoder bit-rot)
    BLACK = "black"  # zeroed frame (dropped/blanked capture)
    FREEZE = "freeze"  # repeat the last good frame (stalled stream)
    SHAPE = "shape"  # wrong dimensions (partial/resynced decode)
    DELAY = "delay"  # frame is fine, but arrives late (link/decode stall)
    EOF = "eof"  # (False, None): end-of-stream / connection lost
    EXCEPTION = "exception"  # raise inside read(): a hard decode error


class FaultInjectionError(RuntimeError):
    """Raised by an injected EXCEPTION fault to simulate a hard decode failure."""


@dataclass
class FaultProfile:
    """What faults to inject and when.

    Precedence per frame, highest first:
      1. ``schedule[frame_idx]`` — an exact single-frame event.
      2. the first matching ``windows`` entry — an inclusive ``[start, end]`` burst.
      3. ``probabilities`` — one seeded RNG roll walked over the cumulative mass.

    ``probabilities`` values must sum to <= 1.0; the remaining mass is NONE.
    ``delay_sec`` is how long a DELAY fault stalls; ``corrupt_full`` picks
    whole-frame noise vs. a garbled patch.
    """

    name: str = "none"
    enabled: bool = True
    probabilities: dict[FaultType, float] = field(default_factory=dict)
    schedule: dict[int, FaultType] = field(default_factory=dict)
    windows: list[tuple[int, int, FaultType]] = field(default_factory=list)
    delay_sec: float = 0.5
    corrupt_full: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        # Normalize keys that arrived as plain strings (e.g. from JSON) first,
        # so the sum check below sees canonical FaultType keys.
        self.probabilities = {FaultType(k): float(v) for k, v in self.probabilities.items()}
        self.schedule = {int(k): FaultType(v) for k, v in self.schedule.items()}
        self.windows = [(int(s), int(e), FaultType(t)) for s, e, t in self.windows]
        total = sum(self.probabilities.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"FaultProfile '{self.name}': probabilities sum to {total:.3f} > 1.0")

    @classmethod
    def from_dict(cls, d: dict) -> FaultProfile:
        return cls(**d)


class FaultInjector:
    """Applies a :class:`FaultProfile` to a frame stream, deterministically.

    Stateless except for: the seeded RNG, the last good frame (for FREEZE), and
    per-type hit counters (for the harness report).
    """

    def __init__(self, profile: FaultProfile):
        self.profile = profile
        self._rng = np.random.default_rng(profile.seed)
        self._last_good: np.ndarray | None = None
        self.counts: dict[FaultType, int] = {t: 0 for t in FaultType}

    # -- selection -----------------------------------------------------------

    def _select(self, frame_idx: int) -> FaultType:
        """Decide which fault (if any) fires on this frame."""
        if not self.profile.enabled:
            return FaultType.NONE

        scheduled = self.profile.schedule.get(frame_idx)
        if scheduled is not None:
            return scheduled

        for start, end, ftype in self.profile.windows:
            if start <= frame_idx <= end:
                return ftype

        if not self.profile.probabilities:
            return FaultType.NONE

        # One roll, walked over the cumulative probability mass. Deterministic
        # given the seed and the number of prior rolls.
        u = float(self._rng.random())
        cum = 0.0
        for ftype, p in self.profile.probabilities.items():
            cum += p
            if u < cum:
                return ftype
        return FaultType.NONE

    # -- application ---------------------------------------------------------

    def apply(
        self, ret: bool, frame: np.ndarray | None, frame_idx: int
    ) -> tuple[bool, np.ndarray | None, float]:
        """Transform one read result. Returns ``(ret, frame, delay_sec)``.

        Raises :class:`FaultInjectionError` for an EXCEPTION fault so the caller
        exercises its real error path.
        """
        ftype = self._select(frame_idx)
        self.counts[ftype] += 1

        # Advance the last-good pointer so FREEZE has something to repeat — but
        # NOT on a freeze itself: a stalled stream means no new frame arrived, so
        # the pointer must stay on the previous frame.
        if ret and frame is not None and ftype != FaultType.FREEZE:
            self._last_good = frame

        if ftype == FaultType.NONE:
            return ret, frame, 0.0
        if ftype == FaultType.EOF:
            return False, None, 0.0
        if ftype == FaultType.EXCEPTION:
            raise FaultInjectionError(f"injected decode error at frame {frame_idx}")
        if ftype == FaultType.DELAY:
            return ret, frame, float(self.profile.delay_sec)

        # The remaining faults mutate pixels; with no frame to mutate they no-op.
        if frame is None:
            return ret, frame, 0.0

        if ftype == FaultType.BLACK:
            return ret, np.zeros_like(frame), 0.0
        if ftype == FaultType.FREEZE:
            repeat = self._last_good if self._last_good is not None else frame
            return ret, repeat.copy(), 0.0
        if ftype == FaultType.CORRUPT:
            return ret, self._corrupt(frame), 0.0
        if ftype == FaultType.SHAPE:
            return ret, self._reshape(frame), 0.0

        return ret, frame, 0.0

    # -- pixel mutators ------------------------------------------------------

    def _corrupt(self, frame: np.ndarray) -> np.ndarray:
        """Garble the frame while keeping a valid shape/dtype."""
        if self.profile.corrupt_full:
            return self._rng.integers(0, 256, size=frame.shape, dtype=frame.dtype)
        out = frame.copy()
        h = frame.shape[0]
        # A garbled band across the middle third — a partial-decode look.
        y0, y1 = h // 3, (2 * h) // 3
        out[y0:y1] = self._rng.integers(0, 256, size=out[y0:y1].shape, dtype=frame.dtype)
        return out

    @staticmethod
    def _reshape(frame: np.ndarray) -> np.ndarray:
        """Return a wrong-sized frame (half resolution) to test shape handling."""
        h, w = frame.shape[:2]
        nh, nw = max(1, h // 2), max(1, w // 2)
        return frame[:nh, :nw].copy()

    # -- reporting -----------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Per-fault-type hit counts, only for types that fired at least once."""
        return {t.value: c for t, c in self.counts.items() if c and t != FaultType.NONE}


# --- built-in profiles ------------------------------------------------------
# Named presets the harness exposes via --profile. Each is deterministic given
# its seed; override the seed on the CLI to explore other draws.

PROFILES: dict[str, FaultProfile] = {
    # Baseline: no faults. Isolates pipeline behaviour on a clean loop.
    "clean": FaultProfile(name="clean", enabled=True),
    # Light mix — a soak that occasionally perturbs the stream.
    "smoke": FaultProfile(
        name="smoke",
        probabilities={FaultType.CORRUPT: 0.01, FaultType.BLACK: 0.01, FaultType.DELAY: 0.01},
        delay_sec=0.3,
        seed=1,
    ),
    # Heavy corruption — stress the frame-content robustness path.
    "corruption": FaultProfile(
        name="corruption",
        probabilities={FaultType.CORRUPT: 0.15, FaultType.SHAPE: 0.05},
        seed=2,
    ),
    # A sustained freeze burst + jitter. NOTE: a frozen frame is still a *valid*
    # frame, so localization keeps succeeding on it and the state stays TRACKING
    # (confirmed empirically). This profile stresses freeze/jitter robustness, it
    # does NOT drive LOST — use "blackout" for that.
    "stall": FaultProfile(
        name="stall",
        windows=[(200, 320, FaultType.FREEZE)],
        probabilities={FaultType.DELAY: 0.02},
        delay_sec=0.5,
        seed=3,
    ),
    # A sustained blackout: ~400 frames of unlocalizable black frames. Sized to
    # keep the fix clock starved for well over fix_stale_sec (default 3s) even if
    # black frames fail fast, so the state machine SHOULD go TRACKING -> LOST and
    # then recover once real frames resume. (A 140-frame window sat right at the
    # threshold and did not trip — see max_fix_age_sec in the soak report.)
    "blackout": FaultProfile(
        name="blackout",
        windows=[(200, 600, FaultType.BLACK)],
        seed=6,
    ),
    # Intermittent link loss — an injected EOF. NOTE: EOF ends the stream (the
    # worker breaks and goes IDLE), so this tests graceful shutdown on a mid-
    # stream drop, NOT the LOST path.
    "link-loss": FaultProfile(
        name="link-loss",
        schedule={500: FaultType.EOF},
        seed=4,
    ),
    # A single long link STALL: the stream stays alive but goes silent for ~6s
    # (one frame arrives 6s late). No new fix for well over fix_stale_sec (3s),
    # so the state SHOULD go TRACKING -> LOST and then recover when frames
    # resume. This is the real absence-of-fix test (blackout is bad *content*;
    # this is *no* content for a while).
    "linkstall": FaultProfile(
        name="linkstall",
        schedule={400: FaultType.DELAY},
        delay_sec=6.0,
        seed=7,
    ),
    # Rare hard decode error — exercises the worker/supervisor exception path.
    "decode-error": FaultProfile(
        name="decode-error",
        schedule={750: FaultType.EXCEPTION},
        seed=5,
    ),
}


def get_profile(name_or_dict) -> FaultProfile:
    """Resolve a profile from a preset name or an inline dict/FaultProfile."""
    if isinstance(name_or_dict, FaultProfile):
        return name_or_dict
    if isinstance(name_or_dict, dict):
        return FaultProfile.from_dict(name_or_dict)
    if name_or_dict in PROFILES:
        return PROFILES[name_or_dict]
    raise KeyError(f"unknown fault profile: {name_or_dict!r} (known: {sorted(PROFILES)})")


class FaultInjectingVideoSource(VideoSource):
    """A ``VideoSource`` that reads a real clip through a :class:`FaultInjector`.

    Plugs straight into ``RealtimeTrackingWorker`` (which accepts a pre-built
    ``VideoSource``). For a long soak it can loop the underlying clip so a short
    video sustains an hours-long run; a deliberately-scheduled EOF fault still
    ends the stream regardless of looping.
    """

    def __init__(
        self,
        source: str,
        profile: FaultProfile | str | dict,
        *,
        loop: bool = True,
        config: VideoSourceConfig | None = None,
    ):
        cfg = config or VideoSourceConfig(source=str(source))
        super().__init__(cfg)
        self.injector = FaultInjector(get_profile(profile))
        self.loop = loop
        self.frame_idx = -1
        self.loops_completed = 0
        self._sleep = time.sleep  # bound so tests can stub out real waiting

    def _read_looping(self) -> tuple[bool, np.ndarray | None]:
        """Underlying read, re-seeking to the start on a natural EOF when looping."""
        ret, frame = super().read()
        if not ret and self.loop and self._cap is not None:
            import cv2

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.loops_completed += 1
            ret, frame = super().read()
        return ret, frame

    def read(self) -> tuple[bool, np.ndarray | None]:  # type: ignore[override]
        self.frame_idx += 1
        ret, frame = self._read_looping()
        ret, frame, delay = self.injector.apply(ret, frame, self.frame_idx)
        if delay > 0:
            self._sleep(delay)
        return ret, frame

    def report(self) -> dict:
        """Injection stats for the harness report."""
        return {
            "profile": self.injector.profile.name,
            "frames_read": self.frame_idx + 1,
            "loops_completed": self.loops_completed,
            "faults_injected": self.injector.summary(),
        }
