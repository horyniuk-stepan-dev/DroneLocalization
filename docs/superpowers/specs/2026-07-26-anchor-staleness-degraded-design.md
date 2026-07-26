# Anchor-staleness DEGRADED signal — design

**Date:** 2026-07-26
**Branch:** `hardening/p0`
**Status:** approved, pre-implementation
**Closes:** the content-blind gap recorded in `docs/THREAT_MODEL_AND_SECURITY.md` §4a
and memory `lost-detector-content-blind.md`.

## Problem

The operating-state machine (`CoordinatesBroker.get_operating_state`, P1-9/10) has
two ways to leave `TRACKING`:

- `LOST` — no fix at all for `fix_stale_sec` (a time clock on any emitted fix).
- `DEGRADED` — the last fix's `inliers`/`confidence` fell below a floor.

Measured gap (P3-15 soak, `blackout` profile, 401 black frames): neither fires.
The pipeline keeps emitting positions by **optical-flow propagation** from the
last good keyframe, and those OF fixes **inherit that keyframe's quality
metadata** — so `inliers` stayed frozen at 1318 and the fix clock stayed fresh
(0.67 s vs 0.59 s clean baseline). A camera that degrades to garbage *while still
delivering frames* therefore keeps reporting `TRACKING` with a possibly-wrong
fix. This is a silent-failure mode for a GPS-denied payload.

Root cause: both existing signals inspect the **propagated** fix, not whether a
**fresh keyframe anchor** has actually landed recently.

## Chosen approach (B): anchor-staleness clock

The pipeline already distinguishes a fresh keyframe localization from an OF-coast:
`tracking_worker.py:473` reads `loc_result.get("is_of")` (set `True` only by
`localizer.localize_optical_flow`, localizer.py:959). We add a **second staleness
clock**, symmetric to the existing `LOST` clock, over the *fresh keyframe anchor*:

> Tracking, but no fresh keyframe anchor for longer than `propagation_stale_sec`
> ⇒ `DEGRADED`.

During blackout, keyframe localizations fail (a black frame has no features to
match the DB — confirmed by the frozen inlier count), so the anchor clock ages
past threshold and trips `DEGRADED`, while OF keeps the *fix* clock fresh so it
does not (yet) reach `LOST`.

### Why not approach A (frame-content pixel gate)

Variance/frame-diff heuristics (near-zero variance ⇒ black, identical ⇒ frozen)
need per-deployment threshold tuning and false-positive on legitimately dark
scenes or a static hover. Approach B reuses the pipeline's own trusted success
signal, needs no pixel thresholds, and expresses the operationally meaningful
truth ("no real re-localization in N seconds").

## Implementation

Additive, flag-gated, default OFF (= current behavior). Four touch points:

1. **`src/workers/tracking_worker.py`** — new zero-arg signal
   `anchor_fix = pyqtSignal()`. Emit it right after the existing
   `location_found.emit(...)` **only when the fix is a fresh keyframe anchor**,
   i.e. `not loc_result.get("is_of")`. (The `location_found` signal is unchanged —
   changing its arity would break all three existing `connect` sites and the
   `@pyqtSlot` decorators.)

2. **`src/network/coordinates_broker.py`** —
   - `__init__`: `self._last_anchor_mono: float | None = None`.
   - `set_tracking_active(True)`: reset `_last_anchor_mono = None` (mirrors
     `_last_fix_mono`; a new session has no anchor yet).
   - New slot `@pyqtSlot()` `on_anchor_fix(self)`: `_last_anchor_mono = time.monotonic()`.
   - `get_operating_state`: in the `else` branch (fix recent, not `LOST`), after
     the existing inlier/confidence `DEGRADED` check, add: if
     `propagation_stale_sec > 0` and (`_last_anchor_mono is None` or
     `now - _last_anchor_mono > propagation_stale_sec`) ⇒ `DEGRADED`. Include
     `anchor_age_sec` in the returned dict for observability.

3. **Wiring** — connect `anchor_fix → on_anchor_fix` next to the existing
   `location_found → on_location_found` in:
   - `src/core/headless_runner.py:165`
   - `src/gui/mixins/tracking_mixin.py:167` (broker connection)

4. **`config/app.py`** — `NetworkApiConfig`: new field
   `propagation_stale_sec: float = 0.0` (0 = off), documented next to the
   `degraded_*` fields.

### Precedence (unchanged ordering)

`IDLE` > `ACQUIRING` > `LOST` (fix clock) > `DEGRADED` (inliers/confidence OR
anchor-staleness) > `TRACKING`. Anchor-staleness only ever demotes `TRACKING`
to `DEGRADED`; it never masks `LOST`.

### Timing note

With `tracking.keyframe_interval` = 5 at 30 fps input, a fresh anchor lands
roughly every few fixes; the clean soak baseline emitted a fix ~every 0.6 s.
`propagation_stale_sec` must sit comfortably above the observed fresh-anchor
interval (a few× margin) or it false-positives during healthy tracking — hence
default 0 (off); the operator sets it after observing their own anchor cadence.

## Testing

- **Unit (`tests/test_operating_state.py`, new)** — construct
  `CoordinatesBroker(NetworkApiConfig(enabled=False))` (no servers), drive the
  monotonic clocks directly:
  - flag off (`propagation_stale_sec=0`) ⇒ never `DEGRADED` from anchor
    (old behavior preserved).
  - flag on, fresh anchor recent ⇒ `TRACKING`.
  - flag on, anchor older than threshold, fix recent ⇒ `DEGRADED`.
  - anchor stale AND fix stale ⇒ `LOST` (precedence holds).
  - `on_anchor_fix()` refreshes the clock; `set_tracking_active(True)` clears it.
- **Integration (soak harness)** — `blackout` profile with
  `propagation_stale_sec` enabled ⇒ run reaches `DEGRADED`; `clean` profile with
  the same threshold ⇒ stays `TRACKING`. Add a `--propagation-stale-sec` flag to
  `scripts/soak_test.py` (harness-local, like the existing `--degraded-*`).

## Residual risk

A *partially* degraded frame that still produces a false keyframe DB match
(`is_of=False`) would reset the anchor clock and evade `DEGRADED`. Unlikely for
true blackout (no features; confirmed by the frozen inlier count), and a pixel
gate (approach A) would not reliably catch that case either. Documented, not
mitigated here.

## Out of scope

Frame-content pixel heuristics (approach A); any change to the hot path when the
flag is off; resource-level soak scenarios (GPU-OOM/thermal — separate P3-15
follow-up).
