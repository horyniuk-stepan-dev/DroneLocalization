"""HARDENING P3-15: soak / fault-injection harness.

Drives the REAL headless localization pipeline through a fault-injected video
stream and watches how the hardening controls react:

  * operating-state machine (P1-9/10) — did FREEZE/EOF drive it to LOST, and did
    it recover to TRACKING afterwards?
  * per-frame latency (P1-8) — what are the p95/p99/max tails under DELAY faults?
  * process survival — did a CORRUPT/SHAPE/EXCEPTION frame crash the worker?

For a long soak, a short clip is looped so it can run for hours; sprinkle faults
via ``--profile`` and watch for slow leaks / drift / wedging.

Usage (run on the Windows/GPU box — needs torch, PyQt6, a built project):

    python scripts/soak_test.py --project <dir> --source flight.mp4 --profile stall
    python scripts/soak_test.py --project <dir> --source flight.mp4 \
        --profile smoke --duration 3600 --report soak_report.json

Profiles: clean | smoke | corruption | stall | blackout | linkstall | link-loss | decode-error
(see src/utils/fault_injection.py). ``--profile`` also accepts a path to a JSON
file describing a custom FaultProfile.

Only the fault injectors are unit-tested (tests/test_fault_injection.py); this
harness itself must be run against the real pipeline — it is not exercised by
the pure-Python suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- pure report logic (unit-tested) ----------------------------------------


def summarize_transitions(events: list[tuple[float, str]]) -> dict:
    """Reduce a time-ordered list of (elapsed_sec, op_state) samples to a report.

    Returns dwell time per state, the ordered list of state changes, and the
    LOST episodes (enter time + duration) — the numbers that answer "did it lose
    lock, and how long until it recovered?".
    """
    if not events:
        return {"transitions": [], "dwell_sec": {}, "lost_episodes": [], "samples": 0}

    transitions: list[dict] = []
    dwell: dict[str, float] = {}
    lost_episodes: list[dict] = []

    prev_t, prev_state = events[0]
    transitions.append({"at_sec": round(prev_t, 2), "state": prev_state})
    lost_start = prev_t if prev_state == "LOST" else None

    for t, state in events[1:]:
        dwell[prev_state] = dwell.get(prev_state, 0.0) + (t - prev_t)
        if state != prev_state:
            transitions.append({"at_sec": round(t, 2), "state": state})
            if state == "LOST":
                lost_start = t
            elif prev_state == "LOST" and lost_start is not None:
                lost_episodes.append(
                    {"at_sec": round(lost_start, 2), "duration_sec": round(t - lost_start, 2)}
                )
                lost_start = None
        prev_t, prev_state = t, state

    # Close an open final dwell + an unterminated LOST episode.
    last_t = events[-1][0]
    dwell[prev_state] = dwell.get(prev_state, 0.0) + (last_t - prev_t)
    if lost_start is not None:
        lost_episodes.append(
            {
                "at_sec": round(lost_start, 2),
                "duration_sec": round(last_t - lost_start, 2),
                "recovered": False,
            }
        )

    return {
        "transitions": transitions,
        "dwell_sec": {k: round(v, 2) for k, v in dwell.items()},
        "lost_episodes": lost_episodes,
        "samples": len(events),
    }


# --- live monitor ------------------------------------------------------------


class _StateMonitor(threading.Thread):
    """Polls the broker's operating state + the worker's latency tracker."""

    def __init__(self, runner, interval: float):
        super().__init__(daemon=True)
        self._runner = runner
        self._interval = interval
        # NB: not `_stop` — that name shadows threading.Thread's own internal
        # _stop() method and breaks join().
        self._stop_evt = threading.Event()
        self._t0 = time.monotonic()
        self.events: list[tuple[float, str]] = []
        self.latency: dict = {}
        # Diagnostics: how close the fix ever got to going stale, and the
        # threshold it is measured against. Reveals whether an unlocalizable
        # burst actually starved the fix clock or just flew past below threshold.
        self.max_fix_age_sec = 0.0
        self.stale_after_sec = None
        # §4a: how stale the fresh-keyframe-anchor clock ever got. Reveals whether
        # an OF-coast (e.g. blackout) drove TRACKING -> DEGRADED via anchor age.
        self.max_anchor_age_sec = 0.0
        # Worst fix quality seen (from broker._last_position). Tells us what
        # inliers/confidence a degraded burst actually produces — i.e. whether
        # the DEGRADED thresholds could ever catch it.
        self.min_inliers: int | None = None
        self.min_confidence: float | None = None

    def run(self) -> None:
        # Wait for the worker to come up (built inside runner.run()).
        while not self._stop_evt.is_set():
            if getattr(self._runner, "tracking_worker", None) is not None:
                break
            time.sleep(0.05)

        last_state = None
        while not self._stop_evt.wait(self._interval):
            broker = getattr(self._runner, "coordinates_broker", None)
            if broker is None:
                continue
            op = broker.get_operating_state()
            state = op.get("op_state", "UNKNOWN")
            age = op.get("last_fix_age_sec")
            if age is not None:
                self.max_fix_age_sec = max(self.max_fix_age_sec, age)
            if op.get("stale_after_sec") is not None:
                self.stale_after_sec = op["stale_after_sec"]
            anchor_age = op.get("anchor_age_sec")
            if anchor_age is not None:
                self.max_anchor_age_sec = max(self.max_anchor_age_sec, anchor_age)
            lp = getattr(broker, "_last_position", None)
            if lp:
                inl = lp.get("inliers")
                conf = lp.get("confidence")
                if inl is not None:
                    self.min_inliers = (
                        inl if self.min_inliers is None else min(self.min_inliers, inl)
                    )
                if conf is not None:
                    self.min_confidence = (
                        conf if self.min_confidence is None else min(self.min_confidence, conf)
                    )
            elapsed = time.monotonic() - self._t0
            # Record first sample and every change; dense enough for dwell math.
            if state != last_state or not self.events:
                self.events.append((elapsed, state))
                last_state = state

            worker = getattr(self._runner, "tracking_worker", None)
            tracker = getattr(worker, "_latency_tracker", None) if worker else None
            if tracker is not None:
                self.latency = tracker.stats()

    def stop(self) -> None:
        self._stop_evt.set()


# --- harness -----------------------------------------------------------------


def _force_latency_stats_on() -> None:
    """Enable per-frame latency capture for the run (harness-local; not persisted)."""
    import config

    perf = config.APP_CONFIG.setdefault("models", {}).setdefault("performance", {})
    perf["log_latency_stats"] = True
    perf.setdefault("latency_log_interval", 200)


def _load_profile(spec: str):
    """A preset name, or a path to a JSON FaultProfile."""
    from src.utils.fault_injection import get_profile

    p = Path(spec)
    if p.is_file():
        return get_profile(json.loads(p.read_text(encoding="utf-8")))
    return get_profile(spec)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, help="Project directory")
    parser.add_argument("--source", required=True, help="Video source (file/URL)")
    parser.add_argument("--profile", default="smoke", help="Fault profile name or JSON path")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = run until the stream ends)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Do not loop the clip (a short video then ends the run)",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.25, help="Operating-state poll interval, seconds"
    )
    parser.add_argument("--report", default="", help="Write the JSON report to this path")
    parser.add_argument(
        "--degraded-min-inliers",
        type=int,
        default=0,
        help="Flag a fix below this many inliers as DEGRADED (0 = off; tests the "
        "state machine's DEGRADED branch against the content-blind gap)",
    )
    parser.add_argument(
        "--degraded-min-confidence",
        type=float,
        default=0.0,
        help="Flag a fix below this confidence as DEGRADED (0 = off)",
    )
    parser.add_argument(
        "--propagation-stale-sec",
        type=float,
        default=0.0,
        help="Flag TRACKING as DEGRADED after this many seconds with no fresh "
        "keyframe anchor (0 = off; tests the §4a anchor-staleness gate that "
        "closes the content-blind blackout gap)",
    )
    args = parser.parse_args()

    # WinError 1114 workaround (mirrors main.py): torch's DLLs must load BEFORE
    # PoseLib / pyproj / cv2, which `import config` (inside the calls below) pulls
    # in. Import torch first. Kept inside main() so the module's pure report
    # logic stays importable without torch (tests/test_fault_injection.py).
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: F401

    _force_latency_stats_on()

    from src.core.headless_runner import HeadlessRunner
    from src.utils.fault_injection import FaultInjectingVideoSource
    from src.utils.logging_utils import get_logger

    logger = get_logger("soak_test")

    # Apply DEGRADED thresholds to the broker config BEFORE HeadlessRunner builds
    # the broker. Harness-local; not persisted to user_config.json.
    if args.degraded_min_inliers or args.degraded_min_confidence or args.propagation_stale_sec:
        import config

        config.APP_SETTINGS.network_api.degraded_min_inliers = args.degraded_min_inliers
        config.APP_SETTINGS.network_api.degraded_min_confidence = args.degraded_min_confidence
        config.APP_SETTINGS.network_api.propagation_stale_sec = args.propagation_stale_sec
        logger.info(
            f"DEGRADED thresholds set | min_inliers={args.degraded_min_inliers} "
            f"min_confidence={args.degraded_min_confidence} "
            f"propagation_stale_sec={args.propagation_stale_sec}"
        )

    profile = _load_profile(args.profile)
    source = FaultInjectingVideoSource(args.source, profile, loop=not args.no_loop)
    logger.info(
        f"Soak harness | profile={profile.name} | loop={not args.no_loop} | "
        f"duration={args.duration or 'until-eos'}s"
    )

    runner = HeadlessRunner(args.project, source)
    monitor = _StateMonitor(runner, args.sample_interval)

    def _deadline_stopper():
        """Stop the worker after --duration once it is actually running."""
        while getattr(runner, "tracking_worker", None) is None:
            time.sleep(0.05)
        time.sleep(args.duration)
        logger.info(f"Duration {args.duration}s reached — stopping worker")
        try:
            runner.tracking_worker.stop()
        except Exception as e:
            logger.warning(f"stop() failed: {e}")

    monitor.start()
    if args.duration > 0:
        threading.Thread(target=_deadline_stopper, daemon=True).start()

    run_error = None
    started = time.monotonic()
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    except Exception as e:  # a fault that crashed the pipeline is a RESULT, not a bug
        run_error = f"{type(e).__name__}: {e}"
        logger.error(f"Pipeline raised during soak: {run_error}")
    finally:
        monitor.stop()
        monitor.join(timeout=2.0)

    wall = time.monotonic() - started
    report = {
        "profile": profile.name,
        "wall_sec": round(wall, 1),
        "run_error": run_error,
        "injection": source.report(),
        "operating_state": summarize_transitions(monitor.events),
        "fix_clock": {
            "max_fix_age_sec": round(monitor.max_fix_age_sec, 3),
            "stale_after_sec": monitor.stale_after_sec,
            "max_anchor_age_sec": round(monitor.max_anchor_age_sec, 3),
            "propagation_stale_sec": args.propagation_stale_sec,
        },
        "fix_quality": {
            "min_inliers": monitor.min_inliers,
            "min_confidence": (
                round(monitor.min_confidence, 3) if monitor.min_confidence is not None else None
            ),
            "degraded_min_inliers": args.degraded_min_inliers,
            "degraded_min_confidence": args.degraded_min_confidence,
        },
        "latency": monitor.latency,
    }

    print("\n===== SOAK REPORT =====")
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"Wrote report to {args.report}")

    # Non-zero exit if the pipeline crashed or never left ACQUIRING/IDLE.
    reached = set(report["operating_state"]["dwell_sec"])
    healthy = run_error is None and ("TRACKING" in reached or args.duration == 0)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
