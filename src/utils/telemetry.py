import atexit
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _TelemetryTracker:
    def __init__(self):
        self.stats = defaultdict(
            lambda: {"calls": 0, "total_time": 0.0, "min_time": float("inf"), "max_time": 0.0}
        )

    class ProfilerContext:
        def __init__(self, tracker: "_TelemetryTracker", stage_name: str):
            self.tracker = tracker
            self.stage_name = stage_name
            self.start_time = 0.0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.perf_counter() - self.start_time
            st = self.tracker.stats[self.stage_name]
            st["calls"] += 1
            st["total_time"] += elapsed
            if elapsed < st["min_time"]:
                st["min_time"] = elapsed
            if elapsed > st["max_time"]:
                st["max_time"] = elapsed

        def __call__(self, func: Callable) -> Callable:
            """Allows using the context manager as a decorator."""

            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                with self:
                    return func(*args, **kwargs)

            return wrapper

    def profile(self, stage_name: str):
        """
        Returns a context manager / decorator for profiling a stage.
        Usage:
            with Telemetry.profile("yolo"):
                do_something()

            @Telemetry.profile("feature_extraction")
            def do_something_else():
                pass
        """
        return self.ProfilerContext(self, stage_name)

    def get_summary(self) -> dict:
        summary = {}
        for stage, st in self.stats.items():
            if st["calls"] == 0:
                continue
            avg = st["total_time"] / st["calls"]
            summary[stage] = {
                "calls": st["calls"],
                "total_time_s": round(st["total_time"], 4),
                "avg_time_s": round(avg, 4),
                "min_time_s": round(st["min_time"], 4),
                "max_time_s": round(st["max_time"], 4),
            }
        return summary

    def dump_report(self, path="logs/telemetry_report.json"):
        if not self.stats:
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_summary(), f, indent=4)
            logger.info(f"Telemetry report saved to {path} ({len(self.stats)} stages profiled)")
        except Exception as e:
            logger.error(f"Failed to save telemetry report: {e}")


# Singleton instance
Telemetry = _TelemetryTracker()


@atexit.register
def _save_telemetry_on_exit():
    Telemetry.dump_report()
