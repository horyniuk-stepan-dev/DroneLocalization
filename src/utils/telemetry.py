import atexit
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Where the default report goes; tests point it at a temporary folder (root
# conftest.py) so a pytest run never overwrites the application's report.
TELEMETRY_DIR_ENV = "DRONELOC_TELEMETRY_DIR"
# Per-run copies kept in <logs>/telemetry/ (oldest removed beyond this count).
MAX_RUN_REPORTS = 200


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

    @staticmethod
    def _default_report_paths() -> list[Path]:
        """``<logs>/telemetry_report.json`` (latest run) + a per-run copy.

        The per-run copy (``<logs>/telemetry/telemetry_<time>_<pid>.json``) keeps
        earlier runs as baselines; before, every process overwrote the one file.
        ``DRONELOC_TELEMETRY_DIR`` replaces ``<logs>``.
        """
        override = os.environ.get(TELEMETRY_DIR_ENV)
        if override:
            logs = Path(override)
        else:
            from config import user_data_dir

            logs = user_data_dir() / "logs"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_copy = logs / "telemetry" / f"telemetry_{stamp}_{os.getpid()}.json"
        return [logs / "telemetry_report.json", run_copy]

    @staticmethod
    def _prune_run_reports(folder: Path, keep: int = MAX_RUN_REPORTS) -> None:
        reports = sorted(folder.glob("telemetry_*.json"))
        for old in reports[: max(0, len(reports) - keep)]:
            try:
                old.unlink()
            except OSError:
                pass

    def dump_report(self, path: str | None = None):
        if not self.stats:
            return

        targets = [Path(path)] if path is not None else self._default_report_paths()
        summary = self.get_summary()
        for target in targets:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4)
                logger.info(
                    f"Telemetry report saved to {target} ({len(self.stats)} stages profiled)"
                )
            except Exception as e:
                logger.error(f"Failed to save telemetry report {target}: {e}")
        if path is None and len(targets) > 1:
            self._prune_run_reports(targets[1].parent)


# Singleton instance
Telemetry = _TelemetryTracker()


@atexit.register
def _save_telemetry_on_exit():
    Telemetry.dump_report()
