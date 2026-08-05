"""Frame processing latency monitoring module.

Computes frame processing duration statistics (p50, p95, max) to detect
performance dips and estimate the pipeline's computational latency.
"""

import math
from collections import deque
from typing import Any


class LatencyTracker:
    """Rolling window of per-frame latencies with percentile reporting."""

    def __init__(self, window: int = 300, log_interval: int = 100, logger: Any = None):
        self._samples_ms: deque[float] = deque(maxlen=max(1, window))
        self._log_interval = max(1, log_interval)
        self._count = 0
        self._logger = logger

    def record(self, seconds: float) -> None:
        """Record one frame's processing time (in seconds); log on interval."""
        self._samples_ms.append(seconds * 1000.0)
        self._count += 1
        if self._logger is not None and self._count % self._log_interval == 0:
            self._logger.info(self.format_stats())

    def stats(self) -> dict:
        """Nearest-rank percentiles over the current window (empty if no data)."""
        if not self._samples_ms:
            return {}
        ordered = sorted(self._samples_ms)
        n = len(ordered)

        def pct(p: float) -> float:
            rank = min(n - 1, max(0, math.ceil(p / 100.0 * n) - 1))
            return ordered[rank]

        return {
            "n": n,
            "p50_ms": round(pct(50), 1),
            "p95_ms": round(pct(95), 1),
            "p99_ms": round(pct(99), 1),
            "max_ms": round(ordered[-1], 1),
        }

    def format_stats(self) -> str:
        st = self.stats()
        if not st:
            return "latency: (no samples)"
        return (
            f"latency ms | p50={st['p50_ms']} p95={st['p95_ms']} "
            f"p99={st['p99_ms']} max={st['max_ms']} (n={st['n']})"
        )
