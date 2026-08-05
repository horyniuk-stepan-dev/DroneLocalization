"""VRAM budget: LRU eviction of loaded models.

Extracted from ModelManager. Doesn't import torch directly:
memory check and unloading are performed via injected callbacks.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VramBudget:
    """Tracks last usage timestamp of models and evicts LRU models.

    Args:
        free_vram_mb: callable returning currently free VRAM in MB; float("inf") on CPU.
        unload: callable that unloads a model by name.
        default_required_mb: fallback required free VRAM in MB.
        enabled: False on CPU.
    """

    def __init__(
        self,
        free_vram_mb: Callable[[], float],
        unload: Callable[[str], None],
        default_required_mb: float = 2000.0,
        enabled: bool = True,
    ):
        self._free_vram_mb = free_vram_mb
        self._unload = unload
        self.default_required_mb = float(default_required_mb)
        self.enabled = bool(enabled)

        self.usage: dict[str, float] = {}
        self.pinned: set[str] = set()

    # ------------------------------------------------------------------
    # Usage bookkeeping
    # ------------------------------------------------------------------

    def touch(self, name: str) -> None:
        """Marks model as recently used (LRU timestamp)."""
        self.usage[name] = time.time()

    def forget(self, name: str) -> None:
        """Removes model from usage tracking after unloading."""
        self.usage.pop(name, None)

    def pin(self, names: list[str]) -> None:
        """Pins models to prevent eviction."""
        for n in names:
            self.pinned.add(n)
        logger.info(f"Pinned models: {self.pinned}")

    def unpin_all(self) -> None:
        self.pinned.clear()
        logger.info("Unpinned all models")

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def ensure(self, required_mb: float | None = None, loaded: dict | None = None) -> list[str]:
        """Evicts oldest unpinned models until required_mb free VRAM is available.

        Returns list of evicted model names.
        """
        if not self.enabled:
            return []

        req = self.default_required_mb if required_mb is None else float(required_mb)
        evicted: list[str] = []

        while self._free_vram_mb() < req and loaded:
            non_pinned = {k: v for k, v in self.usage.items() if k not in self.pinned}
            if not non_pinned:
                logger.warning("All models pinned, cannot free VRAM. Risk of OOM.")
                return evicted
            least = min(non_pinned, key=lambda k: non_pinned[k])
            self._unload(least)
            self.usage.pop(least, None)
            evicted.append(least)

        return evicted
