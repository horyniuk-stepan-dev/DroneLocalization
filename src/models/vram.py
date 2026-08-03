"""VRAM budget: LRU-виселення завантажених моделей (IMPROVEMENT_PLAN п.1.6).

Витягнуто з ``ModelManager`` без зміни семантики. Модуль свідомо НЕ імпортує
torch: скільки пам'яті вільно й як саме вивантажити модель, він питає через
інжектовані колбеки (``free_vram_mb`` / ``unload``). Завдяки цьому політика
виселення (кого і скільки разів вивантажувати, що робити, коли все закріплене)
тестується в пісочниці без GPU — раніше вона була неперевірною взагалі.

Блокування лишається у ``ModelManager``: ``_model_lock`` має охоплювати
load+evict атомарно, тож клас навмисно не має власного локу, щоб не створювати
другу, слабшу дисципліну блокування поруч із наявною.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VramBudget:
    """Веде час останнього використання моделей і виселяє найдавніші.

    Args:
        free_vram_mb: скільки VRAM вільно зараз; ``float("inf")`` на CPU.
        unload: вивантажує модель за іменем (звільняє посилання + empty_cache).
        default_required_mb: скільки вимагати, якщо виклик не вказав явно.
        enabled: False на CPU — виселяти нема сенсу й нема чого міряти.
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
        """Позначає модель як щойно використану (LRU-мітка)."""
        self.usage[name] = time.time()

    def forget(self, name: str) -> None:
        """Прибирає модель з обліку (викликається після фактичного вивантаження)."""
        self.usage.pop(name, None)

    def pin(self, names: list[str]) -> None:
        """Закріплює моделі — виселенню не підлягають."""
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
        """Виселяє найдавніші незакріплені моделі, доки не звільниться ``required_mb``.

        ``loaded`` — актуальний словник завантажених моделей; цикл зупиняється,
        коли він порожній, інакше на GPU, де пам'ять зайняв хтось інший, вийшов
        би нескінченний цикл.

        Повертає список імен, які було вивантажено (для логів і тестів).
        Якщо всі кандидати закріплені — попереджає й виходить: закріплення
        сильніше за бюджет, це свідомий вибір виклику ``pin()``.
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
            # Гарантія прогресу: якщо ``unload`` не зняв LRU-мітку (модель уже
            # зникла зі словника завантажених, і ``_unload_model_unsafe``
            # мовчки вийшов), наступна ітерація обрала б те саме ім'я — і цикл
            # крутився б вічно, бо вільна пам'ять не зростає. У попередній
            # версії всередині ModelManager цей шлях був так само зациклюваний.
            self.usage.pop(least, None)
            evicted.append(least)

        return evicted
