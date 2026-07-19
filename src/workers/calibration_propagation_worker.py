"""Тонка QThread-обгортка над :class:`PropagationPipeline`.

Уся математика графової пропагації живе в Qt-free ядрі
``src/workers/propagation_pipeline.py``. Тут — лише міст між Qt-сигналами
(``progress`` / ``completed`` / ``error``) і колбеками пайплайна + запуск у
окремому потоці. Поведінка збережена 1:1: ті самі сигнали й тексти прогресу,
``stop()``, ``start()`` / ``isRunning()`` (з QThread), і прямий синхронний
виклик ``_propagate()`` (шлях бенчмарка/тестів).
"""

from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger
from src.workers.propagation_pipeline import PropagationPipeline

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """QThread-адаптер: емітить сигнали з колбеків PropagationPipeline."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, database, calibration, matcher, config=None):
        super().__init__()
        self._pipeline = PropagationPipeline(
            database,
            calibration,
            matcher,
            config=config,
            progress_callback=self.progress.emit,
            error_callback=self.error.emit,
            completed_callback=self.completed.emit,
        )

    def stop(self):
        self._pipeline.stop()

    def _propagate(self):
        # Прямий синхронний виклик (бенчмарк/тести) — делегуємо в пайплайн.
        # Сигнали летять через колбеки, під'єднані в __init__.
        self._pipeline._propagate()

    def run(self):
        try:
            self._pipeline._propagate()
        except Exception as e:
            logger.error(
                f"Graph propagation failed: {e} | "
                f"num_anchors={len(self._pipeline.calibration.anchors)}, "
                f"db_frames={self._pipeline.database.get_num_frames()}",
                exc_info=True,
            )
            self.error.emit(str(e))
