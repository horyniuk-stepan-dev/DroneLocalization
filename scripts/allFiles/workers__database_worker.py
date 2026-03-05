import os
from PyQt6.QtCore import QThread, pyqtSignal
from src.database.database_builder import DatabaseBuilder
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _CancelledError(Exception):
    """Raised internally when user requests cancellation via stop()."""


class DatabaseGenerationWorker(QThread):
    progress         = pyqtSignal(int, str)
    frame_processed  = pyqtSignal(int)
    completed        = pyqtSignal(str)
    error            = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, model_manager, config=None):
        super().__init__()
        self.video_path    = video_path
        self.output_path   = output_path
        self.model_manager = model_manager
        self.config        = config or {}
        self._is_running   = False
        logger.info(f"DatabaseGenerationWorker | {video_path} → {output_path}")

    def run(self):
        self._is_running = True
        logger.info("DatabaseGenerationWorker started")

        try:
            self.progress.emit(0, "Ініціалізація бази даних та моделей...")
            builder = DatabaseBuilder(self.output_path, self.config)

            def update_progress(percent: int):
                if not self._is_running:
                    raise _CancelledError("Обробку скасовано користувачем")
                self.progress.emit(percent, f"Обробка кадрів... {percent}%")

            builder.build_from_video(
                video_path=self.video_path,
                model_manager=self.model_manager,
                progress_callback=update_progress,
            )

            logger.success(f"Database generation completed: {self.output_path}")
            self.completed.emit(self.output_path)

        except _CancelledError as e:
            logger.warning(f"Generation cancelled: {e}")
            # Remove incomplete database to prevent corrupt loads
            if os.path.exists(self.output_path):
                try:
                    os.remove(self.output_path)
                    logger.info(f"Incomplete database removed: {self.output_path}")
                except OSError as rm_err:
                    logger.warning(f"Could not remove incomplete file: {rm_err}")
            self.error.emit(str(e))

        except Exception as e:
            logger.error(f"Database generation failed: {e}", exc_info=True)
            self.error.emit(f"Критична помилка: {e}")

    def stop(self):
        logger.info("Stopping DatabaseGenerationWorker...")
        self._is_running = False
        self.wait()
