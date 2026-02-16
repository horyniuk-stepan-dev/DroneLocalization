from PyQt6.QtCore import QThread, pyqtSignal
from src.database.database_builder import DatabaseBuilder
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseGenerationWorker(QThread):
    progress = pyqtSignal(int, str)
    frame_processed = pyqtSignal(int)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, model_manager, config=None):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_manager = model_manager
        self.config = config or {}
        self._is_running = True

        logger.info(f"DatabaseGenerationWorker initialized")
        logger.info(f"Video: {video_path}")
        logger.info(f"Output: {output_path}")

    def run(self):
        logger.info("DatabaseGenerationWorker thread started")

        try:
            self.progress.emit(0, "Ініціалізація бази даних та моделей...")
            logger.info("Initializing database builder...")

            builder = DatabaseBuilder(self.output_path, self.config)

            def update_progress(percent: int):
                if not self._is_running:
                    logger.warning("Database generation interrupted by user")
                    raise InterruptedError("Обробку скасовано користувачем")
                self.progress.emit(percent, f"Обробка кадрів... {percent}%")

            logger.info("Starting video processing...")
            builder.build_from_video(
                video_path=self.video_path,
                model_manager=self.model_manager,
                progress_callback=update_progress
            )

            if self._is_running:
                self.progress.emit(100, "Базу даних успішно створено!")
                logger.success(f"Database generation completed: {self.output_path}")
                self.completed.emit(self.output_path)

        except InterruptedError as e:
            logger.warning(f"Database generation interrupted: {e}")
            self.error.emit(str(e))
        except Exception as e:
            logger.error(f"Database generation failed: {e}", exc_info=True)
            self.error.emit(f"Критична помилка: {str(e)}")

    def stop(self):
        logger.info("Stopping database generation worker...")
        self._is_running = False
