import shutil
import tempfile
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.database.database_builder import DatabaseBuilder
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseGenerationWorker(QThread):
    """Background thread for generating HDF5 database (XFeat + DINOv2)."""

    progress = pyqtSignal(int, str)
    frame_processed = pyqtSignal(int)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        video_path: str,
        output_path: str,
        model_manager,
        config=None,
        project_manager=None,
        required_frame_ids: set[int] | None = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_manager = model_manager
        self.config = config or {}
        self.project_manager = project_manager
        self.required_frame_ids = {int(frame_id) for frame_id in (required_frame_ids or set())}
        self._is_running = True

        logger.info("DatabaseGenerationWorker initialized")
        logger.info(f"Video: {video_path}")
        logger.info(f"Output: {output_path}")
        if self.required_frame_ids:
            logger.info(
                "Exact calibration anchors required by rebuild: "
                f"{sorted(self.required_frame_ids)}"
            )

    @staticmethod
    def _artifact_pairs(staging_dir: Path, output_path: Path) -> list[tuple[Path, Path]]:
        """Artifacts produced by DatabaseBuilder and their committed locations."""
        return [
            (staging_dir / output_path.name, output_path),
            (staging_dir / "vectors.lance", output_path.parent / "vectors.lance"),
            (
                staging_dir / "database_keypoints.mp4",
                output_path.parent / "database_keypoints.mp4",
            ),
        ]

    @classmethod
    def _promote_staged_database(cls, staging_dir: Path, output_path: Path) -> None:
        """Commit a complete rebuild while preserving the previous DB on failure."""
        pairs = cls._artifact_pairs(staging_dir, output_path)
        required_staged = pairs[:2]
        missing = [str(src) for src, _ in required_staged if not src.exists()]
        if missing:
            raise RuntimeError(f"Database build completed without required artifacts: {missing}")

        backup_dir = Path(
            tempfile.mkdtemp(prefix=".database-backup-", dir=str(output_path.parent))
        )
        backed_up: list[tuple[Path, Path]] = []
        promoted: list[Path] = []
        try:
            for _src, destination in pairs:
                if destination.exists():
                    backup = backup_dir / destination.name
                    destination.replace(backup)
                    backed_up.append((backup, destination))

            for source, destination in pairs:
                if source.exists():
                    source.replace(destination)
                    promoted.append(destination)
        except Exception:
            for destination in reversed(promoted):
                if destination.is_dir():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink(missing_ok=True)
            for backup, destination in reversed(backed_up):
                if backup.exists():
                    backup.replace(destination)
            raise
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)

    def run(self):
        logger.info("DatabaseGenerationWorker thread started")

        staging_dir: Path | None = None
        try:
            self.progress.emit(0, "Initializing database (XFeat + DINOv2)...")
            logger.info("Initializing database builder...")

            final_output = Path(self.output_path).resolve()
            final_output.parent.mkdir(parents=True, exist_ok=True)
            staging_dir = Path(
                tempfile.mkdtemp(prefix=".database-rebuild-", dir=str(final_output.parent))
            )
            staged_output = staging_dir / final_output.name
            logger.info(f"Transactional rebuild staging directory: {staging_dir}")

            builder = DatabaseBuilder(
                output_path=str(staged_output),
                config=self.config,
            )

            def update_progress(percent: int):
                if not self._is_running:
                    logger.warning("Database generation interrupted by user")
                    raise InterruptedError("Processing cancelled by user")
                self.progress.emit(percent, f"Processing frames... {percent}%")

            logger.info("Starting video processing...")
            builder.build_from_video(
                video_path=self.video_path,
                model_manager=self.model_manager,
                progress_callback=update_progress,
                project_manager=self.project_manager,
                required_frame_ids=self.required_frame_ids,
            )

            if self._is_running:
                self._promote_staged_database(staging_dir, final_output)
                self.progress.emit(100, "Database successfully generated!")
                logger.success(f"Database generation completed: {self.output_path}")
                self.completed.emit(self.output_path)

        except InterruptedError as e:
            logger.warning(
                f"Database generation interrupted by user: {e} | video={self.video_path}"
            )
            self.cancelled.emit()
        except Exception as e:
            logger.error(
                f"Database generation failed: {e} | "
                f"video={self.video_path}, output={self.output_path}. "
                f"Check that the video file is valid (MP4/H.264) and disk has sufficient space.",
                exc_info=True,
            )
            self.error.emit(f"Critical error: {str(e)}")
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)

    def stop(self):
        logger.info("Stopping DatabaseGenerationWorker...")
        self._is_running = False
