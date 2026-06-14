import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.project_video_source import ProjectVideoSource
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectSettings:
    """Stores metadata about a Drone Localization Project"""

    project_name: str
    created_at: str
    video_path: str

    # Відносні шляхи файлів джерела 'main'
    # Нові проєкти: sources/main/ — всі джерела в єдиній структурі
    database_filename: str = "sources/main/database.h5"
    calibration_filename: str = "sources/main/calibration.json"

    # Мультиджерельна конфігурація (список dict для JSON-серіалізації)
    video_sources: list[dict[str, Any]] = field(default_factory=list)

    # Optional mission parameters inherited from NewMissionDialog
    altitude_m: float = 100.0
    focal_length_mm: float = 13.2
    sensor_width_mm: float = 8.8
    image_width_px: int = 4000

    # Еталонна роздільність відео, з якого побудована БД.
    # Заповнюється автоматично при побудові бази даних.
    # 0 означає "не встановлено".
    ref_frame_width: int = 0
    ref_frame_height: int = 0

    @classmethod
    def from_dict(cls, data: dict):
        # Фільтруємо тільки відомі поля
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        instance = cls(**filtered)

        # Авто-міграція: якщо немає video_sources — створюємо з поточних полів
        if not instance.video_sources and instance.video_path:
            instance.video_sources = [
                ProjectVideoSource(
                    source_id="main",
                    area_id="area_main",
                    video_path=instance.video_path,
                    database_file=instance.database_filename,
                    calibration_file=instance.calibration_filename,
                    description="Auto-migrated from project settings",
                    enabled=True,
                    priority=0,
                ).to_dict()
            ]
            logger.info(
                f"Auto-migrated project to video_sources format "
                f"(source_id='main', db='{instance.database_filename}')"
            )
        return instance

    def source_configs(self) -> list[ProjectVideoSource]:
        """Повертає список ProjectVideoSource з серіалізованих dicts."""
        return [ProjectVideoSource.from_dict(d) for d in self.video_sources]

    def get_enabled_sources(self) -> list[ProjectVideoSource]:
        """Повертає тільки enabled джерела."""
        return [s for s in self.source_configs() if s.enabled]

    def add_source(self, source: ProjectVideoSource) -> None:
        """Додає нове джерело. Перевіряє унікальність source_id."""
        existing_ids = {s["source_id"] for s in self.video_sources}
        if source.source_id in existing_ids:
            raise ValueError(f"Source ID '{source.source_id}' already exists in project")
        self.video_sources.append(source.to_dict())
        logger.info(f"Added video source: {source.source_id} (area: {source.area_id})")

    def remove_source(self, source_id: str) -> bool:
        """Видаляє джерело за source_id. Повертає True якщо знайдено."""
        before = len(self.video_sources)
        self.video_sources = [s for s in self.video_sources if s.get("source_id") != source_id]
        removed = len(self.video_sources) < before
        if removed:
            logger.info(f"Removed video source: {source_id}")
        return removed

    def get_source(self, source_id: str) -> ProjectVideoSource | None:
        """Повертає конфіг за source_id або None."""
        for d in self.video_sources:
            if d.get("source_id") == source_id:
                return ProjectVideoSource.from_dict(d)
        return None

    def update_source(self, source: ProjectVideoSource) -> None:
        """Оновлює існуюче джерело (шукає за source_id)."""
        for i, d in enumerate(self.video_sources):
            if d.get("source_id") == source.source_id:
                self.video_sources[i] = source.to_dict()
                logger.info(f"Updated video source: {source.source_id}")
                return
        raise ValueError(f"Source ID '{source.source_id}' not found in project")


class ProjectManager:
    """
    Manages the workspace directory, saving/loading project configuration,
    and resolving absolute paths to project files.
    """

    def __init__(self):
        self.project_dir: Path | None = None
        self.settings: ProjectSettings | None = None

    @property
    def is_loaded(self) -> bool:
        return self.project_dir is not None and self.settings is not None

    @property
    def project_name(self) -> str:
        return self.settings.project_name if self.settings else "No Project"

    @property
    def database_path(self) -> str | None:
        if not self.is_loaded:
            return None
        return str(self.project_dir / self.settings.database_filename)

    @property
    def calibration_path(self) -> str | None:
        if not self.is_loaded:
            return None
        return str(self.project_dir / self.settings.calibration_filename)

    def create_project(self, workspace_dir: str, mission_data: dict) -> bool:
        """
        Creates a new project folder inside `workspace_dir` using the mission name.
        Saves the project.json.
        """
        try:
            name = mission_data.get("mission_name", "Untitled_Mission")
            # Create a safe folder name
            safe_folder_name = "".join([c if c.isalnum() else "_" for c in name])
            self.project_dir = Path(workspace_dir) / safe_folder_name

            # Ensure the directory exists
            self.project_dir.mkdir(parents=True, exist_ok=True)

            # Створюємо стандартні підпапки
            (self.project_dir / "sources" / "main").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "panoramas").mkdir(exist_ok=True)
            (self.project_dir / "test_photos").mkdir(exist_ok=True)
            (self.project_dir / "test_videos").mkdir(exist_ok=True)

            self.settings = ProjectSettings(
                project_name=name,
                created_at=datetime.now().isoformat(),
                video_path=mission_data.get("video_path", ""),
                altitude_m=mission_data.get("altitude_m", 100.0),
                focal_length_mm=mission_data.get("focal_length_mm", 13.2),
                sensor_width_mm=mission_data.get("sensor_width_mm", 8.8),
                image_width_px=mission_data.get("image_width_px", 4000),
                # Залишаємо значення за замовчуванням (вже sources/main/)
            )

            # Авто-створюємо video_sources для джерела main
            self.settings.video_sources = [
                ProjectVideoSource(
                    source_id="main",
                    area_id="area_main",
                    video_path=mission_data.get("video_path", ""),
                    database_file="sources/main/database.h5",
                    calibration_file="sources/main/calibration.json",
                    description="Primary video source",
                    enabled=True,
                    priority=0,
                ).to_dict()
            ]

            self.save_project()
            logger.info(f"Project created at: {self.project_dir} (sources/main/ structure)")
            return True

        except Exception as e:
            logger.error(
                f"Failed to create project: {e} | "
                f"workspace_dir={workspace_dir}, mission_name={mission_data.get('mission_name', '?')}. "
                f"Check disk permissions and available space.",
                exc_info=True,
            )
            self.project_dir = None
            self.settings = None
            return False

    def load_project(self, project_dir: str) -> bool:
        """Loads an existing project from the given directory path."""
        try:
            dir_path = Path(project_dir)
            if not dir_path.is_dir():
                logger.error(
                    f"Project directory does not exist: {project_dir}. "
                    f"It may have been moved or deleted."
                )
                return False

            json_file = dir_path / "project.json"
            if not json_file.exists():
                logger.error(
                    f"Missing project.json in: {project_dir}. "
                    f"This directory may not be a valid project or project.json was deleted. "
                    f"Available files: {[f.name for f in dir_path.iterdir() if f.is_file()][:10]}"
                )
                return False

            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            self.settings = ProjectSettings.from_dict(data)
            self.project_dir = dir_path

            logger.info(f"Project loaded successfully: {self.settings.project_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to load project: {e} | dir={project_dir}. "
                f"The project.json file may be corrupted or have invalid format.",
                exc_info=True,
            )
            self.project_dir = None
            self.settings = None
            return False

    def save_project(self) -> bool:
        """Saves current ProjectSettings to project.json"""
        if not self.is_loaded:
            return False

        try:
            json_file = self.project_dir / "project.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(
                f"Failed to save project settings: {e} | "
                f"path={self.project_dir / 'project.json'}. "
                f"Check disk permissions and available space.",
                exc_info=True,
            )
            return False
