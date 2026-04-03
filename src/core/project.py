import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectSettings:
    """Stores metadata about a Drone Localization Project"""

    project_name: str
    created_at: str
    video_path: str

    # Internal relative paths
    database_filename: str = "database.h5"
    calibration_filename: str = "calibration.json"

    # Optional mission parameters inherited from NewMissionDialog
    altitude_m: float = 100.0
    focal_length_mm: float = 13.2
    sensor_width_mm: float = 8.8
    image_width_px: int = 4000

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


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

            # Create subfolders for organization
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
            )

            self.save_project()
            logger.info(f"Project created successfully at: {self.project_dir}")
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
