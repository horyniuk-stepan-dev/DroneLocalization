import json
from pathlib import Path
from datetime import datetime
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProjectRegistry:
    """
    Централізований реєстр усіх проєктів.
    Зберігає шляхи та метадані в JSON файлі у домашній директорії.
    """

    def __init__(self):
        self._registry_dir = Path.home() / ".drone_localizer"
        self._registry_path = self._registry_dir / "projects.json"
        self._projects: list[dict] = []
        self._load()

    def _load(self):
        """Завантажити реєстр з диска."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._projects = data.get('projects', [])
                logger.debug(f"ProjectRegistry loaded: {len(self._projects)} projects")
            except Exception as e:
                logger.warning(f"Failed to load project registry: {e}")
                self._projects = []
        else:
            self._projects = []

    def _save(self):
        """Зберегти реєстр на диск."""
        try:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump({'projects': self._projects}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save project registry: {e}")

    def _find_index(self, project_dir: str) -> int:
        """Знайти індекс проєкту за шляхом."""
        norm = str(Path(project_dir).resolve())
        for i, p in enumerate(self._projects):
            if str(Path(p['path']).resolve()) == norm:
                return i
        return -1

    def register(self, project_dir: str, name: str, video_path: str = ""):
        """Додати або оновити проєкт у реєстрі."""
        idx = self._find_index(project_dir)
        now = datetime.now().isoformat()

        entry = {
            'name': name,
            'path': str(Path(project_dir).resolve()),
            'video_path': video_path,
            'created_at': now,
            'last_opened': now,
            'has_database': (Path(project_dir) / "database.h5").exists(),
            'has_calibration': (Path(project_dir) / "calibration.json").exists(),
        }

        if idx >= 0:
            # Зберігаємо оригінальну дату створення
            entry['created_at'] = self._projects[idx].get('created_at', now)
            self._projects[idx] = entry
        else:
            self._projects.append(entry)

        self._save()
        logger.info(f"Project registered: {name} at {project_dir}")

    def unregister(self, project_dir: str):
        """Видалити проєкт з реєстру (файли НЕ видаляються)."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            removed = self._projects.pop(idx)
            self._save()
            logger.info(f"Project unregistered: {removed['name']}")

    def update_last_opened(self, project_dir: str):
        """Оновити дату останнього відкриття."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            self._projects[idx]['last_opened'] = datetime.now().isoformat()
            self._save()

    def refresh_status(self, project_dir: str):
        """Оновити статус наявності БД та калібрації."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            p = Path(project_dir)
            self._projects[idx]['has_database'] = (p / "database.h5").exists()
            self._projects[idx]['has_calibration'] = (p / "calibration.json").exists()
            self._save()

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Повернути останні відкриті проєкти (відсортовані за датою)."""
        valid = [p for p in self._projects if Path(p['path']).is_dir()]
        valid.sort(key=lambda p: p.get('last_opened', ''), reverse=True)
        return valid[:limit]

    def get_all(self) -> list[dict]:
        """Повернути всі зареєстровані проєкти."""
        return list(self._projects)
