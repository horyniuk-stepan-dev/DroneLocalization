import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ImportResult:
    """Outcome of ``ProjectRegistry.import_projects`` (resolved folder paths)."""

    added: list[str] = field(default_factory=list)
    already_known: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class ProjectRegistry:
    """Centralized project registry storing metadata in JSON file under home directory."""

    # Folders never searched for projects (in addition to hidden ".*" ones).
    _SCAN_SKIP_DIRS = frozenset({"__pycache__", "node_modules"})

    def __init__(self):
        self._registry_dir = Path.home() / ".drone_localizer"
        self._registry_path = self._registry_dir / "projects.json"
        self._projects: list[dict] = []
        self._load()

    def _load(self):
        """Load registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._projects = data.get("projects", [])
                logger.debug(f"ProjectRegistry loaded: {len(self._projects)} projects")
            except Exception as e:
                logger.warning(
                    f"Failed to load project registry: {e} | "
                    f"path={self._registry_path}. "
                    f"Registry will be reset. This is safe but recent project history will be lost."
                )
                self._projects = []
        else:
            self._projects = []

    def _save(self):
        """Save registry to disk."""
        try:
            from src.utils.atomic_io import atomic_write_text

            self._registry_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                str(self._registry_path),
                json.dumps({"projects": self._projects}, indent=2, ensure_ascii=False),
            )
        except Exception as e:
            logger.error(
                f"Failed to save project registry: {e} | "
                f"path={self._registry_path}. "
                f"Check disk permissions for {self._registry_dir}.",
                exc_info=True,
            )

    def _find_index(self, project_dir: str) -> int:
        """Find project index by path."""
        norm = str(Path(project_dir).resolve())
        for i, p in enumerate(self._projects):
            if str(Path(p["path"]).resolve()) == norm:
                return i
        return -1

    def register(self, project_dir: str, name: str, video_path: str = ""):
        """Register or update a project in registry."""
        idx = self._find_index(project_dir)
        now = datetime.now().isoformat()
        p = Path(project_dir)

        entry = {
            "name": name,
            "path": str(Path(project_dir).resolve()),
            "video_path": video_path,
            "created_at": now,
            "last_opened": now,
            "has_database": self._check_has_database(p),
            "has_calibration": self._check_has_calibration(p),
        }

        if idx >= 0:
            # Preserve original creation date
            entry["created_at"] = self._projects[idx].get("created_at", now)
            self._projects[idx] = entry
        else:
            self._projects.append(entry)

        self._save()
        logger.info(f"Project registered: {name} at {project_dir}")

    def unregister(self, project_dir: str):
        """Unregister a project from registry (files are preserved)."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            removed = self._projects.pop(idx)
            self._save()
            logger.info(f"Project unregistered: {removed['name']}")

    def update_last_opened(self, project_dir: str):
        """Update last opened timestamp."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            self._projects[idx]["last_opened"] = datetime.now().isoformat()
            self._save()

    def refresh_status(self, project_dir: str):
        """Refresh database and calibration existence status."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            p = Path(project_dir)
            self._projects[idx]["has_database"] = self._check_has_database(p)
            self._projects[idx]["has_calibration"] = self._check_has_calibration(p)
            self._save()

    @staticmethod
    def _check_has_database(project_dir: Path) -> bool:
        """Checks for database presence in project root or sources subdirectories."""
        if (project_dir / "database.h5").exists():
            return True
        sources_dir = project_dir / "sources"
        if sources_dir.is_dir():
            for sub in sources_dir.iterdir():
                if sub.is_dir() and (sub / "database.h5").exists():
                    return True
        return False

    @staticmethod
    def _check_has_calibration(project_dir: Path) -> bool:
        """Checks for calibration presence in project root or sources subdirectories."""
        if (project_dir / "calibration.json").exists():
            return True
        sources_dir = project_dir / "sources"
        if sources_dir.is_dir():
            for sub in sources_dir.iterdir():
                if sub.is_dir() and (sub / "calibration.json").exists():
                    return True
        return False

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Returns recent projects sorted by last opened date."""
        valid = [p for p in self._projects if Path(p["path"]).is_dir()]
        valid.sort(key=lambda p: p.get("last_opened", ""), reverse=True)
        return valid[:limit]

    def get_all(self) -> list[dict]:
        """Returns all registered projects."""
        return list(self._projects)

    # ── Import of existing project folders ─────────────────────────────────────

    @classmethod
    def find_projects(cls, root: str | Path, max_depth: int = 2) -> list[Path]:
        """Project folders (holding a ``project.json``) at or below ``root``.

        ``root`` itself counts when it is a project. A project folder is not
        searched further: its ``sources/`` are layers, not projects. Hidden and
        unreadable folders are skipped, so a broad root cannot abort the scan.
        ``max_depth`` counts levels below ``root`` (2 = children and grandchildren).
        """
        found: list[Path] = []
        pending: list[tuple[Path, int]] = [(Path(root), 0)]
        while pending:
            folder, depth = pending.pop()
            try:
                if (folder / "project.json").is_file():
                    found.append(folder.resolve())
                    continue
                if depth >= max_depth:
                    continue
                entries = list(folder.iterdir())
            except OSError as e:
                logger.debug(f"Project scan skipped {folder}: {e}")
                continue
            for child in entries:
                if child.name.startswith(".") or child.name in cls._SCAN_SKIP_DIRS:
                    continue
                try:
                    if child.is_dir():
                        pending.append((child, depth + 1))
                except OSError:
                    continue
        return sorted(found, key=lambda p: str(p).casefold())

    @staticmethod
    def _read_manifest(project_dir: Path) -> dict:
        """Plaintext project.json fields, or {} if encrypted or unreadable."""
        try:
            data = json.loads((project_dir / "project.json").read_bytes())
        except (OSError, ValueError) as e:  # encrypted bytes fail to decode/parse
            logger.debug(f"project.json not readable as plaintext in {project_dir}: {e}")
            return {}
        return data if isinstance(data, dict) else {}

    def import_projects(self, project_dirs: list[str | Path]) -> ImportResult:
        """Adds existing project folders to the registry without opening them.

        Used for projects built outside the GUI (e.g. by
        ``scripts/build_multilayer_project.py``), which never reach the registry.
        New entries get the name, video and creation date from project.json
        (folder name if the manifest is encrypted) and ``last_opened`` = now, so
        they appear at the top of the list. Entries already registered only get
        their name and status refreshed; their dates are kept. Folders without
        project.json are skipped.
        """
        result = ImportResult()
        now = datetime.now().isoformat()
        for project_dir in project_dirs:
            p = Path(project_dir).resolve()
            if not (p / "project.json").is_file():
                result.skipped.append(str(p))
                continue
            manifest = self._read_manifest(p)
            update = {
                "name": str(manifest.get("project_name") or p.name),
                "has_database": self._check_has_database(p),
                "has_calibration": self._check_has_calibration(p),
            }
            idx = self._find_index(str(p))
            if idx >= 0:
                self._projects[idx].update(update)
                result.already_known.append(str(p))
                continue
            self._projects.append(
                {
                    "path": str(p),
                    "video_path": str(manifest.get("video_path") or ""),
                    "created_at": str(manifest.get("created_at") or now),
                    "last_opened": now,
                    **update,
                }
            )
            result.added.append(str(p))

        if result.added or result.already_known:
            self._save()
        logger.info(
            f"Projects imported: {len(result.added)} added, "
            f"{len(result.already_known)} already known, {len(result.skipped)} skipped"
        )
        return result
