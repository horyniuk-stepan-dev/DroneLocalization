"""HARDENING P1-6: detect which artifacts of a project are encrypted at rest.

Kept free of Qt so it is unit-testable in the pure-Python suite; the GUI
passphrase dialog is the only consumer that needs a widget toolkit.

Artifacts live per source (``sources/main/database.h5``, see ProjectSettings),
so paths are resolved through the project's own source configuration rather
than guessed from the project root.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.security.at_rest import is_encrypted
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Only the header is needed to classify a file — never read a 67 MB database in.
_HEADER_PROBE_BYTES = 64


def _file_is_encrypted(path: Path) -> bool:
    """True if ``path`` carries the at-rest container header. Missing or
    unreadable files are reported as not encrypted — a load error there is the
    caller's problem, not a passphrase problem."""
    try:
        with open(path, "rb") as f:
            return is_encrypted(f.read(_HEADER_PROBE_BYTES))
    except OSError:
        return False


def _artifacts_for(root: Path, settings) -> list[Path]:
    """Resolve a project's encryptable artifacts from its root and settings.

    Returns existing paths only, ordered cheapest-to-decrypt first so a caller
    verifying a passphrase can use ``[0]`` without decrypting the map database."""
    calibrations: list[Path] = []
    databases: list[Path] = []
    others: list[Path] = []

    sources = settings.get_enabled_sources()
    if not sources:
        # Legacy single-source project: fall back to the top-level filenames.
        calibrations.append(root / settings.calibration_filename)
        databases.append(root / settings.database_filename)

    for source in sources:
        calibrations.append(root / source.calibration_file)
        databases.append(root / source.database_file)

    for db_path in list(databases):
        # Sibling artifacts are named after their database (see DatabaseBuilder).
        others.append(db_path.parent / "vectors.lance")
        others.append(db_path.with_name(db_path.stem + "_keypoints.mp4"))

    resolved: list[Path] = []
    for path in [*calibrations, *databases, *others]:
        if path.is_dir():
            # A lance dataset is a directory: probe its first data file.
            resolved.extend(sorted(p for p in path.rglob("*") if p.is_file())[:1])
        elif path.is_file():
            resolved.append(path)
    return resolved


def candidate_artifacts(project_manager) -> list[Path]:
    """Encryptable artifacts of a *loaded* project (see :func:`_artifacts_for`)."""
    if not project_manager.is_loaded:
        return []
    return _artifacts_for(Path(project_manager.project_dir), project_manager.settings)


def project_is_encrypted(project_dir: str | Path) -> bool:
    """True if the project at ``project_dir`` holds any encrypted artifact.

    Path-based and quiet — it reads ``project.json`` directly rather than going
    through ``ProjectManager``, which logs a line per load. The project picker
    calls this once per listed row, so noise and cost both matter; it returns on
    the first encrypted artifact instead of collecting them all. A directory
    that is not a project counts as not encrypted."""
    from src.core.project import ProjectSettings

    root = Path(project_dir)
    manifest = root / "project.json"
    if not manifest.is_file():
        return False
    try:
        settings = ProjectSettings.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return any(_file_is_encrypted(p) for p in _artifacts_for(root, settings))


def find_encrypted_artifacts(project_manager) -> list[Path]:
    """Encrypted artifacts of the loaded project, cheapest-to-decrypt first.

    Empty list means the project is plaintext and must load exactly as before —
    no passphrase prompt, no behaviour change."""
    found = [p for p in candidate_artifacts(project_manager) if _file_is_encrypted(p)]
    if found:
        logger.info(
            f"Encrypted-at-rest artifacts detected in "
            f"'{project_manager.project_name}': {[p.name for p in found]}"
        )
    return found
