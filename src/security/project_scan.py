"""HARDENING P1-6: detect encrypted projects on disk and keep them immutable.

Kept free of Qt so it is unit-testable in the pure-Python suite; the GUI
passphrase dialog is the only consumer that needs a widget toolkit.

An encrypted deployment copy has EVERY file encrypted, ``project.json``
included, so the manifest header is the marker: one 7-byte read tells you
whether a directory is an encrypted copy, before anything is loaded.

Copies built before that (plaintext manifest, encrypted artifacts) still open —
the artifact scan below is kept as a fallback. Artifacts live per source
(``sources/main/database.h5``, see ProjectSettings), so their paths are resolved
through the project's own source configuration rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.security.at_rest import EncryptionError, is_encrypted
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Only the header is needed to classify a file — never read a 67 MB database in.
_HEADER_PROBE_BYTES = 64


class EncryptedProjectWriteError(EncryptionError):
    """Raised when something tries to write into an encrypted deployment copy."""


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


def encrypted_artifacts_at(project_dir: str | Path) -> list[Path]:
    """Encrypted artifacts of the project at ``project_dir``, cheapest first.

    Works from a path alone — the project cannot be loaded before the passphrase
    is known, since the manifest itself may be encrypted. An empty list means the
    project is plaintext and must load exactly as before: no prompt, no change.

    Side-effect free and quiet: it reads ``project.json`` directly rather than
    going through ``ProjectManager``, which logs a line per load, because the
    project picker calls this once per listed row."""
    from src.core.project import ProjectSettings

    root = Path(project_dir)
    manifest = root / "project.json"
    if not manifest.is_file():
        return []

    if _file_is_encrypted(manifest):
        # Fully encrypted copy: the manifest is both the marker and the cheapest
        # possible verification target (a few hundred bytes).
        return [manifest]

    try:
        settings = ProjectSettings.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return []
    return [p for p in _artifacts_for(root, settings) if _file_is_encrypted(p)]


def project_is_encrypted(project_dir: str | Path) -> bool:
    """True if the project at ``project_dir`` is an encrypted deployment copy."""
    return bool(encrypted_artifacts_at(project_dir))


def find_project_root(path: str | Path) -> Path | None:
    """Nearest ancestor directory (or ``path`` itself) holding a ``project.json``.

    Write guards get a target file path, not a project, so they walk up to find
    which project — if any — they are about to write into."""
    start = Path(path)
    start = start if start.is_dir() else start.parent
    for candidate in [start, *start.parents]:
        if (candidate / "project.json").is_file():
            return candidate
    return None


def assert_project_writable(path: str | Path) -> None:
    """Refuse any write that lands inside an encrypted deployment copy.

    The copy is immutable by design: the ground station keeps the plaintext
    master and the drone carries ciphertext. Without this guard a normal rebuild
    or calibration save silently writes plaintext into a project the operator
    believes is encrypted — observed in the field, hence the hard refusal.

    Writes outside any project, or into a plaintext project, are unaffected."""
    root = find_project_root(path)
    if root is None or not project_is_encrypted(root):
        return
    logger.error(f"Refused write into encrypted project '{root.name}': {path}")
    raise EncryptedProjectWriteError(
        f"'{root.name}' is an encrypted deployment copy and cannot be modified. "
        f"Rebuild, recalibrate and re-run propagation on the plaintext master "
        f"project, then build a fresh encrypted copy from it."
    )
