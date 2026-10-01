"""Multi-calibration manager.

Stores dict[source_id -> MultiAnchorCalibration] for multi-source projects.

Invariant: a layer whose calibration file exists on disk is never handed out
as an empty in-memory calibration — the first anchor save would overwrite the
file and silently drop every earlier anchor.  ``get_or_load`` enforces it for
layers that were disabled or added after the project was opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.core.layer_status import CALIBRATION_OWNER_KEY
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.core.project_video_source import ProjectVideoSource

logger = get_logger(__name__)


def load_layer_calibration(src: ProjectVideoSource, project_dir: Path) -> MultiAnchorCalibration:
    """Reads a layer's calibration file; empty calibration if the file is absent.

    Raises on an unreadable file so callers never mistake a broken file for an
    empty one (and later overwrite it).
    """
    cal = MultiAnchorCalibration()
    calib_path = Path(project_dir) / src.calibration_file
    if calib_path.exists():
        cal.load(str(calib_path))
        logger.info(
            f"Calibration loaded for '{src.source_id}': "
            f"{len(cal.anchors)} anchors from {calib_path}"
        )
    else:
        logger.debug(f"No calibration file for '{src.source_id}' at {calib_path}")
    return cal


def save_layer_calibration(cal: MultiAnchorCalibration, path: str | Path, source_id: str) -> None:
    """Saves ``cal`` to ``path`` stamped with the owning layer id."""
    cal.extra_metadata[CALIBRATION_OWNER_KEY] = source_id
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cal.save(str(path))


class MultiCalibrationManager:
    """Manages multiple calibration instances: dict[source_id -> MultiAnchorCalibration]."""

    def __init__(self) -> None:
        self._calibrations: dict[str, MultiAnchorCalibration] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def get(self, source_id: str) -> MultiAnchorCalibration:
        """Returns calibration for source_id, creating an empty one if missing.

        Prefer ``get_or_load`` when the source config is at hand: this method
        cannot know whether a file for the layer exists on disk.
        """
        if source_id not in self._calibrations:
            self._calibrations[source_id] = MultiAnchorCalibration()
            logger.debug(f"Created empty calibration for source '{source_id}'")
        return self._calibrations[source_id]

    def get_or_load(self, src: ProjectVideoSource, project_dir: Path) -> MultiAnchorCalibration:
        """Returns the in-memory calibration, loading it from disk on first use."""
        cal = self._calibrations.get(src.source_id)
        if cal is None:
            cal = load_layer_calibration(src, project_dir)
            self._calibrations[src.source_id] = cal
        return cal

    def set(self, source_id: str, calibration: MultiAnchorCalibration) -> None:
        """Replaces the calibration object of a layer (e.g. after loading a JSON)."""
        self._calibrations[source_id] = calibration

    def discard(self, source_id: str) -> None:
        """Forgets a layer's in-memory calibration (the file stays on disk)."""
        self._calibrations.pop(source_id, None)

    def load_all(
        self,
        sources: list[ProjectVideoSource],
        project_dir: Path,
    ) -> None:
        """Loads calibration for all enabled sources."""
        self._calibrations.clear()
        for src in sources:
            if not src.enabled:
                continue
            try:
                cal = load_layer_calibration(src, project_dir)
            except Exception as e:
                # Not cached: get_or_load will retry instead of handing out an
                # empty calibration that would overwrite the file on first save.
                logger.error(
                    f"Failed to load calibration for '{src.source_id}' "
                    f"from {project_dir / src.calibration_file}: {e}. "
                    f"The file is left untouched."
                )
                continue
            self._calibrations[src.source_id] = cal

    def save_all(
        self,
        sources: list[ProjectVideoSource],
        project_dir: Path,
    ) -> None:
        """Saves all modified calibrations, creating subdirectories if needed."""
        for src in sources:
            if src.source_id not in self._calibrations:
                continue
            cal = self._calibrations[src.source_id]
            calib_path = project_dir / src.calibration_file
            if not cal.is_calibrated and not calib_path.exists():
                continue
            try:
                save_layer_calibration(cal, calib_path, src.source_id)
            except Exception as e:
                logger.error(f"Failed to save calibration for '{src.source_id}': {e}")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_any_calibrated(self) -> bool:
        """Returns True if at least one source is fully calibrated."""
        return any(cal.is_calibrated for cal in self._calibrations.values())

    @property
    def source_ids(self) -> list[str]:
        """List of loaded calibration source_ids."""
        return list(self._calibrations.keys())

    def items(self) -> list[tuple[str, MultiAnchorCalibration]]:
        return list(self._calibrations.items())

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._calibrations

    def __len__(self) -> int:
        return len(self._calibrations)
