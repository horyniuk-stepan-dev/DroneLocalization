"""Multi-calibration manager.

Stores dict[source_id -> MultiAnchorCalibration] for multi-source projects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.core.project_video_source import ProjectVideoSource

logger = get_logger(__name__)


class MultiCalibrationManager:
    """Manages multiple calibration instances: dict[source_id -> MultiAnchorCalibration]."""

    def __init__(self) -> None:
        self._calibrations: dict[str, MultiAnchorCalibration] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def get(self, source_id: str) -> MultiAnchorCalibration:
        """Returns calibration for source_id, creating an empty one if missing."""
        if source_id not in self._calibrations:
            self._calibrations[source_id] = MultiAnchorCalibration()
            logger.debug(f"Created empty calibration for source '{source_id}'")
        return self._calibrations[source_id]

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
            calib_path = project_dir / src.calibration_file
            cal = MultiAnchorCalibration()
            if calib_path.exists():
                try:
                    cal.load(str(calib_path))
                    logger.info(
                        f"Calibration loaded for '{src.source_id}': "
                        f"{len(cal.anchors)} anchors from {calib_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load calibration for '{src.source_id}' "
                        f"from {calib_path}: {e}. Using empty calibration."
                    )
            else:
                logger.debug(f"No calibration file for '{src.source_id}' at {calib_path}")
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
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cal.save(str(calib_path))
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

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._calibrations
        return source_id in self._calibrations

    def __len__(self) -> int:
        return len(self._calibrations)
