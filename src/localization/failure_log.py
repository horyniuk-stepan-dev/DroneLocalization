"""Failure logging for localization (CSV audit trail).

Extracted from Localizer (IMPROVEMENT_PLAN 1.1). FAILURE_TYPES are unchanged.

Vision-only plan 0.2: every row also carries a coarse ``reason`` (see
``FAILURE_REASONS``), the session ``run_id`` and the last ``coverage_state``.
A file written with the old 4-column header is renamed aside (never deleted)
before the first row with the new header is appended.

Destination: ``DRONELOC_FAILURE_CSV`` overrides the default
``<user_data>/logs/localization_failures.csv`` (tests and replay scripts point it
at their own output); ``off`` / ``none`` / empty disables the CSV.
"""

from __future__ import annotations

import os
import time
import uuid

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FAILURE_TYPES = {
    "out_of_coverage": "out_of_coverage",
    "No candidates": "no_retrieval_candidates",
    "Not enough valid inliers": "insufficient_inliers",
    "No propagated calibration": "no_propagated_affine",
    "Outlier detected": "trajectory_outlier",
    "Coordinate transformation": "transform_error",
    "Footprint outside coverage": "footprint_outside_coverage",
}

# Coarse failure classes (plan 0.2): where in the pipeline the frame was lost.
FAILURE_REASONS = (
    "no_candidates",  # retrieval returned nothing usable
    "too_few_matches",  # candidates found, local matching below min_matches
    "retrieval_only",  # only a retrieval-only candidate position (not a fix)
    "geometry_rejected",  # homography / projection / layer geometry rejected
    "gate_rejected",  # a valid fix rejected by the kinematic / coverage gate
    "no_calibration",  # matched frame has no usable georeference
    "out_of_coverage",  # consecutive-failure counter guard fired
    "optical_flow",  # optical-flow step rejected
    "other",
)

_TYPE_TO_REASON = {
    "no_retrieval_candidates": "no_candidates",
    "insufficient_inliers": "too_few_matches",
    "no_propagated_affine": "no_calibration",
    "trajectory_outlier": "gate_rejected",
    "footprint_outside_coverage": "gate_rejected",
    "transform_error": "geometry_rejected",
    "out_of_coverage": "out_of_coverage",
}

# (substring of the result "error", reason); first match wins.
_ERROR_TO_REASON = (
    ("out_of_coverage", "out_of_coverage"),
    ("OF ", "optical_flow"),
    ("No previous state to apply OF", "optical_flow"),
    ("No candidates", "no_candidates"),
    ("no_verified_candidates", "no_candidates"),
    ("Not enough valid inliers", "too_few_matches"),
    ("Retrieval candidate has no verified geometry", "retrieval_only"),
    ("Outlier detected", "gate_rejected"),
    ("motion_gated", "gate_rejected"),
    ("outside map coverage", "gate_rejected"),
    ("awaiting_confirmation", "gate_rejected"),
    ("provisional or invalid georeference", "no_calibration"),
    ("No propagated calibration", "no_calibration"),
    ("Layer calibration manager missing", "no_calibration"),
    ("Failed to compute transform", "geometry_rejected"),
    ("Coordinate transformation", "geometry_rejected"),
    ("safe reference projection", "geometry_rejected"),
    ("ambiguous_geometry", "geometry_rejected"),
    ("layer_map_disagreement", "geometry_rejected"),
)

HEADER = "timestamp,error_type,inliers,details,reason,run_id,coverage_state"


def classify_failure(error: object) -> str:
    """Coarse reason for a failed localization result's ``error`` text."""
    text = str(error or "")
    if not text:
        return "other"
    for needle, reason in _ERROR_TO_REASON:
        if needle in text:
            return reason
    return "other"


def reason_for_type(error_type: str) -> str:
    return _TYPE_TO_REASON.get(error_type, "other")


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _default_csv_path() -> str | None:
    env = os.environ.get("DRONELOC_FAILURE_CSV")
    if env is not None:
        return None if env.strip().lower() in ("", "off", "none", "0") else env
    from config import user_data_dir

    return str(user_data_dir() / "logs" / "localization_failures.csv")


class FailureLogger:
    """Appends localization failures to a CSV (default logs/localization_failures.csv)."""

    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = csv_path if csv_path is not None else _default_csv_path()
        self.run_id = new_run_id()
        self.coverage_state = ""
        self._header_checked = False

    def new_run(self) -> str:
        self.run_id = new_run_id()
        return self.run_id

    def _ensure_header(self, csv_path: str) -> bool:
        """True when the header must be written (new file or legacy file moved aside)."""
        if not os.path.exists(csv_path):
            return True
        if self._header_checked:
            return False
        self._header_checked = True
        try:
            with open(csv_path, encoding="utf-8") as f:
                first = f.readline().strip()
        except OSError:
            return False
        if first == HEADER:
            return False
        root, ext = os.path.splitext(csv_path)
        legacy = f"{root}.legacy-{time.strftime('%Y%m%d-%H%M%S')}{ext or '.csv'}"
        try:
            os.replace(csv_path, legacy)
            logger.info(f"Failure CSV format changed; old file kept as {legacy}")
            return True
        except OSError as e:
            logger.warning(f"Could not move legacy failure CSV aside ({e}); appending anyway")
            return False

    def log(
        self,
        error_type: str,
        inliers: int = 0,
        details: str = "",
        reason: str | None = None,
    ) -> None:
        csv_path = self.csv_path
        if not csv_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            write_header = self._ensure_header(csv_path)
            with open(csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(HEADER + "\n")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                safe_details = details.replace('"', '""')
                f.write(
                    f'{timestamp},{error_type},{inliers},"{safe_details}",'
                    f"{reason or reason_for_type(error_type)},{self.run_id},"
                    f"{self.coverage_state}\n"
                )
        except Exception as e:
            logger.error(f"Failed to log to localization_failures.csv: {e}")
