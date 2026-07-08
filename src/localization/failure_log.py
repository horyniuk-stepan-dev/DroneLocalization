"""Failure logging for localization (CSV audit trail).

Extracted verbatim from Localizer (IMPROVEMENT_PLAN 1.1). FAILURE_TYPES and the
CSV format are unchanged, so existing log files and call sites keep working.
"""

from __future__ import annotations

import os
import time

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FAILURE_TYPES = {
    "out_of_coverage": "out_of_coverage",
    "No candidates": "no_retrieval_candidates",
    "Not enough valid inliers": "insufficient_inliers",
    "No propagated calibration": "no_propagated_affine",
    "Outlier detected": "trajectory_outlier",
    "Coordinate transformation": "transform_error",
}


class FailureLogger:
    """Appends localization failures to a CSV (default logs/localization_failures.csv)."""

    def __init__(self, csv_path: str = "logs/localization_failures.csv") -> None:
        self.csv_path = csv_path

    def log(self, error_type: str, inliers: int = 0, details: str = "") -> None:
        try:
            csv_path = self.csv_path
            write_header = not os.path.exists(csv_path)
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("timestamp,error_type,inliers,details\n")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                safe_details = details.replace('"', '""')
                f.write(f'{timestamp},{error_type},{inliers},"{safe_details}"\n')
        except Exception as e:
            logger.error(f"Failed to log to localization_failures.csv: {e}")
