"""Run calibration propagation synchronously without the GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_CONFIG
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.database.database_loader import DatabaseLoader
from src.localization.matcher import FeatureMatcher
from src.models.model_manager import ModelManager
from src.workers.propagation_pipeline import PropagationPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    args = parser.parse_args()

    database = DatabaseLoader(str(args.db))
    calibration = MultiAnchorCalibration()
    calibration.load(str(args.calibration))
    manager = ModelManager(config=APP_CONFIG)
    matcher = FeatureMatcher(model_manager=manager, config=APP_CONFIG)
    errors: list[str] = []
    completed = False

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    def complete() -> None:
        nonlocal completed
        completed = True

    pipeline = PropagationPipeline(
        database,
        calibration,
        matcher,
        config=APP_CONFIG,
        progress_callback=progress,
        error_callback=errors.append,
        completed_callback=complete,
    )
    try:
        pipeline._run_propagation()
    finally:
        database.close()

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    if not completed:
        print("ERROR: propagation ended without a completion signal")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
