"""Run calibration propagation synchronously without the GUI."""

from __future__ import annotations

import argparse
import copy
import json
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


def config_overrides(config: dict, items: list[str]) -> dict[tuple[str, str], object]:
    """Parse SECTION.KEY=VALUE items; only keys already present in the config."""
    overrides: dict[tuple[str, str], object] = {}
    for item in items:
        name, sep, text = item.partition("=")
        section, dot, key = name.strip().partition(".")
        if not sep or not dot or not section or not key:
            raise ValueError(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        if not isinstance(config.get(section), dict) or key not in config[section]:
            raise ValueError(f"unknown config key: {section}.{key}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        current = config[section][key]
        if isinstance(current, bool) != isinstance(value, bool) or (
            isinstance(current, int | float) and not isinstance(value, int | float)
        ):
            raise ValueError(f"{section}.{key}: {value!r} does not match type of {current!r}")
        overrides[(section, key)] = value
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument(
        "--anchor-linear-fallback",
        action="store_true",
        help="opt in to the anchor-verified straight-leg motion model",
    )
    parser.add_argument(
        "--pin-exact-anchors",
        action="store_true",
        help="keep surveyed affine transforms exact at every anchor image",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override an existing config key for this run only (JSON value, e.g. "
        "graph_optimization.kinematic_prior_weight=1.0); repeatable",
    )
    args = parser.parse_args()

    config = copy.deepcopy(APP_CONFIG)
    try:
        overrides = config_overrides(config, args.set)
    except ValueError as exc:
        parser.error(str(exc))
    for (section, key), value in overrides.items():
        config[section][key] = value
        print(f"override {section}.{key} = {value!r}")
    if args.anchor_linear_fallback:
        config["graph_optimization"]["anchor_linear_fallback"] = True
    if args.pin_exact_anchors:
        config["graph_optimization"]["pin_exact_anchors"] = True

    database = DatabaseLoader(str(args.db))
    calibration = MultiAnchorCalibration()
    calibration.load(str(args.calibration))
    manager = ModelManager(config=config)
    matcher = FeatureMatcher(model_manager=manager, config=config)
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
        config=config,
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
