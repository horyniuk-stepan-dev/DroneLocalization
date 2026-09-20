"""Build a reference database while preserving every calibration anchor image."""

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
from src.database.database_builder import DatabaseBuilder
from src.models.model_manager import ModelManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="store every sampled slot; useful when turns break interpolation",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    anchors = calibration.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        parser.error("calibration has no anchors")
    required = {int(anchor["frame_id"]) for anchor in anchors}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(APP_CONFIG)
    if args.keep_all:
        config["database"]["keyframe_criterion"] = "step"
        config["database"]["keyframe_min_translation_px"] = 0.0
        config["database"]["keyframe_min_rotation_deg"] = 0.0
    manager = ModelManager(config=config)
    builder = DatabaseBuilder(output_path=str(args.output), config=config)

    def progress(percent: int) -> None:
        print(f"[{percent:3d}%] building database")

    builder.build_from_video(
        video_path=str(args.video),
        model_manager=manager,
        progress_callback=progress,
        save_keypoint_video=False,
        required_frame_ids=required,
    )
    print(f"Built {args.output} with required anchor slots {sorted(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
