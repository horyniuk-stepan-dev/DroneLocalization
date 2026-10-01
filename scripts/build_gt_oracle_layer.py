"""Build GT-oracle copies of simulator reference layers (evaluation only).

Each copy keeps the layer's images, descriptors and LanceDB index, but every
keyframe is pinned to the simulator's ground-truth pose. Replaying a query
against oracle layers measures retrieval + matching + layer handoff alone,
without the error of building the map from sparse anchors.

    python scripts/build_gt_oracle_layer.py --out-root D:/maps/oracle \\
        --layer main D:/proj/sources/main/database.h5 D:/FlightSimulator/output/layer_1000m \\
        --layer 2    D:/proj/sources/2/database.h5    D:/FlightSimulator/output/layer_2000m

    # only measure a layer's map error against GT, write nothing:
    python scripts/build_gt_oracle_layer.py --report-only --layer main DB SIM_DIR

The simulator folder must be the recording the database was built from
(checked through the video SHA-256 in manifest.json). Sources are never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.layer_gt_tools import (  # noqa: E402
    LayerGTError,
    SimulatorRun,
    build_oracle_layer,
    map_error_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--layer",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "DATABASE_H5", "SIMULATOR_RUN_DIR"),
        help="reference layer; repeat for each layer",
    )
    parser.add_argument("--out-root", type=Path, help="oracle copies go to OUT_ROOT/<ID>/")
    parser.add_argument("--report-only", action="store_true", help="measure map error only")
    parser.add_argument(
        "--copy-keypoint-video",
        action="store_true",
        help="also copy <db>_keypoints.mp4 (only needed to open the calibration dialog)",
    )
    args = parser.parse_args(argv)
    if not args.report_only and args.out_root is None:
        parser.error("--out-root is required unless --report-only")
    ids = [layer[0] for layer in args.layer]
    if len(set(ids)) != len(ids):
        parser.error(f"layer IDs must be unique: {ids}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = {}
    failed = False
    for source_id, db, sim_dir in args.layer:
        run = SimulatorRun(Path(sim_dir))
        try:
            if args.report_only:
                summary[source_id] = {"map_error": map_error_report(Path(db), run)}
                continue
            out_dir = args.out_root / source_id
            result = build_oracle_layer(
                Path(db),
                run,
                out_dir,
                source_id=source_id,
                copy_keypoint_video=args.copy_keypoint_video,
            )
            result["map_error_after"] = map_error_report(out_dir / Path(db).name, run)
            summary[source_id] = result
        except (LayerGTError, OSError) as exc:
            failed = True
            summary[source_id] = {"error": str(exc)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
