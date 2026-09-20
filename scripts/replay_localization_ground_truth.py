"""Replay a simulator query flight through Localizer and score it against GT.

Ground truth is read only after each localization call and is never passed to
the Localizer.  The report separates raw visual observations from the smoothed
output, failures, processing age, and altitude bins.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_CONFIG
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.database.database_loader import DatabaseLoader
from src.localization.localizer import Localizer
from src.localization.matcher import FeatureMatcher
from src.models.model_manager import ModelManager
from src.models.wrappers.feature_extractor import FeatureExtractor

GEOD = Geod(ellps="WGS84")


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(float(GEOD.inv(a[1], a[0], b[1], b[0])[2]))


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--every", type=int, default=1, help="evaluate every Nth GT slot")
    parser.add_argument(
        "--disable-smoother",
        action="store_true",
        help="disable the sliding-window trajectory smoother for an ablation run",
    )
    args = parser.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    slots = gt["slots"][:: max(1, args.every)]

    database = DatabaseLoader(str(args.db))
    calibration = MultiAnchorCalibration()
    calibration.load(str(args.calibration))
    manager = ModelManager(config=APP_CONFIG)
    extractor = FeatureExtractor(
        manager.load_local_extractor(), manager.load_dinov2(), manager.device, config=APP_CONFIG
    )
    matcher = FeatureMatcher(model_manager=manager, config=APP_CONFIG)
    config = copy.deepcopy(APP_CONFIG)
    if args.disable_smoother:
        config.setdefault("tracking", {})["smoother_enabled"] = False
    config["_model_manager"] = manager
    localizer = Localizer(
        database,
        extractor,
        matcher,
        calibration,
        config=config,
        ref_frame_width=int(database.metadata.get("frame_width", 0)),
        ref_frame_height=int(database.metadata.get("frame_height", 0)),
    )

    rows: list[dict] = []
    capture = cv2.VideoCapture(str(args.video))
    try:
        previous_timestamp = None
        for index, slot in enumerate(slots):
            video_frame = int(slot["video_frame"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            ok, frame = capture.read()
            if not ok or frame is None:
                rows.append({"slot": int(slot["slot"]), "success": False, "error": "decode_failed"})
                continue
            timestamp = float(slot.get("timestamp", index))
            dt = 1.0 if previous_timestamp is None else max(1e-3, timestamp - previous_timestamp)
            previous_timestamp = timestamp
            started = time.perf_counter()
            # No GT field, altitude, camera pose or heading enters this call.
            result = localizer.localize_frame(frame, dt=dt, timestamp=timestamp)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            target = tuple(map(float, slot["ground_center_gps"]))
            row = {
                "slot": int(slot["slot"]),
                "video_frame": video_frame,
                "timestamp": timestamp,
                "camera_agl_eval_only": float(slot["camera_agl"]),
                "success": bool(result.get("success")),
                "status": result.get("status"),
                "error": result.get("error"),
                "matched_frame": result.get("matched_frame"),
                "confidence": result.get("confidence"),
                "scale_ratio": result.get("scale_ratio"),
                "rotation_deg": result.get("rotation_deg"),
                "processing_ms": elapsed_ms,
                "raw_error_m": None,
                "smoothed_error_m": None,
            }
            if result.get("success"):
                if result.get("raw_lat") is not None:
                    row["raw_error_m"] = distance_m(
                        (float(result["raw_lat"]), float(result["raw_lon"])), target
                    )
                row["smoothed_error_m"] = distance_m(
                    (float(result["lat"]), float(result["lon"])), target
                )
            rows.append(row)
            print(
                f"slot={row['slot']:>3} agl={row['camera_agl_eval_only']:>7.1f} "
                f"ok={row['success']} scale={row['scale_ratio']} "
                f"raw_err={row['raw_error_m']} ms={elapsed_ms:.0f}"
            )
    finally:
        capture.release()
        database.close()

    successes = [row for row in rows if row.get("success")]
    raw_errors = [float(row["raw_error_m"]) for row in successes if row.get("raw_error_m") is not None]
    smooth_errors = [
        float(row["smoothed_error_m"])
        for row in successes
        if row.get("smoothed_error_m") is not None
    ]
    latencies = [float(row["processing_ms"]) for row in rows if row.get("processing_ms") is not None]
    bins = {}
    for lo, hi in ((0, 600), (600, 800), (800, 1000), (1000, math.inf)):
        selected = [
            float(row["raw_error_m"])
            for row in successes
            if lo <= row["camera_agl_eval_only"] < hi and row.get("raw_error_m") is not None
        ]
        attempted = sum(lo <= row.get("camera_agl_eval_only", -1) < hi for row in rows)
        bins[f"{lo}-{hi if math.isfinite(hi) else 'inf'}m"] = {
            "attempted": attempted,
            "confirmed": len(selected),
            "raw_error_m": stats(selected),
        }
    report = {
        "inputs": {
            "db": str(args.db.resolve()),
            "calibration": str(args.calibration.resolve()),
            "video": str(args.video.resolve()),
            "ground_truth": str(args.gt.resolve()),
        },
        "attempted": len(rows),
        "confirmed": len(successes),
        "confirmed_rate": len(successes) / max(1, len(rows)),
        "raw_error_m": stats(raw_errors),
        "smoothed_error_m": stats(smooth_errors),
        "processing_ms": stats(latencies),
        "altitude_bins": bins,
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
