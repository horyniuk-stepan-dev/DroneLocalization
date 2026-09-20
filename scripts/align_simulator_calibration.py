"""Align simulator anchors to the exact keyframe slots present in a database.

This is an offline simulator-only bridge for a database that was built before
``database.required_frame_ids`` existed.  A missing anchor is moved to the
nearest unused keyframe, but its affine is recomputed from that target slot's
ground truth.  The original affine is never copied to a different image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from src.geometry.coordinates import CoordinateConverter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _qa_for_target(template: dict, slot: dict, affine: np.ndarray, converter) -> dict:
    qa = dict(template.get("qa_data", {}))
    points_2d = qa.get("points_2d") or [
        [640.0, 360.0],
        [256.0, 144.0],
        [1024.0, 144.0],
        [1024.0, 576.0],
        [256.0, 576.0],
    ]
    px = np.asarray(points_2d, dtype=np.float64)
    metric = px @ affine[:, :2].T + affine[:, 2]
    gps = [list(map(float, converter.metric_to_gps(float(x), float(y)))) for x, y in metric]
    rmse = float(slot.get("rmse_m", 0.0))
    qa.update(
        {
            "rmse_m": rmse,
            "median_err_m": rmse,
            "max_err_m": rmse,
            "inliers_count": len(points_2d),
            "points_2d": points_2d,
            "points_metric": metric.tolist(),
            "points_gps": gps,
            "transform_type": "simulator_ground_truth_db_aligned",
            "projection_mode": "WEB_MERCATOR",
            "updated_at": datetime.now().isoformat(),
            "notes": (
                f"Exact affine for DB slot {slot['slot']}; simulator reason: "
                f"{slot.get('anchor_reason') or 'db-keyframe-alignment'}"
            ),
        }
    )
    return qa


def align(db_path: Path, source_calibration: Path, gt_path: Path, output: Path, video: Path | None):
    source = json.loads(source_calibration.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    slots = {int(row["slot"]): row for row in gt["slots"]}

    with h5py.File(db_path, "r+") as db:
        metadata = db["metadata"]
        if list(map(int, gt["frame_size"])) != [
            int(metadata.attrs["frame_width"]),
            int(metadata.attrs["frame_height"]),
        ]:
            raise ValueError("Ground-truth frame size does not match the database")
        if int(gt.get("frame_step", 1)) != int(metadata.attrs.get("frame_step", 1)):
            raise ValueError("Ground-truth frame_step does not match the database")
        keyframes = [int(i) for i in metadata["frame_index_map"][:]]
        if video is not None:
            video = video.resolve()
            stat = video.stat()
            metadata.attrs["source_path"] = str(video)
            metadata.attrs["source_size_bytes"] = int(stat.st_size)
            metadata.attrs["source_mtime_ns"] = int(stat.st_mtime_ns)
            metadata.attrs["source_sha256"] = _sha256(video)

    used: set[int] = set()
    mapping: list[dict] = []
    anchors: list[dict] = []
    converter = CoordinateConverter.from_metadata(source["projection"])
    for anchor in source.get("anchors", []):
        original = int(anchor["frame_id"])
        candidates = sorted((fid for fid in keyframes if fid not in used), key=lambda fid: (abs(fid - original), fid))
        if not candidates:
            raise ValueError("Not enough unique database keyframes for calibration anchors")
        target = original if original in candidates else candidates[0]
        if target not in slots:
            raise ValueError(f"Ground truth has no slot {target}")
        used.add(target)
        slot = slots[target]
        affine = np.asarray(slot["affine"], dtype=np.float64)
        anchors.append(
            {
                "frame_id": target,
                "affine_matrix": affine.tolist(),
                "qa_data": _qa_for_target(anchor, slot, affine, converter),
            }
        )
        mapping.append({"source_anchor": original, "db_anchor": target, "delta_slots": target - original})

    output_data = dict(source)
    output_data["version"] = "2.4"
    output_data["anchors"] = sorted(anchors, key=lambda row: row["frame_id"])
    output_data["database_alignment"] = {
        "method": "nearest_unused_keyframe_with_target_slot_ground_truth_affine",
        "database": str(db_path.resolve()),
        "database_sha256": _sha256(db_path),
        "ground_truth": str(gt_path.resolve()),
        "ground_truth_sha256": _sha256(gt_path),
        "mapping": mapping,
    }

    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = output.with_name(f"{output.name}.{stamp}.bak")
        shutil.copy2(output, backup)
        print(f"backup: {backup}")
    _atomic_json(output, output_data)
    print("anchor mapping:")
    for item in mapping:
        marker = " (remapped with target GT affine)" if item["delta_slots"] else ""
        print(f"  {item['source_anchor']:>4} -> {item['db_anchor']:>4}{marker}")
    print(f"written: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-calibration", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    align(args.db, args.source_calibration, args.gt, args.output, args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
