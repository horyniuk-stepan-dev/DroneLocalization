"""Simulator ground truth for reference layers: map-error report and GT-oracle copies.

A reference layer recorded by FlightSimulator comes with ``ground_truth.json``
(one pixel->map affine per database slot) and ``manifest.json`` (SHA-256 of the
video). This module

* measures how far a propagated layer's georeference is from that truth, and
* builds a *GT-oracle* copy of a layer: the same images, descriptors and index,
  but every keyframe pinned to its true pose. Replaying a query against oracle
  layers isolates retrieval, matching and layer handoff from map construction.

Oracle copies are evaluation artefacts. They use information a real mission
never has and must not be shipped as maps.

Only h5py / numpy / pyproj are needed (no torch).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.multi_anchor_calibration import (  # noqa: E402
    AnchorCalibration,
    MultiAnchorCalibration,
)
from src.core.layer_status import CALIBRATION_OWNER_KEY  # noqa: E402
from src.geometry.calibration_provenance import CalibrationOrigin, GeoreferenceStatus  # noqa: E402
from src.geometry.coordinates import CoordinateConverter  # noqa: E402

ORACLE_OPTIMIZER = "gt_oracle"


class LayerGTError(ValueError):
    """The ground truth cannot be applied to this database (wrong layer, units, ...)."""


@dataclass(frozen=True)
class SimulatorRun:
    """One FlightSimulator recording folder (reference layer)."""

    folder: Path

    @property
    def ground_truth(self) -> Path:
        return self.folder / "ground_truth.json"

    @property
    def manifest(self) -> Path:
        return self.folder / "manifest.json"

    def load_gt(self) -> dict:
        if not self.ground_truth.is_file():
            raise LayerGTError(f"no ground_truth.json in {self.folder}")
        return json.loads(self.ground_truth.read_text(encoding="utf-8"))

    def video_sha256(self) -> str | None:
        if not self.manifest.is_file():
            return None
        data = json.loads(self.manifest.read_text(encoding="utf-8"))
        return ((data.get("files") or {}).get("video") or {}).get("sha256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attr_str(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def gt_affines(gt: dict, num_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2, 3) GT affines and a validity mask indexed by database slot."""
    affines = np.zeros((num_slots, 2, 3), dtype=np.float64)
    valid = np.zeros(num_slots, dtype=bool)
    for slot in gt.get("slots", []):
        sid = int(slot["slot"])
        if not 0 <= sid < num_slots or not slot.get("ground_center_valid", True):
            continue
        matrix = np.asarray(slot.get("affine"), dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            continue
        affines[sid] = matrix
        valid[sid] = True
    return affines, valid


def _centre(affine: np.ndarray, width: int, height: int) -> np.ndarray:
    return affine @ np.array([width / 2.0, height / 2.0, 1.0])


def check_compatibility(db: h5py.File, gt: dict, run: SimulatorRun | None) -> dict:
    """Raises LayerGTError unless ``gt`` describes exactly the video/slots of ``db``."""
    meta = db["metadata"].attrs
    step = int(meta.get("frame_step", 0))
    width, height = int(meta.get("frame_width", 0)), int(meta.get("frame_height", 0))
    if int(gt.get("frame_step", -1)) != step:
        raise LayerGTError(f"frame_step: GT {gt.get('frame_step')} != database {step}")
    if list(gt.get("frame_size") or []) != [width, height]:
        raise LayerGTError(f"frame_size: GT {gt.get('frame_size')} != database {[width, height]}")

    info = {"frame_step": step, "frame_size": [width, height]}
    db_sha = meta.get("source_sha256")
    run_sha = run.video_sha256() if run is not None else None
    if db_sha is not None and run_sha is not None:
        if _attr_str(db_sha) != run_sha:
            raise LayerGTError(
                "ground truth belongs to another recording: manifest video sha256 "
                f"{run_sha[:12]}… != database source {_attr_str(db_sha)[:12]}…"
            )
        info["video_identity"] = "sha256 match"
    else:
        info["video_identity"] = "not checked (sha256 missing)"

    projection = gt.get("projection") or {}
    if "calibration" in db and "projection_json" in db["calibration"].attrs:
        db_projection = json.loads(_attr_str(db["calibration"].attrs["projection_json"]))
        if db_projection.get("mode") != projection.get("mode"):
            raise LayerGTError(
                f"projection: GT {projection.get('mode')} != database {db_projection.get('mode')}"
            )
    info["projection"] = projection
    return info


def anchor_consistency_m(db: h5py.File, affines: np.ndarray, valid: np.ndarray) -> dict | None:
    """Centre distance (projected metres) between the database's anchor slots and GT.

    Anchors were produced from this very GT, so a large value means the GT and
    the database do not describe the same slots (wrong layer, shifted slots).
    """
    if "calibration" not in db or "frame_origin" not in db["calibration"]:
        return None
    grp = db["calibration"]
    origin = grp["frame_origin"][:]
    db_affine = grp["frame_affine"][:]
    meta = db["metadata"].attrs
    width, height = int(meta["frame_width"]), int(meta["frame_height"])
    ids = [i for i in np.where(origin == int(CalibrationOrigin.ANCHOR))[0] if valid[i]]
    if not ids:
        return None
    dist = [
        float(
            np.linalg.norm(
                _centre(db_affine[i], width, height) - _centre(affines[i], width, height)
            )
        )
        for i in ids
    ]
    return {"n": len(ids), "median_m": float(np.median(dist)), "max_m": float(np.max(dist))}


def map_error_report(db_path: Path, run: SimulatorRun) -> dict:
    """Geodesic error of the database's per-slot GPS (frame_gps) against GT centres."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    gt = run.load_gt()
    by_slot = {int(s["slot"]): s for s in gt.get("slots", [])}
    with h5py.File(db_path, "r") as db:
        check_compatibility(db, gt, run)
        if "frame_gps" not in db or "calibration" not in db:
            return {"propagated": False}
        gps = db["frame_gps"][:]
        grp = db["calibration"]
        status = grp["frame_georef_status"][:] if "frame_georef_status" in grp else None
        origin = grp["frame_origin"][:] if "frame_origin" in grp else None
        keyframe = db["local_features"]["kp_counts"][:] > 0
    err = np.full(len(gps), np.nan)
    for i in range(len(gps)):
        slot = by_slot.get(i)
        if slot is None or not slot.get("ground_center_valid") or not np.isfinite(gps[i]).all():
            continue
        lat, lon = slot["ground_center_gps"]
        err[i] = abs(geod.inv(lon, lat, gps[i, 1], gps[i, 0])[2])

    def stats(mask):
        values = err[mask & np.isfinite(err)]
        if not values.size:
            return {"n": 0}
        return {
            "n": int(values.size),
            "median_m": round(float(np.median(values)), 2),
            "p95_m": round(float(np.percentile(values, 95)), 2),
            "max_m": round(float(values.max()), 2),
        }

    supported = (
        status == int(GeoreferenceStatus.SUPPORTED) if status is not None else np.isfinite(err)
    )
    report = {
        "propagated": True,
        "keyframes": int(keyframe.sum()),
        "supported_keyframes": int((supported & keyframe).sum()),
        "supported": stats(supported & keyframe),
    }
    if origin is not None:
        for code in (CalibrationOrigin.ANCHOR, CalibrationOrigin.OPTIMIZED):
            report[f"supported_{code.name.lower()}"] = stats(
                supported & keyframe & (origin == int(code))
            )
    return report


def _replace(group: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, compression="gzip")


def build_oracle_layer(
    database: Path,
    run: SimulatorRun,
    out_dir: Path,
    *,
    source_id: str,
    copy_keypoint_video: bool = False,
    max_anchor_disagreement_m: float = 25.0,
) -> dict:
    """Copy ``database`` (+ vectors.lance) into ``out_dir`` and pin keyframes to GT.

    Every image that has local features gets its GT affine, status SUPPORTED and
    origin ANCHOR; other slots are marked invalid. The source is never modified;
    an existing ``out_dir`` is refused.
    """
    database = Path(database)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise LayerGTError(f"output already exists: {out_dir}")
    gt = run.load_gt()

    with h5py.File(database, "r") as db:
        info = check_compatibility(db, gt, run)
        num_slots = int(db["metadata"].attrs["num_frames"])
        affines, valid = gt_affines(gt, num_slots)
        consistency = anchor_consistency_m(db, affines, valid)
        if consistency is not None and consistency["median_m"] > max_anchor_disagreement_m:
            raise LayerGTError(
                f"database anchors are {consistency['median_m']:.1f} m (median) from GT "
                f"— GT does not describe these slots"
            )
        keyframe = db["local_features"]["kp_counts"][:] > 0
        width = int(db["metadata"].attrs["frame_width"])
        height = int(db["metadata"].attrs["frame_height"])
        projection = (
            json.loads(_attr_str(db["calibration"].attrs["projection_json"]))
            if "calibration" in db and "projection_json" in db["calibration"].attrs
            else dict(gt.get("projection") or {"mode": "WEB_MERCATOR", "reference_gps": None})
        )

    pinned = valid & keyframe
    ids = np.where(pinned)[0]
    if not ids.size:
        raise LayerGTError("no keyframe slot has a valid GT affine")

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        target = partial / database.name
        shutil.copy2(database, target)
        lance = database.parent / "vectors.lance"
        if lance.exists():
            shutil.copytree(lance, partial / "vectors.lance")
        video = database.with_name(database.stem + "_keypoints.mp4")
        if copy_keypoint_video and video.exists():
            shutil.copy2(video, partial / video.name)

        converter = CoordinateConverter.from_metadata(projection)
        anchors = [
            AnchorCalibration(
                int(i),
                affines[i].copy(),
                {"transform_type": ORACLE_OPTIMIZER, "notes": "pinned to simulator GT"},
            )
            for i in ids
        ]
        anchors_json = json.dumps([a.to_dict() for a in anchors], ensure_ascii=False)
        frame_gps = np.full((num_slots, 2), np.nan)
        for i in ids:
            x, y = _centre(affines[i], width, height)
            frame_gps[i] = converter.metric_to_gps(float(x), float(y))

        gt_sha = _file_sha256(run.ground_truth)
        with h5py.File(target, "a") as db:
            for scratch in (
                "_calibration_pending",
                "_calibration_previous",
                "_frame_gps_pending",
                "_frame_gps_previous",
            ):
                if scratch in db:
                    del db[scratch]
            grp = db.require_group("calibration")
            matches = db["local_features"]["kp_counts"][:].astype(np.int32)
            _replace(grp, "frame_affine", np.where(pinned[:, None, None], affines, 0.0))
            _replace(grp, "frame_valid", pinned.astype(np.uint8))
            _replace(
                grp,
                "frame_georef_status",
                np.where(
                    pinned, int(GeoreferenceStatus.SUPPORTED), int(GeoreferenceStatus.INVALID)
                ).astype(np.uint8),
            )
            _replace(
                grp,
                "frame_origin",
                np.where(
                    pinned, int(CalibrationOrigin.ANCHOR), int(CalibrationOrigin.UNKNOWN)
                ).astype(np.uint8),
            )
            _replace(grp, "frame_rmse", np.zeros(num_slots))
            _replace(grp, "frame_disagreement", np.zeros(num_slots))
            _replace(grp, "frame_matches", np.where(pinned, matches, 0).astype(np.int32))
            _replace(grp, "frame_support_distance_slots", np.zeros(num_slots, dtype=np.int32))
            _replace(grp, "frame_support_anchor_count", pinned.astype(np.uint16))
            _replace(grp, "frame_graph_component", np.where(pinned, 0, -1).astype(np.int32))
            _replace(grp, "frame_anchor_linear_support", np.zeros(num_slots, dtype=np.uint8))
            attrs = {
                "version": "3.0",
                "optimizer": ORACLE_OPTIMIZER,
                "num_anchors": int(ids.size),
                "anchors_json": anchors_json,
                "projection_json": json.dumps(projection),
                "pin_exact_anchors": True,
                "provenance_version": 1,
                "georef_status_version": 1,
                "frame_origin_codes": "0=unknown,1=anchor,2=optimized,3=interpolated,"
                "4=extrapolated,5=anchor_linear_model",
                "frame_georef_status_codes": "0=unknown,1=supported,2=provisional,3=invalid",
                "frame_rmse_units": "reference_pixels",
                "disagreement_kind": "none",
                "num_temporal_edges": 0,
                "num_spatial_edges": 0,
                "component_anchors_json": json.dumps({"0": [int(i) for i in ids]}),
                "anchor_linear_model_intervals_json": "[]",
                "anchor_linear_model_parameters_json": json.dumps({"enabled": False}),
                "oracle_ground_truth": str(run.ground_truth),
                "oracle_ground_truth_sha256": gt_sha,
                "oracle_note": "evaluation only: every keyframe pinned to simulator GT",
            }
            for key, value in attrs.items():
                grp.attrs[key] = value
            _replace(db, "frame_gps", frame_gps)
            db.attrs["georef_generation"] = int(db.attrs.get("georef_generation", 0)) + 1

        calibration = {
            "version": MultiAnchorCalibration.VERSION,
            "projection": projection,
            "frame_size": [width, height],
            "anchors": [a.to_dict() for a in anchors],
            CALIBRATION_OWNER_KEY: source_id,
            "oracle": {
                "ground_truth": str(run.ground_truth),
                "ground_truth_sha256": gt_sha,
                "source_database": str(database),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        }
        (partial / "calibration.json").write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    return {
        "source_id": source_id,
        "out_dir": str(out_dir),
        "pinned_keyframes": int(ids.size),
        "keyframes": int(keyframe.sum()),
        "slots": num_slots,
        "anchor_consistency_before": consistency,
        **info,
    }
