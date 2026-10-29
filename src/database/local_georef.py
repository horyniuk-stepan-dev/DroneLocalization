"""Turn a propagated LOCAL (no-GPS) layer into a geographic one, and back.

The layer's local solution is kept verbatim in ``calibration/local_*`` so the
georeference can be redone with other control points or removed without
re-running propagation. Only quantities in map units are rewritten:
frame transforms, frame_disagreement (scaled) and frame_gps.
"""

from __future__ import annotations

import json
from datetime import datetime

import h5py
import numpy as np
from src.geometry.coordinates import CoordinateConverter
from src.security.project_scan import assert_project_writable
from src.utils.logging_utils import get_logger

from src.geometry.georef_fit import compose_affine, compose_homography

logger = get_logger(__name__)

_BACKUP_DATASETS = ("frame_affine", "frame_homography", "frame_disagreement")
_BACKUP_ATTRS = (
    "coordinate_kind",
    "coordinate_units",
    "projection_json",
    "anchors_json",
    "num_anchors",
    "relative_extent",
    "gauge_frame_id",
)


def _jsonable(v):
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def georef_info(database) -> dict | None:
    """The applied GCP georeference of this layer, or None."""
    path = getattr(database, "db_path", None)
    if not path:
        return None
    lock = getattr(database, "lock", None)
    if lock is not None:
        lock.acquire()
    try:
        f = getattr(database, "db_file", None)
        if f is None or "calibration" not in f:
            return None
        raw = f["calibration"].attrs.get("georef_transform_json")
        return json.loads(_jsonable(raw)) if raw is not None else None
    except (TypeError, ValueError, KeyError):
        return None
    finally:
        if lock is not None:
            lock.release()


def _frame_centers(affines, valid, w, h):
    c = np.array([w / 2.0, h / 2.0, 1.0])
    out = np.full((len(affines), 2), np.nan)
    out[valid] = affines[valid] @ c
    return out


def _swap_in(grp, name, data):
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data, compression="gzip")


def _rewrite(database, mutate) -> None:
    db_path = database.db_path
    assert_project_writable(db_path)
    database.lock.acquire()
    try:
        database.close()
        with h5py.File(db_path, "a") as f:
            if "calibration" not in f:
                raise ValueError("Шар не має пропагації")
            mutate(f)
            f.attrs["georef_generation"] = int(f.attrs.get("georef_generation", 0)) + 1
            f.flush()
    finally:
        try:
            database._load_hot_data()
        finally:
            database.lock.release()


def apply_local_georef(
    database,
    matrix: np.ndarray,
    converter: CoordinateConverter,
    *,
    kind: str,
    gcps: list[dict],
    rmse_m: float,
    anchors: list[dict],
) -> int:
    """Map every local frame transform through ``matrix`` (local -> metric).

    Returns the number of frames that received GPS.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (2, 3) or not np.isfinite(M).all():
        raise ValueError("Invalid georeference matrix")
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    w = int(database.metadata.get("frame_width", 0))
    h = int(database.metadata.get("frame_height", 0))
    if min(w, h) <= 0:
        raise ValueError("Database is missing frame dimensions")
    result = {}

    def mutate(f):
        grp = f["calibration"]
        if "local_frame_affine" not in grp:
            if grp.attrs.get("coordinate_kind") != "local_planar":
                raise ValueError("Шар не є локальною картою X/Y")
            for name in _BACKUP_DATASETS:
                if name in grp:
                    grp.create_dataset("local_" + name, data=grp[name][:], compression="gzip")
            grp.attrs["local_backup_attrs_json"] = json.dumps(
                {k: _jsonable(grp.attrs[k]) for k in _BACKUP_ATTRS if k in grp.attrs}
            )
        valid = grp["frame_valid"][:].astype(bool)
        local = grp["local_frame_affine"][:]
        geo = local.copy()
        for fid in np.flatnonzero(valid):
            geo[fid] = compose_affine(M, local[fid])
        _swap_in(grp, "frame_affine", geo)
        if "local_frame_homography" in grp:
            H = grp["local_frame_homography"][:]
            Hg = H.copy()
            for fid in np.flatnonzero(valid):
                Hg[fid] = compose_homography(M, H[fid])
            _swap_in(grp, "frame_homography", Hg)
        if "local_frame_disagreement" in grp:
            _swap_in(grp, "frame_disagreement", grp["local_frame_disagreement"][:] * scale)
        for k in ("relative_extent", "gauge_frame_id"):
            if k in grp.attrs:
                del grp.attrs[k]
        grp.attrs["coordinate_kind"] = "geographic"
        grp.attrs["coordinate_units"] = "projected_metres"
        grp.attrs["projection_json"] = json.dumps(converter.export_metadata())
        grp.attrs["anchors_json"] = json.dumps(anchors, ensure_ascii=False)
        grp.attrs["num_anchors"] = len(anchors)
        grp.attrs["georef_transform_json"] = json.dumps(
            {
                "kind": kind,
                "matrix": M.tolist(),
                "metres_per_unit": scale,
                "rmse_m": float(rmse_m),
                "gcps": gcps,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        )
        centers = _frame_centers(geo, valid, w, h)
        gps = np.full_like(centers, np.nan)
        ok = valid & np.isfinite(centers).all(axis=1)
        if ok.any():
            lat, lon = converter.metric_to_gps_array(centers[ok, 0], centers[ok, 1])
            gps[ok, 0], gps[ok, 1] = lat, lon
        _swap_in(f, "frame_gps", gps)
        result["n"] = int(ok.sum())

    _rewrite(database, mutate)
    logger.success(
        f"Local map georeferenced ({kind}, {len(gcps)} GCPs, RMSE {rmse_m:.2f} m): "
        f"{result['n']} frames with GPS"
    )
    return result["n"]


def reset_local_georef(database) -> None:
    """Restore the pure local X/Y solution saved by ``apply_local_georef``."""

    def mutate(f):
        grp = f["calibration"]
        if "local_frame_affine" not in grp:
            raise ValueError("Шар не має збереженої локальної карти")
        for name in _BACKUP_DATASETS:
            if "local_" + name in grp:
                _swap_in(grp, name, grp["local_" + name][:])
                del grp["local_" + name]
        attrs = json.loads(_jsonable(grp.attrs["local_backup_attrs_json"]))
        for k in _BACKUP_ATTRS:
            if k in attrs:
                grp.attrs[k] = attrs[k]
            elif k in grp.attrs:
                del grp.attrs[k]
        for k in ("georef_transform_json", "local_backup_attrs_json"):
            if k in grp.attrs:
                del grp.attrs[k]
        n = len(grp["frame_valid"])
        _swap_in(f, "frame_gps", np.full((n, 2), np.nan))

    _rewrite(database, mutate)
    logger.info("Georeference removed: layer is a local X/Y map again")
