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
    """Replace ``name`` with ``data``; the new dataset is fully written first.

    The old object is only unlinked after the replacement exists under a
    scratch name, so a failure while writing never leaves ``name`` missing.
    """
    scratch = name + "__pending"
    if scratch in grp:
        del grp[scratch]
    grp.create_dataset(scratch, data=data, compression="gzip")
    if name in grp:
        del grp[name]
    grp.move(scratch, name)


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
    frame_matrices: dict[int, np.ndarray] | None = None,
    extra: dict | None = None,
) -> int:
    """Map every local frame transform through ``matrix`` (local -> metric).

    ``frame_matrices`` (frame id -> 2x3) overrides ``matrix`` per frame — the
    non-rigid graph georeference; frames missing from it use ``matrix``.
    ``extra`` is merged into the stored ``georef_transform_json``.
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
        # Compute everything in memory first: any failure (projection, bad
        # matrix) must happen before the first byte of the layer is touched.
        valid = grp["frame_valid"][:].astype(bool)
        local = grp["local_frame_affine"][:]
        per_frame = {}
        for fid in np.flatnonzero(valid):
            Mf = M
            if frame_matrices is not None and int(fid) in frame_matrices:
                Mf = np.asarray(frame_matrices[int(fid)], dtype=np.float64)
                if Mf.shape != (2, 3) or not np.isfinite(Mf).all():
                    raise ValueError(f"Invalid georeference matrix for frame {fid}")
            per_frame[int(fid)] = Mf
        geo = local.copy()
        for fid, Mf in per_frame.items():
            geo[fid] = compose_affine(Mf, local[fid])
        Hg = None
        if "local_frame_homography" in grp:
            H = grp["local_frame_homography"][:]
            Hg = H.copy()
            for fid, Mf in per_frame.items():
                Hg[fid] = compose_homography(Mf, H[fid])
        disagreement = None
        if "local_frame_disagreement" in grp:
            disagreement = grp["local_frame_disagreement"][:] * scale
            if frame_matrices is not None and disagreement.ndim >= 1:
                for fid, Mf in per_frame.items():
                    if fid < len(disagreement):
                        sf = float(np.sqrt(abs(np.linalg.det(Mf[:, :2]))))
                        disagreement[fid] = disagreement[fid] / scale * sf
        centers = _frame_centers(geo, valid, w, h)
        gps = np.full_like(centers, np.nan)
        ok = valid & np.isfinite(centers).all(axis=1)
        if ok.any():
            lat, lon = converter.metric_to_gps_array(centers[ok, 0], centers[ok, 1])
            gps[ok, 0], gps[ok, 1] = lat, lon
        ground_scale = 1.0
        if ok.any():
            ground_scale = float(converter.ground_scale_factor(float(np.nanmean(gps[ok, 0]))))
        transform_json = json.dumps(
            {
                "kind": kind,
                "matrix": M.tolist(),
                # Projected (Web-Mercator) metres per local unit, as stored in
                # frame_affine; ground_metres_per_unit is the real-world scale.
                "metres_per_unit": scale,
                "ground_metres_per_unit": scale * ground_scale,
                "rmse_m": float(rmse_m),
                "gcps": gcps,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                **(extra or {}),
            },
            ensure_ascii=False,
        )
        projection_json = json.dumps(converter.export_metadata())

        # Write phase. The layer is marked geographic only as the very last
        # step; if anything below is interrupted it stays "local_planar" and
        # the backup in local_* restores it.
        grp.attrs["coordinate_kind"] = "georef_in_progress"
        _swap_in(grp, "frame_affine", geo)
        if Hg is not None:
            _swap_in(grp, "frame_homography", Hg)
        if disagreement is not None:
            _swap_in(grp, "frame_disagreement", disagreement)
        _swap_in(f, "frame_gps", gps)
        for k in ("relative_extent", "gauge_frame_id"):
            if k in grp.attrs:
                del grp.attrs[k]
        grp.attrs["coordinate_units"] = "projected_metres"
        grp.attrs["projection_json"] = projection_json
        grp.attrs["anchors_json"] = json.dumps(anchors, ensure_ascii=False)
        grp.attrs["num_anchors"] = len(anchors)
        grp.attrs["georef_transform_json"] = transform_json
        grp.attrs["coordinate_kind"] = "geographic"
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
