"""Measure propagation accuracy against the simulator's ground truth.

Every number the propagation log prints — Drift, RMSE, matching, residuals —
measures the graph's INTERNAL consistency, and every one of them improves when
constraints are removed. Tuning against them drives you toward an empty graph.
This script measures the only thing that matters: how far each frame's centre
ended up from where the simulator says it actually was.

Usage:
    python scripts/validate_vs_ground_truth.py \
        --db "D:/My Projects/TEST/topnew/sources/main/database.h5" \
        --gt "D:/My Projects/FlightSimulator/ground_truth.json"

Optional:
    --csv out.csv        per-slot errors for plotting
    --gaps 83-124,132-173,...   slot ranges flagged by anchor_gap_check, to get
                                a separate error figure for them
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def _row(label: str, e: np.ndarray) -> str:
    if e.size == 0:
        return f"  {label:<22} —"
    return (
        f"  {label:<22} n={e.size:>5}  median={np.median(e):7.2f}  "
        f"p95={_pct(e, 95):8.2f}  max={e.max():9.2f}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to database.h5")
    ap.add_argument("--gt", required=True, help="path to ground_truth.json")
    ap.add_argument("--csv", default="", help="optional per-slot CSV output")
    ap.add_argument("--gaps", default="", help="e.g. 83-124,132-173")
    ap.add_argument("--video", default="", help="verify DB source SHA-256 against this video")
    ap.add_argument(
        "--max-supported-p95",
        type=float,
        default=None,
        help="optional failing gate for supported-slot surface error in ground metres",
    )
    args = ap.parse_args()

    import h5py

    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    slots = {int(s["slot"]): s for s in gt["slots"]}
    fw, fh = gt["frame_size"]
    cx, cy = fw / 2.0, fh / 2.0

    with h5py.File(args.db, "r") as f:
        if "calibration" not in f or "frame_affine" not in f["calibration"]:
            print("ERROR: database has no propagation data — run propagation first.")
            return 2
        affine = f["calibration"]["frame_affine"][:]
        valid = f["calibration"]["frame_valid"][:].astype(bool)
        n_anchors = int(f["calibration"].attrs.get("num_anchors", 0))
        n_temporal = int(f["calibration"].attrs.get("num_temporal_edges", 0))
        n_spatial = int(f["calibration"].attrs.get("num_spatial_edges", 0))
        status = (
            f["calibration"]["frame_georef_status"][:]
            if "frame_georef_status" in f["calibration"]
            else np.zeros(len(valid), dtype=np.uint8)
        )
        try:
            saved_anchors = json.loads(f["calibration"].attrs.get("anchors_json", "[]"))
            db_anchor_ids = {int(anchor["frame_id"]) for anchor in saved_anchors}
        except Exception:
            db_anchor_ids = set()
        source_sha256 = str(f["metadata"].attrs.get("source_sha256", ""))
        source_path = str(f["metadata"].attrs.get("source_path", ""))

    if args.video:
        actual_source_sha = _sha256(Path(args.video))
        if not source_sha256:
            print("ERROR: database has no source_sha256; rebuild or register source identity.")
            return 2
        if actual_source_sha != source_sha256:
            print(
                "ERROR: database/video source identity mismatch\n"
                f"  DB:    {source_sha256}\n"
                f"  video: {actual_source_sha}"
            )
            return 2

    # Mercator -> ground metres. Distances in EPSG:3857 are inflated by 1/cos(lat).
    try:
        from pyproj import Transformer

        t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        mid = slots[sorted(slots)[len(slots) // 2]]["center_mercator"]
        _, lat_mid = t.transform(mid[0], mid[1])
        k_ground = math.cos(math.radians(lat_mid))
    except Exception as e:  # noqa: BLE001
        print(f"(pyproj unavailable: {e} — reporting projection metres)")
        lat_mid, k_ground = float("nan"), 1.0

    ids, err, err_affine_model, d_ang, d_scale, georef_status = [], [], [], [], [], []
    for sid, s in sorted(slots.items()):
        if sid >= len(affine) or not valid[sid]:
            continue
        M = affine[sid]
        c = M[:, :2] @ np.array([cx, cy]) + M[:, 2]
        g = np.asarray(s["center_mercator"], dtype=np.float64)
        g_affine = np.asarray(s.get("affine_center_mercator", g), dtype=np.float64)
        ids.append(sid)
        err.append(float(np.linalg.norm(c - g)) * k_ground)
        err_affine_model.append(float(np.linalg.norm(c - g_affine)) * k_ground)
        georef_status.append(int(status[sid]) if sid < len(status) else 0)

        ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        d = ang - float(s["angle_deg"])
        d_ang.append(abs((d + 180.0) % 360.0 - 180.0))
        sx = float(np.hypot(M[0, 0], M[1, 0]))
        d_scale.append(abs(sx / max(float(s["sx"]), 1e-9) - 1.0) * 100.0)

    ids = np.asarray(ids)
    err = np.asarray(err)
    err_affine_model = np.asarray(err_affine_model)
    d_ang = np.asarray(d_ang)
    d_scale = np.asarray(d_scale)
    georef_status = np.asarray(georef_status, dtype=np.uint8)
    if err.size == 0:
        print("ERROR: no overlapping slots between DB and ground truth.")
        return 2

    is_anchor = np.array([int(i) in db_anchor_ids for i in ids])
    supported = georef_status == 1
    provisional = georef_status == 2
    invalid = georef_status == 3
    unknown = georef_status == 0

    print(f"\ndb      : {args.db}")
    print(f"gt      : {args.gt}")
    print(f"graph   : {n_anchors} anchors, {n_temporal} temporal + {n_spatial} spatial edges")
    print(f"slots   : {err.size} compared (of {len(slots)} in GT)")
    print(f"source  : {source_path or 'unrecorded'}")
    print(f"GT sha  : {_sha256(Path(args.gt))}")
    print(f"latitude: {lat_mid:.4f}  ->  ground = mercator x {k_ground:.4f}\n")

    print("CENTRE ERROR VS TRUE SURFACE INTERSECTION (ground metres)")
    print(_row("all slots", err))
    print(_row("at anchors", err[is_anchor]))
    print(_row("between anchors", err[~is_anchor]))
    print(_row("supported", err[supported]))
    print(_row("provisional", err[provisional]))
    print(_row("invalid", err[invalid]))
    print(_row("unknown", err[unknown]))
    print(
        "  coverage               "
        f"supported={supported.sum()}/{err.size} ({supported.mean():.1%}), "
        f"provisional={provisional.sum()}, invalid={invalid.sum()}, unknown={unknown.sum()}"
    )

    print("\nAFFINE-MODEL CENTRE ERROR (isolates graph/calibration error)")
    print(_row("all slots", err_affine_model))
    print(_row("supported", err_affine_model[supported]))
    print(_row("provisional", err_affine_model[provisional]))
    print(_row("invalid", err_affine_model[invalid]))

    for spec in filter(None, args.gaps.split(",")):
        a, b = (int(x) for x in spec.split("-"))
        m = (ids >= a) & (ids <= b)
        print(_row(f"gap {a}-{b}", err[m]))

    print("\nORIENTATION / SCALE")
    print(_row("angle error (deg)", d_ang))
    print(_row("scale error (%)", d_scale))

    worst = ids[np.argsort(err)[::-1][:10]]
    print("\nworst slots: " + ", ".join(f"#{i}({err[ids == i][0]:.0f}m)" for i in worst))

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("slot,is_anchor,georef_status,surface_err_m,affine_model_err_m,angle_err_deg,scale_err_pct\n")
            for i, e, em, st, a, s in zip(
                ids, err, err_affine_model, georef_status, d_ang, d_scale
            ):
                f.write(
                    f"{i},{int(i in db_anchor_ids)},{st},{e:.3f},{em:.3f},{a:.4f},{s:.4f}\n"
                )
        print(f"\nper-slot CSV written: {args.csv}")

    if args.max_supported_p95 is not None:
        observed = _pct(err[supported], 95)
        if not np.isfinite(observed) or observed > args.max_supported_p95:
            print(
                f"\nFAIL: supported p95={observed:.3f} m exceeds "
                f"{args.max_supported_p95:.3f} m"
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
