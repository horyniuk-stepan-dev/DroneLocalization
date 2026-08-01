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
import json
import math
import sys

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to database.h5")
    ap.add_argument("--gt", required=True, help="path to ground_truth.json")
    ap.add_argument("--csv", default="", help="optional per-slot CSV output")
    ap.add_argument("--gaps", default="", help="e.g. 83-124,132-173")
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

    ids, err, d_ang, d_scale = [], [], [], []
    for sid, s in sorted(slots.items()):
        if sid >= len(affine) or not valid[sid]:
            continue
        M = affine[sid]
        c = M[:, :2] @ np.array([cx, cy]) + M[:, 2]
        g = np.asarray(s["center_mercator"], dtype=np.float64)
        ids.append(sid)
        err.append(float(np.linalg.norm(c - g)) * k_ground)

        ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        d = ang - float(s["angle_deg"])
        d_ang.append(abs((d + 180.0) % 360.0 - 180.0))
        sx = float(np.hypot(M[0, 0], M[1, 0]))
        d_scale.append(abs(sx / max(float(s["sx"]), 1e-9) - 1.0) * 100.0)

    ids = np.asarray(ids)
    err = np.asarray(err)
    d_ang = np.asarray(d_ang)
    d_scale = np.asarray(d_scale)
    if err.size == 0:
        print("ERROR: no overlapping slots between DB and ground truth.")
        return 2

    is_anchor = np.array([bool(slots[int(i)].get("is_anchor")) for i in ids])

    print(f"\ndb      : {args.db}")
    print(f"gt      : {args.gt}")
    print(f"graph   : {n_anchors} anchors, {n_temporal} temporal + {n_spatial} spatial edges")
    print(f"slots   : {err.size} compared (of {len(slots)} in GT)")
    print(f"latitude: {lat_mid:.4f}  ->  ground = mercator x {k_ground:.4f}\n")

    print("CENTRE ERROR (ground metres) — THE number to tune against")
    print(_row("all slots", err))
    print(_row("at anchors", err[is_anchor]))
    print(_row("between anchors", err[~is_anchor]))

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
            f.write("slot,is_anchor,err_m,angle_err_deg,scale_err_pct\n")
            for i, e, a, s in zip(ids, err, d_ang, d_scale):
                f.write(f"{i},{int(slots[int(i)].get('is_anchor', False))},{e:.3f},{a:.4f},{s:.4f}\n")
        print(f"\nper-slot CSV written: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
