"""Split keyframe failures under appearance change: retrieval miss vs match miss.

Vision-only plan 2.5. For the failed keyframes of a tracking replay report
(scripts/replay_tracking_ground_truth.py) this re-runs each frame cold
(reset_session, no priors) and answers, per frame:

* positives: reference frames (all layers) whose ground footprint contains
  the ground-truth ground centre;
* retrieval: is a positive among the localizer's retrieved candidates
  (DebugCollector.retrieval_candidates of the chosen layer), at which rank;
* oracle match: ALIKED + LightGlue + RANSAC of the query against the nearest
  positive of every layer at angle 0 and every scale of the pyramid — can the
  matcher see the place at all;
* class: ``ok`` (localizer succeeded cold), ``retrieval_miss`` (oracle >= --min-inliers
  but no positive retrieved), ``match_miss`` (positive retrieved, localizer
  still failed), ``unmatchable`` (oracle < --min-inliers).

Angle 0 assumes the north-up query of the route3min scenarios (yaw 0) and the
heading-hold 0 reference layers of testtopboch_max.

    python scripts/diagnose_appearance_gap.py \
      --source main D:/.../sources/main/database.h5 D:/.../sources/main/calibration.json \
      --source 2 ... --source 3 ... \
      --video D:/.../video.mp4 --frames-gt D:/.../video.frames.jsonl \
      --report D:/.../reports/wayback_set/vision@wayback_2014-02-20.json \
      --json D:/.../reports/gap_2014.json --max-frames 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _xy(lat, lon, lat0, lon0):
    k = 6_371_008.8 * math.pi / 180.0
    return (lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k


def layer_footprints(database, calibration):
    """[(frame_id, (4, 2) lat/lon corners, centre lat/lon)] of georeferenced frames."""
    from src.geometry.frame_transform import get_frame_transform, project_frame_points

    conv = calibration.converter
    out = []
    for fid in range(int(database.get_num_frames())):
        T = get_frame_transform(database, fid)
        if T is None:
            continue
        h, w = database.get_frame_size(fid)
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], dtype=np.float64)
        try:
            T = np.asarray(T, dtype=np.float64)
            m = (
                project_frame_points(T, pts, w, h)
                if T.shape == (3, 3)
                else pts @ T[:, :2].T + T[:, 2]
            )
            lat, lon = conv.metric_to_gps_array(m[:, 0], m[:, 1])
        except Exception:
            continue
        ll = np.column_stack([lat, lon])
        if np.isfinite(ll).all():
            out.append((fid, ll[:4], ll[4]))
    return out


def positives(fps, lat, lon):
    """Frame ids whose footprint contains (lat, lon), nearest centre first."""
    import cv2

    hits = []
    for fid, poly, centre in fps:
        p = np.array([_xy(a, b, lat, lon) for a, b in poly], dtype=np.float32)
        if cv2.pointPolygonTest(p.reshape(-1, 1, 2), (0.0, 0.0), False) >= 0:
            cx, cy = _xy(centre[0], centre[1], lat, lon)
            hits.append((math.hypot(cx, cy), fid))
    return [fid for _, fid in sorted(hits)]


def oracle_inliers(loc, frame_rgb, database, fid, scales):
    """Best RANSAC inlier count of the query vs reference frame ``fid`` over scales."""
    import cv2

    ref = database.get_local_features(fid)
    norm, _ = loc.normalizer.normalize(frame_rgb)
    best = (0, None)
    for sc in scales:
        img = norm
        if abs(sc - 1.0) > 1e-3:
            img, _ = loc._scale_manager.normalize(norm, float(sc))
        q = loc.feature_extractor.extract_local_features(img)
        mq, mr = loc.matcher.match(q, ref)
        if len(mq) < 8:
            continue
        H, mask = cv2.findHomography(
            np.asarray(mq, np.float32), np.asarray(mr, np.float32), cv2.USAC_MAGSAC, 3.0
        )
        n = int(mask.sum()) if mask is not None else 0
        if n > best[0]:
            best = (n, float(sc))
    return best


def main(argv=None) -> int:
    from scripts.replay_multilayer_ground_truth import (
        build_localizer,
        config_overrides,
        source_specs,
    )
    from scripts.replay_tracking_ground_truth import load_frames_gt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source", action="append", nargs=3, required=True, metavar=("ID", "DB", "CAL")
    )
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--frames-gt", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path, help="replay_tracking_ground_truth JSON")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=60, help="evenly sampled failed keyframes")
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--scales", default="0.5,0.7,1.0,1.4,2.0")
    ap.add_argument("--pose-top-k", type=int, default=50)
    ap.add_argument("--set", dest="config_set", action="append", metavar="SECTION.KEY=VALUE")
    args = ap.parse_args(argv)
    os.environ.setdefault("DRONELOC_FAILURE_CSV", "off")

    import cv2

    from src.localization.debug_collector import DebugCollector

    sources = source_specs(args.source)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failed = [
        r["frame"]
        for r in report["tracking_rows"]
        if r["path"] == "kf" and not r["success"] and r.get("gt_valid")
    ]
    failed = sorted(set(failed))
    if len(failed) > args.max_frames:
        idx = np.linspace(0, len(failed) - 1, args.max_frames).round().astype(int)
        failed = [failed[i] for i in sorted(set(idx))]
    gt = load_frames_gt(args.frames_gt)
    scales = [float(s) for s in args.scales.split(",")]

    loc, dbm, _inv = build_localizer(
        sources,
        disable_smoother=True,
        set_overrides=config_overrides(args.config_set),
        legacy_search=True,
    )
    t0 = time.perf_counter()
    fps = {}
    for sp in sources:
        sid = sp.source_id
        fps[sid] = layer_footprints(dbm.get_database(sid), loc.calib_manager.get(sid))
    print(f"footprints: { {k: len(v) for k, v in fps.items()} } ({time.perf_counter() - t0:.1f} s)")

    cap = cv2.VideoCapture(str(args.video))
    rows = []
    try:
        for n, fr in enumerate(failed, 1):
            g = gt.get(fr, {})
            point = g.get("ground_center_gps")
            if not point:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lat, lon = float(point[0]), float(point[1])
            pos = {sid: positives(v, lat, lon) for sid, v in fps.items()}

            loc.reset_session()
            col = DebugCollector(want_matches=False, want_dino_pca=False, want_depth=False)
            res = loc.localize_frame(rgb, collector=col, timestamp=float(g.get("timestamp_s", 0.0)))
            sid = getattr(col, "retrieval_source", None) or getattr(loc, "_active_source_id", None)
            cands = [int(c) for c, _ in (col.retrieval_candidates or [])]
            ranks = [i for i, c in enumerate(cands) if c in set(pos.get(sid, []))]
            oracle = {}
            for lid, ids in pos.items():
                # nearest positive that has stored local features (slots
                # without keypoints carry a georeference but cannot be matched)
                for fid in ids[:10]:
                    try:
                        oracle[lid] = (
                            *oracle_inliers(loc, rgb, dbm.get_database(lid), fid, scales),
                            fid,
                        )
                        break
                    except ValueError:
                        continue
            best_layer, (best_inl, _best_sc, _best_fid) = max(
                oracle.items(), key=lambda kv: kv[1][0], default=(None, (0, None, None))
            )
            # Retrieval at the true pose (angle 0, oracle scale, oracle layer):
            # does the global descriptor find the place when angle/scale are right?
            pose_rank = None
            if best_layer is not None and best_inl >= args.min_inliers:
                o_sc = oracle[best_layer][1] or 1.0
                norm, _ = loc.normalizer.normalize(rgb)
                img = (
                    norm if abs(o_sc - 1.0) < 1e-3 else loc._scale_manager.normalize(norm, o_sc)[0]
                )
                desc = loc.feature_extractor.extract_global_descriptor(img)
                got = dbm.get_matches_by_source(desc, args.pose_top_k, source_ids=[best_layer])
                ids = [int(c) for c, _ in got.get(best_layer, [])]
                hit = [i for i, c in enumerate(ids) if c in set(pos[best_layer])]
                pose_rank = hit[0] if hit else None
            if res.get("success"):
                cls = "ok"
            elif best_inl < args.min_inliers:
                cls = "unmatchable"
            elif ranks:
                cls = "match_miss"
            else:
                cls = "retrieval_miss"
            row = {
                "frame": fr,
                "t": g.get("timestamp_s"),
                "agl": g.get("camera_agl_m"),
                "class": cls,
                "cold_success": bool(res.get("success")),
                "error": res.get("error"),
                "retrieval_layer": sid,
                "retrieval_angle": col.global_angle,
                "retrieval_scale": col.scale,
                "retrieval_top_score": col.global_score if hasattr(col, "global_score") else None,
                "positive_rank": ranks[0] if ranks else None,
                "pose_retrieval_rank": pose_rank,
                "positives_per_layer": {k: len(v) for k, v in pos.items()},
                "oracle": {
                    k: {"inliers": v[0], "scale": v[1], "frame": v[2]} for k, v in oracle.items()
                },
                "oracle_best_layer": best_layer,
                "oracle_best_inliers": best_inl,
            }
            rows.append(row)
            print(
                f"[{n}/{len(failed)}] frame {fr} t={row['t']} {cls:15} layer={sid} "
                f"rank={row['positive_rank']} oracle={best_layer}:{best_inl}"
            )
    finally:
        cap.release()
        dbm.close_all()

    summary = {
        "frames": len(rows),
        "classes": dict(Counter(r["class"] for r in rows)),
        "oracle_layers": dict(
            Counter(
                r["oracle_best_layer"] for r in rows if r["oracle_best_inliers"] >= args.min_inliers
            )
        ),
        "min_inliers": args.min_inliers,
        # among frames whose place is matchable: positive rank when retrieving
        # at the true angle / scale / layer (None = not in --pose-top-k)
        "pose_rank_le_11": sum(
            1
            for r in rows
            if r["pose_retrieval_rank"] is not None and r["pose_retrieval_rank"] < 12
        ),
        "pose_rank_12_plus": sum(
            1
            for r in rows
            if r["pose_retrieval_rank"] is not None and r["pose_retrieval_rank"] >= 12
        ),
        "pose_rank_none": sum(
            1
            for r in rows
            if r["pose_retrieval_rank"] is None and r["oracle_best_inliers"] >= args.min_inliers
        ),
        "retrieved_angle_off_0": sum(
            1
            for r in rows
            if min(
                abs((r["retrieval_angle"] or 0) % 360), 360 - abs((r["retrieval_angle"] or 0) % 360)
            )
            > 30
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {"inputs": vars(args) | {"source": args.source}, "summary": summary, "rows": rows},
            indent=2,
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
