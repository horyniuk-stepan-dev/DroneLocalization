"""Replay a simulator flight through the FULL live tracking loop and score it.

LOCALIZER_IMPROVEMENT_PLAN 0.2. Unlike replay_localization_ground_truth.py
(keyframes only), this runs every video frame the way TrackingWorker does:
a keyframe every ``tracking.keyframe_interval`` frames, keyframe-anchored LK
optical flow in between (src/tracking/keyframe_flow.py — the same code the
worker runs), the worker's dt rules, of_stride. That makes the OF path, the
smoother's odometry edges and the OF noise model measurable offline.

Ground truth comes from FlightSimulator's per-frame ``video.frames.jsonl`` and
is read only after each call; nothing from it reaches the Localizer.
Telemetry priors (heading -> yaw hint, height -> altitude hint) are passed
exactly like the worker does when ``flight_data.source`` is set, e.g.
``--set flight_data.source=csv --set flight_data.csv_path=.../telemetry.csv
--set flight_data.csv_preset=flightsim``. Difference from the live worker: no
YOLO static mask.

Scorecard (JSON ``summary``):
- final_error_m / raw_error_m — median, p95, max (output = what the GUI shows)
- jumps — consecutive outputs whose step deviates from the true step by more
  than --jump-m (default 10 m, the project accuracy target)
- per-path latency (keyframe / OF) and per-stage inclusive timings of the
  extractor, matcher and homography calls (GPU-synchronised with --sync-cuda)
The report's ``rows`` holds keyframe rows only (``slot``/``gt_valid``/
``confirmed``/``raw_error_m``/``processing_ms``), so scripts/compare_replays.py
can bootstrap A/B intervals on it unchanged; every processed frame (keyframe
and OF) is in ``tracking_rows`` and feeds the summary.

Sources and overrides work exactly like replay_multilayer_ground_truth.py
(same localizer construction, same AppConfig-validated ``--set``); layer search
follows user_config unless ``--layer-search-mode on|off``. Flags are A/B-tested
WITHOUT editing user_config.json:

    python scripts/replay_tracking_ground_truth.py ^
        --source main D:/maps/m/database.h5 D:/maps/m/calibration.json ^
        --source 2 D:/maps/2/database.h5 D:/maps/2/calibration.json ^
        --video flight/video.mp4 --frames-gt flight/video.frames.jsonl ^
        --json reports/odom_huber.json --set tracking.smoother_odom_huber_k=1.2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── pure helpers (unit-tested without torch) ─────────────────────────────────


def local_vector_m(a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    """East/north metres from a to b (equirectangular, fine over one step)."""
    lat0 = math.radians((a[0] + b[0]) / 2.0)
    r = 6371008.8
    de = math.radians(b[1] - a[1]) * r * math.cos(lat0)
    dn = math.radians(b[0] - a[0]) * r
    return np.array([de, dn])


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(local_vector_m(a, b)))


def stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not arr.size:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def step_errors(rows: list[dict]) -> list[dict]:
    """|est step - true step| between consecutive successful outputs that both
    have valid GT. A failure in between does not break the chain: a jump on
    re-acquisition is exactly what must be counted."""
    out = []
    prev = None
    for row in rows:
        if not (row.get("success") and row.get("gt_valid") and row.get("lat") is not None):
            continue
        if prev is not None:
            est = local_vector_m((prev["lat"], prev["lon"]), (row["lat"], row["lon"]))
            true = local_vector_m((prev["gt_lat"], prev["gt_lon"]), (row["gt_lat"], row["gt_lon"]))
            out.append(
                {
                    "frame": row["frame"],
                    "path": row["path"],
                    "step_error_m": float(np.linalg.norm(est - true)),
                }
            )
        prev = row
    return out


class WorkerClock:
    """dt rules of TrackingWorker.run (offline video, non-live)."""

    def __init__(self, keyframe_interval: int, frame_duration: float, of_local_speed: bool):
        self.ki = int(keyframe_interval)
        self.fd = float(frame_duration)
        self.of_local_speed = bool(of_local_speed)
        self.last_loc = -1.0
        self.last_kf = -1.0
        self.last_of = -1.0

    def dt(self, t: float, keyframe_path: bool) -> float:
        if keyframe_path:
            if self.last_kf < 0:
                return self.ki * self.fd
            d = t - self.last_kf
            return d if d > 0 else self.ki * self.fd
        base = self.last_of if self.of_local_speed else self.last_loc
        if base < 0:
            return self.fd
        d = t - base
        return d if d > 0 else self.fd


class StageTimer:
    """Wraps bound methods with inclusive wall-clock timing per frame."""

    def __init__(self, sync=None):
        self.sync = sync
        self.frame: dict[str, float] = {}

    def wrap(self, owner, name: str, label: str) -> None:
        fn = getattr(owner, name, None)
        if fn is None or not callable(fn):
            return

        def timed(*args, **kwargs):
            if self.sync:
                self.sync()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if self.sync:
                    self.sync()
                self.frame[label] = self.frame.get(label, 0.0) + (time.perf_counter() - t0) * 1e3

        setattr(owner, name, timed)

    def take(self) -> dict[str, float]:
        out, self.frame = self.frame, {}
        return out


def load_frames_gt(path: Path) -> dict[int, dict]:
    gt = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rec = json.loads(line)
                gt[int(rec["frame_index"])] = rec
    return gt


def summarize(rows: list[dict], jump_m: float) -> dict:
    def sel(path=None):
        return [r for r in rows if path is None or r["path"] == path]

    steps = step_errors(rows)
    summary = {}
    for name, path in (("all", None), ("keyframe", "kf"), ("optical_flow", "of")):
        rs = sel(path)
        ok = [r for r in rs if r.get("success")]
        summary[name] = {
            "attempted": len(rs),
            "success": len(ok),
            "success_rate": len(ok) / max(1, len(rs)),
            "final_error_m": stats(r.get("final_error_m") for r in ok),
            "processing_ms": stats(r.get("processing_ms") for r in rs),
        }
    for name, path in (("keyframe", "kf"), ("optical_flow", "of")):
        summary[name]["raw_error_m"] = stats(
            r.get("raw_error_m") for r in rows if r["path"] == path and r.get("success")
        )
    summary["step_error_m"] = stats(s["step_error_m"] for s in steps)
    summary["jumps"] = sum(s["step_error_m"] > jump_m for s in steps)
    summary["jump_m"] = jump_m
    summary["jump_frames"] = [s["frame"] for s in steps if s["step_error_m"] > jump_m][:50]
    stage: dict[str, list[float]] = {}
    for r in rows:
        for k, v in (r.get("stages_ms") or {}).items():
            stage.setdefault(f"{r['path']}:{k}", []).append(v)
    summary["stages_ms"] = {k: stats(v) for k, v in sorted(stage.items())}
    return summary


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from scripts.replay_multilayer_ground_truth import (
        build_localizer,
        config_overrides,
        layer_search_overrides,
        source_specs,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "DATABASE_H5", "CALIBRATION_JSON"),
        help="reference source; repeat for each layer (first = priority 0)",
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frames-gt", required=True, type=Path, help="video.frames.jsonl")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--jump-m", type=float, default=10.0)
    parser.add_argument(
        "--set",
        dest="config_set",
        action="append",
        metavar="SECTION.KEY=VALUE",
        help="config override for this run only (validated against AppConfig)",
    )
    parser.add_argument("--layer-search", action="append", metavar="KEY=VALUE")
    parser.add_argument(
        "--layer-search-mode",
        choices=["config", "on", "off"],
        default="config",
        help="config = whatever user_config says (what the app does)",
    )
    parser.add_argument(
        "--sync-cuda",
        action="store_true",
        help="cuda.synchronize around timed stages (honest stage times, slightly slower)",
    )
    args = parser.parse_args(argv)
    try:
        sources = source_specs(args.source)
        set_overrides = config_overrides(args.config_set)
        ls_overrides = layer_search_overrides(args.layer_search)
        for path in (args.video, args.frames_gt):
            if not path.is_file():
                raise ValueError(f"input file does not exist: {path}")
    except ValueError as exc:
        parser.error(str(exc))

    import cv2

    from config import APP_CONFIG, get_cfg
    from src.flight_data import build_flight_prior
    from src.geometry.transformations import GeometryTransforms
    from src.tracking.keyframe_flow import FlowChain, KeyframeFlow

    if args.layer_search_mode == "config":
        legacy = not bool(get_cfg(APP_CONFIG, "localization.layer_search.enabled", False))
    else:
        legacy = args.layer_search_mode == "off"
    if legacy and ls_overrides:
        parser.error("--layer-search does nothing while layer search is off")

    frames_gt = load_frames_gt(args.frames_gt)
    localizer, db_manager, _inventory = build_localizer(
        sources,
        disable_smoother=False,
        layer_search_overrides=ls_overrides,
        set_overrides=set_overrides,
        legacy_search=legacy,
    )
    config = localizer.config
    flight_prior = build_flight_prior(config)  # None unless flight_data.source is set
    extractor = localizer.feature_extractor
    matcher = localizer.matcher

    sync = None
    if args.sync_cuda:
        import torch

        if torch.cuda.is_available():
            sync = torch.cuda.synchronize
    timer = StageTimer(sync)
    for name in (
        "extract_global_descriptor",
        "extract_global_descriptors_multi",
        "extract_patch_tokens",
        "extract_local_features",
    ):
        timer.wrap(extractor, name, name)
    for name in ("match", "match_accuracy", "match_mnn", "match_sift"):
        timer.wrap(matcher, name, f"lightglue.{name}")
    original_h = GeometryTransforms.estimate_homography

    def timed_h(*a, **k):
        t0 = time.perf_counter()
        try:
            return original_h(*a, **k)
        finally:
            timer.frame["homography"] = (
                timer.frame.get("homography", 0.0) + (time.perf_counter() - t0) * 1e3
            )

    GeometryTransforms.estimate_homography = staticmethod(timed_h)

    ki = int(get_cfg(config, "tracking.keyframe_interval", 5))
    of_stride = max(1, int(get_cfg(config, "tracking.of_stride", 1)))
    flow = KeyframeFlow(
        fb_check=get_cfg(config, "tracking.of_fb_check", False),
        fb_max_px=get_cfg(config, "tracking.of_fb_max_px", 2.0),
        half_res=bool(get_cfg(config, "tracking.of_half_res", False)),
        win_px=get_cfg(config, "tracking.of_lk_win_px", 15),
        max_level=get_cfg(config, "tracking.of_lk_max_level", 2),
        max_anchor_age_s=get_cfg(config, "tracking.of_max_anchor_age_s", 0.0),
    )
    chain = FlowChain(
        enabled=get_cfg(config, "tracking.of_chain_on_keyframe_fail", False),
        max_s=get_cfg(config, "tracking.of_chain_max_s", 30.0),
    )
    retry_every = max(1, int(get_cfg(config, "tracking.keyframe_retry_every", 1)))

    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    clock = WorkerClock(ki, 1.0 / fps, bool(get_cfg(config, "tracking.of_local_speed", False)))
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    rows: list[dict] = []
    frame_idx = 0  # worker-relative index (keyframe cadence)
    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            video_frame = args.start_frame + frame_idx
            gt = frames_gt.get(video_frame, {})
            t = float(gt.get("timestamp_s", video_frame / fps))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = frame_idx % ki == 0
            flow.expire(t)
            retry_skip = not is_keyframe and not flow.has_anchor and frame_idx % retry_every != 0
            kf_path = (is_keyframe or not flow.has_anchor) and not retry_skip
            dt = clock.dt(t, kf_path)
            of_computed = (not kf_path) and (of_stride <= 1 or frame_idx % of_stride == 0)
            if of_computed or is_keyframe:
                clock.last_of = t

            result = {"success": False, "error": "Not processed"}
            path = "kf" if kf_path else "of"
            flow_quality = None
            timer.take()
            started = time.perf_counter()
            if kf_path:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hints = {}
                if flight_prior is not None:
                    hints["yaw_hint_deg"] = flight_prior.yaw_hint_deg(t)
                    altitude = flight_prior.altitude_m(t)
                    if altitude is not None:
                        hints["altitude_m"] = altitude
                try:
                    result = localizer.localize_frame(rgb, dt=dt, timestamp=t, **hints)
                except Exception as exc:  # mirror the worker: log and continue
                    result = {"success": False, "error": str(exc)}
                clock.last_kf = t
                if result.get("success") and result.get("fallback_mode") != "retrieval_only":
                    flow.anchor(gray, t)
                    chain.note_keyframe_success(t)
                elif not result.get("success"):
                    chain.try_rebase(localizer, flow, t)
            elif of_computed and not retry_skip:
                meas = flow.track(gray)
                if meas is not None:
                    flow_quality = meas.flow_quality
                    try:
                        result = localizer.localize_optical_flow(
                            meas.dx_px,
                            meas.dy_px,
                            dt=dt,
                            rot_width=frame.shape[1],
                            rot_height=frame.shape[0],
                            flow_affine=meas.flow_affine,
                            flow_quality=meas.flow_quality,
                            timestamp=t,
                        )
                    except Exception as exc:
                        result = {"success": False, "error": str(exc)}
                    if result.get("success"):
                        chain.note_of_success(gray, meas.flow_affine, t)
            else:
                path = "skip"
            elapsed = (time.perf_counter() - started) * 1e3
            stages = timer.take()
            frame_idx += 1
            if path == "skip":
                continue

            success = bool(result.get("success")) and result.get("matched_frame", -1) != -1
            if success:
                clock.last_loc = t
                try:  # TrackingWorker does the same after every success
                    db_manager.set_active_by_gps(result["lat"], result["lon"])
                    db_manager.update_retriever_positions(result["lat"], result["lon"])
                except Exception:
                    pass
            gt_point = gt.get("ground_center_gps")
            gt_valid = bool(gt.get("ground_center_valid", gt_point is not None)) and gt_point
            row = {
                "frame": video_frame,
                "slot": video_frame,
                "path": path,
                "t": t,
                "dt": dt,
                "success": success,
                "confirmed": success,
                "error": result.get("error"),
                "fallback_mode": result.get("fallback_mode"),
                "fix_source": result.get("fix_source") if success else None,
                "matched_frame": result.get("matched_frame"),
                "source_id": result.get("source_id", getattr(localizer, "_active_source_id", None)),
                "confidence": result.get("confidence"),
                "flow_quality": flow_quality,
                "processing_ms": elapsed,
                "stages_ms": stages,
                "gt_valid": bool(gt_valid),
                "gt_lat": float(gt_point[0]) if gt_valid else None,
                "gt_lon": float(gt_point[1]) if gt_valid else None,
                "camera_agl_eval_only": gt.get("camera_agl_m"),
                "lat": result.get("lat") if success else None,
                "lon": result.get("lon") if success else None,
                "raw_lat": result.get("raw_lat") if success else None,
                "raw_lon": result.get("raw_lon") if success else None,
                "final_error_m": None,
                "raw_error_m": None,
            }
            if success and gt_valid:
                target = (row["gt_lat"], row["gt_lon"])
                row["final_error_m"] = distance_m((float(row["lat"]), float(row["lon"])), target)
                if result.get("raw_lat") is not None:
                    row["raw_error_m"] = distance_m(
                        (float(result["raw_lat"]), float(result["raw_lon"])), target
                    )
            rows.append(row)
            if path == "kf":
                print(
                    f"frame={video_frame:>6} kf ok={success} "
                    f"raw={row['raw_error_m']} final={row['final_error_m']} ms={elapsed:.0f}"
                )
    finally:
        capture.release()
        db_manager.close_all()
        if flight_prior is not None:
            flight_prior.close()
        GeometryTransforms.estimate_homography = original_h

    report = {
        "inputs": {
            "sources": [
                {
                    "source_id": sp.source_id,
                    "database": str(sp.database),
                    "calibration": str(sp.calibration),
                }
                for sp in sources
            ],
            "video": str(args.video.resolve()),
            "frames_gt": str(args.frames_gt.resolve()),
            "config_overrides": set_overrides,
            "layer_search_overrides": ls_overrides,
            "legacy_search": legacy,
            "keyframe_interval": ki,
            "of_stride": of_stride,
        },
        "summary": summarize(rows, args.jump_m),
        "rows": [r for r in rows if r["path"] == "kf"],
        "tracking_rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        flat = [{k: v for k, v in r.items() if k != "stages_ms"} for r in rows]
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for r in flat for k in r}))
            writer.writeheader()
            writer.writerows(flat)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
