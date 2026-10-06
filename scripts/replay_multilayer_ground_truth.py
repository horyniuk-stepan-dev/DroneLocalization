"""Replay one query flight against one or more reference scale layers.

Only the decoded image and its timestamp enter Localizer. Query ground truth,
including camera altitude, is read by the evaluator after each localization call.
This is a chronological, unpaced replay; processing_ms is call duration, not
live frame age or a real-time scheduling guarantee.

Example (repeat --source for every reference layer)::

    python scripts/replay_multilayer_ground_truth.py \
      --source high D:/maps/high/database.h5 D:/maps/high/calibration.json \
      --source low D:/maps/low/database.h5 D:/maps/low/calibration.json \
      --video D:/query/video.mp4 --gt D:/query/ground_truth.json \
      --json D:/reports/layers.json --csv D:/reports/layers.csv

A single source is accepted for a smoke run, but cannot demonstrate handoff.
Ablations override ``localization.layer_search`` keys and record them in the
report inputs, e.g. ``--layer-search motion_gate=true --layer-search early_stop=true``.
Other config keys are overridden with ``--set SECTION.KEY=VALUE`` (nested keys
with dots, e.g. ``--set localization.use_patchify=true``). ``--legacy-search``
replays the multi-source path the app uses while ``layer_search.enabled`` is
false; its successful results count as confirmed. Rows carry the estimated and
GT coordinates, and the summary scores the step error between consecutive
confirmed fixes (how far a step deviates from the true motion, i.e. "jumps").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One dotted-path override implementation for the replay, sweep and build scripts.
from scripts.propagation_sweep import apply_overrides as apply_config_overrides  # noqa: E402


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    database: Path
    calibration: Path


def to_rgb(frame):
    """cv2 decodes BGR; the database builder and the tracking worker hand the
    localizer RGB (cvtColor BGR2RGB). Until 2026-10-04 the replay passed BGR, so
    every replay measured a colour-swapped query against RGB maps."""
    import numpy as np

    if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "DATABASE_H5", "CALIBRATION_JSON"),
        help="reference source; repeat for each layer",
    )
    parser.add_argument("--video", required=True, type=Path, help="independent query video")
    parser.add_argument("--gt", required=True, type=Path, help="query ground_truth.json with slots")
    parser.add_argument("--json", required=True, type=Path, help="detailed JSON report")
    parser.add_argument("--csv", required=True, type=Path, help="one row per evaluated slot")
    parser.add_argument("--every", type=positive_int, default=1, help="replay every Nth GT slot")
    parser.add_argument(
        "--source-validation",
        action="append",
        nargs=3,
        metavar=("ID", "TRACK", "REFERENCE_VALIDATION_CSV"),
        help="declare reference-map construction track (track_a_direct_gt or "
        "track_b_propagated) and attach independent reference GT validation",
    )
    parser.add_argument(
        "--max-slots",
        type=positive_int,
        help="stop after this many selected slots (for a clearly marked smoke run)",
    )
    parser.add_argument(
        "--altitude-edge",
        action="append",
        type=positive_float,
        default=None,
        help="upper edge of an AGL bin in metres; repeat in ascending order "
        "(default: 600, 800, 1000)",
    )
    parser.add_argument(
        "--false-confirmation-m",
        type=positive_float,
        help="optional, predeclared raw visual error threshold for false confirmations",
    )
    parser.add_argument("--disable-smoother", action="store_true")
    parser.add_argument(
        "--max-verifications", type=positive_int,
        help="override geometry budget for a controlled replay ablation",
    )
    parser.add_argument(
        "--layer-search",
        action="append",
        metavar="KEY=VALUE",
        help="override localization.layer_search.KEY (JSON value, e.g. true, 30.0) "
        "for a controlled ablation; repeatable, recorded in the report inputs",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="config_set",
        metavar="SECTION.KEY=VALUE",
        help="override an existing config key outside localization.layer_search "
        "(JSON value; nested keys with dots, e.g. localization.use_patchify=true); "
        "repeatable, validated against AppConfig, recorded in the report inputs",
    )
    parser.add_argument(
        "--flight-data",
        type=Path,
        default=None,
        help="telemetry CSV whose heading feeds the rotation prior (yaw hint) of every "
        "slot, like flight_data.source=csv in the app; GT is still never passed",
    )
    parser.add_argument(
        "--flight-data-preset",
        choices=["flightsim", "generic"],
        default="flightsim",
        help="column layout of --flight-data (flightsim = FlightSimulator telemetry.csv)",
    )
    parser.add_argument(
        "--legacy-search",
        action="store_true",
        help="replay with localization.layer_search.enabled=false (the multi-source "
        "path the app uses when layer search is off); a successful result counts "
        "as confirmed",
    )
    args = parser.parse_args(argv)
    try:
        args.layer_search_overrides = layer_search_overrides(args.layer_search)
        args.config_overrides = config_overrides(args.config_set)
        if args.legacy_search and (args.layer_search_overrides or args.max_verifications):
            raise ValueError("--layer-search/--max-verifications do nothing with --legacy-search")
        if (
            args.max_verifications is not None
            and "max_verifications" in args.layer_search_overrides
        ):
            raise ValueError("give max_verifications once: --max-verifications or --layer-search")
        args.sources = source_specs(args.source)
        args.validations = source_validation_specs(
            args.source_validation, {source.source_id for source in args.sources}
        )
        args.altitude_edges = altitude_edges(args.altitude_edge)
        for path in (args.video, args.gt) + ((args.flight_data,) if args.flight_data else ()):
            if not path.is_file():
                raise ValueError(f"input file does not exist: {path}")
    except ValueError as exc:
        parser.error(str(exc))
    return args


# The replay itself enables layer search and requires schema-validated maps.
REPLAY_FIXED_LAYER_SEARCH_KEYS = frozenset({"enabled", "require_schema"})
# Candidates kept per retrieval call and source in rows["retrieval_calls"].
RETRIEVAL_CALL_TOP = 5


def layer_search_overrides(raw: Iterable[str] | None) -> dict:
    from config.localization import LayerSearchConfig

    overrides = {}
    for item in raw or ():
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--layer-search expects KEY=VALUE, got {item!r}")
        if key not in LayerSearchConfig.model_fields:
            raise ValueError(f"unknown localization.layer_search key: {key!r}")
        if key in REPLAY_FIXED_LAYER_SEARCH_KEYS:
            raise ValueError(f"localization.layer_search.{key} is fixed by the replay")
        if key in overrides:
            raise ValueError(f"--layer-search {key} given more than once")
        try:
            overrides[key] = json.loads(text)
        except json.JSONDecodeError:
            overrides[key] = text
    try:
        validated = LayerSearchConfig(**overrides)
    except Exception as exc:  # pydantic.ValidationError; keep the CLI message short
        raise ValueError(f"invalid --layer-search value: {exc}") from exc
    return {key: getattr(validated, key) for key in overrides}


def config_overrides(raw: Iterable[str] | None) -> dict:
    """Parse SECTION.KEY=VALUE items; layer_search keys go through --layer-search."""
    overrides = {}
    for item in raw or ():
        name, sep, text = item.partition("=")
        name = name.strip()
        if not sep or "." not in name or not all(name.split(".")):
            raise ValueError(f"--set expects SECTION.KEY=VALUE, got {item!r}")
        if name == "localization.layer_search" or name.startswith("localization.layer_search."):
            raise ValueError(f"use --layer-search or --legacy-search for {name}")
        if name in overrides:
            raise ValueError(f"--set {name} given more than once")
        try:
            overrides[name] = json.loads(text)
        except json.JSONDecodeError:
            overrides[name] = text
    if not overrides:
        return {}
    from config import APP_CONFIG, AppConfig

    config = apply_config_overrides(APP_CONFIG, overrides)
    try:
        validated = AppConfig(**config).model_dump()
    except Exception as exc:  # pydantic.ValidationError; keep the CLI message short
        raise ValueError(f"invalid --set value: {exc}") from exc
    result = {}
    for name in overrides:
        node = validated
        for part in name.split("."):
            node = node[part]
        result[name] = node
    return result


def source_specs(raw: Iterable[list[str]]) -> list[SourceSpec]:
    specs = []
    seen = set()
    for source_id, db_text, cal_text in raw:
        if not source_id.strip() or source_id in seen:
            raise ValueError(f"source ID must be non-empty and unique: {source_id!r}")
        seen.add(source_id)
        database = Path(db_text).resolve()
        calibration = Path(cal_text).resolve()
        if not database.is_file():
            raise ValueError(f"database does not exist: {database}")
        if not calibration.is_file():
            raise ValueError(f"calibration does not exist: {calibration}")
        specs.append(SourceSpec(source_id, database, calibration))
    return specs


def source_validation_specs(raw: list[list[str]] | None, known_sources: set[str]) -> dict:
    """Summarize reference-map GT evidence; this is never query GT."""
    validations = {}
    for source_id, track, csv_text in raw or []:
        if source_id not in known_sources or source_id in validations:
            raise ValueError(f"source-validation ID is unknown or duplicated: {source_id!r}")
        if track not in {"track_a_direct_gt", "track_b_propagated"}:
            raise ValueError(f"unsupported map track: {track!r}")
        path = Path(csv_text).resolve()
        if not path.is_file():
            raise ValueError(f"reference validation CSV does not exist: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"georef_status", "surface_err_m"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"reference validation CSV is missing required columns: {path}")
            rows = list(reader)
        supported = [row for row in rows if int(row["georef_status"]) == 1]
        validations[source_id] = {
            "declared_map_track": track,
            "track_a_declared": track == "track_a_direct_gt",
            "reference_validation_csv": str(path),
            "reference_validation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reference_slots_evaluated": len(rows),
            "reference_supported_slots": len(supported),
            "reference_supported_surface_error_m": stats(
                float(row["surface_err_m"]) for row in supported if row["surface_err_m"]
            ),
        }
    return validations


def altitude_edges(values: list[float] | None) -> list[float]:
    edges = [600.0, 800.0, 1000.0] if values is None else values
    if any(not math.isfinite(edge) or edge <= 0 for edge in edges):
        raise ValueError("altitude edges must be finite positive metres")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("altitude edges must be strictly increasing")
    return edges


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    from pyproj import Geod

    return abs(float(Geod(ellps="WGS84").inv(a[1], a[0], b[1], b[0])[2]))


def stats(values: Iterable[float]) -> dict:
    sorted_values = sorted(
        float(value) for value in values if value is not None and math.isfinite(value)
    )
    if not sorted_values:
        return {"n": 0, "median": None, "p95": None, "max": None}

    def percentile(q: float) -> float:
        position = (len(sorted_values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)

    return {
        "n": len(sorted_values),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": sorted_values[-1],
    }


def score_result(
    slot: dict,
    result: dict,
    processing_ms: float | None,
    previous_source_id: str | None,
    known_sources: set[str],
    *,
    distance: Callable = distance_m,
    false_confirmation_m: float | None = None,
) -> dict:
    """Score a completed call; no GT value is passed to the localizer."""
    reported_source = result.get("source_id")
    confirmed = bool(result.get("success")) and result.get("status") == "confirmed"
    confirmed = confirmed and reported_source in known_sources
    confirmed = confirmed and all(result.get(key) is not None for key in ("lat", "lon"))
    status = result.get("status") or "failed"
    error = result.get("error")
    if result.get("success") and not confirmed:
        status = "invalid_result"
        error = "successful result lacks confirmed status, known source, or coordinates"

    gt_point = slot.get("ground_center_gps")
    gt_valid = bool(slot.get("ground_center_valid", gt_point is not None)) and gt_point is not None
    target = tuple(map(float, gt_point)) if gt_valid else None
    altitude = slot.get("camera_agl")
    row = {
        "slot": int(slot["slot"]),
        "video_frame": int(slot["video_frame"]),
        "timestamp": float(slot["timestamp"]),
        "camera_agl_eval_only": float(altitude) if altitude is not None else None,
        "gt_valid": gt_valid,
        "confirmed": confirmed,
        "status": status,
        "error": str(error) if error is not None else None,
        "source_id": reported_source if confirmed else None,
        "reported_source_id": reported_source,
        "previous_confirmed_source_id": previous_source_id,
        "bootstrap": confirmed and previous_source_id is None,
        "handoff": confirmed
        and previous_source_id is not None
        and reported_source != previous_source_id,
        "source_changed_reported": result.get("source_changed"),
        "layer_state": result.get("layer_state"),
        "pending_source_id": result.get("replay_pending_source_id"),
        "handoff_reason": result.get("replay_handoff_reason"),
        "matched_frame": result.get("matched_frame"),
        "matched_region": result.get("matched_region"),
        "inliers": result.get("inliers"),
        "confidence": result.get("confidence"),
        "scale_ratio": result.get("scale_ratio"),
        "rotation_deg": result.get("rotation_deg"),
        "fallback_mode": result.get("fallback_mode"),
        "lat": None,
        "lon": None,
        "raw_lat": None,
        "raw_lon": None,
        "gt_lat": target[0] if target is not None else None,
        "gt_lon": target[1] if target is not None else None,
        "processing_ms": processing_ms,
        "raw_error_m": None,
        "final_error_m": None,
        "false_confirmation": None,
        "exception_type": result.get("exception_type"),
        "observed_candidates": result.get("replay_observed_candidates", []),
        "retrieved_candidates_by_source": result.get("replay_retrieved_candidates_by_source", {}),
        "retrieval_calls": result.get("replay_retrieval_calls", []),
        "observed_sources": sorted(
            {candidate["source_id"] for candidate in result.get("replay_observed_candidates", [])}
        ),
    }
    search = result.get("search") or {}
    for key in (
        "verifications",
        "hypotheses",
        "budget_exhausted",
        "early_stop",
        "elapsed_ms",
        "sources_probed",
        "sources_available",
    ):
        row[f"search_{key}"] = search.get(key)
    if confirmed:
        row["lat"], row["lon"] = float(result["lat"]), float(result["lon"])
        if result.get("raw_lat") is not None and result.get("raw_lon") is not None:
            row["raw_lat"], row["raw_lon"] = float(result["raw_lat"]), float(result["raw_lon"])
    if confirmed and gt_valid:
        if result.get("raw_lat") is not None and result.get("raw_lon") is not None:
            row["raw_error_m"] = distance(
                (float(result["raw_lat"]), float(result["raw_lon"])), target
            )
        row["final_error_m"] = distance((float(result["lat"]), float(result["lon"])), target)
        if false_confirmation_m is not None and row["raw_error_m"] is not None:
            row["false_confirmation"] = row["raw_error_m"] > false_confirmation_m
    return row


def capture_observations(localizer) -> Callable[[], tuple[list[dict], dict]]:
    """Tap source-scoped retrieval and verified observations without changing decisions."""
    search = getattr(localizer, "_layer_search", None)
    if search is None:
        return lambda: ([], {}, [])
    original = search.search
    latest: list[dict] = []
    retrieved: dict[str, dict] = {}
    # One entry per retrieval call (one query view): top candidates per source with
    # their scores, for threshold calibration (scripts/retrieval_threshold_report.py).
    calls: list[dict] = []

    manager = getattr(localizer, "db_manager", None)
    if manager is not None:
        original_matches = manager.get_matches_by_source

        def tapped_matches(*args, **kwargs):
            groups = original_matches(*args, **kwargs)
            calls.append(
                {
                    sid: [
                        [int(frame), round(float(score), 4)]
                        for frame, score in candidates[:RETRIEVAL_CALL_TOP]
                    ]
                    for sid, candidates in groups.items()
                }
            )
            for sid, candidates in groups.items():
                record = retrieved.setdefault(sid, {"count": 0, "frames": set(), "top_score": None})
                record["count"] += len(candidates)
                record["frames"].update(int(frame) for frame, _ in candidates)
                for _, score in candidates:
                    score = float(score)
                    if record["top_score"] is None or score > record["top_score"]:
                        record["top_score"] = score
            return groups

        manager.get_matches_by_source = tapped_matches

        # The legacy (non-layer) path retrieves through get_best_match: one source
        # per call. Recorded the same way so thresholds can be calibrated for it.
        original_best = getattr(manager, "get_best_match", None)
        if original_best is not None:

            def tapped_best(*args, **kwargs):
                out = original_best(*args, **kwargs)
                sid, candidates = out
                if sid is not None and candidates:
                    calls.append(
                        {
                            sid: [
                                [int(frame), round(float(score), 4)]
                                for frame, score in candidates[:RETRIEVAL_CALL_TOP]
                            ]
                        }
                    )
                return out

            manager.get_best_match = tapped_best

    def tapped(*args, **kwargs):
        nonlocal latest
        observations = original(*args, **kwargs)
        latest = []
        for observation in observations:
            verification = observation.prepared[0]
            latest.append(
                {
                    "source_id": observation.source_id,
                    "frame": int(verification.candidate_id),
                    "quality": float(observation.quality),
                    "inliers": int(verification.inliers),
                    "rmse_px": float(verification.rmse),
                    "predicted_gps": list(map(float, observation.gps)),
                }
            )
        return observations

    def take_latest():
        nonlocal latest, retrieved, calls
        captured = latest
        candidates = {
            sid: {
                "count": record["count"],
                "unique_frames": sorted(record["frames"]),
                "top_score": record["top_score"],
            }
            for sid, record in retrieved.items()
        }
        taken = calls
        latest = []
        retrieved = {}
        calls = []
        return captured, candidates, taken

    search.search = tapped
    return take_latest


def replay_slots(
    slots: list[dict],
    capture,
    localizer,
    known_sources: set[str],
    *,
    clock: Callable[[], float] = time.perf_counter,
    distance: Callable = distance_m,
    false_confirmation_m: float | None = None,
    legacy: bool = False,
    flight_prior=None,
) -> list[dict]:
    import cv2

    rows = []
    previous_timestamp = None
    previous_source = None
    observed = capture_observations(localizer)
    for index, slot in enumerate(slots):
        video_frame = int(slot["video_frame"])
        timestamp = float(slot.get("timestamp", index))
        dt = 1.0 if previous_timestamp is None else max(1e-3, timestamp - previous_timestamp)
        previous_timestamp = timestamp
        capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ok, frame = capture.read()
        elapsed_ms = None
        if not ok or frame is None:
            result = {"success": False, "status": "decode_failed", "error": "decode_failed"}
        else:
            started = clock()
            try:
                # Image and time only: no query GT, altitude, or camera pose.
                # Optional telemetry heading (--flight-data) acts as a prior only.
                hint = flight_prior.yaw_hint_deg(timestamp) if flight_prior is not None else None
                extra = {"yaw_hint_deg": hint} if hint is not None else {}
                result = localizer.localize_frame(
                    to_rgb(frame), dt=dt, timestamp=timestamp, **extra
                )
            except Exception as exc:
                result = {
                    "success": False,
                    "status": "exception",
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            elapsed_ms = (clock() - started) * 1000.0
            result = dict(result)
            if legacy and result.get("success") and result.get("status") is None:
                # The legacy path has no confirmation state: what it returns is
                # what the app shows.
                result["status"] = "confirmed"
            taken = observed()
            verified, retrieved = taken[0], taken[1]
            result["replay_retrieval_calls"] = taken[2] if len(taken) > 2 else []
            result["replay_observed_candidates"] = verified
            result["replay_retrieved_candidates_by_source"] = retrieved
            handoff = getattr(getattr(localizer, "_layer_search", None), "handoff", None)
            if handoff is not None:
                result["replay_pending_source_id"] = handoff.pending
                result["replay_handoff_reason"] = handoff.reason
        scoring_slot = dict(slot, timestamp=timestamp)
        row = score_result(
            scoring_slot,
            result,
            elapsed_ms,
            previous_source,
            known_sources,
            distance=distance,
            false_confirmation_m=false_confirmation_m,
        )
        rows.append(row)
        if row["confirmed"]:
            previous_source = row["source_id"]
        print(
            f"slot={row['slot']:>4} status={row['status']} "
            f"source={row['source_id']} raw_err={row['raw_error_m']} "
            f"ms={elapsed_ms:.0f}"
            if elapsed_ms is not None
            else f"slot={row['slot']:>4} status=decode_failed"
        )
    return rows


def step_errors_m(rows: list[dict], lat_key: str, lon_key: str) -> list[float]:
    """|estimated step - GT step| between consecutive confirmed fixes, local metres."""
    points = [
        row
        for row in rows
        if row["confirmed"] and row.get(lat_key) is not None and row.get("gt_lat") is not None
    ]
    errors = []
    for a, b in zip(points, points[1:]):
        metres_per_deg_lon = 111_320.0 * math.cos(math.radians(b["gt_lat"]))
        dx = (b[lon_key] - a[lon_key]) - (b["gt_lon"] - a["gt_lon"])
        dy = (b[lat_key] - a[lat_key]) - (b["gt_lat"] - a["gt_lat"])
        errors.append(math.hypot(dx * metres_per_deg_lon, dy * 110_574.0))
    return errors


def summarize(rows: list[dict], edges: list[float], false_confirmation_m: float | None) -> dict:
    confirmed = [row for row in rows if row["confirmed"]]
    timed = [row["processing_ms"] for row in rows if row["processing_ms"] is not None]
    handoffs = [
        {
            "slot": row["slot"],
            "timestamp": row["timestamp"],
            "from_source_id": row["previous_confirmed_source_id"],
            "to_source_id": row["source_id"],
        }
        for row in rows
        if row["handoff"]
    ]
    boundaries = [0.0, *edges, math.inf]
    bins = {}
    for lo, hi in zip(boundaries, boundaries[1:]):
        selected = [
            row
            for row in rows
            if row["camera_agl_eval_only"] is not None and lo <= row["camera_agl_eval_only"] < hi
        ]
        label = f"{lo:g}-{hi:g}m" if math.isfinite(hi) else f"{lo:g}-infm"
        bins[label] = {
            "attempted": len(selected),
            "confirmed": sum(row["confirmed"] for row in selected),
            "raw_error_m": stats(row["raw_error_m"] for row in selected),
            "final_error_m": stats(row["final_error_m"] for row in selected),
            "processing_ms": stats(row["processing_ms"] for row in selected),
        }
    run = longest_unconfirmed_run = 0
    for row in rows:
        run = 0 if row["confirmed"] else run + 1
        longest_unconfirmed_run = max(longest_unconfirmed_run, run)
    gaps = [b["timestamp"] - a["timestamp"] for a, b in zip(confirmed, confirmed[1:])]
    source_counts = Counter(row["source_id"] for row in confirmed)
    observation_counts = Counter(
        candidate["source_id"] for row in rows for candidate in row["observed_candidates"]
    )
    retrieval_counts = Counter()
    for row in rows:
        retrieval_counts.update(
            {sid: item["count"] for sid, item in row["retrieved_candidates_by_source"].items()}
        )
    failures = Counter(row["error"] or row["status"] for row in rows if not row["confirmed"])
    false_evaluable = [row for row in confirmed if row["false_confirmation"] is not None]
    raw_steps = step_errors_m(rows, "raw_lat", "raw_lon")
    return {
        "attempted": len(rows),
        "confirmed": len(confirmed),
        "failed": len(rows) - len(confirmed),
        "confirmed_rate": len(confirmed) / len(rows) if rows else None,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "failure_reasons": dict(failures),
        "confirmed_by_source": dict(source_counts),
        "verified_observations_by_source": dict(observation_counts),
        "retrieved_hypotheses_by_source": dict(retrieval_counts),
        "search_verifications": stats(row["search_verifications"] for row in rows),
        "search_hypotheses": stats(row["search_hypotheses"] for row in rows),
        "search_elapsed_ms": stats(row["search_elapsed_ms"] for row in rows),
        "search_sources_probed_per_frame": stats(
            len(row["search_sources_probed"])
            for row in rows
            if row["search_sources_probed"] is not None
        ),
        "search_max_sources_available": max(
            (
                row["search_sources_available"]
                for row in rows
                if row["search_sources_available"] is not None
            ),
            default=None,
        ),
        "search_budget_exhausted_frames": sum(
            row["search_budget_exhausted"] is True for row in rows
        ),
        "search_early_stop_frames": sum(row.get("search_early_stop") is True for row in rows),
        "confirmed_handoff_reasons": dict(
            Counter(row["handoff_reason"] for row in confirmed if row.get("handoff_reason"))
        ),
        "handoff_count": len(handoffs),
        "handoffs": handoffs,
        "max_consecutive_unconfirmed_slots": longest_unconfirmed_run,
        "max_confirmed_timestamp_gap_s": max(gaps, default=None),
        "raw_error_m": stats(row["raw_error_m"] for row in confirmed),
        "final_error_m": stats(row["final_error_m"] for row in confirmed),
        "raw_step_error_m": stats(raw_steps),
        "raw_step_errors_over_threshold": sum(error > false_confirmation_m for error in raw_steps)
        if false_confirmation_m is not None
        else None,
        "final_step_error_m": stats(step_errors_m(rows, "lat", "lon")),
        "confirmed_fallback_modes": dict(
            Counter(row["fallback_mode"] for row in confirmed if row.get("fallback_mode"))
        ),
        "processing_ms": stats(timed),
        "warm_processing_ms": stats(timed[1:]),
        "altitude_bins": bins,
        "unknown_altitude_attempted": sum(row["camera_agl_eval_only"] is None for row in rows),
        "false_confirmation_threshold_m": false_confirmation_m,
        "false_confirmations": sum(row["false_confirmation"] is True for row in false_evaluable)
        if false_confirmation_m is not None
        else None,
        "false_confirmation_evaluable": len(false_evaluable)
        if false_confirmation_m is not None
        else None,
    }


def build_localizer(
    sources: list[SourceSpec],
    disable_smoother: bool,
    max_verifications: int | None = None,
    layer_search_overrides: dict | None = None,
    set_overrides: dict | None = None,
    legacy_search: bool = False,
):
    from config import APP_CONFIG
    from src.calibration.multi_calibration_manager import MultiCalibrationManager
    from src.core.project_video_source import ProjectVideoSource
    from src.database.multi_database_manager import MultiDatabaseManager
    from src.localization.localizer import Localizer
    from src.localization.matcher import FeatureMatcher
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor

    config = apply_config_overrides(APP_CONFIG, set_overrides or {})
    layer_config = config.setdefault("localization", {}).setdefault("layer_search", {})
    layer_config["enabled"] = not legacy_search
    layer_config["require_schema"] = True
    if max_verifications is not None:
        layer_config["max_verifications"] = max_verifications
    layer_config.update(layer_search_overrides or {})
    if disable_smoother:
        config.setdefault("tracking", {})["smoother_enabled"] = False
    projects = [
        ProjectVideoSource(
            source_id=spec.source_id,
            area_id="replay",
            video_path="",
            database_file=str(spec.database),
            calibration_file=str(spec.calibration),
        )
        for spec in sources
    ]
    db_manager = MultiDatabaseManager(projects, Path.cwd(), config=config)
    try:
        expected = {source.source_id for source in sources}
        loaded = set(db_manager.all_source_ids)
        if loaded != expected:
            raise ValueError(
                f"database load incomplete: expected {sorted(expected)}, loaded {sorted(loaded)}"
            )
        fingerprints = {
            sid: db_manager.get_database(sid).metadata.get("schema_fingerprint") for sid in expected
        }
        if (
            any(not value for value in fingerprints.values())
            or len(set(fingerprints.values())) != 1
        ):
            raise ValueError(f"source descriptor schemas are missing or disagree: {fingerprints}")
        inventory = {}
        for sid in expected:
            database = db_manager.get_database(sid)
            if database.lance_table is not None:
                vector_count = int(database.lance_table.count_rows())
                backend = "lancedb"
            elif database.global_descriptors is not None:
                vector_count = len(database.global_descriptors)
                backend = "faiss_or_geo_aware"
            else:
                raise ValueError(f"source {sid!r} has no searchable descriptor index")
            frame_statuses = getattr(database, "frame_georef_status", None)
            inventory[sid] = {
                "schema_fingerprint": fingerprints[sid],
                "indexed_vectors": vector_count,
                "retrieval_backend": backend,
                "reference_video_sha256": database.metadata.get("source_sha256"),
                "database_georef_status_counts": dict(
                    Counter(str(int(value)) for value in frame_statuses)
                ) if frame_statuses is not None else None,
            }
        calibrations = MultiCalibrationManager()
        calibrations.load_all(projects, Path.cwd())
        missing = [sid for sid in expected if not calibrations.get(sid).is_calibrated]
        if missing:
            raise ValueError(f"calibration could not be loaded for sources: {sorted(missing)}")
        model_manager = ModelManager(config=config)
        extractor = FeatureExtractor(
            model_manager.load_local_extractor(),
            model_manager.load_dinov2(),
            model_manager.device,
            config=config,
        )
        matcher = FeatureMatcher(model_manager=model_manager, config=config)
        config["_model_manager"] = model_manager
        first = sources[0].source_id
        database = db_manager.get_database(first)
        localizer = Localizer(
            database,
            extractor,
            matcher,
            calibrations.get(first),
            config=config,
            ref_frame_width=int(database.metadata.get("frame_width", 0)),
            ref_frame_height=int(database.metadata.get("frame_height", 0)),
            db_manager=db_manager,
            calib_manager=calibrations,
        )
        return localizer, db_manager, inventory
    except BaseException:
        db_manager.close_all()
        raise


def write_report(report: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = report["rows"]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["observed_candidates"] = json.dumps(row["observed_candidates"])
            flat["observed_sources"] = json.dumps(row["observed_sources"])
            flat["retrieved_candidates_by_source"] = json.dumps(
                row["retrieved_candidates_by_source"]
            )
            flat["search_sources_probed"] = json.dumps(row["search_sources_probed"])
            writer.writerow(flat)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    slots = gt["slots"][:: args.every]
    if args.max_slots is not None:
        slots = slots[: args.max_slots]
    if not slots:
        raise ValueError("query ground truth contains no selected slots")
    import cv2

    localizer, databases, inventory = build_localizer(
        args.sources,
        args.disable_smoother,
        args.max_verifications,
        args.layer_search_overrides,
        args.config_overrides,
        args.legacy_search,
    )
    capture = cv2.VideoCapture(str(args.video))
    try:
        if not capture.isOpened():
            raise ValueError(f"query video could not be opened: {args.video}")
        flight_prior = None
        if args.flight_data is not None:
            from src.flight_data import CsvFlightData, FlightPrior

            flight_prior = FlightPrior(
                CsvFlightData(args.flight_data, preset=args.flight_data_preset),
                localizer.config,
            )
        rows = replay_slots(
            slots,
            capture,
            localizer,
            {source.source_id for source in args.sources},
            false_confirmation_m=args.false_confirmation_m,
            legacy=args.legacy_search,
            flight_prior=flight_prior,
        )
    finally:
        capture.release()
        databases.close_all()
    summary = summarize(rows, args.altitude_edges, args.false_confirmation_m)
    report = {
        "inputs": {
            "sources": [
                {
                    "source_id": spec.source_id,
                    "database": str(spec.database),
                    "calibration": str(spec.calibration),
                    **inventory[spec.source_id],
                    **args.validations.get(spec.source_id, {
                        "declared_map_track": "unspecified",
                        "track_a_declared": False,
                        "reference_validation_csv": None,
                    }),
                }
                for spec in args.sources
            ],
            "video": str(args.video.resolve()),
            "ground_truth": str(args.gt.resolve()),
            "gt_version": gt.get("version"),
            "every": args.every,
            "max_slots": args.max_slots,
            "disable_smoother": args.disable_smoother,
            "max_verifications": args.max_verifications,
            "layer_search_overrides": args.layer_search_overrides,
            "config_overrides": args.config_overrides,
            "legacy_search": args.legacy_search,
            "flight_data": str(args.flight_data) if args.flight_data else None,
            "flight_data_preset": args.flight_data_preset if args.flight_data else None,
        },
        "timing_mode": "chronological_unpaced; processing_ms excludes video decode",
        "single_source_smoke_run": len(args.sources) == 1,
        "total_indexed_vectors": sum(item["indexed_vectors"] for item in inventory.values()),
        **summary,
        "rows": rows,
    }
    write_report(report, args.json, args.csv)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 2 if any(row["exception_type"] is not None for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
