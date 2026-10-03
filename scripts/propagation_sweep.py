"""Sweep calibration-propagation settings on one layer with a persistent match cache.

Pairwise matching (LightGlue + homography RANSAC) dominates propagation time and
does not depend on the pose-graph settings, so every ``_match_and_build_edge``
result is cached by (from slot, to slot, matching settings). A cache filled once
on the GPU machine lets later sweeps run without loading any model
(``--cache-only``); a pair missing from the cache then counts as a failed match
and is reported. ``--light`` additionally skips the local-feature datasets in the
working copy (cache-only runs never read them).

Every variant runs on a fresh working copy of the database under ``--work``; the
layer's own database is only read. With ``--gt-run`` each variant is scored
against simulator ground truth: map error of supported keyframes and "jumps" —
how much the map error changes between consecutive keyframes.

    python scripts/propagation_sweep.py \\
        --db "D:/P/sources/main/database.h5" --calibration "D:/P/sources/main/calibration.json" \\
        --gt-run "D:/FlightSimulator/output/run_1000m" --work "D:/P/experiments/sweep_main" \\
        --variants variants.json

variants.json: [{"name": "base", "set": {}},
                {"name": "nogap", "set": {"graph_optimization.anchor_gap_check": false}}]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_VERSION = 1
HEAVY_FEATURE_DATASETS = frozenset({"descriptors", "keypoints", "coords_2d"})
# Config that changes what a pairwise match returns; any change is a cache miss.
MATCH_SECTIONS = ("localization", "homography", "models")


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Deep copy of ``config`` with dotted-path overrides of existing keys only.

    Any depth (``graph_optimization.isotropy_weight``, ``models.vlad.enabled``). The new
    value must have the type of the current one; a None leaf accepts any value.
    """
    result = copy.deepcopy(config)
    for name, value in overrides.items():
        *parents, key = name.split(".")
        node = result
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not parents or not isinstance(node, dict) or key not in node:
            raise ValueError(f"unknown config key: {name}")
        current = node[key]
        if current is not None and (
            isinstance(current, dict)
            or isinstance(current, bool) != isinstance(value, bool)
            or (isinstance(current, int | float) and not isinstance(value, int | float))
            or (isinstance(current, str) and not isinstance(value, str))
            or (isinstance(current, list) and not isinstance(value, list))
        ):
            raise ValueError(f"{name}: {value!r} does not match type of {current!r}")
        node[key] = value
    return result


def load_variants(path: Path, config: dict) -> list[dict]:
    variants = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants file must hold a non-empty JSON list")
    names = set()
    for variant in variants:
        name = variant.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"variant names must be unique non-empty strings: {name!r}")
        names.add(name)
        overrides = variant.setdefault("set", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"{name}: 'set' must be an object")
        apply_overrides(config, overrides)  # validate before any work starts
    return variants


def match_settings_key(config: dict) -> str:
    relevant = {section: config.get(section) for section in MATCH_SECTIONS}
    relevant["mnn_fallback"] = (config.get("propagation") or {}).get("mnn_fallback")
    text = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def database_identity(db_path: Path) -> dict:
    import h5py

    with h5py.File(db_path, "r") as db:
        meta = dict(db["metadata"].attrs) if "metadata" in db else {}
        keyframes = int((db["local_features"]["kp_counts"][:] > 0).sum())
    return {
        "source_sha256": str(meta.get("source_sha256")),
        "creation_date": str(meta.get("creation_date")),
        "keyframes": keyframes,
    }


class MatchCache:
    def __init__(self, path: Path, identity: dict):
        self.path = Path(path)
        self.identity = identity
        self.pairs: dict = {}
        self.slots: list[int] | None = None
        self.dirty = False
        if self.path.exists():
            with self.path.open("rb") as stream:
                data = pickle.load(stream)
            if data.get("version") != CACHE_VERSION or data.get("identity") != identity:
                raise ValueError(
                    f"{self.path} belongs to another database or cache version "
                    f"({data.get('identity')} vs {identity})"
                )
            self.pairs = data["pairs"]
            self.slots = data.get("slots")

    def save(self) -> None:
        if not self.dirty:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            pickle.dump(
                {
                    "version": CACHE_VERSION,
                    "identity": self.identity,
                    "slots": self.slots,
                    "pairs": self.pairs,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, self.path)
        self.dirty = False


def light_copy(src: Path, dst: Path) -> None:
    """Copy an HDF5 database without the per-keypoint datasets (kp_counts is kept)."""
    import h5py

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for key, value in fin.attrs.items():
            fout.attrs[key] = value
        for name, item in fin.items():
            if name != "local_features":
                fin.copy(item, fout, name=name)
                continue
            group = fout.create_group(name)
            for key, value in item.attrs.items():
                group.attrs[key] = value
            for child in item:
                if child not in HEAVY_FEATURE_DATASETS:
                    item.copy(item[child], group, name=child)


def prepare_working_copy(db: Path, work: Path, light: bool) -> Path:
    """Fresh working copy per variant (the light template is built once)."""
    work.mkdir(parents=True, exist_ok=True)
    source = db
    if light:
        source = work / "pristine_light.h5"
        if not source.exists():
            tmp = source.with_suffix(".tmp")
            light_copy(db, tmp)
            os.replace(tmp, source)
    lance = db.parent / "vectors.lance"
    if lance.is_dir() and not (work / "vectors.lance").exists():
        shutil.copytree(lance, work / "vectors.lance")
    working = work / "database.h5"
    shutil.copyfile(source, working)
    return working


class LazyMatchers:
    """One FeatureMatcher per matching-settings key, created only on a cache miss."""

    def __init__(self):
        self._matchers: dict[str, object] = {}

    def get(self, key: str, config: dict):
        if key not in self._matchers:
            from src.localization.matcher import FeatureMatcher
            from src.models.model_manager import ModelManager

            manager = ModelManager(config=config)
            self._matchers[key] = FeatureMatcher(model_manager=manager, config=config)
        return self._matchers[key]


class _MatcherProxy:
    def __init__(self, factory):
        self._factory = factory

    def __getattr__(self, name):
        return getattr(self._factory(), name)


def map_jumps(db_path: Path, gt_slots: dict) -> dict:
    """Change of the map error vector between consecutive supported keyframes (m)."""
    import h5py
    import numpy as np

    with h5py.File(db_path, "r") as db:
        gps = db["frame_gps"][:]
        status = db["calibration"]["frame_georef_status"][:]
        keyframe = db["local_features"]["kp_counts"][:] > 0
    vectors = []
    for slot in np.nonzero(keyframe & (status == 1))[0]:
        truth = gt_slots.get(int(slot))
        if (
            truth is None
            or not truth.get("ground_center_valid")
            or not np.isfinite(gps[slot]).all()
        ):
            continue
        lat_gt, lon_gt = truth["ground_center_gps"]
        east = (gps[slot, 1] - lon_gt) * 111_320.0 * math.cos(math.radians(lat_gt))
        north = (gps[slot, 0] - lat_gt) * 110_540.0
        vectors.append((int(slot), east, north))
    if len(vectors) < 2:
        return {"n": 0}
    steps = np.array([math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(vectors, vectors[1:])])
    return {
        "n": int(steps.size),
        "p95_m": round(float(np.percentile(steps, 95)), 2),
        "max_m": round(float(steps.max()), 2),
        "over_10m": int((steps > 10.0).sum()),
    }


def run_variant(variant, args, base_config, cache, matchers, gt_slots, run) -> dict:
    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
    from src.database.database_loader import DatabaseLoader
    from src.workers.propagation_pipeline import PropagationPipeline

    config = apply_overrides(base_config, variant["set"])
    if args.cache_only and config.get("propagation", {}).get("rotation_retry"):
        raise ValueError("propagation.rotation_retry needs the matcher; not with --cache-only")
    settings = match_settings_key(config)
    working = prepare_working_copy(args.db, args.work, args.light)
    stats = {"hits": 0, "misses": 0, "new": 0}
    errors: list[str] = []
    completed = []
    started = time.monotonic()

    database = DatabaseLoader(str(working))
    calibration = MultiAnchorCalibration()
    calibration.load(str(args.calibration))
    pipeline = PropagationPipeline(
        database,
        calibration,
        _MatcherProxy(lambda: matchers.get(settings, config)),
        config=config,
        progress_callback=lambda *_: None,
        error_callback=errors.append,
        completed_callback=lambda: completed.append(True),
    )
    identity: dict[int, int] = {}
    original_prefetch = pipeline._prefetch_features
    original_match = pipeline._match_and_build_edge

    def prefetch(num_frames):
        if args.light:
            if cache.slots is None:
                raise ValueError("--light needs a cache filled by a full run (slot list missing)")
            features = {slot: {"slot": slot} for slot in cache.slots}
        else:
            features = original_prefetch(num_frames)
            slots = sorted(features)
            if cache.slots is None:
                cache.slots, cache.dirty = slots, True
            elif cache.slots != slots:
                raise ValueError("prefetched keyframes differ from the cache's slot list")
        identity.clear()
        identity.update({id(value): slot for slot, value in features.items()})
        return features

    def match(features_a, features_b):
        key = (identity.get(id(features_a)), identity.get(id(features_b)), settings)
        if None in key[:2]:
            raise RuntimeError("match called with features outside the prefetched set")
        if key in cache.pairs:
            stats["hits"] += 1
            return cache.pairs[key]
        if args.cache_only:
            stats["misses"] += 1
            return None
        result = original_match(features_a, features_b)
        cache.pairs[key] = result
        cache.dirty = True
        stats["new"] += 1
        return result

    pipeline._prefetch_features = prefetch
    pipeline._match_and_build_edge = match
    try:
        pipeline._run_propagation()
    finally:
        database.close()
    record = {
        "name": variant["name"],
        "set": variant["set"],
        "completed": bool(completed) and not errors,
        "errors": errors,
        "cache": stats,
        "seconds": round(time.monotonic() - started, 1),
    }
    if record["completed"] and run is not None:
        from scripts.layer_gt_tools import map_error_report

        report = map_error_report(working, run)
        record["map_error"] = {
            key: report.get(key)
            for key in ("supported_keyframes", "supported", "supported_optimized")
        }
        record["jumps"] = map_jumps(working, gt_slots)
    return record


def summary_line(record: dict) -> str:
    if not record["completed"]:
        return f"{record['name']:28s} FAILED {record['errors'][:1]}"
    opt = (record.get("map_error") or {}).get("supported_optimized") or {}
    jumps = record.get("jumps") or {}
    miss = record["cache"]["misses"]
    return (
        f"{record['name']:28s} optimized med {opt.get('median_m', float('nan')):7.1f}"
        f"  p95 {opt.get('p95_m', float('nan')):7.1f}  max {opt.get('max_m', float('nan')):7.1f}"
        f"  | jumps p95 {jumps.get('p95_m', float('nan')):6.1f} max {jumps.get('max_m', float('nan')):6.1f}"
        f" >10m {jumps.get('over_10m', '-')}"
        f"  | {record['seconds']:.0f}s" + (f"  MISSES {miss}" if miss else "")
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path, help="working folder")
    parser.add_argument("--variants", required=True, type=Path, help="JSON list of variants")
    parser.add_argument("--gt-run", type=Path, help="simulator run folder (ground_truth.json)")
    parser.add_argument("--cache", type=Path, help="default: WORK/match_cache.pkl")
    parser.add_argument("--results", type=Path, help="default: WORK/results.jsonl")
    parser.add_argument("--cache-only", action="store_true", help="never load a matcher")
    parser.add_argument(
        "--light", action="store_true", help="working copy without keypoint data (cache-only)"
    )
    args = parser.parse_args(argv)
    if args.light:
        args.cache_only = True
    args.cache = args.cache or args.work / "match_cache.pkl"
    args.results = args.results or args.work / "results.jsonl"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    from config import APP_CONFIG

    variants = load_variants(args.variants, APP_CONFIG)
    cache = MatchCache(args.cache, database_identity(args.db))
    run = gt_slots = None
    if args.gt_run is not None:
        from scripts.layer_gt_tools import SimulatorRun

        run = SimulatorRun(args.gt_run)
        gt_slots = {int(s["slot"]): s for s in run.load_gt().get("slots", [])}
    matchers = LazyMatchers()
    args.work.mkdir(parents=True, exist_ok=True)
    lines = []
    for variant in variants:
        try:
            record = run_variant(variant, args, APP_CONFIG, cache, matchers, gt_slots, run)
        finally:
            cache.save()
        with args.results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        lines.append(summary_line(record))
        print(lines[-1], flush=True)
    print("\n".join(["", "=== summary ==="] + lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
