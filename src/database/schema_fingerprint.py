"""Database schema fingerprint — single source of truth for interchangeability.

Two databases are *interchangeable* (mutually queryable / mergeable) only if they
were built with the same structure- and content-type-defining settings: same
global/local models, descriptor dimensions, keypoint budget, sampling scale,
frame step, SIFT policy, VLAD settings. None of those may depend on the machine's
compute power — see ``src/utils/hardware_profile.TUNABLE_KEYS``, whose allow-list
deliberately excludes every field used here.

This module turns that fixed set of settings into a short, stable hash that the
builder writes into each database's metadata and the loader/manager check on
open, so an incompatible database is detected instead of silently corrupting
matches. Pure stdlib (hashlib, json) — safe to import anywhere.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

# Ordered, fixed list of structure/content-defining fields. The order is part of
# the contract; appending a new field only changes the fingerprint of databases
# built afterwards (existing databases keep the fingerprint stored in them).
SCHEMA_FIELDS: tuple[str, ...] = (
    "schema_version",
    "global_backend",
    "descriptor_dim",
    "vlad_enabled",
    "vlad_pca_dim",
    "local_extractor",
    "local_descriptor_dim",
    "max_keypoints_stored",
    "keypoint_video_scale",
    "frame_step",
    "store_sift_features",
    "sift_max_keypoints",
    # CPU-resize before DINO uses cv2.INTER_AREA instead of torchvision Resize(antialias)
    # Different filter creates different descriptor values -> databases are not interchangeable.
    "dino_cpu_resize",
    # VLAD identity: content hash of the vocabulary file, ViT layer, low-norm filter.
    # Two VLAD databases built with different vocabularies have the same descriptor
    # dimension but incomparable descriptors.
    "vlad_vocab",
    "vlad_layer",
    "vlad_low_norm_fraction",
)

# Fields that are None while VLAD is off and are then left out of the hash, so
# every database built without VLAD keeps the fingerprint it was built with.
OPTIONAL_FIELDS: tuple[str, ...] = ("vlad_vocab", "vlad_layer", "vlad_low_norm_fraction")

# Recorded in schema_components but NOT hashed (so existing fingerprints stay valid):
# the DINO input resolution. Descriptors from different resolutions are not
# comparable; databases built before 2026-10-06 lack it and are not checked.
CHECKED_EXTRA_FIELDS: tuple[str, ...] = ("dino_input_size",)


@functools.lru_cache(maxsize=8)
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def vocab_identity(path: Any) -> str:
    """Content hash of a VLAD vocabulary file (16 hex chars), or "missing".

    Content, not path: the same vocabulary copied to another machine matches.
    Cached per (path, mtime, size), so runtime checks stay cheap.
    """
    if not path:
        return "missing"
    try:
        resolved = Path(path).resolve()
        stat = resolved.stat()
    except OSError:
        return "missing"
    return _file_digest(str(resolved), stat.st_mtime_ns, stat.st_size)


def compute_fingerprint(components: dict[str, Any]) -> str:
    """Short deterministic hash of the schema-defining components (16 hex chars)."""
    canonical = {
        k: components.get(k)
        for k in SCHEMA_FIELDS
        if k not in OPTIONAL_FIELDS or components.get(k) is not None
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_components(
    config: Any,
    *,
    descriptor_dim: int,
    local_descriptor_dim: int,
    schema_version: str = "v2",
) -> dict[str, Any]:
    """Collect schema components from a config + the two build-time-derived dims.

    ``config`` may be a Pydantic ``AppConfig`` or a plain dict; access goes
    through ``config.get_cfg`` so both work. ``descriptor_dim`` and
    ``local_descriptor_dim`` are passed in because they are resolved from the
    actual loaded models at build time, not read from config.
    """
    from config import get_cfg

    def g(path: str, default: Any) -> Any:
        return get_cfg(config, path, default)

    # global_descriptor lives at top level ("models.global_descriptor" never existed).
    backend = g("global_descriptor.backend", "dinov3")

    vlad_enabled = bool(g("models.vlad.enabled", False))
    vlad_layer = g("models.vlad.layer", None)
    default_size = 224 if backend == "dinov3" else 336
    return {
        "dino_input_size": int(g(f"global_descriptor.{backend}.input_size", default_size)),
        "schema_version": schema_version,
        "global_backend": backend,
        "descriptor_dim": int(descriptor_dim),
        "vlad_enabled": vlad_enabled,
        "vlad_pca_dim": int(g("models.vlad.pca_dim", 512)),
        "local_extractor": g("models.local_extractor", "aliked"),
        "local_descriptor_dim": int(local_descriptor_dim),
        "max_keypoints_stored": int(g("database.max_keypoints_stored", 2048)),
        "keypoint_video_scale": float(g("database.keypoint_video_scale", 0.5)),
        "frame_step": int(g("database.frame_step", 30)),
        "store_sift_features": bool(g("database.store_sift_features", False)),
        "sift_max_keypoints": int(g("database.sift_max_keypoints", 2048)),
        "dino_cpu_resize": bool(g("models.performance.dino_cpu_resize", False)),
        "vlad_vocab": vocab_identity(g("models.vlad.vocab_path", None)) if vlad_enabled else None,
        "vlad_layer": (int(vlad_layer) if vlad_layer is not None else "last")
        if vlad_enabled
        else None,
        "vlad_low_norm_fraction": float(g("models.vlad.low_norm_fraction", 0.0))
        if vlad_enabled
        else None,
    }


def vlad_mismatch(stored: Any, runtime: dict[str, Any]) -> str | None:
    """Why a database's VLAD settings differ from the runtime ones, or None.

    ``stored`` is the database's ``schema_components`` (None/non-dict for older
    builders: unknown, not a mismatch). A database without VLAD fields counts as
    built without VLAD; a VLAD database built before the vocabulary hash was
    recorded has no ``vlad_vocab`` and therefore never matches a VLAD runtime.
    """
    if not isinstance(stored, dict):
        return None
    stored_on = bool(stored.get("vlad_enabled", False))
    runtime_on = bool(runtime.get("vlad_enabled", False))
    if stored_on != runtime_on:
        return f"vlad_enabled: database {stored_on} != runtime {runtime_on}"
    if not runtime_on:
        return None
    diffs = [
        f"{key}: database {stored.get(key)!r} != runtime {runtime.get(key)!r}"
        for key in OPTIONAL_FIELDS
        if stored.get(key) != runtime.get(key)
    ]
    return "; ".join(diffs) or None


def extra_mismatch(stored: Any, runtime: dict[str, Any]) -> str | None:
    """Differences in the unhashed checked fields (e.g. DINO input size), or None.

    A field missing on either side (older database) is not a mismatch.
    """
    if not isinstance(stored, dict):
        return None
    diffs = [
        f"{key}: database {stored.get(key)!r} != runtime {runtime.get(key)!r}"
        for key in CHECKED_EXTRA_FIELDS
        if stored.get(key) is not None
        and runtime.get(key) is not None
        and stored.get(key) != runtime.get(key)
    ]
    return "; ".join(diffs) or None


def describe(components: dict[str, Any]) -> str:
    """One-line human-readable rendering of the components."""
    return ", ".join(f"{k}={components.get(k)}" for k in SCHEMA_FIELDS)


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Human-readable list of differing fields between two component dicts."""
    return [f"{k}: {a.get(k)!r} != {b.get(k)!r}" for k in SCHEMA_FIELDS if a.get(k) != b.get(k)]
