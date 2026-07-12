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

import hashlib
import json
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
)


def compute_fingerprint(components: dict[str, Any]) -> str:
    """Short deterministic hash of the schema-defining components (16 hex chars)."""
    canonical = {k: components.get(k) for k in SCHEMA_FIELDS}
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

    # global_descriptor lives at top level; fall back to models.* for safety.
    backend = g("global_descriptor.backend", None)
    if backend is None:
        backend = g("models.global_descriptor.backend", "dinov3")

    return {
        "schema_version": schema_version,
        "global_backend": backend,
        "descriptor_dim": int(descriptor_dim),
        "vlad_enabled": bool(g("models.vlad.enabled", False)),
        "vlad_pca_dim": int(g("models.vlad.pca_dim", 512)),
        "local_extractor": g("models.local_extractor", "aliked"),
        "local_descriptor_dim": int(local_descriptor_dim),
        "max_keypoints_stored": int(g("database.max_keypoints_stored", 2048)),
        "keypoint_video_scale": float(g("database.keypoint_video_scale", 0.5)),
        "frame_step": int(g("database.frame_step", 30)),
        "store_sift_features": bool(g("database.store_sift_features", False)),
        "sift_max_keypoints": int(g("database.sift_max_keypoints", 2048)),
    }


def describe(components: dict[str, Any]) -> str:
    """One-line human-readable rendering of the components."""
    return ", ".join(f"{k}={components.get(k)}" for k in SCHEMA_FIELDS)


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Human-readable list of differing fields between two component dicts."""
    return [
        f"{k}: {a.get(k)!r} != {b.get(k)!r}"
        for k in SCHEMA_FIELDS
        if a.get(k) != b.get(k)
    ]
