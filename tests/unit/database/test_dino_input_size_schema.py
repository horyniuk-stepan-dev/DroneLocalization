"""DINO input size: recorded in schema_components, checked, not part of the fingerprint."""

import copy

from config import APP_CONFIG
from src.database.schema_fingerprint import (
    build_components,
    compute_fingerprint,
    extra_mismatch,
)


def _components(size):
    cfg = copy.deepcopy(APP_CONFIG)
    backend = cfg["global_descriptor"]["backend"]
    cfg["global_descriptor"][backend]["input_size"] = size
    return build_components(cfg, descriptor_dim=1024, local_descriptor_dim=128)


def test_input_size_recorded_but_not_hashed():
    a, b = _components(224), _components(336)
    assert a["dino_input_size"] == 224 and b["dino_input_size"] == 336
    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_extra_mismatch():
    a, b = _components(224), _components(336)
    assert extra_mismatch(a, a) is None
    assert "dino_input_size" in extra_mismatch(a, b)
    old = {k: v for k, v in a.items() if k != "dino_input_size"}  # database built before the field
    assert extra_mismatch(old, b) is None
    assert extra_mismatch(None, b) is None
