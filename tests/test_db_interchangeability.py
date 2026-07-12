"""Guards for database interchangeability across machines.

The contract (docs/DB_INTERCHANGEABILITY.md): hardware may change SPEED knobs but
must never change any setting that defines a database's structure or content-type,
so databases built on different machines stay interchangeable. These tests fail if
that invariant is ever broken.
"""

from __future__ import annotations

from src.database import schema_fingerprint as SF
from src.utils.hardware_profile import TUNABLE_KEYS, HardwareProfile

# Keys that define a database's structure/content-type. Auto-tune must never
# touch any of these, or databases stop being interchangeable across machines.
STRUCTURE_DEFINING_KEYS = frozenset({
    "models.local_extractor",
    "global_descriptor.backend",
    "models.global_descriptor.backend",
    "models.vlad.enabled",
    "models.vlad.pca_dim",
    "database.max_keypoints_stored",
    "database.keypoint_video_scale",
    "database.frame_step",
    "database.store_sift_features",
    "database.sift_max_keypoints",
})

# (tier, physical_cores, vram_gb, ampere_plus)
_TIERS = [
    ("low", 2, 0.0, False),
    ("mid", 6, 8.0, False),
    ("high", 12, 12.0, True),
    ("ultra", 32, 24.0, True),
]


def _profile(tier: str, cores: int, vram: float, ampere: bool) -> HardwareProfile:
    hp = HardwareProfile()
    hp.tier = tier
    hp.cpu.physical_cores = cores
    hp.cpu.logical_threads = cores * 2
    hp.gpu.available = vram > 0
    hp.gpu.vram_total_gb = vram
    hp.gpu.is_ampere_plus = ampere
    return hp


def test_tunable_and_structure_keys_are_disjoint():
    assert TUNABLE_KEYS.isdisjoint(STRUCTURE_DEFINING_KEYS)


def test_auto_tune_only_proposes_tunable_keys_on_every_tier():
    for tier, cores, vram, ampere in _TIERS:
        overrides = _profile(tier, cores, vram, ampere).auto_tune({})
        leaked = set(overrides) - TUNABLE_KEYS
        assert not leaked, f"{tier}: auto_tune proposed non-tunable keys {leaked}"
        struct = set(overrides) & STRUCTURE_DEFINING_KEYS
        assert not struct, f"{tier}: auto_tune touched structure keys {struct}"


def test_auto_tune_guard_is_fail_closed():
    """Even if a structure key currently holds its default, it is never proposed."""
    hp = _profile("ultra", 32, 24.0, True)
    overrides = hp.auto_tune({"models": {"local_extractor": "aliked"}})
    assert "models.local_extractor" not in overrides


def test_default_local_extractor_is_hardware_independent():
    from config.models import CANONICAL_LOCAL_EXTRACTOR, get_default_local_extractor

    assert get_default_local_extractor() == get_default_local_extractor()
    assert get_default_local_extractor() == CANONICAL_LOCAL_EXTRACTOR


def _components(local_extractor="aliked", descriptor_dim=1024, local_dim=128):
    # Build directly from a plain dict so the test needs no config file on disk.
    cfg = {
        "global_descriptor": {"backend": "dinov3"},
        "models": {"local_extractor": local_extractor, "vlad": {"enabled": False, "pca_dim": 512}},
        "database": {
            "max_keypoints_stored": 2048,
            "keypoint_video_scale": 0.5,
            "frame_step": 30,
            "store_sift_features": False,
            "sift_max_keypoints": 2048,
        },
    }
    return SF.build_components(cfg, descriptor_dim=descriptor_dim, local_descriptor_dim=local_dim)


def test_fingerprint_is_deterministic_and_hardware_independent():
    a = SF.compute_fingerprint(_components())
    b = SF.compute_fingerprint(_components())
    assert a == b


def test_fingerprint_detects_incompatible_extractor():
    aliked = SF.compute_fingerprint(_components(local_extractor="aliked"))
    rdd = SF.compute_fingerprint(_components(local_extractor="rdd"))
    assert aliked != rdd


def test_fingerprint_detects_descriptor_dim_change():
    d1024 = SF.compute_fingerprint(_components(descriptor_dim=1024))
    d512 = SF.compute_fingerprint(_components(descriptor_dim=512))
    assert d1024 != d512
