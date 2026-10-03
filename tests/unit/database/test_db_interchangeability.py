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
STRUCTURE_DEFINING_KEYS = frozenset(
    {
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
    }
)

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


def _vlad_components(vocab_path, enabled=True, layer=None, low_norm=0.0):
    cfg = {
        "global_descriptor": {"backend": "dinov3"},
        "models": {
            "local_extractor": "aliked",
            "vlad": {
                "enabled": enabled,
                "pca_dim": 256,
                "vocab_path": str(vocab_path),
                "layer": layer,
                "low_norm_fraction": low_norm,
            },
        },
    }
    return SF.build_components(cfg, descriptor_dim=256, local_descriptor_dim=128)


def test_fingerprint_without_vlad_ignores_vlad_identity_fields(tmp_path):
    vocab = tmp_path / "vocab.npz"
    vocab.write_bytes(b"vocabulary A")
    off = _vlad_components(vocab, enabled=False)
    assert all(off[key] is None for key in SF.OPTIONAL_FIELDS)
    legacy = {k: v for k, v in off.items() if k not in SF.OPTIONAL_FIELDS}
    # a database built before the VLAD identity fields existed keeps its fingerprint
    assert SF.compute_fingerprint(off) == SF.compute_fingerprint(legacy)


def test_fingerprint_tracks_vlad_vocabulary_content_not_path(tmp_path):
    a, a_copy, b = tmp_path / "a.npz", tmp_path / "copy" / "a.npz", tmp_path / "b.npz"
    a_copy.parent.mkdir()
    a.write_bytes(b"vocabulary A")
    a_copy.write_bytes(b"vocabulary A")
    b.write_bytes(b"vocabulary B")
    fa = SF.compute_fingerprint(_vlad_components(a))
    assert fa == SF.compute_fingerprint(_vlad_components(a_copy))
    assert fa != SF.compute_fingerprint(_vlad_components(b))
    assert fa != SF.compute_fingerprint(_vlad_components(a, layer=20))
    assert fa != SF.compute_fingerprint(_vlad_components(a, low_norm=0.1))
    assert _vlad_components(tmp_path / "absent.npz")["vlad_vocab"] == "missing"


def test_vlad_mismatch_rules(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    a.write_bytes(b"vocabulary A")
    b.write_bytes(b"vocabulary B")
    runtime_a = _vlad_components(a)
    runtime_off = _vlad_components(a, enabled=False)
    plain = {"vlad_enabled": False, "descriptor_dim": 1024}  # pre-identity database
    assert SF.vlad_mismatch(None, runtime_a) is None  # older builder: unknown
    assert SF.vlad_mismatch(plain, runtime_off) is None
    assert "vlad_enabled" in SF.vlad_mismatch(plain, runtime_a)
    assert "vlad_enabled" in SF.vlad_mismatch(runtime_a, runtime_off)
    assert SF.vlad_mismatch(runtime_a, _vlad_components(a)) is None
    assert "vlad_vocab" in SF.vlad_mismatch(_vlad_components(b), runtime_a)
    old_vlad = {"vlad_enabled": True, "vlad_pca_dim": 256}  # VLAD before vocab hashing
    assert "vlad_vocab" in SF.vlad_mismatch(old_vlad, runtime_a)


def test_manager_skips_database_built_with_another_vocabulary(tmp_path):
    import json
    from types import SimpleNamespace

    from src.database.multi_database_manager import MultiDatabaseManager

    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    a.write_bytes(b"vocabulary A")
    b.write_bytes(b"vocabulary B")
    manager = MultiDatabaseManager.__new__(MultiDatabaseManager)
    manager._config = {"models": {"vlad": {"enabled": True, "vocab_path": str(a)}}}

    def loader(components):
        return SimpleNamespace(metadata={"schema_components": json.dumps(components)})

    same = _vlad_components(a)
    other = _vlad_components(b)
    assert manager._vlad_mismatch(loader(same)) is None
    assert "vlad_vocab" in manager._vlad_mismatch(loader(other))
    assert manager._vlad_mismatch(SimpleNamespace(metadata={})) is None
