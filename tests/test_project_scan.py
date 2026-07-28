"""HARDENING P1-6: encrypted-project detection, passphrase injection, write guard.

Covers the Qt-free half of the feature: deciding whether a project needs a
passphrase before it is loaded, the ``set/clear/verify`` API the dialog uses
instead of the blocking ``getpass`` path, and the guard that keeps an encrypted
deployment copy immutable.
"""

from __future__ import annotations

import json

import pytest

from src.core.project import ProjectManager
from src.security import at_rest
from src.security.at_rest import encrypt_bytes
from src.security.project_scan import (
    EncryptedProjectWriteError,
    assert_project_writable,
    encrypted_artifacts_at,
    find_project_root,
    project_is_encrypted,
)

PW = "map-passphrase"

MANIFEST = {
    "project_name": "scan-test",
    "created_at": "2026-07-28T00:00:00",
    "video_path": "flight.mp4",
    "database_filename": "sources/main/database.h5",
    "calibration_filename": "sources/main/calibration.json",
}


@pytest.fixture(autouse=True)
def _no_cached_passphrase(monkeypatch):
    """The passphrase cache is process-wide; never let it leak between tests."""
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)


def _make_project(root, *, mode: str):
    """Build a per-source project tree.

    ``mode``: ``plain`` (nothing encrypted), ``full`` (every file encrypted, the
    current copy builder), or ``legacy`` (artifacts encrypted, manifest still
    plaintext — copies built before the no-allowlist change).
    """
    src_dir = root / "sources" / "main"
    src_dir.mkdir(parents=True)

    def maybe(data: bytes, *, artifact: bool) -> bytes:
        if mode == "full":
            return encrypt_bytes(data, PW)
        if mode == "legacy" and artifact:
            return encrypt_bytes(data, PW)
        return data

    (src_dir / "calibration.json").write_bytes(maybe(b'{"anchors": []}', artifact=True))
    (src_dir / "database.h5").write_bytes(maybe(b"\x89HDF\r\n\x1a\nfake", artifact=True))
    (src_dir / "database_keypoints.mp4").write_bytes(maybe(b"kp-video", artifact=True))
    (src_dir / "vectors.lance").mkdir()
    (src_dir / "vectors.lance" / "data-0.lance").write_bytes(maybe(b"vec", artifact=True))
    (root / "project.json").write_bytes(
        maybe(json.dumps(MANIFEST).encode("utf-8"), artifact=False)
    )
    return root


# ── Detection ────────────────────────────────────────────────────────────────


def test_plaintext_project_needs_no_passphrase(tmp_path):
    _make_project(tmp_path, mode="plain")
    assert encrypted_artifacts_at(tmp_path) == []
    assert project_is_encrypted(tmp_path) is False


def test_fully_encrypted_project_verifies_against_the_manifest(tmp_path):
    _make_project(tmp_path, mode="full")
    found = encrypted_artifacts_at(tmp_path)
    # The manifest is the marker AND the cheapest verification target — the
    # dialog uses found[0], which must never be the map database.
    assert [p.name for p in found] == ["project.json"]
    assert project_is_encrypted(tmp_path) is True


def test_legacy_copy_with_plaintext_manifest_still_detected(tmp_path):
    """Copies built before the no-allowlist change must keep opening."""
    _make_project(tmp_path, mode="legacy")
    found = encrypted_artifacts_at(tmp_path)
    assert [p.name for p in found] == [
        "calibration.json",
        "database.h5",
        "data-0.lance",
        "database_keypoints.mp4",
    ]
    assert found[0].name == "calibration.json"  # cheapest first


def test_detection_tolerates_non_projects(tmp_path):
    assert project_is_encrypted(tmp_path / "does-not-exist") is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert project_is_encrypted(empty) is False

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "project.json").write_text("{not json", encoding="utf-8")
    assert project_is_encrypted(broken) is False


# ── Loading ──────────────────────────────────────────────────────────────────


def test_load_project_decrypts_the_manifest(tmp_path):
    _make_project(tmp_path, mode="full")
    at_rest.set_passphrase(PW)

    pm = ProjectManager()
    assert pm.load_project(str(tmp_path)) is True
    assert pm.is_encrypted is True
    assert pm.settings.project_name == "scan-test"


def test_is_encrypted_flag_covers_legacy_copies(tmp_path):
    """Regression: the flag drives every GUI pre-check, so a manifest-only test
    let a legacy copy through — the refusal then surfaced from deep inside a
    worker, after propagation had already run, and crashed the app on rebuild."""
    _make_project(tmp_path, mode="legacy")
    pm = ProjectManager()
    assert pm.load_project(str(tmp_path)) is True
    assert pm.is_encrypted is True  # plaintext manifest, encrypted artifacts


def test_load_plaintext_project_is_unchanged(tmp_path):
    _make_project(tmp_path, mode="plain")
    pm = ProjectManager()
    assert pm.load_project(str(tmp_path)) is True
    assert pm.is_encrypted is False


# ── Write guard ──────────────────────────────────────────────────────────────


def test_find_project_root_walks_up(tmp_path):
    _make_project(tmp_path, mode="plain")
    deep = tmp_path / "sources" / "main" / "calibration.json"
    assert find_project_root(deep) == tmp_path
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_returns_none_outside_a_project(tmp_path):
    loose = tmp_path / "loose.txt"
    loose.write_text("x", encoding="utf-8")
    assert find_project_root(loose) is None


def test_guard_refuses_writes_into_an_encrypted_project(tmp_path):
    _make_project(tmp_path, mode="full")
    with pytest.raises(EncryptedProjectWriteError):
        assert_project_writable(tmp_path / "sources" / "main" / "calibration.json")
    with pytest.raises(EncryptedProjectWriteError):
        assert_project_writable(tmp_path / "project.json")


def test_guard_refuses_writes_into_a_legacy_encrypted_copy(tmp_path):
    _make_project(tmp_path, mode="legacy")
    with pytest.raises(EncryptedProjectWriteError):
        assert_project_writable(tmp_path / "sources" / "main" / "database.h5")


def test_guard_allows_plaintext_projects_and_non_projects(tmp_path):
    _make_project(tmp_path, mode="plain")
    assert_project_writable(tmp_path / "sources" / "main" / "database.h5")  # no raise

    outside = tmp_path.parent / "not-a-project.txt"
    outside.write_text("x", encoding="utf-8")
    assert_project_writable(outside)  # no raise


def test_save_project_refuses_on_an_encrypted_copy(tmp_path):
    _make_project(tmp_path, mode="full")
    at_rest.set_passphrase(PW)
    pm = ProjectManager()
    assert pm.load_project(str(tmp_path))

    # save_project swallows exceptions and reports failure — the point is that
    # the manifest is NOT overwritten with plaintext.
    before = (tmp_path / "project.json").read_bytes()
    assert pm.save_project() is False
    assert (tmp_path / "project.json").read_bytes() == before


# ── Passphrase injection ─────────────────────────────────────────────────────


def test_verify_passphrase_accepts_correct_and_rejects_wrong(tmp_path):
    _make_project(tmp_path, mode="full")
    artifact = str(encrypted_artifacts_at(tmp_path)[0])
    assert at_rest.verify_passphrase(artifact, PW) is True
    assert at_rest.verify_passphrase(artifact, "wrong") is False
    # Verification alone must not cache anything.
    assert at_rest._CACHED_PASSPHRASE is None


def test_verify_passphrase_rejects_plaintext_and_missing(tmp_path):
    plain = tmp_path / "plain.json"
    plain.write_bytes(b"{}")
    assert at_rest.verify_passphrase(str(plain), PW) is False
    assert at_rest.verify_passphrase(str(tmp_path / "nope.json"), PW) is False


def test_set_passphrase_is_used_by_get_passphrase(monkeypatch):
    at_rest.set_passphrase(PW)
    # get_passphrase must be satisfied by the cache, never reaching getpass.
    monkeypatch.setattr(
        at_rest.getpass, "getpass", lambda *a, **k: pytest.fail("getpass was called")
    )
    assert at_rest.get_passphrase() == PW


def test_clear_passphrase_drops_the_cache():
    at_rest.set_passphrase(PW)
    at_rest.clear_passphrase()
    assert at_rest._CACHED_PASSPHRASE is None


def test_set_passphrase_rejects_empty():
    with pytest.raises(at_rest.EncryptionError):
        at_rest.set_passphrase("")
