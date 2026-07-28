"""HARDENING P1-6: encrypted-artifact detection + passphrase injection.

Covers the Qt-free half of the GUI passphrase feature: deciding whether a
project needs a passphrase at all, and the ``set/clear/verify`` API the dialog
uses instead of the blocking ``getpass`` path.
"""

from __future__ import annotations

import json

import pytest

from src.core.project import ProjectManager
from src.security import at_rest
from src.security.at_rest import encrypt_bytes
from src.security.project_scan import (
    candidate_artifacts,
    find_encrypted_artifacts,
    project_is_encrypted,
)

PW = "map-passphrase"


@pytest.fixture(autouse=True)
def _no_cached_passphrase(monkeypatch):
    """The passphrase cache is process-wide; never let it leak between tests."""
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.delenv("DRONELOC_PASSPHRASE", raising=False)


def _make_project(root, *, encrypted: bool) -> ProjectManager:
    """Build a realistic per-source project tree and load it."""
    src_dir = root / "sources" / "main"
    src_dir.mkdir(parents=True)

    calib = b'{"anchors": [{"lat": 50.4, "lon": 30.5}]}'
    db = b"\x89HDF\r\n\x1a\nfake-h5"
    lance = b"vector-bytes"
    if encrypted:
        calib, db, lance = (encrypt_bytes(b, PW) for b in (calib, db, lance))

    (src_dir / "calibration.json").write_bytes(calib)
    (src_dir / "database.h5").write_bytes(db)
    (src_dir / "database_keypoints.mp4").write_bytes(b"kp-video")
    (src_dir / "vectors.lance").mkdir()
    (src_dir / "vectors.lance" / "data-0.lance").write_bytes(lance)

    (root / "project.json").write_text(
        json.dumps(
            {
                "project_name": "scan-test",
                "created_at": "2026-07-28T00:00:00",
                "video_path": "flight.mp4",
                "database_filename": "sources/main/database.h5",
                "calibration_filename": "sources/main/calibration.json",
            }
        ),
        encoding="utf-8",
    )

    pm = ProjectManager()
    assert pm.load_project(str(root))
    return pm


def test_candidates_resolve_per_source_paths(tmp_path):
    pm = _make_project(tmp_path, encrypted=False)
    names = {p.name for p in candidate_artifacts(pm)}
    assert "calibration.json" in names
    assert "database.h5" in names
    assert "database_keypoints.mp4" in names
    assert "data-0.lance" in names  # lance dir probed one file deep


def test_plaintext_project_needs_no_passphrase(tmp_path):
    pm = _make_project(tmp_path, encrypted=False)
    assert find_encrypted_artifacts(pm) == []


def test_encrypted_project_is_detected_calibration_first(tmp_path):
    pm = _make_project(tmp_path, encrypted=True)
    found = find_encrypted_artifacts(pm)
    assert [p.name for p in found] == [
        "calibration.json",
        "database.h5",
        "data-0.lance",
    ]
    # The dialog verifies against found[0]; it must be the cheapest artifact,
    # never the map database.
    assert found[0].name == "calibration.json"


def test_unloaded_project_manager_yields_nothing():
    assert find_encrypted_artifacts(ProjectManager()) == []


def test_verify_passphrase_accepts_correct_and_rejects_wrong(tmp_path):
    pm = _make_project(tmp_path, encrypted=True)
    artifact = str(find_encrypted_artifacts(pm)[0])
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


def test_project_is_encrypted_path_based(tmp_path):
    """The project picker probes by path, without loading via ProjectManager."""
    plain = tmp_path / "plain"
    _make_project(plain, encrypted=False)
    enc = tmp_path / "enc"
    _make_project(enc, encrypted=True)

    assert project_is_encrypted(str(plain)) is False
    assert project_is_encrypted(str(enc)) is True


def test_project_is_encrypted_tolerates_non_projects(tmp_path):
    """A missing folder, a plain folder, or a corrupt manifest must not raise —
    the picker lists stale registry entries too."""
    assert project_is_encrypted(str(tmp_path / "does-not-exist")) is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert project_is_encrypted(str(empty)) is False

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "project.json").write_text("{not json", encoding="utf-8")
    assert project_is_encrypted(str(broken)) is False
