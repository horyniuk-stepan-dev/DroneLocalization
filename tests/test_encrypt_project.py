"""HARDENING P1-6: encrypted-copy builder (non-destructive, no allowlist).

Every file in the copy is encrypted — project.json included, which is what makes
the copy self-identifying and immutable. The source master is never modified.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.security.at_rest import decrypt_bytes, is_encrypted

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "encrypt_project", _REPO / "scripts" / "encrypt_project.py"
)
encrypt_project = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encrypt_project)

PW = "copy-passphrase"


def _make_project(root: Path) -> dict[str, bytes]:
    """A realistic per-source tree, including files no allowlist would list."""
    files = {
        "project.json": b'{"project_name": "p"}',
        "sources/main/calibration.json": b'{"anchors":[]}',
        "sources/main/database.h5": b"\x89HDF\r\n\x1a\nfake-h5-bytes",
        "sources/main/database_graph.geojson": b'{"type":"FeatureCollection"}',
        "sources/main/database_keypoints.mp4": b"fake-mp4",
        "sources/main/vectors.lance/data-0.lance": b"vector-bytes",
        "sources/area2/db_area2.h5": b"area2-h5",
        "panoramas/pano_01.jpg": b"jpeg-bytes",
        "notes.txt": b"operator notes about the area",
    }
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return files


def test_every_file_is_encrypted(tmp_path):
    src = tmp_path / "master"
    src.mkdir()
    files = _make_project(src)
    dst = tmp_path / "encrypted"

    summary = encrypt_project.build_encrypted_copy(str(src), str(dst), PW)

    for rel, expected in files.items():
        blob = (dst / rel).read_bytes()
        assert is_encrypted(blob), f"{rel} left in plaintext"
        assert decrypt_bytes(blob, PW) == expected, f"{rel} did not round-trip"

    assert summary["total"] == len(files)
    assert set(summary["encrypted"]) == {str(Path(r)) for r in files}


def test_manifest_is_encrypted_so_the_copy_is_self_identifying(tmp_path):
    """project.json carries the header — that is the marker the app detects."""
    src = tmp_path / "master"
    src.mkdir()
    _make_project(src)
    dst = tmp_path / "encrypted"

    encrypt_project.build_encrypted_copy(str(src), str(dst), PW)

    assert is_encrypted((dst / "project.json").read_bytes())


def test_source_master_untouched(tmp_path):
    src = tmp_path / "master"
    src.mkdir()
    files = _make_project(src)
    dst = tmp_path / "encrypted"

    encrypt_project.build_encrypted_copy(str(src), str(dst), PW)

    for rel, expected in files.items():
        assert (src / rel).read_bytes() == expected, f"{rel} modified in the master"


def test_summary_contract(tmp_path):
    """The GUI worker and the done-handler read these keys by name.

    Regression: renaming ``copied`` to ``total`` left a stale key in the worker's
    log line, which raised a KeyError *after* a perfectly good copy was built —
    the operator saw a failure dialog for a build that had fully succeeded."""
    src = tmp_path / "master"
    src.mkdir()
    files = _make_project(src)

    summary = encrypt_project.build_encrypted_copy(str(src), str(tmp_path / "out"), PW)

    assert set(summary) == {"encrypted", "total"}
    assert isinstance(summary["encrypted"], list)
    assert summary["total"] == len(summary["encrypted"]) == len(files)


def test_refuses_existing_output(tmp_path):
    src = tmp_path / "master"
    src.mkdir()
    _make_project(src)
    dst = tmp_path / "out"
    dst.mkdir()  # already exists
    with pytest.raises(SystemExit):
        encrypt_project.build_encrypted_copy(str(src), str(dst), PW)
