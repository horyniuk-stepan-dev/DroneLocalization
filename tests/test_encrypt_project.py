"""HARDENING P1-6: encrypted-copy builder (non-destructive).

Tests the pure ``build_encrypted_copy`` function: every sensitive artifact is
encrypted into the copy, non-sensitive files are copied verbatim, and the source
master is never modified.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.security.at_rest import is_encrypted

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("encrypt_project", _REPO / "scripts" / "encrypt_project.py")
encrypt_project = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encrypt_project)

PW = "copy-passphrase"


def _make_project(root: Path) -> None:
    (root / "sources" / "main").mkdir(parents=True)
    (root / "calibration.json").write_bytes(b'{"anchors":[]}')
    (root / "database.h5").write_bytes(b"\x89HDF\r\n\x1a\nfake-h5-bytes")
    (root / "database_graph.geojson").write_bytes(b'{"type":"FeatureCollection"}')
    (root / "database_keypoints.mp4").write_bytes(b"fake-mp4")
    (root / "project.json").write_bytes(b'{"name":"p"}')  # non-sensitive
    (root / "vectors.lance").mkdir()
    (root / "vectors.lance" / "data-0.lance").write_bytes(b"vector-bytes")


def test_copy_encrypts_sensitive_and_preserves_source(tmp_path):
    src = tmp_path / "master"
    src.mkdir()
    _make_project(src)
    dst = tmp_path / "encrypted"

    summary = encrypt_project.build_encrypted_copy(str(src), str(dst), PW)

    # Sensitive artifacts encrypted in the copy.
    for rel in ["calibration.json", "database.h5", "database_graph.geojson",
                "database_keypoints.mp4", "vectors.lance/data-0.lance"]:
        assert is_encrypted((dst / rel).read_bytes()), f"{rel} not encrypted"

    # Non-sensitive files copied verbatim.
    assert (dst / "project.json").read_bytes() == b'{"name":"p"}'
    assert not is_encrypted((dst / "project.json").read_bytes())

    # Source master untouched (still plaintext).
    assert (src / "calibration.json").read_bytes() == b'{"anchors":[]}'
    assert not is_encrypted((src / "database.h5").read_bytes())

    assert set(summary["encrypted"]) == {
        "calibration.json", "database.h5", "database_graph.geojson",
        "database_keypoints.mp4", str(Path("vectors.lance") / "data-0.lance"),
    }


def test_encrypts_per_source_artifacts(tmp_path):
    """Real projects keep artifacts under sources/<id>/, not at the root — a
    root-only match would ship the geo-anchors in plaintext."""
    src = tmp_path / "master"
    (src / "sources" / "main").mkdir(parents=True)
    (src / "sources" / "area2").mkdir(parents=True)
    (src / "project.json").write_bytes(b'{"name":"p"}')
    (src / "sources" / "main" / "calibration.json").write_bytes(b'{"anchors":[1]}')
    (src / "sources" / "main" / "database.h5").write_bytes(b"main-h5")
    (src / "sources" / "main" / "database_keypoints.mp4").write_bytes(b"main-mp4")
    (src / "sources" / "area2" / "calibration.json").write_bytes(b'{"anchors":[2]}')
    (src / "sources" / "area2" / "db_area2.h5").write_bytes(b"area2-h5")
    # Keypoint videos are named after their database, not always "database_*".
    (src / "sources" / "area2" / "db_area2_keypoints.mp4").write_bytes(b"area2-mp4")
    (src / "sources" / "main" / "vectors.lance").mkdir()
    (src / "sources" / "main" / "vectors.lance" / "data-0.lance").write_bytes(b"vec")

    dst = tmp_path / "encrypted"
    summary = encrypt_project.build_encrypted_copy(str(src), str(dst), PW)

    for rel in [
        "sources/main/calibration.json",
        "sources/main/database.h5",
        "sources/main/database_keypoints.mp4",
        "sources/area2/calibration.json",
        "sources/area2/db_area2.h5",
        "sources/area2/db_area2_keypoints.mp4",
        "sources/main/vectors.lance/data-0.lance",
    ]:
        assert is_encrypted((dst / rel).read_bytes()), f"{rel} not encrypted"

    assert len(summary["encrypted"]) == 7
    # project.json stays readable so the app can locate the sources.
    assert (dst / "project.json").read_bytes() == b'{"name":"p"}'


def test_refuses_existing_output(tmp_path):
    src = tmp_path / "master"
    src.mkdir()
    _make_project(src)
    dst = tmp_path / "out"
    dst.mkdir()  # already exists
    with pytest.raises(SystemExit):
        encrypt_project.build_encrypted_copy(str(src), str(dst), PW)
