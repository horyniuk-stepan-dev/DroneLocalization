"""HARDENING P1-6 SP2: transparent decrypt-on-open for the HDF5 map.

Tests the ``open_maybe_encrypted_h5`` helper in isolation (no full DatabaseLoader
build): a plaintext DB opens on the lazy path unchanged; an encrypted DB is
decrypted whole into RAM and served from a BytesIO; a wrong passphrase fails
closed. Runs on the Windows side (imports h5py/lancedb).
"""

from __future__ import annotations

import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.database.database_loader import (
    DatabaseLoader,
    materialize_maybe_encrypted_lance,
    open_maybe_encrypted_h5,
    wipe_tree,
)
from src.security.at_rest import EncryptionError, encrypt_bytes


def _make_h5(path) -> bytes:
    with h5py.File(path, "w") as f:
        f.create_dataset("global_descriptors", data=np.arange(12).reshape(3, 4).astype("float32"))
        f.attrs["n"] = 3
    return path.read_bytes()


def test_plaintext_h5_opens_on_lazy_path(tmp_path):
    p = tmp_path / "db.h5"
    _make_h5(p)
    hf, buf = open_maybe_encrypted_h5(str(p))
    try:
        assert buf is None  # plaintext: no RAM buffer, lazy h5py path unchanged
        assert hf["global_descriptors"][:].sum() == 66.0
    finally:
        hf.close()


def test_encrypted_h5_opens_with_passphrase(tmp_path, monkeypatch):
    raw = _make_h5(tmp_path / "plain.h5")
    enc = tmp_path / "db.h5"
    enc.write_bytes(encrypt_bytes(raw, "pw"))

    monkeypatch.setattr("src.security.at_rest._CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setenv("DRONELOC_PASSPHRASE", "pw")

    hf, buf = open_maybe_encrypted_h5(str(enc))
    try:
        assert buf is not None  # decrypted into RAM
        assert hf["global_descriptors"][:].sum() == 66.0
        assert hf.attrs["n"] == 3
    finally:
        hf.close()


def test_encrypted_h5_wrong_passphrase_fails_closed(tmp_path, monkeypatch):
    raw = _make_h5(tmp_path / "plain.h5")
    enc = tmp_path / "db.h5"
    enc.write_bytes(encrypt_bytes(raw, "right"))

    monkeypatch.setattr("src.security.at_rest._CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setenv("DRONELOC_PASSPHRASE", "wrong")

    with pytest.raises(EncryptionError):
        open_maybe_encrypted_h5(str(enc))


# ── SP3: LanceDB index ────────────────────────────────────────────────────────


def _make_lance_dir(root, *, encrypted: bool, passphrase: str = "pw"):
    """A lance dataset is a directory tree, not a single file."""
    root.mkdir()
    (root / "_versions").mkdir()
    payload = {
        "data/part-0.lance": b"lance-data-bytes",
        "_versions/1.manifest": b"manifest-bytes",
    }
    for rel, data in payload.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encrypt_bytes(data, passphrase) if encrypted else data)
    return payload


def test_plaintext_lance_opens_in_place(tmp_path):
    lance = tmp_path / "vectors.lance"
    _make_lance_dir(lance, encrypted=False)

    open_path, tempdir = materialize_maybe_encrypted_lance(lance)

    assert tempdir is None  # no temp copy for a plaintext index
    assert open_path == str(lance)


def test_encrypted_lance_materialized_to_temp_dir(tmp_path, monkeypatch):
    lance = tmp_path / "vectors.lance"
    payload = _make_lance_dir(lance, encrypted=True)

    monkeypatch.setattr("src.security.at_rest._CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setenv("DRONELOC_PASSPHRASE", "pw")

    open_path, tempdir = materialize_maybe_encrypted_lance(lance)
    try:
        assert tempdir is not None and open_path == tempdir
        # Layout preserved and every file decrypted.
        for rel, expected in payload.items():
            assert (Path(tempdir) / rel).read_bytes() == expected
    finally:
        wipe_tree(tempdir)

    assert not Path(tempdir).exists()  # wiped on close
    # Source index untouched (still ciphertext).
    assert (lance / "data" / "part-0.lance").read_bytes() != payload["data/part-0.lance"]


def test_encrypted_lance_wrong_passphrase_leaves_no_plaintext(tmp_path, monkeypatch):
    lance = tmp_path / "vectors.lance"
    _make_lance_dir(lance, encrypted=True, passphrase="right")

    monkeypatch.setattr("src.security.at_rest._CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setenv("DRONELOC_PASSPHRASE", "wrong")

    before = set(Path(tempfile.gettempdir()).glob("droneloc_lance_*"))
    with pytest.raises(EncryptionError):
        materialize_maybe_encrypted_lance(lance)
    after = set(Path(tempfile.gettempdir()).glob("droneloc_lance_*"))
    assert after == before  # fails closed: no lingering decrypted temp dir


def test_empty_lance_dir_opens_in_place(tmp_path):
    lance = tmp_path / "vectors.lance"
    lance.mkdir()
    open_path, tempdir = materialize_maybe_encrypted_lance(lance)
    assert tempdir is None
    assert open_path == str(lance)


def test_close_wipes_lance_tempdir(tmp_path):
    """Regression: a clean app exit must leave no decrypted vectors on disk.

    The first GUI run of the encrypted copy left a populated droneloc_lance_*
    directory behind because nothing closed the databases on shutdown."""
    loader = object.__new__(DatabaseLoader)
    loader._lock = threading.RLock()
    loader.db_file = None
    loader._decrypted_buf = None
    loader._size_cache = {}
    loader._feature_cache = OrderedDict()

    tempdir = tempfile.mkdtemp(prefix="droneloc_lance_")
    (Path(tempdir) / "data").mkdir()
    (Path(tempdir) / "data" / "part-0.lance").write_bytes(b"plaintext-global-vectors")
    loader._lance_tempdir = tempdir
    loader.lance_table = "open-dataset-handle"

    loader.close()

    # The handle is dropped before the wipe, or the open dataset pins the files.
    assert loader.lance_table is None
    assert loader._lance_tempdir is None
    assert not Path(tempdir).exists()
