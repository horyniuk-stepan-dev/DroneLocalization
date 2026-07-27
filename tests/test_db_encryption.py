"""HARDENING P1-6 SP2: transparent decrypt-on-open for the HDF5 map.

Tests the ``open_maybe_encrypted_h5`` helper in isolation (no full DatabaseLoader
build): a plaintext DB opens on the lazy path unchanged; an encrypted DB is
decrypted whole into RAM and served from a BytesIO; a wrong passphrase fails
closed. Runs on the Windows side (imports h5py/lancedb).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.database.database_loader import open_maybe_encrypted_h5
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
