"""HARDENING P1-6 (encryption-at-rest), Sub-project 1: crypto foundation.

Unit tests for src/security/at_rest.py — the passphrase-derived AES-256-GCM
container reused by all encryption-at-rest sub-projects. Pure-Python (only
depends on `cryptography`); no torch/Qt.
"""

from __future__ import annotations

import pytest

from src.security import at_rest

PW = "correct horse battery staple"
PLAINTEXT = b'{"version": "2.3", "anchors": [{"frame_id": 0}]}'


# --- round-trip --------------------------------------------------------------


def test_round_trip_returns_original():
    container = at_rest.encrypt_bytes(PLAINTEXT, PW)
    assert at_rest.decrypt_bytes(container, PW) == PLAINTEXT


def test_container_is_marked_encrypted():
    container = at_rest.encrypt_bytes(PLAINTEXT, PW)
    assert at_rest.is_encrypted(container)
    assert container.startswith(at_rest.MAGIC)


def test_plaintext_is_not_marked_encrypted():
    assert not at_rest.is_encrypted(PLAINTEXT)
    assert not at_rest.is_encrypted(b"")
    assert not at_rest.is_encrypted(b"DLE")  # shorter than MAGIC


def test_fresh_salt_and_nonce_each_time():
    a = at_rest.encrypt_bytes(PLAINTEXT, PW)
    b = at_rest.encrypt_bytes(PLAINTEXT, PW)
    assert a != b  # same plaintext + passphrase, different container


# --- fail-closed on bad input ------------------------------------------------


def test_wrong_passphrase_raises():
    container = at_rest.encrypt_bytes(PLAINTEXT, PW)
    with pytest.raises(at_rest.EncryptionError):
        at_rest.decrypt_bytes(container, "wrong passphrase")


def test_tampered_ciphertext_raises():
    container = bytearray(at_rest.encrypt_bytes(PLAINTEXT, PW))
    container[-1] ^= 0xFF  # flip a byte in the tag/ciphertext
    with pytest.raises(at_rest.EncryptionError):
        at_rest.decrypt_bytes(bytes(container), PW)


def test_truncated_container_raises():
    container = at_rest.encrypt_bytes(PLAINTEXT, PW)
    with pytest.raises(at_rest.EncryptionError):
        at_rest.decrypt_bytes(container[:20], PW)


def test_non_container_raises():
    with pytest.raises(at_rest.EncryptionError):
        at_rest.decrypt_bytes(PLAINTEXT, PW)  # no magic header


# --- passphrase source -------------------------------------------------------


def test_get_passphrase_reads_env(monkeypatch):
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setenv("DRONELOC_PASSPHRASE", "from-env")
    assert at_rest.get_passphrase() == "from-env"


def test_get_passphrase_raises_when_unset_and_no_tty(monkeypatch):
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.delenv("DRONELOC_PASSPHRASE", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(at_rest.EncryptionError):
        at_rest.get_passphrase()
