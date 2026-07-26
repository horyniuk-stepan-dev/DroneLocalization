"""HARDENING P1-6: passphrase-derived encryption-at-rest for map artifacts.

Threat: airframe capture — an adversary who recovers the payload must not read
the mission's operational area or map. The key is *never* stored on the device;
it is derived from an operator passphrase at load time (Scrypt), so a captured,
powered-off payload yields only authenticated ciphertext.

Self-describing container (so a plaintext project stays byte-for-byte unchanged
and encrypted artifacts are auto-detected on load):

    MAGIC(7) | version(1) | salt(16) | nonce(12) | AES-256-GCM(ciphertext‖tag)

This module is the crypto foundation reused by every encryption-at-rest
sub-project (geo-anchors, the h5 map, the lance index). It depends only on
`cryptography` — no torch/Qt — so it is unit-testable in the pure-Python suite.
"""

from __future__ import annotations

import getpass
import os
import sys

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"DLENC1\0"
_VERSION = 1
_SALT_LEN = 16
_NONCE_LEN = 12
_TAG_LEN = 16
_HEADER_LEN = len(MAGIC) + 1 + _SALT_LEN + _NONCE_LEN  # 36

# Scrypt work factors (memory-hard). n=2**15 → ~32 MB, ~100 ms on a modern CPU:
# strong against offline brute force, negligible against a legitimate one-shot
# load. Bumping n means re-encrypting existing artifacts, hence pinned here.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32  # AES-256

# Process-wide passphrase cache so multiple artifact loads prompt/read only once.
_CACHED_PASSPHRASE: str | None = None


class EncryptionError(Exception):
    """Fail-closed error for any at-rest crypto failure (bad passphrase, tamper,
    malformed container, or a missing passphrase). Never yields partial plaintext."""


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte AES-256 key from a passphrase + salt via Scrypt."""
    kdf = Scrypt(salt=salt, length=_KEY_LEN, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(passphrase.encode("utf-8"))


def is_encrypted(data: bytes) -> bool:
    """True iff ``data`` is one of our containers (cheap header check)."""
    return data[: len(MAGIC)] == MAGIC and len(data) >= len(MAGIC)


def encrypt_bytes(plaintext: bytes, passphrase: str) -> bytes:
    """Encrypt ``plaintext`` into a self-describing container. Fresh random salt +
    nonce every call, so the same input never produces the same ciphertext."""
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)  # appends 16-byte tag
    return MAGIC + bytes([_VERSION]) + salt + nonce + ciphertext


def decrypt_bytes(container: bytes, passphrase: str) -> bytes:
    """Authenticate + decrypt a container. Wrong passphrase, tampering, or a
    malformed container all raise :class:`EncryptionError` (fail-closed)."""
    if not is_encrypted(container):
        raise EncryptionError("not an encrypted container (bad magic)")
    if len(container) < _HEADER_LEN + _TAG_LEN:
        raise EncryptionError("truncated container")
    version = container[len(MAGIC)]
    if version != _VERSION:
        raise EncryptionError(f"unsupported container version {version}")

    off = len(MAGIC) + 1
    salt = container[off : off + _SALT_LEN]
    nonce = container[off + _SALT_LEN : _HEADER_LEN]
    ciphertext = container[_HEADER_LEN:]

    key = derive_key(passphrase, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as e:
        raise EncryptionError("wrong passphrase or corrupted data") from e


def get_passphrase() -> str:
    """Resolve the map passphrase: env ``DRONELOC_PASSPHRASE`` (headless/supervised
    — the parent holds it and passes it to restarted children), else an
    interactive prompt on a TTY. Fail-closed if neither is available. Cached
    process-wide after the first successful resolution."""
    global _CACHED_PASSPHRASE
    if _CACHED_PASSPHRASE is not None:
        return _CACHED_PASSPHRASE

    pw = os.environ.get("DRONELOC_PASSPHRASE")
    if not pw and sys.stdin is not None and sys.stdin.isatty():
        pw = getpass.getpass("Enter map decryption passphrase: ")
    if not pw:
        raise EncryptionError(
            "encrypted artifact found but no passphrase available — "
            "set DRONELOC_PASSPHRASE or run interactively"
        )

    _CACHED_PASSPHRASE = pw
    return pw
