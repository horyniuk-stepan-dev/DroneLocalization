"""Module for project artifact and data encryption (Encryption-at-Rest).

Protects disk data from unauthorized access in case of physical device loss.
The encryption key is derived from the operator passphrase during loading (Scrypt),
so a powered-off or captured device contains only authenticated AES-256-GCM ciphertext.

Encrypted container format:

    MAGIC(7) | version(1) | salt(16) | nonce(12) | AES-256-GCM(ciphertext‖tag)

Uses cryptographic primitives from `cryptography` without depending on PyTorch or Qt.
"""

from __future__ import annotations

import getpass
import os
import sys
import tempfile
from pathlib import Path

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


def encrypt_file(src_path: str, dst_path: str, passphrase: str) -> None:
    """Encrypt ``src_path`` into an at-rest container at ``dst_path`` (atomic
    write). Used by the encrypted-copy builder. ``dst`` may equal ``src`` for
    in-place, but the copy model keeps the plaintext master untouched."""
    container = encrypt_bytes(Path(src_path).read_bytes(), passphrase)
    dst = Path(dst_path)
    tmp = dst.with_name(dst.name + ".enc-tmp")
    with open(tmp, "wb") as f:
        f.write(container)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def decrypt_to_tempfile(src_path: str, passphrase: str, *, suffix: str = "") -> str:
    """Decrypt an encrypted file to a fresh temp file and return its path. For
    artifacts a library must open by path (e.g. a video, a lance data file). The
    caller MUST :func:`wipe_file` it when done. Fails closed (never leaves a
    partial plaintext temp on a bad passphrase)."""
    plaintext = decrypt_bytes(Path(src_path).read_bytes(), passphrase)
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="dlmap_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(plaintext)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        wipe_file(tmp)
        raise
    return tmp


def wipe_file(path: str) -> None:
    """Best-effort secure delete: overwrite with random bytes, then unlink. Note:
    on SSD/copy-on-write/journaling filesystems this does NOT guarantee the old
    blocks are physically erased (that needs full-disk encryption or a device
    secure-erase); it raises the bar against casual recovery of a decrypted temp."""
    p = Path(path)
    if not p.exists():
        return
    try:
        size = p.stat().st_size
        with open(p, "r+b") as f:
            f.write(os.urandom(size))
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass  # overwrite is best-effort; still unlink below
    p.unlink(missing_ok=True)


def stdin_is_tty() -> bool:
    """True if stdin is an interactive terminal we may prompt on.

    ``isatty()`` itself raises on a closed or detached stream — a real state for
    a Windows service or a frozen GUI build — so probing it must never be the
    thing that crashes a decrypt. Anything unreadable counts as "not a terminal",
    which routes the caller to the pipe and then to fail-closed."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (OSError, ValueError, AttributeError):
        return False


def read_passphrase_from_stdin() -> str:
    """Read one line of passphrase from a non-TTY stdin, or "" if unavailable.

    The supervised-child channel: the parent prompts once and pipes the
    passphrase to each restarted child (see ``main.py::_run_supervised``). A pipe
    is invisible in process listings, is not inherited by grandchildren, does not
    reach crash dumps, and is consumed once — the properties an environment
    variable lacks.

    Never blocks a normal run: it is only reached when stdin is *not* a terminal,
    and an empty or unreadable stdin returns "" so the caller fails closed."""
    if stdin_is_tty():
        return ""
    try:
        return sys.stdin.readline().strip()
    except (OSError, ValueError, AttributeError):
        # Closed, detached, or a capturing test runner — treat as "no passphrase".
        return ""


def get_passphrase() -> str:
    """Resolve the map passphrase: the process-wide cache (filled by the GUI
    dialog via :func:`set_passphrase`), an interactive prompt on a TTY, or one
    line piped in on a non-TTY stdin. Fail-closed if none of those yields one.

    There is deliberately **no environment-variable channel**. An env var is
    readable by any process running as the same user (Process Explorer,
    ``Win32_Process``, ``/proc/<pid>/environ``), is inherited by every child
    process, and can surface in crash dumps — for a passphrase whose whole
    purpose is that it is never stored on the device, that is the wrong trade.
    The stdin pipe above covers the non-interactive case instead."""
    global _CACHED_PASSPHRASE
    if _CACHED_PASSPHRASE is not None:
        return _CACHED_PASSPHRASE

    if stdin_is_tty():
        pw = getpass.getpass("Enter map decryption passphrase: ")
    else:
        pw = read_passphrase_from_stdin()
    if not pw:
        raise EncryptionError(
            "encrypted artifact found but no passphrase available — "
            "run interactively, pipe it on stdin, or enter it in the application"
        )

    _CACHED_PASSPHRASE = pw
    return pw


def prompt_and_verify_passphrase(artifact_path: str, *, attempts: int = 3) -> bool:
    """Prompt on the TTY until ``artifact_path`` decrypts, or attempts run out.

    The console counterpart of the GUI passphrase dialog, with the same contract:
    verify first, cache only on success, give the operator more than one try. A
    typo must not abort a headless run, and must not leave a wrong passphrase in
    the cache to break every later load.

    Returns True once the passphrase is cached, False if the operator gave up,
    exhausted the attempts, or no passphrase could be obtained at all."""
    if _CACHED_PASSPHRASE is not None:
        return True

    if not stdin_is_tty():
        # Supervised child: a single line piped in by the parent. No retries —
        # there is no operator on the other end, only a pipe that closes once.
        pw = read_passphrase_from_stdin()
        if pw and verify_passphrase(artifact_path, pw):
            set_passphrase(pw)
            return True
        return False

    for remaining in range(attempts, 0, -1):
        try:
            pw = getpass.getpass("Enter map decryption passphrase: ")
        except (KeyboardInterrupt, EOFError):
            print()  # leave the cancelled prompt on its own line
            return False
        if pw and verify_passphrase(artifact_path, pw):
            set_passphrase(pw)
            return True
        if remaining > 1:
            print(f"Wrong passphrase — {remaining - 1} attempt(s) left.")
    return False


def set_passphrase(passphrase: str) -> None:
    """Inject a passphrase into the process-wide cache.

    Injection point for callers that resolve the passphrase themselves — the GUI
    dialog above all: ``get_passphrase`` would otherwise block on ``getpass``
    (stdin looks like a TTY under a GUI launch) with the prompt invisible behind
    the window. Callers should verify the passphrase before injecting it, since
    a wrong one poisons every subsequent load in the process."""
    global _CACHED_PASSPHRASE
    if not passphrase:
        raise EncryptionError("refusing to cache an empty passphrase")
    _CACHED_PASSPHRASE = passphrase


def clear_passphrase() -> None:
    """Drop the cached passphrase. Called when switching projects, so a passphrase
    entered for one project never silently decrypts another."""
    global _CACHED_PASSPHRASE
    _CACHED_PASSPHRASE = None


def verify_passphrase(path: str, passphrase: str) -> bool:
    """Return True if ``passphrase`` decrypts the encrypted artifact at ``path``.

    Used to validate operator input before caching it. Decrypts in full, so pass
    the smallest encrypted artifact available (calibration.json, a few KB) rather
    than the map database. Returns False for a wrong passphrase, a tampered or
    malformed container, or a plaintext file (nothing to verify against)."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False
    if not is_encrypted(data):
        return False
    try:
        decrypt_bytes(data, passphrase)
    except EncryptionError:
        return False
    return True
