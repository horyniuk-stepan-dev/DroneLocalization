"""HARDENING P1-6 (encryption-at-rest), Sub-project 1: crypto foundation.

Unit tests for src/security/at_rest.py — the passphrase-derived AES-256-GCM
container reused by all encryption-at-rest sub-projects. Pure-Python (only
depends on `cryptography`); no torch/Qt.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

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


def test_get_passphrase_reads_the_injected_cache(monkeypatch):
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    at_rest.set_passphrase("injected")
    assert at_rest.get_passphrase() == "injected"


def _piped_stdin(monkeypatch, text: str):
    """Replace stdin with a non-TTY stream, as a supervised child sees it."""
    monkeypatch.setattr(at_rest.sys, "stdin", io.StringIO(text))


def test_get_passphrase_ignores_the_environment(monkeypatch):
    """No env-var channel: it is readable by any same-user process, inherited by
    every child, and can land in a crash dump."""
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setenv("DRONELOC_PASSPHRASE", "from-env")
    _piped_stdin(monkeypatch, "")
    with pytest.raises(at_rest.EncryptionError):
        at_rest.get_passphrase()


def test_get_passphrase_raises_when_unset_and_no_tty(monkeypatch):
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    _piped_stdin(monkeypatch, "")
    with pytest.raises(at_rest.EncryptionError):
        at_rest.get_passphrase()


# --- stdin pipe (supervised child) -------------------------------------------


def test_get_passphrase_reads_a_piped_line(monkeypatch):
    """The supervisor pipes the passphrase to each restarted child."""
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    _piped_stdin(monkeypatch, "piped-secret\n")
    assert at_rest.get_passphrase() == "piped-secret"


def test_read_passphrase_from_stdin_is_empty_on_a_tty(monkeypatch):
    """On a terminal the caller must prompt, not silently consume input."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert at_rest.read_passphrase_from_stdin() == ""


def test_read_passphrase_from_stdin_survives_a_closed_stream(monkeypatch):
    stream = io.StringIO("x")
    stream.close()
    monkeypatch.setattr(at_rest.sys, "stdin", stream)
    assert at_rest.read_passphrase_from_stdin() == ""  # fail-closed, no raise


# --- file-level helpers (copy builder + SP3 temp-decrypt) --------------------


def test_encrypt_file_then_decrypt_tempfile_round_trips(tmp_path):
    src = tmp_path / "map.bin"
    src.write_bytes(PLAINTEXT)
    dst = tmp_path / "map.bin.enc"

    at_rest.encrypt_file(str(src), str(dst), PW)
    assert at_rest.is_encrypted(dst.read_bytes())
    assert src.read_bytes() == PLAINTEXT  # source untouched (copy model)

    tmp = at_rest.decrypt_to_tempfile(str(dst), PW, suffix=".bin")
    try:
        assert Path(tmp).read_bytes() == PLAINTEXT
    finally:
        at_rest.wipe_file(tmp)
        assert not Path(tmp).exists()


def test_decrypt_to_tempfile_wrong_passphrase_leaves_no_temp(tmp_path):
    src = tmp_path / "map.bin"
    src.write_bytes(PLAINTEXT)
    dst = tmp_path / "map.bin.enc"
    at_rest.encrypt_file(str(src), str(dst), PW)

    tdir = tempfile.gettempdir()
    before = {n for n in os.listdir(tdir) if n.startswith("dlmap_")}
    with pytest.raises(at_rest.EncryptionError):
        at_rest.decrypt_to_tempfile(str(dst), "wrong")
    after = {n for n in os.listdir(tdir) if n.startswith("dlmap_")}
    assert before == after  # no lingering plaintext temp from the failed decrypt


def test_wipe_file_is_idempotent_on_missing(tmp_path):
    at_rest.wipe_file(str(tmp_path / "does-not-exist"))  # must not raise


# --- console prompt with retries (headless) ----------------------------------


@pytest.fixture
def encrypted_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(at_rest, "_CACHED_PASSPHRASE", None, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    p = tmp_path / "calibration.json"
    p.write_bytes(at_rest.encrypt_bytes(b'{"anchors": []}', "right"))
    return str(p)


def _answers(monkeypatch, *replies):
    """Feed getpass a scripted sequence of operator inputs."""
    seq = iter(replies)
    monkeypatch.setattr(at_rest.getpass, "getpass", lambda *a, **k: next(seq))


def test_prompt_accepts_correct_passphrase(encrypted_artifact, monkeypatch):
    _answers(monkeypatch, "right")
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is True
    assert at_rest._CACHED_PASSPHRASE == "right"


def test_prompt_retries_after_a_typo(encrypted_artifact, monkeypatch):
    """A typo must not abort a headless run."""
    _answers(monkeypatch, "typo", "right")
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is True
    assert at_rest._CACHED_PASSPHRASE == "right"


def test_prompt_gives_up_after_the_attempt_limit(encrypted_artifact, monkeypatch):
    _answers(monkeypatch, "no", "nope", "still-no")
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact, attempts=3) is False
    # A wrong passphrase must never be cached — it would break every later load.
    assert at_rest._CACHED_PASSPHRASE is None


def test_prompt_returns_false_on_ctrl_c(encrypted_artifact, monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(at_rest.getpass, "getpass", interrupt)
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is False
    assert at_rest._CACHED_PASSPHRASE is None


def test_prompt_returns_false_without_a_tty(encrypted_artifact, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        at_rest.getpass, "getpass", lambda *a, **k: pytest.fail("prompted without a TTY")
    )
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is False


def test_prompt_is_a_noop_when_already_cached(encrypted_artifact, monkeypatch):
    at_rest.set_passphrase("right")
    monkeypatch.setattr(
        at_rest.getpass, "getpass", lambda *a, **k: pytest.fail("prompted despite cache")
    )
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is True


def test_prompt_accepts_a_piped_passphrase(encrypted_artifact, monkeypatch):
    """Supervised child: verified in one shot, no prompt, no retries."""
    monkeypatch.setattr(at_rest.sys, "stdin", io.StringIO("right\n"))
    monkeypatch.setattr(
        at_rest.getpass, "getpass", lambda *a, **k: pytest.fail("prompted on a pipe")
    )
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is True
    assert at_rest._CACHED_PASSPHRASE == "right"


def test_prompt_rejects_a_wrong_piped_passphrase(encrypted_artifact, monkeypatch):
    """No operator on the other end of a pipe — one shot, then fail closed."""
    monkeypatch.setattr(at_rest.sys, "stdin", io.StringIO("wrong\n"))
    assert at_rest.prompt_and_verify_passphrase(encrypted_artifact) is False
    assert at_rest._CACHED_PASSPHRASE is None
