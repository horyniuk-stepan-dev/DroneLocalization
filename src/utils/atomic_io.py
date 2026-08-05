"""
Atomic file write: tempfile in the same directory + os.replace.

Rationale: a plain open(path, 'w') under concurrent writes or a process crash
leaves the file truncated/corrupted (real case: 470 trailing null bytes in
config.py after concurrent saves). os.replace is atomic on POSIX and Windows
(NTFS), so the reader always sees either the old or the new version.
"""

import os
import tempfile


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Atomically write bytes to a file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text to a file."""
    atomic_write_bytes(path, text.encode(encoding))
