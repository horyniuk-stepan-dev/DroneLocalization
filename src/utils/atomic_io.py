"""
Атомарний запис файлів: tempfile у тій самій директорії + os.replace.

Навіщо: прямий open(path, "w") при конкурентному записі або краші процесу
залишає файл обрізаним/зіпсованим (реальний випадок: 470 хвостових null-байтів
у config.py після конкурентного збереження). os.replace — атомарний на
POSIX і Windows (NTFS), тому читач завжди бачить або стару, або нову версію.
"""

import os
import tempfile


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Атомарно записує bytes у файл."""
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
    """Атомарно записує текст у файл."""
    atomic_write_bytes(path, text.encode(encoding))
