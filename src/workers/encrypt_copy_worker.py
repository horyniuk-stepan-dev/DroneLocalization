"""HARDENING P1-6: background worker for building an encrypted project copy.

Encrypting the map database and the lance index runs to tens of seconds (Scrypt
plus whole-file AES-GCM over hundreds of MB), so it must not run on the GUI
thread. The plaintext master is never modified — see ``build_encrypted_copy``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EncryptCopyWorker(QThread):
    """Фоновий потік для створення зашифрованої копії проєкту."""

    progress = pyqtSignal(str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, src_dir: str, dst_dir: str, passphrase: str):
        super().__init__()
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self._passphrase = passphrase

        logger.info(f"EncryptCopyWorker initialized: {src_dir} -> {dst_dir}")

    def run(self):
        try:
            self.progress.emit("Шифрування проєкту...")
            # scripts/ is not a package; add the repo root so the CLI builder is
            # importable from the GUI without duplicating its logic.
            repo_root = str(Path(__file__).resolve().parents[2])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from scripts.encrypt_project import build_encrypted_copy

            summary = build_encrypted_copy(self.src_dir, self.dst_dir, self._passphrase)
            logger.info(f"Encrypted copy built: {summary['total']} file(s) encrypted")
            self.completed.emit(summary)

        except Exception as e:
            logger.error(f"Failed to build encrypted copy: {e}", exc_info=True)
            self.error.emit(str(e))
        finally:
            # Do not keep the passphrase alive in the thread object.
            self._passphrase = ""
