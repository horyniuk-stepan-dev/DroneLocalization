"""HARDENING P1-6: passphrase prompts for encryption-at-rest, GUI side.

Two dialogs:

* ``PassphraseDialog`` — asks for the passphrase of an encrypted project at load
  time, verifies it against a real artifact, and injects it into the at-rest
  cache only once it is known to be correct.
* ``NewPassphraseDialog`` — asks (twice) for the passphrase of a new encrypted
  copy.

The GUI must inject explicitly: ``at_rest.get_passphrase`` falls back to
``getpass`` when stdin looks like a TTY, which under a GUI launch blocks the
process on a prompt the operator cannot see.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from src.security.at_rest import set_passphrase, verify_passphrase
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 3


class PassphraseDialog(QDialog):
    """Modal passphrase prompt for an encrypted project.

    Verifies each attempt against ``verify_artifact`` (pass the cheapest one —
    calibration.json, not the map database) and caches the passphrase process-wide
    only after a successful decryption, so a typo can never poison later loads.
    Rejects after ``MAX_ATTEMPTS`` failures; the caller must then abort the load.
    """

    def __init__(self, project_name: str, verify_artifact: Path, parent=None):
        super().__init__(parent)
        self.verify_artifact = Path(verify_artifact)
        self.attempts_left = MAX_ATTEMPTS

        self.setWindowTitle("Проєкт зашифровано")
        self.setMinimumWidth(420)
        self._init_ui(project_name)

    def _init_ui(self, project_name: str):
        layout = QVBoxLayout(self)

        header = QLabel(
            f"Проєкт «{project_name}» містить зашифровані дані карти.\n"
            f"Введіть пароль, щоб завантажити його."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.EchoMode.Password)
        self.input.setPlaceholderText("Пароль карти")
        self.input.returnPressed.connect(self._on_accept)
        layout.addWidget(self.input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.input.setFocus()

    def _on_accept(self):
        pw = self.input.text()
        if not pw:
            self.error_label.setText("Пароль не може бути порожнім.")
            return

        # Scrypt is deliberately slow (~100 ms); show the operator we are busy.
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = verify_passphrase(str(self.verify_artifact), pw)
        finally:
            self.unsetCursor()

        if ok:
            set_passphrase(pw)
            logger.info("Map passphrase accepted; cached for this session.")
            self.accept()
            return

        self.attempts_left -= 1
        self.input.clear()
        if self.attempts_left <= 0:
            logger.warning("Map passphrase rejected: attempts exhausted.")
            self.reject()
            return
        self.error_label.setText(f"Невірний пароль. Залишилось спроб: {self.attempts_left}.")


class NewPassphraseDialog(QDialog):
    """Prompt for the passphrase of a NEW encrypted copy (entered twice).

    The result is exposed via ``passphrase`` and deliberately NOT cached: the
    session keeps working against the plaintext master after the copy is built.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.passphrase: str | None = None

        self.setWindowTitle("Пароль зашифрованої копії")
        self.setMinimumWidth(420)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel(
            "Задайте пароль для зашифрованої копії проєкту.\n"
            "Його неможливо відновити — збережіть у безпечному місці."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.EchoMode.Password)
        self.input.setPlaceholderText("Новий пароль")
        layout.addWidget(self.input)

        self.confirm = QLineEdit()
        self.confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm.setPlaceholderText("Підтвердіть пароль")
        self.confirm.returnPressed.connect(self._on_accept)
        layout.addWidget(self.confirm)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.input.setFocus()

    def _on_accept(self):
        pw = self.input.text()
        if not pw:
            self.error_label.setText("Пароль не може бути порожнім.")
            return
        if pw != self.confirm.text():
            self.error_label.setText("Паролі не збігаються.")
            self.confirm.clear()
            return
        self.passphrase = pw
        self.accept()
