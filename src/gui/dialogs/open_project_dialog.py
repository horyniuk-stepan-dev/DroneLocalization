import os
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QFileDialog, QLineEdit,
    QMessageBox, QGroupBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from src.core.project_registry import ProjectRegistry
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OpenProjectDialog(QDialog):
    """
    Діалог вибору проєкту зі списку нещодавніх.
    Замінює голий QFileDialog.getExistingDirectory.
    """

    def __init__(self, registry: ProjectRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.selected_path: str | None = None

        self.setWindowTitle("Відкрити проєкт")
        self.setMinimumSize(600, 450)
        self._init_ui()
        self._populate_list()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Пошук
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Пошук за назвою проєкту...")
        self.search_input.textChanged.connect(self._on_search)
        search_row.addWidget(self.search_input)
        layout.addLayout(search_row)

        # Список проєктів
        self.project_list = QListWidget()
        self.project_list.setAlternatingRowColors(True)
        self.project_list.setStyleSheet(
            "QListWidget { font-size: 13px; }"
            "QListWidget::item { padding: 8px 6px; }"
            "QListWidget::item:selected { background: #1565C0; color: white; }"
        )
        self.project_list.itemDoubleClicked.connect(self._on_double_click)
        self.project_list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.project_list, stretch=1)

        # Preview панель
        self.preview_group = QGroupBox("Деталі проєкту")
        preview_layout = QVBoxLayout(self.preview_group)
        self.lbl_preview = QLabel("Виберіть проєкт зі списку")
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color: #666; font-size: 12px;")
        preview_layout.addWidget(self.lbl_preview)
        layout.addWidget(self.preview_group)

        # Кнопки
        buttons_row = QHBoxLayout()

        self.btn_browse = QPushButton("📂 Інша папка...")
        self.btn_browse.setToolTip("Відкрити проєкт з довільної папки")
        self.btn_browse.clicked.connect(self._on_browse)

        self.btn_remove = QPushButton("🗑 Видалити зі списку")
        self.btn_remove.setToolTip("Видаляє лише зі списку, файли залишаються")
        self.btn_remove.setStyleSheet("color: #b71c1c;")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove)

        self.btn_open = QPushButton("✅ Відкрити")
        self.btn_open.setStyleSheet(
            "background: #1565C0; color: white; font-weight: bold; padding: 8px 20px;"
        )
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._on_open)

        self.btn_cancel = QPushButton("Скасувати")
        self.btn_cancel.clicked.connect(self.reject)

        buttons_row.addWidget(self.btn_browse)
        buttons_row.addWidget(self.btn_remove)
        buttons_row.addStretch()
        buttons_row.addWidget(self.btn_cancel)
        buttons_row.addWidget(self.btn_open)
        layout.addLayout(buttons_row)

    def _populate_list(self, filter_text: str = ""):
        """Заповнити список проєктів."""
        self.project_list.clear()
        projects = self.registry.get_recent(limit=50)

        for proj in projects:
            name = proj.get('name', 'Без назви')
            if filter_text and filter_text.lower() not in name.lower():
                continue

            # Статус-іконки
            has_db = proj.get('has_database', False)
            has_cal = proj.get('has_calibration', False)
            status = ""
            if has_db and has_cal:
                status = "✅"
            elif has_db:
                status = "⚠️ без калібрування"
            else:
                status = "❌ без бази"

            # Формат дати
            last = proj.get('last_opened', '')
            try:
                dt = datetime.fromisoformat(last)
                date_str = dt.strftime("%d.%m.%Y %H:%M")
            except (ValueError, TypeError):
                date_str = "—"

            item_text = f"{status}  {name}   [останній: {date_str}]"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, proj)

            # Позначаємо недоступні проєкти
            if not Path(proj['path']).is_dir():
                item.setForeground(QColor("#aaa"))
                item.setToolTip("⚠ Папка проєкту не знайдена")

            self.project_list.addItem(item)

        if self.project_list.count() == 0:
            item = QListWidgetItem("    (немає проєктів — створіть новий або відкрийте папку)")
            item.setForeground(QColor("#999"))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.project_list.addItem(item)

    def _on_search(self, text: str):
        self._populate_list(filter_text=text)

    def _on_selection_changed(self, current: QListWidgetItem, _previous):
        if current is None:
            self.btn_open.setEnabled(False)
            self.btn_remove.setEnabled(False)
            self.lbl_preview.setText("Виберіть проєкт зі списку")
            return

        proj = current.data(Qt.ItemDataRole.UserRole)
        if proj is None:
            self.btn_open.setEnabled(False)
            self.btn_remove.setEnabled(False)
            return

        self.btn_open.setEnabled(True)
        self.btn_remove.setEnabled(True)

        # Preview
        path = proj.get('path', '')
        video = proj.get('video_path', '—')
        created = proj.get('created_at', '—')
        try:
            created = datetime.fromisoformat(created).strftime("%d.%m.%Y %H:%M")
        except (ValueError, TypeError):
            pass

        db_size = "—"
        db_path = Path(path) / "database.h5"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            db_size = f"{size_mb:.1f} MB"

        self.lbl_preview.setText(
            f"<b>Назва:</b> {proj.get('name', '—')}<br>"
            f"<b>Шлях:</b> {path}<br>"
            f"<b>Відео:</b> {Path(video).name if video else '—'}<br>"
            f"<b>Створено:</b> {created}<br>"
            f"<b>База даних:</b> {'✅ ' + db_size if proj.get('has_database') else '❌ відсутня'}<br>"
            f"<b>Калібрація:</b> {'✅ є' if proj.get('has_calibration') else '❌ відсутня'}"
        )

    def _on_double_click(self, item: QListWidgetItem):
        proj = item.data(Qt.ItemDataRole.UserRole)
        if proj and Path(proj['path']).is_dir():
            self.selected_path = proj['path']
            self.accept()

    def _on_open(self):
        current = self.project_list.currentItem()
        if current:
            proj = current.data(Qt.ItemDataRole.UserRole)
            if proj:
                if not Path(proj['path']).is_dir():
                    QMessageBox.warning(self, "Помилка", f"Папка проєкту не знайдена:\n{proj['path']}")
                    return
                self.selected_path = proj['path']
                self.accept()

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Виберіть папку проєкту", "")
        if path:
            self.selected_path = path
            self.accept()

    def _on_remove(self):
        current = self.project_list.currentItem()
        if not current:
            return
        proj = current.data(Qt.ItemDataRole.UserRole)
        if not proj:
            return

        reply = QMessageBox.question(
            self, "Підтвердження",
            f"Видалити «{proj['name']}» зі списку?\n\n"
            f"Файли проєкту НЕ будуть видалені.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.registry.unregister(proj['path'])
            self._populate_list(self.search_input.text())

    def get_selected_path(self) -> str | None:
        return self.selected_path
