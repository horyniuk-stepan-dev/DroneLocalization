from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.core.project_registry import ProjectRegistry
from src.security.project_scan import project_is_encrypted
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OpenProjectDialog(QDialog):
    """Dialog for selecting a project from the list of recent projects.

    Replaces bare QFileDialog.getExistingDirectory.
    """

    def __init__(self, registry: ProjectRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.selected_path: str | None = None
        # HARDENING P1-6: encryption state per project path. The list is rebuilt
        # on every keystroke in the search box, so the disk probe is cached.
        self._encrypted_cache: dict[str, bool] = {}

        self.setWindowTitle("Відкрити проєкт")
        self.setMinimumSize(600, 450)
        self._init_ui()
        self._populate_list()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Search
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Пошук за назвою проєкту...")
        self.search_input.textChanged.connect(self._on_search)
        search_row.addWidget(self.search_input)
        layout.addLayout(search_row)

        # Project list
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

        # Preview panel
        self.preview_group = QGroupBox("Деталі проєкту")
        preview_layout = QVBoxLayout(self.preview_group)
        self.lbl_preview = QLabel("Виберіть проєкт зі списку")
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color: #666; font-size: 12px;")
        preview_layout.addWidget(self.lbl_preview)
        layout.addWidget(self.preview_group)

        # Buttons
        buttons_row = QHBoxLayout()

        self.btn_browse = QPushButton("📂 Інша папка...")
        self.btn_browse.setToolTip(
            "Відкрити проєкт з довільної папки. Якщо обрана папка сама не є проєктом, "
            "проєкти всередині неї буде імпортовано до списку"
        )
        self.btn_browse.clicked.connect(self._on_browse)

        self.btn_import = QPushButton("📥 Імпортувати...")
        self.btn_import.setToolTip(
            "Додати до списку наявні проєкти (папки з project.json), не відкриваючи їх.\n"
            "Можна обрати саму папку проєкту або папку, що містить кілька проєктів "
            "(пошук до 2 рівнів вкладеності)."
        )
        self.btn_import.clicked.connect(self._on_import)

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
        buttons_row.addWidget(self.btn_import)
        buttons_row.addWidget(self.btn_remove)
        buttons_row.addStretch()
        buttons_row.addWidget(self.btn_cancel)
        buttons_row.addWidget(self.btn_open)
        layout.addLayout(buttons_row)

    def _is_encrypted(self, path: str) -> bool:
        """Cached encryption probe — see ``_encrypted_cache``."""
        if not path:
            return False
        if path not in self._encrypted_cache:
            try:
                self._encrypted_cache[path] = project_is_encrypted(path)
            except Exception as e:  # a broken project must not break the picker
                logger.debug(f"Encryption probe failed for {path}: {e}")
                self._encrypted_cache[path] = False
        return self._encrypted_cache[path]

    def _populate_list(self, filter_text: str = ""):
        """Populates the list of projects."""
        self.project_list.clear()
        projects = self.registry.get_recent(limit=50)

        for proj in projects:
            name = proj.get("name", "Без назви")
            if filter_text and filter_text.lower() not in name.lower():
                continue

            # Status indicators
            has_db = proj.get("has_database", False)
            has_cal = proj.get("has_calibration", False)
            status = ""
            if has_db and has_cal:
                status = "✅"
            elif has_db:
                status = "⚠️ без калібрування"
            else:
                status = "❌ без бази"

            # Date format
            last = proj.get("last_opened", "")
            try:
                dt = datetime.fromisoformat(last)
                date_str = dt.strftime("%d.%m.%Y %H:%M")
            except (ValueError, TypeError):
                date_str = "—"

            lock = "🔒 " if self._is_encrypted(proj.get("path", "")) else ""
            item_text = f"{status}  {lock}{name}   [останній: {date_str}]"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, proj)
            if lock:
                item.setToolTip("🔒 Зашифрований проєкт — при відкритті запитає пароль карти")

            # Flag unavailable projects
            if not Path(proj["path"]).is_dir():
                item.setForeground(QColor("#aaa"))
                item.setToolTip("⚠ Папка проєкту не знайдена")

            self.project_list.addItem(item)

        if self.project_list.count() == 0:
            item = QListWidgetItem(
                "    (немає проєктів — створіть новий, відкрийте або імпортуйте папку)"
            )
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
        path = proj.get("path", "")
        video = proj.get("video_path", "—")
        created = proj.get("created_at", "—")
        try:
            created = datetime.fromisoformat(created).strftime("%d.%m.%Y %H:%M")
        except (ValueError, TypeError):
            pass

        # Legacy projects keep database.h5 in the root, layered ones under sources/<id>/.
        db_size = "—"
        try:
            root = Path(path)
            db_files = [root / "database.h5", *root.glob("sources/*/database.h5")]
            sizes = [f.stat().st_size for f in db_files if f.is_file()]
        except OSError:
            sizes = []
        if sizes:
            db_size = f"{sum(sizes) / (1024 * 1024):.1f} MB"
            if len(sizes) > 1:
                db_size += f" (шарів: {len(sizes)})"

        self.lbl_preview.setText(
            f"<b>Назва:</b> {proj.get('name', '—')}<br>"
            f"<b>Шлях:</b> {path}<br>"
            f"<b>Відео:</b> {Path(video).name if video else '—'}<br>"
            f"<b>Створено:</b> {created}<br>"
            f"<b>База даних:</b> {'✅ ' + db_size if proj.get('has_database') else '❌ відсутня'}<br>"
            f"<b>Калібрація:</b> {'✅ є' if proj.get('has_calibration') else '❌ відсутня'}<br>"
            f"<b>Шифрування:</b> "
            + (
                "🔒 зашифровано (потрібен пароль карти)"
                if self._is_encrypted(path)
                else "відкритий текст"
            )
        )

    def _on_double_click(self, item: QListWidgetItem):
        proj = item.data(Qt.ItemDataRole.UserRole)
        if proj and Path(proj["path"]).is_dir():
            self.selected_path = proj["path"]
            self.accept()

    def _on_open(self):
        current = self.project_list.currentItem()
        if current:
            proj = current.data(Qt.ItemDataRole.UserRole)
            if proj:
                if not Path(proj["path"]).is_dir():
                    QMessageBox.warning(
                        self, "Помилка", f"Папка проєкту не знайдена:\n{proj['path']}"
                    )
                    return
                self.selected_path = proj["path"]
                self.accept()

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Виберіть папку проєкту", "")
        if not path:
            return
        if (Path(path) / "project.json").is_file():
            self.selected_path = path
            self.accept()
            return
        # The usual slip is picking the folder that holds the projects: instead of
        # failing on load, list what is inside and let the user pick.
        self._import_from(path)

    def _default_import_dir(self) -> str:
        """Parent folder of the most recent project — projects usually sit together."""
        for proj in self.registry.get_recent(limit=1):
            parent = Path(proj.get("path", "")).parent
            if parent.is_dir():
                return str(parent)
        return ""

    def _on_import(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка проєкту або папка з проєктами", self._default_import_dir()
        )
        if path:
            self._import_from(path)

    def _import_from(self, folder: str):
        """Registers every project found at/below ``folder`` and selects the first."""
        projects = ProjectRegistry.find_projects(folder)
        if not projects:
            QMessageBox.warning(
                self,
                "Імпорт проєктів",
                f"У папці не знайдено жодного проєкту (файлу project.json):\n{folder}\n\n"
                f"Пошук охоплює саму папку і до 2 рівнів вкладених папок.",
            )
            return

        result = self.registry.import_projects(projects)
        self.search_input.clear()
        self._populate_list()
        self._select_path(result.added[0] if result.added else projects[0])

        names = [p.name for p in projects]
        shown = "\n".join(f"  • {n}" for n in names[:15])
        if len(names) > 15:
            shown += f"\n  … і ще {len(names) - 15}"
        QMessageBox.information(
            self,
            "Імпорт проєктів",
            f"Знайдено проєктів: {len(projects)}\n"
            f"Додано до списку: {len(result.added)}\n"
            f"Вже були у списку: {len(result.already_known)}\n\n{shown}",
        )

    def _select_path(self, path: str | Path):
        """Makes the list item of the project at ``path`` current, if listed."""
        target = str(Path(path).resolve()).casefold()
        for row in range(self.project_list.count()):
            item = self.project_list.item(row)
            proj = item.data(Qt.ItemDataRole.UserRole)
            if proj and str(Path(proj.get("path", "")).resolve()).casefold() == target:
                self.project_list.setCurrentItem(item)
                self.project_list.scrollToItem(item)
                return

    def _on_remove(self):
        current = self.project_list.currentItem()
        if not current:
            return
        proj = current.data(Qt.ItemDataRole.UserRole)
        if not proj:
            return

        reply = QMessageBox.question(
            self,
            "Підтвердження",
            f"Видалити «{proj['name']}» зі списку?\n\nФайли проєкту НЕ будуть видалені.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.registry.unregister(proj["path"])
            self._populate_list(self.search_input.text())

    def get_selected_path(self) -> str | None:
        return self.selected_path
