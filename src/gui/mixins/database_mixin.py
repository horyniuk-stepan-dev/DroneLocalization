from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.calibration.multi_calibration_manager import (
    MultiCalibrationManager,
    save_layer_calibration,
)
from src.core.export_results import ResultExporter
from src.core.layer_status import (
    compute_layer_status,
    find_layer_conflicts,
    find_path_conflicts,
)
from src.core.project_registry import ProjectRegistry
from src.core.project_video_source import ProjectVideoSource
from src.database.database_loader import DatabaseLoader
from src.database.multi_database_manager import MultiDatabaseManager
from src.geometry.coordinates import CoordinateConverter
from src.gui.dialogs.new_mission_dialog import NewMissionDialog
from src.gui.dialogs.open_project_dialog import OpenProjectDialog
from src.gui.dialogs.passphrase_dialog import NewPassphraseDialog, PassphraseDialog
from src.security.at_rest import clear_passphrase
from src.security.project_scan import (
    encrypted_artifacts_at,
    find_project_root,
    project_is_encrypted,
)
from src.utils.logging_utils import get_logger
from src.workers.database_worker import DatabaseGenerationWorker
from src.workers.encrypt_copy_worker import EncryptCopyWorker

logger = get_logger(__name__)


class DatabaseMixin:
    # ── Project registry (initialised once) ─────────────────────────────────────

    def _get_registry(self) -> ProjectRegistry:
        if not hasattr(self, "_project_registry"):
            self._project_registry = ProjectRegistry()
        return self._project_registry

    # ── New mission ────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_new_mission(self):
        dialog = NewMissionDialog(self)
        if not dialog.exec():
            return

        mission_data = dialog.get_mission_data()
        workspace_dir = mission_data.get("workspace_dir")
        video_path = mission_data.get("video_path")

        if not workspace_dir or not video_path:
            return

        # A new project must not inherit the previous project's databases,
        # managers or anchors: the old db_manager used to capture the new DB and
        # the old anchors were saved into the new project's calibration.
        self._reset_session_state()

        # Create project directory structure
        if not self.project_manager.create_project(workspace_dir, mission_data):
            QMessageBox.critical(self, "Помилка", "Не вдалося створити проєкт!")
            return
        self.active_source_id = "main"

        # Register in the project registry
        self._get_registry().register(
            project_dir=str(self.project_manager.project_dir),
            name=self.project_manager.project_name,
            video_path=video_path,
        )

        self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
        self._start_database_generation(
            video_path, self.project_manager.database_path, source_id="main"
        )

    # ── Layer (video source) state ───────────────────────────────────────────
    #
    # One layer = one ProjectVideoSource with its own video, database.h5 and
    # calibration.json. ``active_source_id`` is the single source of truth for
    # "which layer the buttons act on"; self.database / self.calibration are
    # always that layer's live objects — the same instances db_manager and
    # calib_manager hold — and only _activate_source switches them, together.

    def _is_multi_source(self) -> bool:
        return getattr(self, "db_manager", None) is not None

    def _wants_multi_source(self) -> bool:
        """The rule the project loader has always used to choose the mode."""
        settings = self.project_manager.settings if self.project_manager.is_loaded else None
        if settings is None:
            return False
        sources = settings.get_enabled_sources()
        return len(sources) > 1 or any(s.source_id != "main" for s in sources)

    def _source_config(self, source_id: str | None) -> ProjectVideoSource | None:
        if not source_id or not self.project_manager.is_loaded:
            return None
        settings = self.project_manager.settings
        return settings.get_source(source_id) if settings else None

    def _layer_ops_busy(self, action: str = "", quiet: bool = False) -> bool:
        """True (and says why) while a worker that uses the layer databases runs."""
        running = []
        for attr, what in (
            ("db_worker", "генерація БД"),
            ("propagation_worker", "пропагація GPS"),
            ("tracking_worker", "відстеження"),
        ):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                running.append(what)
        if not running:
            return False
        msg = f"{action or 'Дія'}: недоступно, поки виконується {', '.join(running)}."
        if quiet:
            self.status_bar.showMessage(msg, 6000)
        else:
            QMessageBox.information(self, "Зачекайте", msg)
        return True

    def _activate_source(self, source_id: str, *, quiet: bool = False) -> bool:
        """Makes ``source_id`` the active layer; its DB and calibration switch together."""
        src = self._source_config(source_id)
        if src is None:
            if not quiet:
                QMessageBox.warning(self, "Помилка", f"Шар '{source_id}' не знайдено в проєкті!")
            return False

        if self._is_multi_source():
            try:
                # get_or_load: a layer disabled at open time or added later still
                # gets its anchors from disk instead of an empty object whose first
                # save would overwrite the file.
                calibration = self.calib_manager.get_or_load(src, self.project_manager.project_dir)
            except Exception as e:
                logger.error(f"Cannot read calibration of layer '{source_id}': {e}", exc_info=True)
                if not quiet:
                    QMessageBox.critical(
                        self,
                        "Помилка калібрування",
                        f"Не вдалося прочитати калібрування шару «{source_id}»:\n{e}\n\n"
                        f"Файл не змінено. Виправте або перейменуйте його і повторіть.",
                    )
                return False
            self.database = self.db_manager.get_database(source_id)  # None: no DB yet
            self.calibration = calibration
        elif source_id != (self.active_source_id or self._single_layer_id()):
            # Single-source mode holds exactly one layer; the other rows are
            # disabled layers. Switching would pair this DB with their files.
            if not quiet:
                QMessageBox.information(
                    self,
                    "Шар вимкнено",
                    f"Шар «{source_id}» вимкнено. Увімкніть його "
                    f"(ПКМ по шару → Увімкнути), щоб з ним працювати.",
                )
            return False

        self.active_source_id = source_id
        if self.database is not None and self.database.converter is not None:
            self.calibration.converter = self.database.converter
        if getattr(self, "map_widget", None):
            self.map_widget.set_coordinate_mode(self.calibration.converter.mode, reset=True)
        logger.info(
            f"Active layer: '{source_id}' (db={'loaded' if self.database else 'none'}, "
            f"anchors={len(self.calibration.anchors)})"
        )
        self._update_project_info_panel()
        return True

    def _single_layer_id(self) -> str:
        """The layer single-source mode loads: 'main', else the first one."""
        settings = self.project_manager.settings if self.project_manager.is_loaded else None
        if settings is None or settings.get_source("main") is not None:
            return "main"
        first = next(iter(settings.source_configs()), None)
        return first.source_id if first is not None else "main"

    def _ensure_active(self, source_id: str) -> bool:
        return source_id == self.active_source_id or self._activate_source(source_id)

    def _close_layer_objects(self) -> None:
        """Closes every open layer database and forgets all per-layer objects."""
        try:
            if getattr(self, "database", None) is not None:
                self.database.close()
            if getattr(self, "db_manager", None) is not None:
                self.db_manager.close_all()
        except Exception as e:
            logger.warning(f"Error closing layer databases: {e}")
        self.db_manager = None
        self.calib_manager = None
        self.database = None
        self.calibration = MultiAnchorCalibration()
        self.active_source_id = None

    def _reset_session_state(self) -> None:
        """Drops everything that belongs to the previously open project."""
        self._close_layer_objects()
        if getattr(self, "map_widget", None):
            self.map_widget.clear_trajectory()
            self.map_widget.clear_verification_markers()
        if hasattr(self, "_tracking_results"):
            self._tracking_results = []

    def _load_project_layers(self, prefer_source_id: str | None = None) -> None:
        """(Re)creates the per-layer objects of the loaded project and activates one.

        Used on project open and whenever enabling/adding/removing a layer changes
        the mode, so a new layer is usable immediately — same result as reopening.
        """
        settings = self.project_manager.settings
        project_dir = self.project_manager.project_dir
        self._close_layer_objects()

        if self._wants_multi_source():
            sources = settings.get_enabled_sources()
            self.db_manager = MultiDatabaseManager(sources, project_dir, config=self.config)
            self.calib_manager = MultiCalibrationManager()
            self.calib_manager.load_all(sources, project_dir)
            candidates = [prefer_source_id, *self.db_manager.all_source_ids]
            candidates += [s.source_id for s in sources]
            for sid in candidates:
                if sid and self._activate_source(sid, quiet=True):
                    break
            logger.info(
                f"Multi-source project loaded: {self.db_manager.num_databases} databases, "
                f"sources={self.db_manager.all_source_ids}, active='{self.active_source_id}'"
            )
        else:
            src = settings.get_source(self._single_layer_id())
            if src is not None:
                sid = src.source_id
                db_path = Path(project_dir) / src.database_file
                cal_path = Path(project_dir) / src.calibration_file
            else:
                sid = "main"
                db_path = Path(self.project_manager.database_path)
                cal_path = Path(self.project_manager.calibration_path)
            self.database = DatabaseLoader(str(db_path)) if db_path.exists() else None
            self.calibration = MultiAnchorCalibration()
            if cal_path.exists():
                self.calibration.load(str(cal_path))
            self.active_source_id = sid

        # Projection: the DB (what propagation used) has priority over the JSON.
        if self.database is not None and self.database.converter is not None:
            self.calibration.converter = self.database.converter
        if getattr(self, "map_widget", None):
            self.map_widget.set_coordinate_mode(self.calibration.converter.mode, reset=True)
        self._update_project_info_panel()

    # ── Database generation ────────────────────────────────────────────────────────

    def _find_source_id_by_db_path(self, db_path: str) -> str | None:
        """Знаходить source_id, чий database_file відповідає db_path."""
        if not self.project_manager.is_loaded or not self.project_manager.settings:
            return None
        project_dir = self.project_manager.project_dir
        try:
            target = Path(db_path).resolve()
        except OSError:
            return None
        for src in self.project_manager.settings.video_sources or []:
            sid = src.get("source_id") if isinstance(src, dict) else src.source_id
            db_file = src.get("database_file") if isinstance(src, dict) else src.database_file
            if not sid or not db_file:
                continue
            try:
                if (Path(project_dir) / db_file).resolve() == target:
                    return sid
            except OSError:
                continue
        return None

    def _start_database_generation(
        self,
        video_path: str,
        save_path: str,
        required_frame_ids: set[int] | None = None,
        source_id: str | None = None,
    ):
        if self._refuse_if_encrypted_project("Генерація бази даних"):
            return

        sid = source_id or self._find_source_id_by_db_path(save_path)
        self._db_build_source_id = sid

        # Do NOT initialize WEB_MERCATOR when starting database generation.
        # UTM converter will be initialized automatically after first GPS anchor.
        # (Build actions activate the target layer first, so this is its calibration.)
        if not self.calibration.is_calibrated and self.calibration.converter.mode != "LOCAL":
            self.calibration.converter = CoordinateConverter(
                "UTM"
            )  # ref_gps=None → auto on first anchor

        self.control_panel.btn_new_mission.setEnabled(False)
        self.control_panel.btn_load_db.setEnabled(False)
        self.control_panel.update_progress(0)
        self.control_panel.set_db_generation_running(True)

        # CRITICAL: release exactly the database file that is about to be
        # overwritten. Other layers stay open — closing the active loader while
        # building another layer used to leave a dead HDF5 handle in db_manager.
        if self._is_multi_source() and sid:
            if self.database is not None and self.database is self.db_manager.get_database(sid):
                self.database = None
            # Unload also before overwriting vectors.lance
            self.db_manager.unload_source(sid)
        elif self.database is not None:
            try:
                self.database.close()
                logger.info("Current database closed before starting new generation.")
            except Exception as e:
                logger.warning(f"Could not close database: {e}")
            self.database = None

        self.db_worker = DatabaseGenerationWorker(
            video_path=video_path,
            output_path=save_path,
            model_manager=self.model_manager,
            config=self.config,
            project_manager=self.project_manager,
            required_frame_ids=required_frame_ids,
        )
        self.db_worker.progress.connect(self.on_db_progress)
        self.db_worker.completed.connect(self.on_db_completed)
        self.db_worker.error.connect(self.on_db_error)
        self.db_worker.cancelled.connect(self.on_db_cancelled)

        # Connect stop button
        self.control_panel.stop_db_generation_clicked.connect(self.on_stop_db_generation)

        self._update_project_info_panel()
        self.db_worker.start()

    @pyqtSlot()
    def on_stop_db_generation(self):
        if hasattr(self, "db_worker") and self.db_worker and self.db_worker.isRunning():
            self.control_panel.update_status("Зупинка... (чекаємо завершення кадру)")
            self.db_worker.stop()

    @pyqtSlot(int, str)
    def on_db_progress(self, percent: int, message: str):
        self.control_panel.update_progress(percent)
        self.control_panel.update_status(message)

    @pyqtSlot(str)
    def on_db_completed(self, db_path: str):
        self.control_panel.set_db_generation_running(False)
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.current_database_path = db_path
        sid = self._db_build_source_id or self._find_source_id_by_db_path(db_path)
        self._db_build_source_id = None

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self._is_multi_source() and sid:
                # Fresh loader + LanceDB handle for the rebuilt layer, then switch
                # database AND calibration to it together.
                src = self._source_config(sid)
                if src is not None and src.enabled:
                    self.db_manager.reload_source(src)
                self._activate_source(sid)
            else:
                if self.database:
                    self.database.close()
                self.database = DatabaseLoader(db_path)
                if sid:
                    self.active_source_id = sid
        finally:
            QApplication.restoreOverrideCursor()
        layer = sid or "main"
        self.control_panel.update_progress(100)
        self.control_panel.update_status(f"Базу шару «{layer}» успішно створено")
        self.status_bar.showMessage(
            f"Проєкт: {self.project_manager.project_name} | Шар: {layer} | База: {db_path}"
        )

        self._after_layer_data_changed()

        QMessageBox.information(self, "Успіх", f"Базу даних шару «{layer}» успішно згенеровано!")

    @pyqtSlot(str)
    def on_db_error(self, error_msg: str):
        self._db_build_source_id = None
        self.control_panel.set_db_generation_running(False)
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.control_panel.update_progress(0)
        self.control_panel.update_status("Помилка генерації")
        self._update_project_info_panel()
        QMessageBox.critical(self, "Помилка", f"Помилка генерації:\n{error_msg}")

    @pyqtSlot()
    def on_db_cancelled(self):
        self._db_build_source_id = None
        self.control_panel.set_db_generation_running(False)
        self.control_panel.update_status("Генерацію скасовано користувачем")
        self.control_panel.update_progress(0)
        self._update_project_info_panel()

    # ── Project opening ────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_load_database(self):
        dialog = OpenProjectDialog(self._get_registry(), parent=self)
        if not dialog.exec():
            self.status_bar.showMessage("Вибір проєкту скасовано")
            return

        path = dialog.get_selected_path()
        if not path:
            return

        self._open_project(path)

    # ── Encrypted project export ───────────────────────────────────────────────

    @pyqtSlot()
    def on_create_encrypted_copy(self):
        """Create an encrypted copy of the current project (master remains unchanged)."""
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Warning", "Please open the project first!")
            return

        src_dir = Path(self.project_manager.project_dir)
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Save encrypted copy to", str(src_dir.parent)
        )
        if not parent_dir:
            return

        dst_dir = Path(parent_dir) / f"{src_dir.name}_encrypted"
        if dst_dir.exists():
            QMessageBox.critical(
                self, "Error", f"Directory already exists (overwriting not allowed):\n{dst_dir}"
            )
            return

        dialog = NewPassphraseDialog(self)
        if not dialog.exec() or not dialog.passphrase:
            return

        self.status_bar.showMessage(f"Creating encrypted copy: {dst_dir.name}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self._encrypt_worker = EncryptCopyWorker(str(src_dir), str(dst_dir), dialog.passphrase)
        self._encrypt_worker.progress.connect(self.status_bar.showMessage)
        self._encrypt_worker.completed.connect(self._on_encrypted_copy_done)
        self._encrypt_worker.error.connect(self._on_encrypted_copy_error)
        self._encrypt_worker.start()

    @pyqtSlot(dict)
    def _on_encrypted_copy_done(self, summary: dict):
        QApplication.restoreOverrideCursor()
        self.status_bar.showMessage("Encrypted copy created")
        if not summary["encrypted"]:
            QMessageBox.warning(
                self,
                "Warning",
                "Copy created, but source project is empty — nothing to encrypt.",
            )
            return
        QMessageBox.information(
            self,
            "Done",
            f"Encrypted files: {summary['total']} (all, without exceptions)\n\n"
            f"Original project is not modified. Copy is immutable: application will refuse "
            f"to write to it — rebuilds and calibration must be done on the master.\n\n"
            f"Passphrase cannot be recovered — save it in a safe place.",
        )

    @pyqtSlot(str)
    def _on_encrypted_copy_error(self, message: str):
        QApplication.restoreOverrideCursor()
        self.status_bar.showMessage("Encrypted copy creation failed")
        QMessageBox.critical(self, "Error", f"Failed to create copy:\n{message}")

    def _refuse_if_encrypted_project(self, action: str) -> bool:
        """True (and shows why) if ``action`` would write into an encrypted copy.

        The write guards in the core raise regardless — this only turns the
        refusal into a clear message before any work starts, instead of an
        exception surfacing from a worker thread."""
        if not getattr(self.project_manager, "is_encrypted", False):
            return False
        QMessageBox.critical(
            self,
            "Encrypted project",
            f"{action} is impossible: this is an encrypted copy for deployment, "
            f"it is immutable.\n\nPerform this action on an open master project, "
            f"and then create a new encrypted copy from it.",
        )
        return True

    def _prompt_passphrase_if_encrypted(self, path: str) -> bool:
        """Ask for the map passphrase if the project at ``path`` is encrypted.

        Runs BEFORE the project is loaded: a fully encrypted copy encrypts
        project.json too, so ``load_project`` cannot even parse the manifest
        without the passphrase. The project's display name is unknown at this
        point for the same reason — the folder name is used instead.

        Returns True when loading may proceed: either the project is plaintext
        (no prompt at all — behaviour identical to before this feature) or the
        operator supplied a passphrase that provably decrypts an artifact.
        Returns False if the operator cancelled or exhausted their attempts, in
        which case the caller must abort the load rather than fail deep inside
        h5py with an opaque error."""
        encrypted = encrypted_artifacts_at(path)
        if not encrypted:
            return True

        dialog = PassphraseDialog(Path(path).name, encrypted[0], parent=self)
        if dialog.exec():
            return True

        clear_passphrase()
        self.status_bar.showMessage("Loading cancelled: passphrase required")
        return False

    def _open_project(self, path: str):
        """Load project by path (used for recent menu as well)."""
        # A passphrase belongs to one project only — never let the previous one
        # silently decrypt (or fail against) the project being opened now.
        clear_passphrase()

        # The passphrase must be resolved BEFORE loading: an encrypted copy
        # encrypts project.json itself, so the manifest is unparseable without it.
        if not self._prompt_passphrase_if_encrypted(path):
            return

        if not self.project_manager.load_project(path):
            QMessageBox.critical(self, "Error", "Selected folder is not a valid project!")
            return

        # Nothing of the previous project may survive — including on the
        # "generate missing database" path below, where the old db_manager used
        # to reload the OLD project's database under the new project's name.
        self._reset_session_state()
        self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")

        loaded = False
        try:
            settings = self.project_manager.settings
            if not self._wants_multi_source():
                src = settings.get_source("main") or next(iter(settings.source_configs()), None)
                db_path = (
                    str(self.project_manager.project_dir / src.database_file)
                    if src is not None
                    else self.project_manager.database_path
                )
                if not Path(db_path).exists():
                    video_path = src.video_path if src is not None else settings.video_path
                    reply = QMessageBox.question(
                        self,
                        "Database missing",
                        f"Project '{self.project_manager.project_name}' has no generated database.\n\n"
                        f"Generate database now from video:\n{Path(video_path).name}?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.active_source_id = src.source_id if src is not None else "main"
                        self._start_database_generation(
                            video_path, db_path, source_id=self.active_source_id
                        )
                    else:
                        self.status_bar.showMessage("Loading cancelled: missing database")
                    return
            # Multi-source projects open even if some (or all) layers have no DB
            # yet: such layers are listed as "Без БД" and can be built from the table.

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._load_project_layers()

            # Update registry
            registry = self._get_registry()
            registry.register(
                project_dir=str(self.project_manager.project_dir),
                name=self.project_manager.project_name,
                video_path=self.project_manager.settings.video_path
                if self.project_manager.settings
                else "",
            )

            layer = self.active_source_id or "main"
            if self.database and self.database.is_propagated:
                n_valid = int(self.database.frame_valid.sum())
                n_total = self.database.get_num_frames()
                kind = "Local X/Y" if self.calibration.converter.mode == "LOCAL" else "GPS"
                self.status_bar.showMessage(
                    f"Project: {self.project_manager.project_name} | layer '{layer}' "
                    f"({kind}: {n_valid}/{n_total} frames)"
                )
            else:
                self.status_bar.showMessage(
                    f"Project: {self.project_manager.project_name} | layer '{layer}' "
                    f"(no GPS propagation)"
                )
            self.control_panel.update_status("Project loaded")
            self._update_project_info_panel()
            loaded = True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project database:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

        if loaded:
            self._warn_layer_conflicts()

    # ── Propagation check ───────────────────────────────────────────────────────

    @pyqtSlot()
    def on_verify_propagation(self):
        if not self.database or not self.database.is_propagated:
            QMessageBox.warning(self, "Warning", "Propagation data missing or project not loaded!")
            return

        num_frames = self.database.get_num_frames()
        frame_valid = self.database.frame_valid
        frame_affine = self.database.frame_affine

        # Get frame dimensions from metadata
        width = self.database.metadata.get("frame_width", 1920)
        height = self.database.metadata.get("frame_height", 1080)

        # Frame centre in pixels
        center_px = np.array([[width / 2, height / 2]], dtype=np.float32)

        points_to_show = []

        # Collect valid frames only (stride-5 for map rendering performance)
        step = max(1, num_frames // 200)  # Max ~200 points to avoid slowing down the binder

        for i in range(0, num_frames, step):
            if frame_valid[i]:
                # Apply affine matrix (2x3)
                M = frame_affine[i]
                # Metric = M * [x, y, 1]^T
                metric_x = M[0, 0] * center_px[0, 0] + M[0, 1] * center_px[0, 1] + M[0, 2]
                metric_y = M[1, 0] * center_px[0, 0] + M[1, 1] * center_px[0, 1] + M[1, 2]

                lat, lon = self.calibration.converter.metric_to_gps(
                    float(metric_x), float(metric_y)
                )
                points_to_show.append({"lat": float(lat), "lon": float(lon), "label": str(i)})

        if not points_to_show:
            QMessageBox.information(self, "Information", "No frames with valid coordinates found.")
            return

        self.map_widget.show_verification_markers(points_to_show)
        self.status_bar.showMessage(f"Displayed {len(points_to_show)} verification points on map.")

    # ── Database regeneration ────────────────────────────────────────────────────

    @pyqtSlot()
    def on_rebuild_database(self):
        """Rebuilds the database of the ACTIVE layer (not always the main one)."""
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Warning", "Please load the project first!")
            return

        # Before the confirmation prompt AND before the calibration save below —
        # that save is a write into the project and would otherwise raise.
        if self._refuse_if_encrypted_project("Database rebuild"):
            return
        if self._layer_ops_busy("Перегенерація бази"):
            return

        sid = self._get_current_source_id()
        src = self._source_config(sid)
        if src is not None:
            if not src.enabled:
                QMessageBox.information(
                    self, "Шар вимкнено", f"Увімкніть шар «{sid}» перед побудовою його бази."
                )
                return
            video_path = src.video_path
            db_path = str(self.project_manager.project_dir / src.database_file)
        else:
            video_path = self.project_manager.settings.video_path
            db_path = self.project_manager.database_path

        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(
                self,
                "Warning",
                f"Video of layer '{sid}' not found:\n{video_path}\n\n"
                "Check the video path in project settings.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Database rebuild",
            f"The database of layer «{sid}» will be overwritten!\n\n"
            f"Video: {Path(video_path).name}\n"
            f"DB: {db_path}\n"
            f"Calibration of this layer will be saved.\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Save calibration before regeneration (into THIS layer's file)
        if self.calibration.is_calibrated:
            calib_path = self._get_calibration_save_path()
            if calib_path:
                save_layer_calibration(self.calibration, calib_path, sid)
                logger.info(f"Calibration of layer '{sid}' saved before rebuild: {calib_path}")

        required_frame_ids = {int(anchor.frame_id) for anchor in self.calibration.anchors}
        if required_frame_ids:
            logger.info(
                "Rebuild will preserve exact calibration anchor slots: "
                f"{sorted(required_frame_ids)}"
            )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._start_database_generation(
            video_path,
            db_path,
            required_frame_ids=required_frame_ids,
            source_id=sid,
        )

    # ── Results export ───────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_export_results(self):
        if not hasattr(self, "_tracking_results") or not self._tracking_results:
            QMessageBox.warning(self, "Warning", "No results to export!\n\nPerform tracking first.")
            return

        local_results = any(r.get("coordinate_kind") == "local_planar" for r in self._tracking_results)
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export results",
            "tracking_results",
            "CSV (*.csv)" if local_results else "CSV (*.csv);;GeoJSON (*.geojson);;KML (*.kml)",
        )
        if not path:
            return

        # The destination is operator-chosen, so it may land inside an encrypted
        # copy — where the exported track would sit in plaintext.
        export_root = find_project_root(path)
        if export_root is not None and project_is_encrypted(export_root):
            QMessageBox.critical(
                self,
                "Encrypted project",
                "Export into an encrypted copy is impossible: the mission track would "
                "be stored there in plain text.\n\nChoose a directory outside the project.",
            )
            return

        try:
            if path.endswith(".csv") or "CSV" in selected_filter:
                if not path.endswith(".csv"):
                    path += ".csv"
                ResultExporter.export_csv(self._tracking_results, path)
                if hasattr(self, "_object_tracking_results") and self._object_tracking_results:
                    obj_path = path.replace(".csv", "_objects.csv")
                    ResultExporter.export_objects_csv(self._object_tracking_results, obj_path)
            elif path.endswith(".geojson") or "GeoJSON" in selected_filter:
                if not path.endswith(".geojson"):
                    path += ".geojson"
                ResultExporter.export_geojson(self._tracking_results, path)
                if hasattr(self, "_object_tracking_results") and self._object_tracking_results:
                    obj_path = path.replace(".geojson", "_objects.geojson")
                    ResultExporter.export_objects_geojson(self._object_tracking_results, obj_path)
            elif path.endswith(".kml") or "KML" in selected_filter:
                if not path.endswith(".kml"):
                    path += ".kml"
                name = (
                    self.project_manager.project_name
                    if self.project_manager.is_loaded
                    else "Drone Track"
                )
                ResultExporter.export_kml(self._tracking_results, path, name=name)

            self.status_bar.showMessage(f"Results exported: {path}")
            QMessageBox.information(
                self, "Success", f"Exported {len(self._tracking_results)} points\n\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export error:\n{e}")

    # ── Info panel ──────────────────────────────────────────────────────────────

    def _after_layer_data_changed(self) -> None:
        """Anchors, propagation or a layer's DB changed: refresh every status view."""
        if self.project_manager.is_loaded:
            try:
                self._get_registry().refresh_status(str(self.project_manager.project_dir))
            except Exception as e:
                logger.warning(f"Project registry refresh failed: {e}")
        self._update_project_info_panel()

    def _update_project_info_panel(self):
        """Update the project info panel (data of the ACTIVE layer) and the layer table."""
        if not self.project_manager.is_loaded:
            self.control_panel.update_project_info()
            return

        sid = self._get_current_source_id()
        src = self._source_config(sid)
        num_frames = self.database.get_num_frames() if self.database else None
        num_anchors = len(self.calibration.anchors) if self.calibration else None
        num_propagated = None
        db_size_mb = None

        if self.database and self.database.is_propagated:
            num_propagated = int(self.database.frame_valid.sum())

        if src is not None:
            db_path = str(self.project_manager.project_dir / src.database_file)
            video_path = src.video_path
        else:
            db_path = self.project_manager.database_path
            video_path = self.project_manager.settings.video_path
        if db_path and Path(db_path).exists():
            db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)

        self.control_panel.update_project_info(
            project_name=self.project_manager.project_name,
            video_path=video_path,
            num_frames=num_frames,
            num_anchors=num_anchors,
            num_propagated=num_propagated,
            db_size_mb=db_size_mb,
            layer_id=sid,
        )

        # Update layer table
        self._refresh_sources_panel()

    def _layer_snapshot(self):
        """Live (database, calibration) of every layer + detected cross-layer conflicts."""
        settings = self.project_manager.settings
        project_dir = self.project_manager.project_dir
        active_id = self._get_current_source_id()
        sources = settings.source_configs()
        databases: dict = {}
        calibrations: dict = {}
        for src in sources:
            sid = src.source_id
            db = cal = None
            if self._is_multi_source():
                db = self.db_manager.get_database(sid)
                if src.enabled:
                    try:
                        cal = self.calib_manager.get_or_load(src, project_dir)
                    except Exception as e:
                        logger.warning(f"Layer '{sid}': calibration unreadable: {e}")
            elif sid == active_id:
                db, cal = self.database, self.calibration
            databases[sid] = db
            calibrations[sid] = cal
        num_frames = {sid: db.get_num_frames() for sid, db in databases.items() if db is not None}
        conflicts = find_layer_conflicts(
            sources,
            project_dir,
            {sid: cal for sid, cal in calibrations.items() if cal is not None},
            num_frames,
        )
        return sources, databases, calibrations, conflicts

    def _layer_rows(self) -> list[dict]:
        project_dir = self.project_manager.project_dir
        active_id = self._get_current_source_id()
        sources, databases, calibrations, conflicts = self._layer_snapshot()
        conflict_msgs: dict[str, list[str]] = {}
        for c in conflicts:
            for sid in c.source_ids:
                conflict_msgs.setdefault(sid, []).append(c.message)

        rows = []
        for src in sources:
            sid = src.source_id
            st = compute_layer_status(
                src, project_dir, database=databases.get(sid), calibration=calibrations.get(sid)
            )
            lines = [f"Шар: {sid}" + (f" — {src.description}" if src.description else "")]
            lines.append(f"Зона: {src.area_id} · пріоритет {src.priority}")
            lines.append(f"Відео: {src.video_path}")
            lines.append(f"БД: {src.database_file}")
            lines.append(f"Калібрування: {src.calibration_file}")
            layer = src.scale_layer
            if layer is not None and layer.nominal_gsd_m_per_px:
                lines.append(f"GSD ≈ {layer.nominal_gsd_m_per_px:.3f} м/px ({layer.scale_quality})")
            lines.append(f"Статус: {st.label} — {st.hint}")
            lines.extend(f"⚠ {msg}" for msg in conflict_msgs.get(sid, []))
            rows.append(
                {
                    "source_id": sid,
                    "area_id": src.area_id,
                    "anchors": st.anchors_text,
                    "gps": st.gps_text,
                    "label": st.label,
                    "state": st.state.value,
                    "tooltip": "\n".join(lines),
                    "enabled": st.enabled,
                    "db_loaded": st.db_loaded,
                    "has_db_file": st.has_db_file,
                    "num_anchors": st.num_anchors or 0,
                    "is_active": sid == active_id,
                    "conflict": sid in conflict_msgs,
                }
            )
        return rows

    def _refresh_sources_panel(self):
        """Updates the layer table and the active-layer badge in ControlPanel."""
        if not self.project_manager.is_loaded or not self.project_manager.settings:
            return
        active_id = self._get_current_source_id()
        src = self._source_config(active_id)
        video_path = src.video_path if src is not None else self.project_manager.settings.video_path
        self.control_panel.set_active_source(active_id, video_path or "")

        try:
            rows = self._layer_rows()
        except Exception as e:
            logger.error(f"Layer table refresh failed: {e}", exc_info=True)
            return
        self.control_panel.update_sources_list(rows)
        self.control_panel.set_active_layer_context(active_id, len(rows))

    def _warn_layer_conflicts(self) -> None:
        """Warns (once, on open) if layer files look mixed up. Changes nothing."""
        if not self.project_manager.is_loaded:
            return
        try:
            conflicts = self._layer_snapshot()[3]
        except Exception as e:
            logger.warning(f"Layer conflict check failed: {e}")
            return
        if not conflicts:
            return
        for c in conflicts:
            logger.error(f"Layer conflict ({c.kind}): {c.message}")
        QMessageBox.warning(
            self,
            "Можливе змішування шарів",
            "Схоже, що файли шарів переплутані:\n\n"
            + "\n".join(f"• {c.message}" for c in conflicts)
            + "\n\nАвтоматично нічого не змінено. Перевірте калібрування цих шарів "
            "(ПКМ по шару → Калібрувати) або завантажте правильний JSON "
            "(ПКМ → Завантажити калібрування).",
        )

    # ── Layer slots ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_add_video_source(self):
        """Slot for 'Add layer' button."""
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Помилка", "Спочатку відкрийте або створіть проєкт!")
            return
        if self._refuse_if_encrypted_project("Додавання шару"):
            return
        if self._layer_ops_busy("Додавання шару"):
            return

        from src.gui.dialogs.add_video_source_dialog import AddVideoSourceDialog

        settings = self.project_manager.settings
        project_dir = self.project_manager.project_dir
        existing_areas = sorted(
            {src.get("area_id", "") for src in settings.video_sources or []} - {""}
        )

        dialog = AddVideoSourceDialog(existing_area_ids=existing_areas, parent=self)
        if not dialog.exec():
            return

        new_source = dialog.get_source_config()
        sid = new_source.source_id

        # IDs become folder names; on Windows "Low" and "low" are one folder.
        taken = {str(s.get("source_id", "")).casefold() for s in settings.video_sources or []}
        if sid.casefold() in taken:
            QMessageBox.warning(
                self,
                "Помилка",
                f"Шар з ID '{sid}' вже існує в проєкті (ID не розрізняють регістр)!",
            )
            return
        clashes = [
            c
            for c in find_path_conflicts([*settings.source_configs(), new_source], project_dir)
            if sid in c.source_ids
        ]
        if clashes:
            QMessageBox.warning(
                self,
                "Конфлікт файлів",
                "Новий шар ділив би файли з існуючим:\n\n"
                + "\n".join(f"• {c.message}" for c in clashes),
            )
            return

        settings.add_source(new_source)
        self.project_manager.save_project()
        (project_dir / "sources" / sid).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Layer added: {sid} (area={new_source.area_id}, "
            f"video={Path(new_source.video_path).name})"
        )

        # A second layer switches the project to multi-source mode right away
        # (previously the project had to be reopened before the layer worked).
        if self._wants_multi_source() != self._is_multi_source():
            self._reload_layers_keeping_active()
        self._after_layer_data_changed()

        reply = QMessageBox.question(
            self,
            "Новий шар",
            f"Шар «{sid}» додано до проєкту.\n\n"
            f"Побудувати для нього базу даних зараз?\n"
            f"Відео: {Path(new_source.video_path).name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.on_source_action(sid, "build_db")
        else:
            self.status_bar.showMessage(
                f"Шар «{sid}» додано. Побудуйте його БД через ПКМ по шару в таблиці."
            )

    def _reload_layers_keeping_active(self, exclude: str | None = None) -> None:
        prefer = self.active_source_id if self.active_source_id != exclude else None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._load_project_layers(prefer_source_id=prefer)
        except Exception as e:
            logger.error(f"Reloading project layers failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося перезавантажити шари:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    @pyqtSlot(str)
    def on_active_source_changed(self, source_id: str):
        """Table click: make the clicked layer the active one."""
        if not self.project_manager.is_loaded or source_id == self.active_source_id:
            return
        if self._layer_ops_busy("Зміна активного шару", quiet=True):
            self._refresh_sources_panel()  # put the selection back on the active layer
            return
        if self._activate_source(source_id):
            self.status_bar.showMessage(f"Активний шар: {source_id}")
        else:
            self._refresh_sources_panel()

    @pyqtSlot(str, str)
    def on_source_action(self, source_id: str, action: str):
        """Layer table actions: each one acts on exactly the clicked layer."""
        if not self.project_manager.is_loaded:
            return

        settings = self.project_manager.settings
        source = settings.get_source(source_id)
        if source is None:
            QMessageBox.warning(self, "Помилка", f"Шар '{source_id}' не знайдено!")
            return

        if action == "activate":
            self.on_active_source_changed(source_id)
            return

        if action in ("calibrate", "load_calibration", "propagate", "build_db"):
            if self._layer_ops_busy():
                return
            if action == "build_db" and not source.enabled:
                QMessageBox.information(
                    self, "Шар вимкнено", f"Увімкніть шар «{source_id}» перед побудовою його бази."
                )
                return
            # Activate first: every handler below works on the active layer only.
            if not self._ensure_active(source_id):
                return
            if action == "calibrate":
                self.on_calibrate()
            elif action == "load_calibration":
                self.on_load_calibration()
            elif action == "propagate":
                self.on_run_propagation()
            else:
                self._build_layer_database(source)
            return

        if action == "toggle":
            if self._refuse_if_encrypted_project("Увімкнення/вимкнення шару"):
                return
            if self._layer_ops_busy("Увімкнення/вимкнення шару"):
                return
            source.enabled = not source.enabled
            settings.update_source(source)
            self.project_manager.save_project()

            if self._wants_multi_source() != self._is_multi_source():
                self._reload_layers_keeping_active(exclude=None if source.enabled else source_id)
            elif self._is_multi_source():
                self.db_manager.toggle_source(source)
                if source_id == self.active_source_id:
                    # Its loader was just opened/closed: re-pair DB + calibration.
                    fallback = source_id
                    if not source.enabled:
                        fallback = next(
                            (s for s in self.db_manager.all_source_ids if s != source_id),
                            source_id,
                        )
                    self._activate_source(fallback, quiet=True)

            self._after_layer_data_changed()
            state = "увімкнено" if source.enabled else "вимкнено"
            self.status_bar.showMessage(f"Шар «{source_id}» {state}")
            return

        if action == "remove":
            if self._refuse_if_encrypted_project("Видалення шару"):
                return
            if self._layer_ops_busy("Видалення шару"):
                return
            if len(settings.video_sources or []) <= 1:
                QMessageBox.information(
                    self, "Видалення шару", "Це єдиний шар проєкту — його не можна видалити."
                )
                return
            reply = QMessageBox.question(
                self,
                "Видалення шару",
                f"Видалити шар «{source_id}» з проєкту?\n\n"
                f"Файли бази та калібрування НЕ будуть видалені з диску.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            settings.remove_source(source_id)
            self.project_manager.save_project()
            if self._wants_multi_source() != self._is_multi_source():
                self._reload_layers_keeping_active(exclude=source_id)
            elif self._is_multi_source():
                self.db_manager.unload_source(source_id)
                self.calib_manager.discard(source_id)
                if source_id == self.active_source_id:
                    self.active_source_id = None
                    remaining = [*self.db_manager.all_source_ids]
                    remaining += [s.source_id for s in settings.source_configs()]
                    for sid in remaining:
                        if self._activate_source(sid, quiet=True):
                            break
            self._after_layer_data_changed()
            self.status_bar.showMessage(f"Шар «{source_id}» видалено з проєкту")

    def _build_layer_database(self, source: ProjectVideoSource) -> None:
        """Builds (or, if it exists, rebuilds) the DB of the already active layer."""
        db_path = self.project_manager.project_dir / source.database_file
        if db_path.exists():
            # Same path as the "rebuild" button: confirmation + anchor slots kept.
            self.on_rebuild_database()
            return
        if not source.video_path or not Path(source.video_path).exists():
            QMessageBox.warning(
                self,
                "Відео не знайдено",
                f"Відео шару «{source.source_id}» не знайдено:\n{source.video_path}",
            )
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_database_generation(source.video_path, str(db_path), source_id=source.source_id)
