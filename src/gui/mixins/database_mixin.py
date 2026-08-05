from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.calibration.multi_calibration_manager import MultiCalibrationManager
from src.core.export_results import ResultExporter
from src.core.project_registry import ProjectRegistry
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

        # Create project directory structure
        if not self.project_manager.create_project(workspace_dir, mission_data):
            QMessageBox.critical(self, "Помилка", "Не вдалося створити проєкт!")
            return

        # Register in the project registry
        self._get_registry().register(
            project_dir=str(self.project_manager.project_dir),
            name=self.project_manager.project_name,
            video_path=video_path,
        )

        self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
        self._start_database_generation(video_path, self.project_manager.database_path)

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

    def _start_database_generation(self, video_path: str, save_path: str):
        if self._refuse_if_encrypted_project("Генерація бази даних"):
            return

        # Do NOT initialize WEB_MERCATOR when starting database generation.
        # UTM converter will be initialized automatically after first GPS anchor.
        if not self.calibration.is_calibrated:
            self.calibration.converter = CoordinateConverter(
                "UTM"
            )  # ref_gps=None → auto on first anchor

        self.control_panel.btn_new_mission.setEnabled(False)
        self.control_panel.btn_load_db.setEnabled(False)
        self.control_panel.update_progress(0)
        self.control_panel.set_db_generation_running(True)

        # CRITICAL: Close and release the database file handle before overwriting/truncating it
        if hasattr(self, "database") and self.database:
            try:
                self.database.close()
                logger.info("Current database closed before starting new generation.")
            except Exception as e:
                logger.warning(f"Could not close database: {e}")
        self.database = None

        # Unload source from multi-manager before overwriting vectors.lance
        if getattr(self, "db_manager", None):
            sid = self._find_source_id_by_db_path(save_path)
            if sid:
                self.db_manager.unload_source(sid)

        self.db_worker = DatabaseGenerationWorker(
            video_path=video_path,
            output_path=save_path,
            model_manager=self.model_manager,
            config=self.config,
            project_manager=self.project_manager,
        )
        self.db_worker.progress.connect(self.on_db_progress)
        self.db_worker.completed.connect(self.on_db_completed)
        self.db_worker.error.connect(self.on_db_error)
        self.db_worker.cancelled.connect(self.on_db_cancelled)

        # Connect stop button
        self.control_panel.stop_db_generation_clicked.connect(self.on_stop_db_generation)

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

        if self.database:
            self.database.close()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Reload source in db_manager for fresh LanceDB handle
            reloaded = False
            if getattr(self, "db_manager", None):
                sid = self._find_source_id_by_db_path(db_path)
                src = (
                    self.project_manager.settings.get_source(sid)
                    if sid and self.project_manager.settings
                    else None
                )
                if src is not None and self.db_manager.reload_source(src):
                    self.database = self.db_manager.get_database(sid)
                    reloaded = True

            if not reloaded:
                self.database = DatabaseLoader(db_path)
        finally:
            QApplication.restoreOverrideCursor()
        self.control_panel.update_progress(100)
        self.control_panel.update_status("Базу успішно створено")
        self.status_bar.showMessage(
            f"Проєкт: {self.project_manager.project_name} | База: {db_path}"
        )

        # Update registry and info panel
        if self.project_manager.is_loaded:
            self._get_registry().refresh_status(str(self.project_manager.project_dir))
        self._update_project_info_panel()

        QMessageBox.information(self, "Успіх", "Проєкт та базу даних успішно згенеровано!")

    @pyqtSlot(str)
    def on_db_error(self, error_msg: str):
        self.control_panel.set_db_generation_running(False)
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.control_panel.update_progress(0)
        self.control_panel.update_status("Помилка генерації")
        QMessageBox.critical(self, "Помилка", f"Помилка генерації:\n{error_msg}")

    @pyqtSlot()
    def on_db_cancelled(self):
        self.control_panel.set_db_generation_running(False)
        self.control_panel.update_status("Генерацію скасовано користувачем")
        self.control_panel.update_progress(0)

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

        try:
            db_path = self.project_manager.database_path

            # Check whether the database file exists
            if not Path(db_path).exists():
                video_path = self.project_manager.settings.video_path
                reply = QMessageBox.question(
                    self,
                    "Database missing",
                    f"Project '{self.project_manager.project_name}' has no generated database.\n\n"
                    f"Generate database now from video:\n{Path(video_path).name}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.setWindowTitle(
                        f"Drone Topometric Localizer - {self.project_manager.project_name}"
                    )
                    self._start_database_generation(video_path, db_path)
                    return
                else:
                    self.status_bar.showMessage("Loading cancelled: missing database")
                    return

            if self.database:
                self.database.close()
            # Shut down previous multi-source managers
            if hasattr(self, "db_manager") and self.db_manager:
                self.db_manager.close_all()

            # Clear previous project state
            if hasattr(self, "calibration") and self.calibration:
                self.calibration.clear()

            if hasattr(self, "map_widget") and self.map_widget:
                self.map_widget.clear_trajectory()
                self.map_widget.clear_verification_markers()

            if hasattr(self, "_tracking_results"):
                self._tracking_results = []

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            sources = self.project_manager.settings.get_enabled_sources()
            is_multi = len(sources) > 1 or any(s.source_id != "main" for s in sources)
            if is_multi and len(sources) > 0:
                # Multi-source mode
                project_dir = self.project_manager.project_dir
                self.db_manager = MultiDatabaseManager(sources, project_dir, config=self.config)
                self.calib_manager = MultiCalibrationManager()
                self.calib_manager.load_all(sources, project_dir)

                # self.database — first source for UI compatibility
                first_id = (
                    self.db_manager.all_source_ids[0] if self.db_manager.all_source_ids else None
                )
                if first_id:
                    self.database = self.db_manager.get_database(first_id)
                    self.calibration = self.calib_manager.get(first_id)
                else:
                    raise RuntimeError("Multi-source project: no databases loaded")

                logger.info(
                    f"Multi-source project loaded: {self.db_manager.num_databases} databases, "
                    f"sources={self.db_manager.all_source_ids}"
                )
            else:
                # Single-source mode (backwards compatibility)
                self.db_manager = None
                self.calib_manager = None
                self.database = DatabaseLoader(db_path)

            self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")

            # Update registry
            registry = self._get_registry()
            registry.register(
                project_dir=str(self.project_manager.project_dir),
                name=self.project_manager.project_name,
                video_path=self.project_manager.settings.video_path
                if self.project_manager.settings
                else "",
            )

            # Load calibration if present (single mode)
            if self.calib_manager is None:
                calib_path = self.project_manager.calibration_path
                if calib_path and Path(calib_path).exists():
                    self.calibration.load(calib_path)

            # Sync converter (DB priority, then calibration file)
            if self.database and self.database.converter is not None:
                self.calibration.converter = self.database.converter
            elif self.calibration.converter and self.calibration.converter.is_initialized:
                pass  # converter loaded from calibration.json

            if self.database and self.database.is_propagated:
                n_valid = int(self.database.frame_valid.sum())
                n_total = self.database.get_num_frames()
                self.status_bar.showMessage(
                    f"Project: {self.project_manager.project_name} (GPS: {n_valid}/{n_total} frames)"
                )
            else:
                self.status_bar.showMessage(
                    f"Project: {self.project_manager.project_name} (no GPS propagation)"
                )
            self.control_panel.update_status("Project loaded")
            self._update_project_info_panel()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project database:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

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
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Warning", "Please load the project first!")
            return

        # Before the confirmation prompt AND before the calibration save below —
        # that save is a write into the project and would otherwise raise.
        if self._refuse_if_encrypted_project("Database rebuild"):
            return

        video_path = self.project_manager.settings.video_path
        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(
                self,
                "Warning",
                f"Project video not found:\n{video_path}\n\n"
                "Check the video path in project settings.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Database rebuild",
            f"The database will be overwritten!\n\n"
            f"Video: {Path(video_path).name}\n"
            f"Calibration will be saved.\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Save calibration before regeneration
        if self.calibration.is_calibrated:
            calib_path = (
                self._get_calibration_save_path()
                if hasattr(self, "_get_calibration_save_path")
                else self.project_manager.calibration_path
            )
            if calib_path:
                self.calibration.save(calib_path)
                logger.info(f"Calibration saved before rebuild: {calib_path}")

        self._start_database_generation(video_path, self.project_manager.database_path)

    # ── Results export ───────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_export_results(self):
        if not hasattr(self, "_tracking_results") or not self._tracking_results:
            QMessageBox.warning(self, "Warning", "No results to export!\n\nPerform tracking first.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export results",
            "tracking_results",
            "CSV (*.csv);;GeoJSON (*.geojson);;KML (*.kml)",
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

    def _update_project_info_panel(self):
        """Update the project info panel in control_panel."""
        if not self.project_manager.is_loaded:
            self.control_panel.update_project_info()
            return

        num_frames = self.database.get_num_frames() if self.database else None
        num_anchors = len(self.calibration.anchors) if self.calibration else None
        num_propagated = None
        db_size_mb = None

        if self.database and self.database.is_propagated:
            num_propagated = int(self.database.frame_valid.sum())

        db_path = self.project_manager.database_path
        if db_path and Path(db_path).exists():
            db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)

        self.control_panel.update_project_info(
            project_name=self.project_manager.project_name,
            video_path=self.project_manager.settings.video_path
            if self.project_manager.settings
            else None,
            num_frames=num_frames,
            num_anchors=num_anchors,
            num_propagated=num_propagated,
            db_size_mb=db_size_mb,
        )

        # Update video sources panel
        self._refresh_sources_panel()

    def _refresh_sources_panel(self):
        """Updates video sources table and active source badge in ControlPanel."""
        if not self.project_manager.is_loaded or not self.project_manager.settings:
            return
        sources_raw = self.project_manager.settings.video_sources or []
        project_dir = (
            str(self.project_manager.project_dir) if self.project_manager.project_dir else ""
        )

        # Get active source ID
        active_id = self._get_current_source_id()

        # Get video_path for active source
        video_path = ""
        for src_dict in sources_raw:
            if src_dict.get("source_id") == active_id:
                video_path = src_dict.get("video_path", "")
                break

        # Fallback to default video_path if missing
        if not video_path and self.project_manager.settings:
            video_path = self.project_manager.settings.video_path

        self.control_panel.set_active_source(active_id, video_path or "")

        # Check which sources are propagated
        propagated_ids: set[str] = set()
        if hasattr(self, "db_manager") and self.db_manager:
            for sid in self.db_manager.all_source_ids:
                db = self.db_manager.get_database(sid)
                if db and db.is_propagated:
                    propagated_ids.add(sid)
        elif hasattr(self, "database") and self.database and self.database.is_propagated:
            propagated_ids.add("main")

        self.control_panel.update_sources_list(
            sources_raw,
            project_dir=project_dir,
            active_source_id=active_id,
            propagated_source_ids=propagated_ids,
        )

    # ── Multi-source slots ────────────────────────────────────────────────────

    @pyqtSlot()
    def on_add_video_source(self):
        """Slot for 'Add Source' button."""
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Помилка", "Спочатку відкрийте або створіть проєкт!")
            return

        from src.gui.dialogs.add_video_source_dialog import AddVideoSourceDialog

        # Collect existing area_ids
        existing_areas = set()
        for src in self.project_manager.settings.video_sources or []:
            area = src.get("area_id", "")
            if area:
                existing_areas.add(area)

        dialog = AddVideoSourceDialog(existing_area_ids=sorted(existing_areas), parent=self)
        if not dialog.exec():
            return

        new_source = dialog.get_source_config()

        # Duplicate check
        if self.project_manager.settings.get_source(new_source.source_id) is not None:
            QMessageBox.warning(
                self, "Помилка", f"Джерело з ID '{new_source.source_id}' вже існує в проєкті!"
            )
            return

        # Add to project
        self.project_manager.settings.add_source(new_source)
        self.project_manager.save_project()

        # Create directory for this source
        source_dir = self.project_manager.project_dir / "sources" / new_source.source_id
        source_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Video source added: {new_source.source_id} "
            f"(area={new_source.area_id}, video={Path(new_source.video_path).name})"
        )

        self._refresh_sources_panel()
        self.status_bar.showMessage(
            f"Додано відеоджерело '{new_source.source_id}'. "
            f"Побудуйте БД через контекстне меню таблиці."
        )

    @pyqtSlot(str)
    def on_active_source_changed(self, source_id: str):
        """Обробка зміни активного джерела при кліку в таблиці."""
        if not self.db_manager:
            return

        if source_id not in self.db_manager.all_source_ids:
            # Source is disabled or has no database
            self.database = None
            self.calibration = None
            self.status_bar.showMessage(f"Джерело '{source_id}' вимкнено або недоступне")
            self._update_project_info_panel()
            return

        self.database = self.db_manager.get_database(source_id)
        self.calibration = self.calib_manager.get(source_id)
        logger.info(f"Active source switched to: {source_id}")

        self.status_bar.showMessage(f"Обрано джерело: {source_id}")
        self._update_project_info_panel()
        self._refresh_sources_panel()  # Щоб оновити підсвічування рядка в таблиці

    @pyqtSlot(str, str)
    def on_source_action(self, source_id: str, action: str):
        """Обробка дій з контекстного меню таблиці джерел."""
        if not self.project_manager.is_loaded:
            return

        settings = self.project_manager.settings
        source = settings.get_source(source_id)
        if source is None:
            QMessageBox.warning(self, "Помилка", f"Джерело '{source_id}' не знайдено!")
            return

        if action == "build_db":
            # Generate database for this specific source
            video_path = source.video_path
            db_path = str(self.project_manager.project_dir / source.database_file)
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self._start_database_generation(video_path, db_path)

        elif action == "calibrate":
            # For now: open the standard calibration dialog
            self.status_bar.showMessage(
                f"Для калібрування '{source_id}' використовуйте стандартний калібрувальний інструмент."
            )

        elif action == "toggle":
            source.enabled = not source.enabled
            settings.update_source(source)
            self.project_manager.save_project()

            if hasattr(self, "db_manager") and self.db_manager:
                self.db_manager.toggle_source(source)

                # If the currently active source was disabled, switch to the first available one
                if not source.enabled and self._get_current_source_id() == source_id:
                    avail = self.db_manager.all_source_ids
                    if avail:
                        self.on_active_source_changed(avail[0])
                    else:
                        self.database = None
                        self.calibration = None
                        self._update_project_info_panel()

            self._refresh_sources_panel()
            state = "увімкнено" if source.enabled else "вимкнено"
            self.status_bar.showMessage(f"Джерело '{source_id}' {state}")

        elif action == "remove":
            reply = QMessageBox.question(
                self,
                "Видалення джерела",
                f"Видалити відеоджерело '{source_id}'?\n\n"
                f"Файли бази та калібрації НЕ будуть видалені з диску.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                settings.remove_source(source_id)
                self.project_manager.save_project()
                self._refresh_sources_panel()
                self.status_bar.showMessage(f"Джерело '{source_id}' видалено з проєкту")
