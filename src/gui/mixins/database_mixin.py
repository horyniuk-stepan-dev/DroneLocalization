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
from src.utils.logging_utils import get_logger
from src.workers.database_worker import DatabaseGenerationWorker

logger = get_logger(__name__)


class DatabaseMixin:
    # ── Реєстр проєктів (ініціалізується один раз) ───────────────────────────

    def _get_registry(self) -> ProjectRegistry:
        if not hasattr(self, "_project_registry"):
            self._project_registry = ProjectRegistry()
        return self._project_registry

    # ── Нова місія ────────────────────────────────────────────────────────────

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

        # Створюємо структуру проєкту
        if not self.project_manager.create_project(workspace_dir, mission_data):
            QMessageBox.critical(self, "Помилка", "Не вдалося створити проєкт!")
            return

        # Реєструємо в реєстрі
        self._get_registry().register(
            project_dir=str(self.project_manager.project_dir),
            name=self.project_manager.project_name,
            video_path=video_path,
        )

        self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
        self._start_database_generation(video_path, self.project_manager.database_path)

    # ── Генерація бази ────────────────────────────────────────────────────────

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
        # ВИПРАВЛЕННЯ: НЕ ініціалізуємо WEB_MERCATOR при старті генерації бази.
        # UTM-конвертер буде ініціалізований автоматично після отримання першого
        # GPS-якоря у CalibrationMixin (через _on_first_gps_anchor або еквівалент),
        # щоб забезпечити ізотропний евклідів простір для всієї геометричної математики.
        # WEB_MERCATOR залишається лише як відображальний шар у MapWidget.
        #
        # Якщо якорів ще немає, залишаємо конвертер у стані "not initialized" (UTM, без ref),
        # щоб перший GPS-якір автоматично зафіксував зону UTM.
        if not self.calibration.is_calibrated:
            self.calibration.converter = CoordinateConverter(
                "UTM"
            )  # ref_gps=None → авто при першому якорі

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

        # CRITICAL: Вивантажуємо це джерело і з мульти-менеджера, інакше його
        # retriever триматиме stale handle на vectors.lance, який зараз буде
        # перезаписано (→ "LanceDB query failed: Not found" при трекінгу).
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
            # У мульти-режимі перезавантажуємо джерело всередині db_manager,
            # щоб retriever отримав СВІЖИЙ LanceDB handle (stale handle після
            # перезапису vectors.lance ламає весь vector search).
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

        # Оновити реєстр та інфо-панель
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

    # ── Відкриття проєкту ─────────────────────────────────────────────────────

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

    def _open_project(self, path: str):
        """Завантажити проєкт за шляхом (використовується і для recent menu)."""
        if not self.project_manager.load_project(path):
            QMessageBox.critical(self, "Помилка", "Обрана папка не є валідним проєктом!")
            return

        try:
            db_path = self.project_manager.database_path

            # НОВЕ: Перевірка наявності бази даних
            if not Path(db_path).exists():
                video_path = self.project_manager.settings.video_path
                reply = QMessageBox.question(
                    self,
                    "База даних відсутня",
                    f"Проєкт '{self.project_manager.project_name}' не має згенерованої бази даних.\n\n"
                    f"Згенерувати базу зараз з відео:\n{Path(video_path).name}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.setWindowTitle(
                        f"Drone Topometric Localizer - {self.project_manager.project_name}"
                    )
                    self._start_database_generation(video_path, db_path)
                    return
                else:
                    self.status_bar.showMessage("Завантаження скасовано: відсутня база даних")
                    return

            if self.database:
                self.database.close()
            # Закриваємо попередні мульти-менеджери
            if hasattr(self, "db_manager") and self.db_manager:
                self.db_manager.close_all()

            # Очищення стану попереднього проєкту
            if hasattr(self, "calibration") and self.calibration:
                self.calibration.clear()

            if hasattr(self, "map_widget") and self.map_widget:
                self.map_widget.clear_trajectory()
                self.map_widget.clear_verification_markers()

            if hasattr(self, "_tracking_results"):
                self._tracking_results = []

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                # Перевіряємо чи проєкт мультиджерельний
                sources = self.project_manager.settings.get_enabled_sources()
                is_multi = len(sources) > 1 or any(
                    s.source_id != "main" for s in sources
                )

                if is_multi and len(sources) > 0:
                    # Мультиджерельний режим
                    project_dir = self.project_manager.project_dir
                    self.db_manager = MultiDatabaseManager(
                        sources, project_dir, config=self.config
                    )
                    self.calib_manager = MultiCalibrationManager()
                    self.calib_manager.load_all(sources, project_dir)

                    # self.database — перше джерело для сумісності з UI
                    first_id = self.db_manager.all_source_ids[0] if self.db_manager.all_source_ids else None
                    if first_id:
                        self.database = self.db_manager.get_database(first_id)
                        self.calibration = self.calib_manager.get(first_id)
                    else:
                        raise RuntimeError("Мультиджерельний проєкт: жодна база не завантажена")

                    logger.info(
                        f"Multi-source project loaded: {self.db_manager.num_databases} databases, "
                        f"sources={self.db_manager.all_source_ids}"
                    )
                else:
                    # Single-source режим (зворотна сумісність)
                    self.db_manager = None
                    self.calib_manager = None
                    self.database = DatabaseLoader(db_path)
            finally:
                QApplication.restoreOverrideCursor()
            self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")

            # Оновити реєстр (завжди викликаємо register для збереження нових проєктів)
            registry = self._get_registry()
            registry.register(
                project_dir=str(self.project_manager.project_dir),
                name=self.project_manager.project_name,
                video_path=self.project_manager.settings.video_path
                if self.project_manager.settings
                else "",
            )

            # Завантажити калібрацію якщо є (single-mode)
            if self.calib_manager is None:
                calib_path = self.project_manager.calibration_path
                if calib_path and Path(calib_path).exists():
                    self.calibration.load(calib_path)

            # Bug C: Синхронізація конвертера (пріоритет — БД, потім файл калібрації)
            if self.database and self.database.converter is not None:
                self.calibration.converter = self.database.converter
            elif self.calibration.converter and self.calibration.converter.is_initialized:
                pass  # конвертер вже завантажений з calibration.json

            if self.database.is_propagated:
                n_valid = int(self.database.frame_valid.sum())
                n_total = self.database.get_num_frames()
                self.status_bar.showMessage(
                    f"Проєкт: {self.project_manager.project_name} (GPS: {n_valid}/{n_total} кадрів)"
                )
            else:
                self.status_bar.showMessage(
                    f"Проєкт: {self.project_manager.project_name} (без GPS пропагації)"
                )
            self.control_panel.update_status("Проєкт завантажено")
            self._update_project_info_panel()

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити базу проєкту:\n{e}")

    # ── Перевірка пропагації ─────────────────────────────────────────────────

    @pyqtSlot()
    def on_verify_propagation(self):
        if not self.database or not self.database.is_propagated:
            QMessageBox.warning(
                self, "Увага", "Дані пропагації відсутні або проєкт не завантажено!"
            )
            return

        num_frames = self.database.get_num_frames()
        frame_valid = self.database.frame_valid
        frame_affine = self.database.frame_affine

        # Отримуємо розміри кадру з метаданих
        width = self.database.metadata.get("frame_width", 1920)
        height = self.database.metadata.get("frame_height", 1080)

        # Центр кадру в пікселях
        center_px = np.array([[width / 2, height / 2]], dtype=np.float32)

        points_to_show = []

        # Збираємо тільки валідні кадри (з кроком 5 для продуктивності на карті)
        step = max(1, num_frames // 200)  # Максимум ~200 точок щоб не гальмував біндер

        for i in range(0, num_frames, step):
            if frame_valid[i]:
                # Приміняємо афінну матрицю (2x3)
                M = frame_affine[i]
                # Metric = M * [x, y, 1]^T
                metric_x = M[0, 0] * center_px[0, 0] + M[0, 1] * center_px[0, 1] + M[0, 2]
                metric_y = M[1, 0] * center_px[0, 0] + M[1, 1] * center_px[0, 1] + M[1, 2]

                lat, lon = self.calibration.converter.metric_to_gps(
                    float(metric_x), float(metric_y)
                )
                points_to_show.append({"lat": float(lat), "lon": float(lon), "label": str(i)})

        if not points_to_show:
            QMessageBox.information(
                self, "Інформація", "Не знайдено жодного кадру з валідними координатами."
            )
            return

        self.map_widget.show_verification_markers(points_to_show)
        self.status_bar.showMessage(f"Відображено {len(points_to_show)} точок перевірки на карті.")

    # ── Перегенерація бази ────────────────────────────────────────────────────

    @pyqtSlot()
    def on_rebuild_database(self):
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте проєкт!")
            return

        video_path = self.project_manager.settings.video_path
        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(
                self,
                "Увага",
                f"Відео проєкту не знайдено:\n{video_path}\n\n"
                "Перевірте шлях до відео у налаштуваннях проєкту.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Перегенерація бази",
            f"Базу даних буде перезаписано!\n\n"
            f"Відео: {Path(video_path).name}\n"
            f"Калібрація буде збережена.\n\n"
            f"Продовжити?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Зберігаємо калібрацію перед перегенерацією
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

    # ── Експорт результатів ───────────────────────────────────────────────────

    @pyqtSlot()
    def on_export_results(self):
        if not hasattr(self, "_tracking_results") or not self._tracking_results:
            QMessageBox.warning(
                self, "Увага", "Немає результатів для експорту!\n\nСпочатку виконайте відстеження."
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Експорт результатів",
            "tracking_results",
            "CSV (*.csv);;GeoJSON (*.geojson);;KML (*.kml)",
        )
        if not path:
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

            self.status_bar.showMessage(f"Результати експортовано: {path}")
            QMessageBox.information(
                self, "Успіх", f"Експортовано {len(self._tracking_results)} точок\n\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Помилка експорту:\n{e}")

    # ── Інфо-панель ───────────────────────────────────────────────────────────

    def _update_project_info_panel(self):
        """Оновити інформаційну панель проєкту у control_panel."""
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

        # Оновлюємо панель відеоджерел
        self._refresh_sources_panel()

    def _refresh_sources_panel(self):
        """Оновлює таблицю відеоджерел та бейдж активного джерела у ControlPanel."""
        if not self.project_manager.is_loaded or not self.project_manager.settings:
            return
        sources_raw = self.project_manager.settings.video_sources or []
        project_dir = str(self.project_manager.project_dir) if self.project_manager.project_dir else ""

        # Визначаємо активне джерело
        active_id = self._get_current_source_id()

        # Отримуємо video_path для активного джерела
        video_path = ""
        for src_dict in sources_raw:
            if src_dict.get("source_id") == active_id:
                video_path = src_dict.get("video_path", "")
                break

        # Якщо в settings не знайдено, fallback на загальний video_path
        if not video_path and self.project_manager.settings:
            video_path = self.project_manager.settings.video_path

        self.control_panel.set_active_source(active_id, video_path or "")

        # Визначаємо які джерела вже мають пропагацію в HDF5
        # (калібрування може бути вбудоване в HDF5 без окремого calibration.json)
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

    # ── Мультиджерельні слоти ─────────────────────────────────────────────────

    @pyqtSlot()
    def on_add_video_source(self):
        """Слот для кнопки 'Додати джерело'."""
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Помилка", "Спочатку відкрийте або створіть проєкт!")
            return

        from src.gui.dialogs.add_video_source_dialog import AddVideoSourceDialog

        # Збираємо існуючі area_id
        existing_areas = set()
        for src in (self.project_manager.settings.video_sources or []):
            area = src.get("area_id", "")
            if area:
                existing_areas.add(area)

        dialog = AddVideoSourceDialog(
            existing_area_ids=sorted(existing_areas), parent=self
        )
        if not dialog.exec():
            return

        new_source = dialog.get_source_config()

        # Перевірка на дублікат
        if self.project_manager.settings.get_source(new_source.source_id) is not None:
            QMessageBox.warning(
                self, "Помилка",
                f"Джерело з ID '{new_source.source_id}' вже існує в проєкті!"
            )
            return

        # Додаємо до проєкту
        self.project_manager.settings.add_source(new_source)
        self.project_manager.save_project()

        # Створюємо директорію для джерела
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
            # Джерело вимкнено або не має БД
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
            # Генерація БД для конкретного джерела
            video_path = source.video_path
            db_path = str(self.project_manager.project_dir / source.database_file)
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self._start_database_generation(video_path, db_path)

        elif action == "calibrate":
            # Поки що — відкриваємо стандартний калібрувальний діалог
            self.status_bar.showMessage(
                f"Для калібрування '{source_id}' використовуйте стандартний калібрувальний інструмент."
            )

        elif action == "toggle":
            source.enabled = not source.enabled
            settings.update_source(source)
            self.project_manager.save_project()

            if hasattr(self, "db_manager") and self.db_manager:
                self.db_manager.toggle_source(source)

                # Якщо вимкнули поточне джерело — перемикаємось на перше доступне
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

