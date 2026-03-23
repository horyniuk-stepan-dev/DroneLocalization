from pathlib import Path

from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication

from src.database.database_loader import DatabaseLoader
from src.workers.database_worker import DatabaseGenerationWorker
from src.gui.dialogs.new_mission_dialog import NewMissionDialog
from src.gui.dialogs.open_project_dialog import OpenProjectDialog
from src.core.project_registry import ProjectRegistry
from src.core.export_results import ResultExporter


from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseMixin:

    # ── Реєстр проєктів (ініціалізується один раз) ───────────────────────────

    def _get_registry(self) -> ProjectRegistry:
        if not hasattr(self, '_project_registry'):
            self._project_registry = ProjectRegistry()
        return self._project_registry

    # ── Нова місія ────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_new_mission(self):
        dialog = NewMissionDialog(self)
        if not dialog.exec():
            return

        mission_data = dialog.get_mission_data()
        workspace_dir = mission_data.get('workspace_dir')
        video_path = mission_data.get('video_path')

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

    def _start_database_generation(self, video_path: str, save_path: str):
        from src.geometry.coordinates import CoordinateConverter
        CoordinateConverter.reset()
        
        self.control_panel.btn_new_mission.setEnabled(False)
        self.control_panel.btn_load_db.setEnabled(False)
        self.control_panel.update_progress(0)
        self.control_panel.set_db_generation_running(True)

        # CRITICAL: Close and release the database file handle before overwriting/truncating it
        if hasattr(self, 'database') and self.database:
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
        if hasattr(self, 'db_worker') and self.db_worker and self.db_worker.isRunning():
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
            self.database = DatabaseLoader(db_path)
        finally:
            QApplication.restoreOverrideCursor()
        self.control_panel.update_progress(100)
        self.control_panel.update_status("Базу успішно створено")
        self.status_bar.showMessage(f"Проєкт: {self.project_manager.project_name} | База: {db_path}")

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

        from src.geometry.coordinates import CoordinateConverter
        CoordinateConverter.reset()

        try:
            db_path = self.project_manager.database_path
            
            # НОВЕ: Перевірка наявності бази даних
            if not Path(db_path).exists():
                video_path = self.project_manager.settings.video_path
                reply = QMessageBox.question(
                    self, "База даних відсутня",
                    f"Проєкт '{self.project_manager.project_name}' не має згенерованої бази даних.\n\n"
                    f"Згенерувати базу зараз з відео:\n{Path(video_path).name}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
                    self._start_database_generation(video_path, db_path)
                    return
                else:
                    self.status_bar.showMessage("Завантаження скасовано: відсутня база даних")
                    return

            if self.database:
                self.database.close()

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.database = DatabaseLoader(db_path)
            finally:
                QApplication.restoreOverrideCursor()
            self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")

            # Оновити реєстр (завжди викликаємо register для збереження нових проєктів)
            registry = self._get_registry()
            registry.register(
                project_dir=str(self.project_manager.project_dir),
                name=self.project_manager.project_name,
                video_path=self.project_manager.settings.video_path if self.project_manager.settings else ""
            )

            # Завантажити калібрацію якщо є
            calib_path = self.project_manager.calibration_path
            if calib_path and Path(calib_path).exists():
                self.calibration.load(calib_path)

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
            QMessageBox.warning(self, "Увага", "Дані пропагації відсутні або проєкт не завантажено!")
            return

        from src.geometry.coordinates import CoordinateConverter
        import numpy as np

        num_frames = self.database.get_num_frames()
        frame_valid = self.database.frame_valid
        frame_affine = self.database.frame_affine
        
        # Отримуємо розміри кадру з метаданих
        width = self.database.metadata.get('frame_width', 1920)
        height = self.database.metadata.get('frame_height', 1080)
        
        # Центр кадру в пікселях
        center_px = np.array([[width / 2, height / 2]], dtype=np.float32)
        
        points_to_show = []
        
        # Збираємо тільки валідні кадри (з кроком 5 для продуктивності на карті)
        step = max(1, num_frames // 200) # Максимум ~200 точок щоб не гальмував біндер
        
        for i in range(0, num_frames, step):
            if frame_valid[i]:
                # Приміняємо афінну матрицю (2x3)
                M = frame_affine[i]
                # Metric = M * [x, y, 1]^T
                metric_x = M[0, 0] * center_px[0, 0] + M[0, 1] * center_px[0, 1] + M[0, 2]
                metric_y = M[1, 0] * center_px[0, 0] + M[1, 1] * center_px[0, 1] + M[1, 2]
                
                lat, lon = CoordinateConverter.metric_to_gps(metric_x, metric_y)
                points_to_show.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'label': str(i)
                })

        if not points_to_show:
            QMessageBox.information(self, "Інформація", "Не знайдено жодного кадру з валідними координатами.")
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
                self, "Увага",
                f"Відео проєкту не знайдено:\n{video_path}\n\n"
                "Перевірте шлях до відео у налаштуваннях проєкту."
            )
            return

        reply = QMessageBox.question(
            self, "Перегенерація бази",
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
            calib_path = self.project_manager.calibration_path
            if calib_path:
                self.calibration.save(calib_path)
                logger.info(f"Calibration saved before rebuild: {calib_path}")

        self._start_database_generation(video_path, self.project_manager.database_path)

    # ── Експорт результатів ───────────────────────────────────────────────────

    @pyqtSlot()
    def on_export_results(self):
        if not hasattr(self, '_tracking_results') or not self._tracking_results:
            QMessageBox.warning(self, "Увага", "Немає результатів для експорту!\n\n"
                                "Спочатку виконайте відстеження.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Експорт результатів", "tracking_results",
            "CSV (*.csv);;GeoJSON (*.geojson);;KML (*.kml)",
        )
        if not path:
            return

        try:
            if path.endswith('.csv') or 'CSV' in selected_filter:
                if not path.endswith('.csv'):
                    path += '.csv'
                ResultExporter.export_csv(self._tracking_results, path)
            elif path.endswith('.geojson') or 'GeoJSON' in selected_filter:
                if not path.endswith('.geojson'):
                    path += '.geojson'
                ResultExporter.export_geojson(self._tracking_results, path)
            elif path.endswith('.kml') or 'KML' in selected_filter:
                if not path.endswith('.kml'):
                    path += '.kml'
                name = self.project_manager.project_name if self.project_manager.is_loaded else "Drone Track"
                ResultExporter.export_kml(self._tracking_results, path, name=name)

            self.status_bar.showMessage(f"Результати експортовано: {path}")
            QMessageBox.information(self, "Успіх",
                                    f"Експортовано {len(self._tracking_results)} точок\n\n{path}")
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
            video_path=self.project_manager.settings.video_path if self.project_manager.settings else None,
            num_frames=num_frames,
            num_anchors=num_anchors,
            num_propagated=num_propagated,
            db_size_mb=db_size_mb,
        )
