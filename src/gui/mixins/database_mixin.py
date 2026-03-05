from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from src.database.database_loader import DatabaseLoader
from src.workers.database_worker import DatabaseGenerationWorker
from src.gui.dialogs.new_mission_dialog import NewMissionDialog


class DatabaseMixin:

    @pyqtSlot()
    def on_new_mission(self):
        # Отримуємо дані про нову місію з діалогу
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
            
        self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
        self._start_database_generation(video_path, self.project_manager.database_path)

    def _start_database_generation(self, video_path: str, save_path: str):
        self.control_panel.btn_new_mission.setEnabled(False)
        self.control_panel.btn_load_db.setEnabled(False)
        self.control_panel.update_progress(0)

        self.db_worker = DatabaseGenerationWorker(
            video_path=video_path,
            output_path=save_path,
            model_manager=self.model_manager,
            config=self.config,     # ← завжди self.config, не APP_CONFIG напряму
        )
        self.db_worker.progress.connect(self.on_db_progress)
        self.db_worker.completed.connect(self.on_db_completed)
        self.db_worker.error.connect(self.on_db_error)
        self.db_worker.start()

    @pyqtSlot(int, str)
    def on_db_progress(self, percent: int, message: str):
        self.control_panel.update_progress(percent)
        self.control_panel.update_status(message)

    @pyqtSlot(str)
    def on_db_completed(self, db_path: str):
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.current_database_path = db_path
        # Закриваємо попередню базу щоб звільнити HDF5 handle
        if self.database:
            self.database.close()
        self.database = DatabaseLoader(db_path)
        self.control_panel.update_progress(100)
        self.control_panel.update_status("Базу успішно створено")
        self.status_bar.showMessage(f"Проєкт: {self.project_manager.project_name} | База: {db_path}")
        QMessageBox.information(self, "Успіх", "Проєкт та базу даних успішно згенеровано!")

    @pyqtSlot(str)
    def on_db_error(self, error_msg: str):
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.control_panel.update_progress(0)
        self.control_panel.update_status("Помилка генерації")
        QMessageBox.critical(self, "Помилка", f"Помилка генерації:\n{error_msg}")

    @pyqtSlot()
    def on_load_database(self):
        path = QFileDialog.getExistingDirectory(
            self, "Виберіть папку проєкту", ""
        )
        if not path:
            self.status_bar.showMessage("Вибір проєкту скасовано")
            return
            
        if not self.project_manager.load_project(path):
            QMessageBox.critical(self, "Помилка", "Обрана папка не є валідним проєктом!")
            return
            
        try:
            db_path = self.project_manager.database_path
            
            # Закриваємо попередню базу щоб звільнити HDF5 handle
            if self.database:
                self.database.close()
                
            self.database = DatabaseLoader(db_path)
            self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
            
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
            
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити базу проєкту:\n{e}")
