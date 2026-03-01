from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from src.database.database_loader import DatabaseLoader
from src.workers.database_worker import DatabaseGenerationWorker
from src.gui.dialogs.new_mission_dialog import NewMissionDialog


class DatabaseMixin:

    @pyqtSlot()
    def on_new_mission(self):
        dialog = NewMissionDialog(self)
        if not dialog.exec():
            return
        video_path = dialog.get_mission_data().get('video_path')
        if not video_path:
            QMessageBox.warning(self, "Помилка", "Виберіть еталонне відео.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти базу HDF5", "", "HDF5 Files (*.h5 *.hdf5)"
        )
        if save_path:
            self._start_database_generation(video_path, save_path)
        else:
            self.status_bar.showMessage("Створення місії скасовано")

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
        self.database = DatabaseLoader(db_path)
        self.control_panel.update_progress(100)
        self.control_panel.update_status("Базу успішно створено")
        self.status_bar.showMessage(f"Нова база: {db_path}")
        QMessageBox.information(self, "Успіх", "Базу даних успішно згенеровано!")

    @pyqtSlot(str)
    def on_db_error(self, error_msg: str):
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.control_panel.update_progress(0)
        self.control_panel.update_status("Помилка генерації")
        QMessageBox.critical(self, "Помилка", f"Помилка генерації:\n{error_msg}")

    @pyqtSlot()
    def on_load_database(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть базу HDF5", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if not path:
            self.status_bar.showMessage("Вибір скасовано")
            return
        try:
            self.database = DatabaseLoader(path)
            self.current_database_path = path
            if self.database.is_propagated:
                n_valid = int(self.database.frame_valid.sum())
                n_total = self.database.get_num_frames()
                self.status_bar.showMessage(f"База: {path} (GPS: {n_valid}/{n_total} кадрів)")
            else:
                self.status_bar.showMessage(f"База: {path} (без GPS пропагації)")
            self.control_panel.update_status("База завантажена")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити базу:\n{e}")
