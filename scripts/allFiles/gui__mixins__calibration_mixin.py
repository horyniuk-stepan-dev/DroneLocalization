import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog
from PyQt6.QtCore import Qt

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FeatureMatcher
from src.workers.calibration_propagation_worker import CalibrationPropagationWorker
from src.gui.dialogs.calibration_dialog import CalibrationDialog


class CalibrationMixin:

    # ── Calibration dialog ───────────────────────────────────────────────────

    @pyqtSlot()
    def on_calibrate(self):
        if not self.database or self.database.db_file is None:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте або створіть базу даних!")
            return

        existing_ids = [a.frame_id for a in self.calibration.anchors]
        dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=existing_ids,
            parent=self,
        )
        dialog.anchor_added.connect(self.on_anchor_added)
        dialog.calibration_complete.connect(self.on_run_propagation)
        dialog.exec()

    @pyqtSlot(object)
    def on_anchor_added(self, anchor_data: dict):
        try:
            points_2d  = anchor_data.get('points_2d')
            points_gps = anchor_data.get('points_gps')
            frame_id   = anchor_data.get('calib_frame_id')

            if not points_2d or not points_gps or len(points_2d) < 3:
                QMessageBox.warning(self, "Помилка", "Потрібно мінімум 3 точки для якоря!")
                return

            pts_2d_np = np.array(points_2d, dtype=np.float32)
            pts_metric = [CoordinateConverter.gps_to_metric(lat, lon) for lat, lon in points_gps]
            pts_metric_np = np.array(pts_metric, dtype=np.float32)

            M, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
            if M is None:
                QMessageBox.critical(self, "Помилка",
                    "Не вдалося обчислити матрицю. Спробуйте розставити точки ширше!")
                return

            self.calibration.add_anchor(frame_id=frame_id, affine_matrix=M)

            if self.database and self.database.db_path:
                calib_path = self.database.db_path.replace('.h5', '_calib.json')
                self.calibration.save(calib_path)

            self.status_bar.showMessage(f"Якір для кадру {frame_id} успішно створено!")

        except Exception as e:
            self.logger.error(f"Failed to add anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося додати якір:\n{e}")

    # ── Propagation ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_run_propagation(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Додайте хоча б один якір калібрування!")
            return
        if not self.database:
            QMessageBox.warning(self, "Увага", "База даних не завантажена!")
            return

        try:
            lg_model = self.model_manager.load_lightglue()
            matcher  = FeatureMatcher(lg_model, self.model_manager.device)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити LightGlue:\n{e}")
            return

        anchor_ids = [a.frame_id for a in self.calibration.anchors]
        n_frames   = self.database.get_num_frames()
        self.logger.info(f"Propagation: {len(anchor_ids)} anchors {anchor_ids}, {n_frames} frames")

        self._propagation_dialog = QProgressDialog(
            f"Пропагація GPS від {len(anchor_ids)} якорів на {n_frames} кадрів...",
            "Скасувати", 0, 100, self,
        )
        self._propagation_dialog.setWindowTitle("Розповсюдження GPS координат")
        self._propagation_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._propagation_dialog.setMinimumDuration(0)
        self._propagation_dialog.setValue(0)

        self.propagation_worker = CalibrationPropagationWorker(
            database=self.database,
            calibration=self.calibration,
            matcher=matcher,
            config=self.config,
        )
        self.propagation_worker.progress.connect(self.on_propagation_progress)
        self.propagation_worker.completed.connect(self.on_propagation_completed)
        self.propagation_worker.error.connect(self.on_propagation_error)
        self._propagation_dialog.canceled.connect(self.propagation_worker.stop)
        self.propagation_worker.start()

    @pyqtSlot(int, str)
    def on_propagation_progress(self, percent: int, message: str):
        dialog = self._propagation_dialog
        if dialog is not None:
            try:
                dialog.setLabelText(message)
                dialog.setValue(percent)
            except Exception:
                pass
        self.status_bar.showMessage(message)

    @pyqtSlot()
    def on_propagation_completed(self):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None

        n_valid = int(self.database.frame_valid.sum()) if self.database.frame_valid is not None else 0
        n_total = self.database.get_num_frames()
        self.status_bar.showMessage(f"✅ Пропагація: {n_valid}/{n_total} кадрів")

        reply = QMessageBox.question(
            self, "Пропагація завершена",
            f"GPS розповсюджено на {n_valid}/{n_total} кадрів!\n\nЗберегти файл калібрування?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.on_save_calibration()

        if self.tracking_worker and self.tracking_worker.isRunning():
            if hasattr(self.tracking_worker, 'localizer'):
                self.tracking_worker.localizer.reset_cache()

    @pyqtSlot(str)
    def on_propagation_error(self, error_msg: str):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None
        self.logger.error(f"Propagation error: {error_msg}")
        QMessageBox.critical(self, "Помилка пропагації", error_msg)

    # ── Save / Load calibration ──────────────────────────────────────────────

    @pyqtSlot()
    def on_save_calibration(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Немає даних для збереження.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти калібрування", "calibration.json", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self.calibration.save(path)
            n = len(self.calibration.anchors)
            self.status_bar.showMessage(f"Калібрування збережено: {path} ({n} якорів)")
            QMessageBox.information(self, "Збережено",
                f"Калібрування збережено!\nЯкорів: {n}\nФайл: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{e}")

    @pyqtSlot()
    def on_load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити калібрування", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self.calibration.load(path)
            ids = [a.frame_id for a in self.calibration.anchors]
            propagated = self.database and self.database.is_propagated
            self.status_bar.showMessage(f"Калібрування: {len(ids)} якорів, кадри {ids}")
            self.control_panel.update_status("Калібрування завантажено")
            QMessageBox.information(self, "Успіх",
                f"Завантажено {len(ids)} якір(ів)!\nКадри: {ids}\n\n"
                f"{'✅ БД вже має дані пропагації.' if propagated else '⚠ Запустіть пропагацію.'}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")
