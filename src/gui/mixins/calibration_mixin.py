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

        # ОНОВЛЕНО: Зберігаємо посилання на діалог як атрибут класу
        self._calib_dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=existing_ids,
            parent=self,
        )
        self._calib_dialog.anchor_added.connect(self.on_anchor_added)
        self._calib_dialog.calibration_complete.connect(self.on_run_propagation)
        self._calib_dialog.exec()

        # Очищуємо посилання після закриття діалогу
        self._calib_dialog = None

    @pyqtSlot(object)
    def on_anchor_added(self, anchor_data: dict):
        try:
            points_2d = anchor_data.get('points_2d')
            points_gps = anchor_data.get('points_gps')
            frame_id = anchor_data.get('calib_frame_id')

            if not points_2d or not points_gps or len(points_2d) < 4:
                QMessageBox.warning(self, "Помилка", "Потрібно мінімум 4 точки для якоря!")
                return

            if not getattr(self.calibration, 'reference_gps', None):
                self.calibration.reference_gps = points_gps[0]

            pts_2d_np = np.array(points_2d, dtype=np.float32)
            pts_metric = [CoordinateConverter.gps_to_metric(lat, lon) for lat, lon in points_gps]
            pts_metric_np = np.array(pts_metric, dtype=np.float32)

            # 1. Спроба обчислити різні типи трансформацій та вибір найкращої (QA)
            # Partial Affine (4 DoF) - зазвичай стабільніше
            M_partial, _ = GeometryTransforms.estimate_affine_partial(pts_2d_np, pts_metric_np)
            
            best_M = M_partial
            best_type = "partial_affine (4-DoF)"
            
            # Якщо точок >= 5, пробуємо повний Affine (6 DoF)
            if len(pts_2d_np) >= 5:
                M_full, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
                if M_full is not None:
                    # Порівнюємо RMSE (на метриці)
                    def calc_rmse(M, src, dst):
                        proj = GeometryTransforms.apply_affine(src, M)
                        err = np.linalg.norm(proj - dst, axis=1)
                        return np.sqrt(np.mean(err**2)), np.max(err)

                    rmse_p, _ = calc_rmse(M_partial, pts_2d_np, pts_metric_np)
                    rmse_f, _ = calc_rmse(M_full, pts_2d_np, pts_metric_np)
                    
                    # Вибираємо повний Affine тільки якщо він дає суттєве покращення (>15%)
                    if rmse_f < rmse_p * 0.85:
                        best_M = M_full
                        best_type = "full_affine (6-DoF)"
                        self.logger.info(f"Selected full affine for anchor {frame_id} (RMSE improvement: {rmse_p:.2f} -> {rmse_f:.2f})")

            if best_M is None:
                QMessageBox.critical(self, "Помилка",
                                     "Не вдалося обчислити матрицю. Спробуйте розставити точки ширше!")
                return

            # 2. Обчислення фінальних QA метрик
            proj_metric = GeometryTransforms.apply_affine(pts_2d_np, best_M)
            errors = np.linalg.norm(proj_metric - pts_metric_np, axis=1)
            rmse_m = np.sqrt(np.mean(errors**2))
            max_err_m = np.max(errors)
            
            # 3. Діалог підтвердження (QA Layer)
            from datetime import datetime
            qa_summary = (
                f"<b>Метрики якості для якоря (кадр {frame_id}):</b><br><br>"
                f"Тип трансформації: <code style='color:blue'>{best_type}</code><br>"
                f"Кількість точок: <b>{len(pts_2d_np)}</b><br>"
                f"Середня похибка (RMSE): <b style='color:{'red' if rmse_m > 3.0 else 'green'}'>{rmse_m:.2f} м</b><br>"
                f"Макс. похибка: <b>{max_err_m:.2f} м</b><br>"
            )
            
            if rmse_m > 3.0 or max_err_m > 6.0:
                qa_summary += "<br><span style='color:red'>⚠ Увага: Похибка перевищує рекомендований поріг (3.0м)!</span>"
                reply = QMessageBox.warning(
                    self, "Якість якоря",
                    qa_summary + "<br><br>Зберегти цей якір з такою похибкою?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            else:
                # Навіть якщо все добре, при додаванні першого якоря можна просто логувати,
                # але користувачу краще бачити результат для впевненості.
                self.logger.success(f"Anchor {frame_id} QA: RMSE={rmse_m:.2f}m, MaxErr={max_err_m:.2f}m")

            qa_data = {
                "rmse_m": rmse_m,
                "max_err_m": max_err_m,
                "num_points": len(pts_2d_np),
                "transform_type": best_type,
                "created_at": datetime.now().isoformat(),
                "points_2d": points_2d,
                "points_gps": points_gps
            }

            self.calibration.add_anchor(frame_id=frame_id, affine_matrix=best_M, qa_data=qa_data)

            if self.project_manager and self.project_manager.is_loaded:
                calib_path = self.project_manager.calibration_path
                self.calibration.save(calib_path)

            self.status_bar.showMessage(f"Якір для кадру {frame_id} створено (RMSE: {rmse_m:.1f}м)")

            # ОНОВЛЕНО: Повідомляємо діалогове вікно про успіх
            if hasattr(self, '_calib_dialog') and self._calib_dialog is not None:
                self._calib_dialog.on_anchor_confirmed(frame_id)

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
            # ОНОВЛЕНО: Ініціалізуємо FeatureMatcher, він автоматично підлаштується
            # під розмірність дескрипторів (64 для XFeat) та використає Numpy L2
            matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося ініціалізувати матчер:\n{e}")
            return

        anchor_ids = [a.frame_id for a in self.calibration.anchors]
        n_frames = self.database.get_num_frames()
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

        # Обчислюємо статистику якості
        num_frames = self.database.get_num_frames()
        valid_mask = self.database.frame_valid
        valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0
        
        avg_rmse = 0.0
        max_rmse = 0.0
        avg_dis = 0.0
        
        if valid_count > 0:
            # Safe access to RMSE
            rmse_data = getattr(self.database, 'frame_rmse', None)
            if rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                avg_rmse = float(np.mean(valid_rmse))
                max_rmse = float(np.max(valid_rmse))
            
            # Safe access to Disagreement
            dis_data = getattr(self.database, 'frame_disagreement', None)
            if dis_data is not None:
                dis_valid = dis_data[valid_mask]
                if np.any(dis_valid > 0):
                    avg_dis = float(np.mean(dis_valid[dis_valid > 0]))
        
        report = (
            f"<b>Пропагаця завершена!</b><br><br>"
            f"Валідних кадрів: <b>{valid_count} / {num_frames}</b> ({valid_count/num_frames*100:.1f}%)<br>"
            f"Середня похибка сітки (RMSE): <b>{avg_rmse:.3f} м</b><br>"
            f"Максимальна похибка: <b>{max_rmse:.3f} м</b><br>"
        )
        if avg_dis > 0:
            report += f"Середня розбіжність (Between): <b>{avg_dis:.3f} м</b><br>"
            
        if avg_rmse > 1.0 or avg_dis > 5.0:
            report += "<br><span style='color:#e65100'>⚠ Порада: Велика розбіжність зазвичай свідчить про низьку якість одного з якорів або складний рельєф.</span>"

        QMessageBox.information(self, "Пропагація", report)
        self.status_bar.showMessage(f"Пропагація готова: {valid_count} кадрів (RMSE: {avg_rmse:.2f}м)")
        
        if self.map_widget:
            self.on_verify_propagation()

    @pyqtSlot()
    def on_verify_propagation(self):
        """Візуалізація та звіт якості пропагації на мапі"""
        if not self.database or not self.database.is_propagated:
            QMessageBox.warning(self, "Увага", "Дані пропагації не знайдено.")
            return

        try:
            self.map_widget.clear_verification_markers()
            num_frames = self.database.get_num_frames()
            step = max(1, num_frames // 25)

            # Отримуємо дані один раз для швидкості та безпеки
            rmse_data = getattr(self.database, 'frame_rmse', None)
            valid_mask = getattr(self.database, 'frame_valid', None)
            
            points_to_show = []

            for i in range(0, num_frames, step):
                affine = self.database.get_frame_affine(i)
                if affine is not None:
                    # Центр кадру
                    w = self.database.metadata.get('frame_width', 1920)
                    h = self.database.metadata.get('frame_height', 1080)
                    mx = affine[0, 0] * (w/2) + affine[0, 1] * (h/2) + affine[0, 2]
                    my = affine[1, 0] * (w/2) + affine[1, 1] * (h/2) + affine[1, 2]
                    
                    lat, lon = CoordinateConverter.metric_to_gps(mx, my)
                    
                    # Безпечне отримання RMSE
                    rmse = 0.0
                    if rmse_data is not None and i < len(rmse_data):
                        rmse = float(rmse_data[i])
                    
                    # Колір за якістю
                    color = "green" if rmse < 1.0 else "orange" if rmse < 3.0 else "red"
                    
                    points_to_show.append({
                        'lat': float(lat),
                        'lon': float(lon),
                        'label': f"Кадр {i} (RMSE: {rmse:.2f}м)",
                        'color': color
                    })

            if points_to_show:
                self.map_widget.show_verification_markers(points_to_show)
            
            # Статистика в статус-бар
            if valid_mask is not None and rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                if len(valid_rmse) > 0:
                    avg_rmse = float(np.mean(valid_rmse))
                    self.status_bar.showMessage(f"Якість пропагації: Середній RMSE = {avg_rmse:.3f} м")
        
        except Exception as e:
            self.logger.error(f"Error in on_verify_propagation: {e}", exc_info=True)
            self.status_bar.showMessage("Помилка візуалізації якості")


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
            
        default_path = "calibration.json"
        if self.project_manager and self.project_manager.is_loaded:
            default_path = str(self.project_manager.project_dir / default_path)
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти калібрування", default_path, "JSON Files (*.json)"
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
        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir)
            
        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити калібрування", default_dir, "JSON Files (*.json);;All Files (*)"
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
