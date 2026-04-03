"""
calibration_mixin.py — ВИПРАВЛЕНА ВЕРСІЯ

Ключові зміни:
- ВИПРАВЛЕННЯ БАГ 2: Змінено логіку вибору трансформації у on_anchor_added.
  Попередня версія: estimate_affine_partial (4-DoF) як пріоритет, estimate_affine (6-DoF)
  лише при покращенні RMSE > 15%. Це призводило до вибору матриці з від'ємним детермінантом
  (дзеркальне відображення pixel→UTM) лише у виняткових випадках.

  Нова поведінка:
  1. При UTM-проекції: завжди використовується estimate_affine (6-DoF) якщо точок >= 4,
     оскільки перетворення pixel→UTM ЗАВЖДИ вимагає матрицю з від'ємним детермінантом
     (вісь Y пікселів ↓, вісь Y UTM ↑). estimate_affine_partial фізично не здатна
     це моделювати (детермінант завжди > 0).
  2. При WEB_MERCATOR: попередня логіка збережена (partial як пріоритет, full як fallback).
  3. Якщо точок < 4 або estimate_affine повернула None — fallback до partial.
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from config.config import get_cfg
from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.gui.dialogs.calibration_dialog import CalibrationDialog
from src.localization.matcher import FeatureMatcher
from src.utils.logging_utils import get_logger
from src.workers.calibration_propagation_worker import CalibrationPropagationWorker

logger = get_logger(__name__)


class CalibrationMixin:
    # ── Calibration dialog ───────────────────────────────────────────────────

    @pyqtSlot()
    def on_calibrate(self):
        if not self.database or self.database.db_file is None:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте або створіть базу даних!")
            return

        anchors_data = [a.to_dict() for a in self.calibration.anchors]

        self._calib_dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=anchors_data,
            parent=self,
        )
        self._calib_dialog.anchor_added.connect(self.on_anchor_added)
        self._calib_dialog.anchor_removed.connect(self.on_anchor_removed)
        self._calib_dialog.calibration_complete.connect(self.on_run_propagation)
        self._calib_dialog.exec()

        self._calib_dialog = None

    @pyqtSlot(object)
    def on_anchor_added(self, anchor_data: dict):
        try:
            points_2d = anchor_data.get("points_2d")
            points_gps = anchor_data.get("points_gps")
            frame_id = anchor_data.get("calib_frame_id")

            if not points_2d or not points_gps or len(points_2d) < 4:
                QMessageBox.warning(self, "Помилка", "Потрібно мінімум 4 точки для якоря!")
                return

            # Налаштування проєкції, якщо вона ще не ініціалізована
            if not self.calibration.converter._initialized:
                mode = get_cfg(self.config, "projection.default_mode", "WEB_MERCATOR")
                reference_gps = points_gps[0] if mode == "UTM" else None
                self.calibration.converter = CoordinateConverter(mode, reference_gps)

            pts_2d_np = np.array(points_2d, dtype=np.float32)
            pts_metric = [
                self.calibration.converter.gps_to_metric(lat, lon) for lat, lon in points_gps
            ]
            pts_metric_np = np.array(pts_metric, dtype=np.float32)

            def calc_metrics(M, src, dst):
                proj = GeometryTransforms.apply_affine(src, M)
                errs = np.linalg.norm(proj - dst, axis=1)
                return (
                    float(np.sqrt(np.mean(errs**2))),
                    float(np.median(errs)),
                    float(np.max(errs)),
                    proj.tolist(),
                )

            # ── ВИПРАВЛЕННЯ БАГ 2: логіка вибору трансформації ─────────────────
            #
            # Система координат pixel vs UTM:
            #   - Піксельна: вісь Y спрямована ВНИЗ (0 = верх кадру)
            #   - UTM:       вісь Y спрямована ВГОРУ (на північ)
            # Тому правильна матриця pixel→UTM ЗАВЖДИ має від'ємний детермінант
            # (вона містить дзеркальне відображення). estimate_affine_partial (4-DoF)
            # генерує лише матриці вигляду [s*cos -s*sin; s*sin s*cos], детермінант = s² > 0.
            # Вона фізично не може відобразити такий простір.
            #
            # Рішення: при UTM-проекції завжди пробуємо estimate_affine (6-DoF) першою.
            # При WEB_MERCATOR зберігаємо стару логіку (partial як пріоритет).

            is_utm = self.calibration.converter._mode == "UTM"

            best_M = None
            best_type = "unknown"
            rmse_p = float("inf")
            median_p = 0.0
            max_p = 0.0
            proj_p = []

            if is_utm:
                # UTM: пріоритет — повна афінна (6-DoF), яка підтримує від'ємний детермінант
                M_full, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
                if M_full is not None:
                    det = float(M_full[0, 0] * M_full[1, 1] - M_full[0, 1] * M_full[1, 0])
                    logger.info(f"Full affine determinant for anchor {frame_id}: {det:.6f}")
                    if det > 0:
                        # Від'ємний детермінант очікується; якщо позитивний — попередження
                        logger.warning(
                            f"Anchor {frame_id}: full affine determinant is POSITIVE ({det:.4f}). "
                            "Pixel→UTM should have negative det (Y-axis flip). "
                            "Check point ordering or projection setup."
                        )
                    rmse_p, median_p, max_p, proj_p = calc_metrics(M_full, pts_2d_np, pts_metric_np)
                    best_M = M_full
                    best_type = "affine_full"
                    logger.info(
                        f"UTM mode: using full affine for anchor {frame_id} (RMSE: {rmse_p:.2f}m, det={det:.4f})"
                    )

                # Fallback до partial якщо full не вийшла (дуже мала к-сть точок або збій)
                if best_M is None:
                    logger.warning(
                        f"Anchor {frame_id}: estimate_affine failed for UTM. "
                        "Falling back to affine_partial — metric accuracy will be degraded."
                    )
                    M_partial, _ = GeometryTransforms.estimate_affine_partial(
                        pts_2d_np, pts_metric_np
                    )
                    if M_partial is not None:
                        rmse_p, median_p, max_p, proj_p = calc_metrics(
                            M_partial, pts_2d_np, pts_metric_np
                        )
                        best_M = M_partial
                        best_type = "affine_partial (UTM fallback)"

            else:
                # WEB_MERCATOR або інші проекції: стара логіка
                M_partial, _ = GeometryTransforms.estimate_affine_partial(pts_2d_np, pts_metric_np)
                best_M = M_partial
                best_type = "affine_partial"

                if M_partial is not None:
                    rmse_p, median_p, max_p, proj_p = calc_metrics(
                        M_partial, pts_2d_np, pts_metric_np
                    )

                # Пробуємо повну афінну якщо точок >= 5
                if len(pts_2d_np) >= 5:
                    M_full, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
                    if M_full is not None:
                        rmse_f, median_f, max_f, proj_f = calc_metrics(
                            M_full, pts_2d_np, pts_metric_np
                        )
                        if rmse_f < rmse_p * 0.85:
                            best_M = M_full
                            best_type = "affine_full"
                            rmse_p, median_p, max_p, proj_p = rmse_f, median_f, max_f, proj_f
                            logger.info(
                                f"Selected full affine for anchor {frame_id} (RMSE: {rmse_f:.2f}m)"
                            )

            if best_M is None:
                QMessageBox.critical(
                    self,
                    "Помилка",
                    "Не вдалося обчислити матрицю. Спробуйте іншу комбінацію точок.",
                )
                return

            # ── Перевірка порогів якості ────────────────────────────────────────
            rmse_threshold = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)
            max_err_threshold = get_cfg(self.config, "projection.anchor_max_error_m", 5.0)

            from datetime import datetime

            severity_color = "green"
            if rmse_p > rmse_threshold:
                severity_color = "red"
            elif rmse_p > rmse_threshold * 0.7:
                severity_color = "orange"

            qa_summary = (
                f"<b>Метрики якості для якоря (кадр {frame_id}):</b><br><br>"
                f"Трансформація: <code style='color:blue'>{best_type}</code><br>"
                f"Кількість точок: <b>{len(pts_2d_np)}</b><br>"
                f"RMSE: <b style='color:{severity_color}'>{rmse_p:.2f} м</b> (поріг: {rmse_threshold}м)<br>"
                f"Медіанна похибка: <b>{median_p:.2f} м</b><br>"
                f"Макс. похибка: <b>{max_p:.2f} м</b> (поріг: {max_err_threshold}м)<br>"
            )

            if rmse_p > rmse_threshold or max_p > max_err_threshold:
                qa_summary += "<br><span style='color:red'>⚠ Увага: Якість прив'язки нижча за рекомендовану!</span>"
                reply = QMessageBox.warning(
                    self,
                    "Якість калібрування",
                    qa_summary + "<br><br>Зберегти цей якір попри високу похибку?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            else:
                logger.success(f"Anchor {frame_id} QA passed: RMSE={rmse_p:.2f}m")

            # ── Збереження результатів ──────────────────────────────────────────
            qa_data = {
                "rmse_m": rmse_p,
                "median_err_m": median_p,
                "max_err_m": max_p,
                "inliers_count": len(pts_2d_np),
                "transform_type": best_type,
                "projection_mode": self.calibration.converter._mode,
                "created_at": datetime.now().isoformat(),
                "points_2d": points_2d,
                "points_gps": points_gps,
                "points_metric": pts_metric,
            }

            self.calibration.add_anchor(frame_id=frame_id, affine_matrix=best_M, qa_data=qa_data)

            if self.project_manager and self.project_manager.is_loaded:
                self.calibration.save(self.project_manager.calibration_path)

            # ── Діагностичний лог по точках ──────────────────────────────────────
            logger.info(f"--- Anchor {frame_id} Point-by-Point Analysis ---")
            for j in range(len(pts_2d_np)):
                p2d = pts_2d_np[j]
                pm = pts_metric_np[j]
                if best_M is not None:
                    trans = GeometryTransforms.apply_affine(p2d.reshape(1, 2), best_M)[0]
                    err = np.linalg.norm(trans - pm)

                    lat_c, lon_c = self.calibration.converter.metric_to_gps(
                        float(trans[0]), float(trans[1])
                    )
                    lat_t, lon_t = points_gps[j][0], points_gps[j][1]

                    dist_err = CoordinateConverter.haversine_distance(
                        (lat_c, lon_c), (lat_t, lon_t)
                    )

                    logger.info(
                        f"  Pt {j}: px={p2d} -> err={err:.3f}м ({dist_err:.3f}м по Хаверсину)"
                    )
                    logger.debug(
                        f"    GPS Calc: ({lat_c:.7f}, {lon_c:.7f}) | Target: ({lat_t:.7f}, {lon_t:.7f})"
                    )

            logger.info(
                f"Anchor {frame_id} QA Summary: {best_type} | points={len(pts_2d_np)} | "
                f"RMSE={rmse_p:.3f}м | MedianErr={median_p:.3f}м | MaxErr={max_p:.3f}м"
            )

            self.status_bar.showMessage(f"Додано якір (кадр {frame_id}, RMSE: {rmse_p:.2f}м)")

            if hasattr(self, "_calib_dialog") and self._calib_dialog is not None:
                self._calib_dialog.on_anchor_confirmed(frame_id)

        except Exception as e:
            logger.error(f"Failed to add anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося додати якір:\n{e}")

    @pyqtSlot(int)
    def on_anchor_removed(self, frame_id: int):
        """Видалення якоря та маркування пропагації застарілою."""
        try:
            if self.calibration.remove_anchor(frame_id):
                if self.project_manager and self.project_manager.is_loaded:
                    self.calibration.save(self.project_manager.calibration_path)

                logger.info(f"Anchor {frame_id} removed from project")
                self.status_bar.showMessage(
                    f"Якір {frame_id} видалено. Потрібно оновити пропагацію.", 5000
                )
        except Exception as e:
            logger.error(f"Failed to remove anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося видалити якір:\n{e}")

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
            matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося ініціалізувати матчер:\n{e}")
            return

        anchor_ids = [a.frame_id for a in self.calibration.anchors]
        n_frames = self.database.get_num_frames()
        logger.info(f"Propagation: {len(anchor_ids)} anchors {anchor_ids}, {n_frames} frames")

        self._propagation_dialog = QProgressDialog(
            f"Пропагація GPS від {len(anchor_ids)} якорів на {n_frames} кадрів...",
            "Скасувати",
            0,
            100,
            self,
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

        num_frames = self.database.get_num_frames()
        valid_mask = self.database.frame_valid
        valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0

        avg_rmse = 0.0
        max_rmse = 0.0
        avg_dis = 0.0
        avg_matches = 0.0

        if valid_count > 0:
            rmse_data = getattr(self.database, "frame_rmse", None)
            if rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                avg_rmse = float(np.mean(valid_rmse))
                max_rmse = float(np.max(valid_rmse))

            dis_data = getattr(self.database, "frame_disagreement", None)
            if dis_data is not None:
                dis_valid = dis_data[valid_mask]
                if np.any(dis_valid > 0):
                    avg_dis = float(np.mean(dis_valid[dis_valid > 0]))

            matches_data = getattr(self.database, "frame_matches", None)
            if matches_data is not None:
                avg_matches = float(np.mean(matches_data[valid_mask]))

        rmse_thresh = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)

        report = (
            f"<b>Пропагація завершена!</b><br><br>"
            f"Валідних кадрів: <b>{valid_count} / {num_frames}</b> ({valid_count / num_frames * 100:.1f}%)<br>"
            f"Середній RMSE (grid): <b style='color:{'green' if avg_rmse < rmse_thresh * 0.5 else 'orange'}'>{avg_rmse:.3f} м</b><br>"
            f"Середній матчинг: <b>{avg_matches:.1f} точок</b><br>"
        )
        if avg_dis > 0:
            report += f"Середня розбіжність (drift): <b style='color:{'red' if avg_dis > 5.0 else 'green'}'>{avg_dis:.3f} м</b><br>"

        if avg_rmse > rmse_thresh or avg_dis > 5.0:
            report += "<br><span style='color:red'>⚠ Увага: Якість у деяких сегментах може бути нестабільною.</span>"
        else:
            report += "<br><span style='color:green'>✅ Результати стабільні. Можна починати локалізацію.</span>"

        QMessageBox.information(self, "Пропагація", report)
        self.status_bar.showMessage(
            f"Пропагація готова: {valid_count} к., RMSE: {avg_rmse:.2f}м, Mat: {avg_matches:.0f}"
        )

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
            step = max(1, num_frames // 30)

            rmse_data = getattr(self.database, "frame_rmse", None)
            dis_data = getattr(self.database, "frame_disagreement", None)
            matches_data = getattr(self.database, "frame_matches", None)
            valid_mask = getattr(self.database, "frame_valid", None)

            points_to_show = []
            _diag_done = False
            for i in range(0, num_frames, step):
                affine = self.database.get_frame_affine(i)
                if affine is not None:
                    w = self.database.metadata.get("frame_width", 1920)
                    h = self.database.metadata.get("frame_height", 1080)

                    if not _diag_done:
                        _diag_done = True
                        logger.warning(f"=== VERIFY DIAG frame={i} ===")
                        logger.warning(f"  frame_width={w}, frame_height={h}")
                        logger.warning(f"  affine=\n{affine}")
                        for lbl, px, py in [
                            ("corner0", 0, 0),
                            ("center", w / 2, h / 2),
                            ("corner2", w, h),
                        ]:
                            mx_d = affine[0, 0] * px + affine[0, 1] * py + affine[0, 2]
                            my_d = affine[1, 0] * px + affine[1, 1] * py + affine[1, 2]
                            lat_d, lon_d = self.calibration.converter.metric_to_gps(
                                float(mx_d), float(my_d)
                            )
                            logger.warning(
                                f"  {lbl}({px},{py}) -> metric({mx_d:.1f},{my_d:.1f}) -> GPS({lat_d:.6f},{lon_d:.6f})"
                            )

                    mx, my = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h / 2) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h / 2) + affine[1, 2],
                    )
                    lat_c, lon_c = self.calibration.converter.metric_to_gps(float(mx), float(my))

                    mx_b, my_b = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h * 0.75) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h * 0.75) + affine[1, 2],
                    )
                    lat_b, lon_b = self.calibration.converter.metric_to_gps(
                        float(mx_b), float(my_b)
                    )

                    rmse = (
                        float(rmse_data[i]) if rmse_data is not None and i < len(rmse_data) else 0.0
                    )
                    dis = float(dis_data[i]) if dis_data is not None and i < len(dis_data) else 0.0
                    matches = (
                        int(matches_data[i])
                        if matches_data is not None and i < len(matches_data)
                        else 0
                    )

                    if (i // step) % 3 == 0:
                        logger.debug(
                            f"Verify Frame {i}: CENTER={lat_c:.6f},{lon_c:.6f} | BOTTOM={lat_b:.6f},{lon_b:.6f} | RMSE={rmse:.2f}m"
                        )

                    color = "green"
                    if rmse > 5.0 or dis > 10.0:
                        color = "red"
                    elif rmse > 2.0 or dis > 3.0:
                        color = "orange"

                    cx_px, cy_px = w / 2, h / 2
                    dw, dh = w * 0.1, h * 0.1
                    pts_px = [
                        (cx_px - dw, cy_px - dh),
                        (cx_px + dw, cy_px - dh),
                        (cx_px + dw, cy_px + dh),
                        (cx_px - dw, cy_px + dh),
                    ]
                    for idx_p, (px, py) in enumerate(pts_px):
                        mx_p, my_p = (
                            affine[0, 0] * px + affine[0, 1] * py + affine[0, 2],
                            affine[1, 0] * px + affine[1, 1] * py + affine[1, 2],
                        )
                        lat_p, lon_p = self.calibration.converter.metric_to_gps(
                            float(mx_p), float(my_p)
                        )
                        points_to_show.append(
                            {
                                "lat": float(lat_p),
                                "lon": float(lon_p),
                                "label": f"Кадр {i} Корнер {idx_p}",
                                "color": "gray",
                            }
                        )

                    points_to_show.append(
                        {
                            "lat": float(lat_c),
                            "lon": float(lon_c),
                            "label": f"Кадр {i} (Центр) | RMSE:{rmse:.1f}м | Mat:{matches}",
                            "color": color,
                        }
                    )
                    points_to_show.append(
                        {
                            "lat": float(lat_b),
                            "lon": float(lon_b),
                            "label": f"Кадр {i} (Низ) | RMSE:{rmse:.1f}м",
                            "color": "blue",
                        }
                    )

            if points_to_show:
                self.map_widget.show_verification_markers(points_to_show)

            if valid_mask is not None and rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                if len(valid_rmse) > 0:
                    avg_rmse = float(np.mean(valid_rmse))
                    self.status_bar.showMessage(f"Пропагація: Середній RMSE = {avg_rmse:.3f} м")

        except Exception as e:
            logger.error(f"Error in on_verify_propagation: {e}", exc_info=True)
            self.status_bar.showMessage("Помилка візуалізації якості")

    @pyqtSlot(str)
    def on_propagation_error(self, error_msg: str):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None
        logger.error(f"Propagation error: {error_msg}")
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
            QMessageBox.information(
                self, "Збережено", f"Калібрування збережено!\nЯкорів: {n}\nФайл: {path}"
            )
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
            QMessageBox.information(
                self,
                "Успіх",
                f"Завантажено {len(ids)} якір(ів)!\nКадри: {ids}\n\n"
                f"{'✅ БД вже має дані пропагації.' if propagated else '⚠ Запустіть пропагацію.'}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")
