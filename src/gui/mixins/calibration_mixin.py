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

from datetime import datetime
from pathlib import Path

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

        # Параметри відповідності кадрів відео ↔ слотів БД.
        # Без них діалог не може конвертувати номери кадрів і якорі
        # прив'язуються до неправильних слотів (див. BUGREPORT №1).
        db_num_frames = self.database.get_num_frames()
        frame_step = int(self.database.metadata.get("frame_step", 0) or 0)
        if frame_step < 1:
            # Старі БД без frame_step у метаданих — беремо з конфіга (може не збігатися!)
            frame_step = int(get_cfg(self.config, "database.frame_step", 30))
            logger.warning(
                f"DB metadata has no 'frame_step' — falling back to config value {frame_step}. "
                f"If the DB was built with a different step, anchor frame ids may be wrong."
            )
        kp_video_path = str(Path(self.database.db_path).with_suffix("")) + "_keypoints.mp4"

        self._calib_dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=anchors_data,
            source_id=self._get_current_source_id(),
            parent=self,
            db_num_frames=db_num_frames,
            frame_step=frame_step,
            keypoints_video_path=kp_video_path,
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

            # ВИПРАВЛЕНО: WEB_MERCATOR-конвертер за замовчуванням завжди
            # "_initialized", тому налаштований projection.default_mode (напр. UTM)
            # мовчки ігнорувався. Перемикаємо режим, поки якорів ще немає.
            mode = str(get_cfg(self.config, "projection.default_mode", "WEB_MERCATOR")).upper()
            conv = self.calibration.converter
            if not conv.is_initialized or (not self.calibration.anchors and conv.mode != mode):
                reference_gps = tuple(points_gps[0]) if mode == "UTM" else None
                self.calibration.converter = CoordinateConverter(mode, reference_gps)
                logger.info(f"Projection initialized for calibration: {mode}")

            # Розмір кадру — потрібен для інтерполяції якорів навколо центру кадру
            fw = int(self.database.metadata.get("frame_width", 0) or 0)
            fh = int(self.database.metadata.get("frame_height", 0) or 0)
            if fw > 0 and fh > 0 and hasattr(self.calibration, "set_frame_size"):
                self.calibration.set_frame_size(fw, fh)

            pts_2d_np = np.array(points_2d, dtype=np.float64)
            pts_metric = [
                self.calibration.converter.gps_to_metric(lat, lon) for lat, lon in points_gps
            ]
            pts_metric_np = np.array(pts_metric, dtype=np.float64)

            def calc_metrics(M, src, dst):
                proj = GeometryTransforms.apply_affine(src, M)
                errs = np.linalg.norm(proj - dst, axis=1)
                return (
                    float(np.sqrt(np.mean(errs**2))),
                    float(np.median(errs)),
                    float(np.max(errs)),
                    proj.tolist(),
                )

            # ── ВИПРАВЛЕНО: детермінований фіт якоря ────────────────────────────
            #
            # Раніше: cv2.estimateAffine2D з RANSAC та порогом 3.0, який
            # інтерпретується в одиницях ПРИЗНАЧЕННЯ (метрах!). Кліки з похибкою
            # 1–3 м опинялися на межі порогу, і недетермінований RANSAC давав
            # РІЗНІ матриці за тих самих точок між запусками → "нестабільний
            # крок калібрації". Для 4–8 перевірених користувачем точок RANSAC
            # недоречний — використовуємо least-squares по всіх точках.
            #
            # Система координат: піксельна вісь Y ↓, метрична (UTM/Mercator) Y ↑,
            # тому фізично коректна матриця pixel→metric ЗАВЖДИ має det < 0.
            best_M = GeometryTransforms.estimate_affine_lsq(pts_2d_np, pts_metric_np)
            best_type = "affine_full_lsq"

            if best_M is None or not GeometryTransforms.is_matrix_valid(best_M):
                QMessageBox.critical(
                    self,
                    "Помилка",
                    "Не вдалося обчислити коректну матрицю за цими точками.\n\n"
                    "Найчастіша причина — точки лежать майже на одній прямій.\n"
                    "Розставте 4–6 точок якомога ширше по всьому кадру.",
                )
                return

            det = float(best_M[0, 0] * best_M[1, 1] - best_M[0, 1] * best_M[1, 0])
            logger.info(f"Anchor {frame_id} affine determinant: {det:.6f}")
            if det > 0:
                # det > 0 фізично неможливий для pixel→map: означає дзеркально
                # переплутані вхідні дані. Раніше тут був лише warning у лог, і
                # такий якір ламав глобальний sign у графовій оптимізації.
                QMessageBox.critical(
                    self,
                    "Помилка калібрування",
                    f"Матриця має додатний детермінант ({det:.4f}) — це фізично "
                    f"неможливо для переходу пікселі → карта (вісь Y має "
                    f"віддзеркалюватися).\n\n"
                    f"Перевірте: чи не переплутані широта/довгота у точках, "
                    f"чи правильним орієнтирам призначені координати.",
                )
                return

            rmse_p, median_p, max_p, proj_p = calc_metrics(best_M, pts_2d_np, pts_metric_np)

            # ── Перевірка порогів якості ────────────────────────────────────────
            rmse_threshold = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)
            max_err_threshold = get_cfg(self.config, "projection.anchor_max_error_m", 5.0)

            # ── B6: Leave-one-out перевірка точок ────────────────────────────────
            # Фітимо матрицю без кожної точки по черзі й міряємо, наскільки
            # "викинута" точка не узгоджується з рештою. Сумарний RMSE маскує
            # одну криву точку (неправильний клік / переплутана координата) —
            # LOO показує її явно.
            suspicious_points: list[tuple[int, float]] = []
            if 5 <= len(pts_2d_np) <= 12:
                loo_errors: list[float] = []
                all_idx = np.arange(len(pts_2d_np))
                for j in range(len(pts_2d_np)):
                    idx = all_idx[all_idx != j]
                    M_loo = GeometryTransforms.estimate_affine_lsq(
                        pts_2d_np[idx], pts_metric_np[idx]
                    )
                    if M_loo is None:
                        loo_errors.append(float("nan"))
                        continue
                    proj_j = GeometryTransforms.apply_affine(
                        pts_2d_np[j].reshape(1, 2), M_loo
                    )[0]
                    loo_errors.append(float(np.linalg.norm(proj_j - pts_metric_np[j])))

                finite = [e for e in loo_errors if np.isfinite(e)]
                if finite:
                    med_loo = float(np.median(finite))
                    for j, e in enumerate(loo_errors):
                        if np.isfinite(e) and e > max(3.0 * med_loo, 2.0 * rmse_threshold):
                            suspicious_points.append((j, e))
                    logger.info(
                        f"Anchor {frame_id} LOO errors (m): "
                        + ", ".join(f"pt{j + 1}={e:.2f}" for j, e in enumerate(loo_errors))
                    )

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

            if suspicious_points:
                pts_txt = ", ".join(f"№{j + 1} ({e:.1f} м)" for j, e in suspicious_points)
                qa_summary += (
                    f"<br><span style='color:red'>⚠ Підозрілі точки (leave-one-out): "
                    f"{pts_txt}.<br>Ймовірно, неправильний клік або переплутана "
                    f"координата — перевірте ці точки.</span>"
                )

            if rmse_p > rmse_threshold or max_p > max_err_threshold or suspicious_points:
                if rmse_p > rmse_threshold or max_p > max_err_threshold:
                    qa_summary += "<br><span style='color:red'>⚠ Увага: Якість прив'язки нижча за рекомендовану!</span>"
                reply = QMessageBox.warning(
                    self,
                    "Якість калібрування",
                    qa_summary + "<br><br>Зберегти цей якір попри зауваження?",
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
                "projection_mode": self.calibration.converter.mode,
                "created_at": datetime.now().isoformat(),
                "points_2d": points_2d,
                "points_gps": points_gps,
                "points_metric": pts_metric,
            }

            self.calibration.add_anchor(frame_id=frame_id, affine_matrix=best_M, qa_data=qa_data)

            if self.project_manager and self.project_manager.is_loaded:
                cal_path = self._get_calibration_save_path()
                if cal_path:
                    self.calibration.save(cal_path)

            if hasattr(self, "_update_project_info_panel"):
                self._update_project_info_panel()

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
                saved_anchor = self.calibration.get_anchor(frame_id)
                self._calib_dialog.on_anchor_confirmed(
                    frame_id, saved_anchor.to_dict() if saved_anchor else None
                )

        except Exception as e:
            logger.error(f"Failed to add anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося додати якір:\n{e}")

    @pyqtSlot(int)
    def on_anchor_removed(self, frame_id: int):
        """Видалення якоря та маркування пропагації застарілою."""
        try:
            if self.calibration.remove_anchor(frame_id):
                if self.project_manager and self.project_manager.is_loaded:
                    cal_path = self._get_calibration_save_path()
                    if cal_path:
                        self.calibration.save(cal_path)

                if hasattr(self, "_update_project_info_panel"):
                    self._update_project_info_panel()

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
        # Взаємне виключення з трекінгом: пропагація перезаписує HDF5.
        tw = getattr(self, "tracking_worker", None)
        if tw is not None and tw.isRunning():
            QMessageBox.warning(
                self, "Увага", "Зупиніть трекінг перед запуском пропагації — "
                "вони використовують одну базу даних."
            )
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

        # УВАГА: frame_rmse з пропагації — це ПІКСЕЛІ репроєкції матчів між
        # кадрами, а не метри (раніше підпис "м" вводив в оману)
        report = (
            f"<b>Пропагація завершена!</b><br><br>"
            f"Валідних кадрів: <b>{valid_count} / {num_frames}</b> ({valid_count / num_frames * 100:.1f}%)<br>"
            f"Середній RMSE матчингу: <b style='color:{'green' if avg_rmse < rmse_thresh * 0.5 else 'orange'}'>{avg_rmse:.3f} px</b><br>"
            f"Середній матчинг: <b>{avg_matches:.1f} точок</b><br>"
        )

        log_msg = (
            f"Пропагація завершена. "
            f"Валідних: {valid_count}/{num_frames} ({valid_count / num_frames * 100:.1f}%), "
            f"RMSE: {avg_rmse:.3f}px, "
            f"Матчинг: {avg_matches:.1f} точок"
        )
        if avg_dis > 0:
            log_msg += f", Drift: {avg_dis:.3f}м"

        logger.info(log_msg)

        if avg_dis > 0:
            report += f"Середня розбіжність (drift): <b style='color:{'red' if avg_dis > 5.0 else 'green'}'>{avg_dis:.3f} м</b><br>"

        if avg_rmse > rmse_thresh or avg_dis > 5.0:
            report += "<br><span style='color:red'>⚠ Увага: Якість у деяких сегментах може бути нестабільною.</span>"
        else:
            report += "<br><span style='color:green'>✅ Результати стабільні. Можна починати локалізацію.</span>"

        QMessageBox.information(self, "Пропагація", report)
        self.status_bar.showMessage(
            f"Пропагація готова: {valid_count} к., RMSE: {avg_rmse:.2f}px, Mat: {avg_matches:.0f}"
        )

        if hasattr(self, "_update_project_info_panel"):
            self._update_project_info_panel()

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
            # A8: тисячі Leaflet-маркерів одним JSON вішають QWebEngine —
            # обмежуємо кількість точок до ~600 із рівномірним кроком
            step = max(1, num_frames // 600)

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

                    # Центр кадру
                    mx, my = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h / 2) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h / 2) + affine[1, 2],
                    )
                    lat_c, lon_c = self.calibration.converter.metric_to_gps(float(mx), float(my))

                    # Низ кадру (замінено 0.75 на h для точнішої орієнтації повного низу)
                    mx_b, my_b = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * h + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * h + affine[1, 2],
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

                    # Відмальовуємо тільки центр кадру
                    points_to_show.append(
                        {
                            "lat": float(lat_c),
                            "lon": float(lon_c),
                            "label": str(i),
                            "color": color,
                        }
                    )

            if points_to_show:
                self.map_widget.show_verification_markers(points_to_show)

            if valid_mask is not None and rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                if len(valid_rmse) > 0:
                    avg_rmse = float(np.mean(valid_rmse))
                    self.status_bar.showMessage(f"Пропагація: Середній RMSE = {avg_rmse:.3f} px")

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

    def _get_current_source_id(self) -> str:
        """Повертає source_id поточного активного джерела (базуючись на db_path)."""
        if not self.project_manager or not self.project_manager.is_loaded or not self.database:
            return "main"

        current_db = str(Path(self.database.db_path).resolve())
        project_dir = self.project_manager.project_dir
        for src_dict in (self.project_manager.settings.video_sources or []):
            db_file = src_dict.get("database_file", "")
            if db_file and str((project_dir / db_file).resolve()) == current_db:
                return src_dict.get("source_id", "main")
        return "main"

    def _get_calibration_save_path(self) -> str | None:
        """Повертає шлях до calibration.json для поточного активного джерела.

        Принцип: зіставляє `self.database.db_path` з `database_file` кожного
        джерела в project settings, щоб знайти відповідний `calibration_file`.
        Fallback: `project_manager.calibration_path` (корінь проєкту).
        """
        if not self.project_manager or not self.project_manager.is_loaded:
            return None

        project_dir = self.project_manager.project_dir

        # Пошук джерела відповідно до поточної БД
        if self.database and self.project_manager.settings:
            current_db = str(Path(self.database.db_path).resolve())
            for src_dict in (self.project_manager.settings.video_sources or []):
                db_file = src_dict.get("database_file", "")
                cal_file = src_dict.get("calibration_file", "")
                if not db_file or not cal_file:
                    continue
                if str((project_dir / db_file).resolve()) == current_db:
                    cal_path = project_dir / cal_file
                    cal_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(
                        f"Calibration path: {cal_path} "
                        f"(source='{src_dict.get('source_id', '?')}')"
                    )
                    return str(cal_path)

        # Fallback — шлях на рівні проєкту
        return self.project_manager.calibration_path

    @pyqtSlot()
    def on_save_calibration(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Немає даних для збереження.")
            return

        # Дефолтний шлях — папка активного джерела
        default_path = self._get_calibration_save_path() or "calibration.json"

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
            self.control_panel.update_status("Калібрування завантажено")

            # Автоматично зберігаємо копію в папці поточного джерела
            # (навіть якщо файл завантажено з іншого місця або скопійовано вручну)
            source_cal_path = self._get_calibration_save_path()
            copied_to_source = False
            if source_cal_path:
                norm_loaded = str(Path(path).resolve())
                norm_source = str(Path(source_cal_path).resolve())
                if norm_loaded != norm_source:
                    # Файл прийшов не з папки джерела — копіюємо туди
                    Path(source_cal_path).parent.mkdir(parents=True, exist_ok=True)
                    self.calibration.save(source_cal_path)
                    copied_to_source = True
                    logger.info(
                        f"Calibration copied to source folder: {source_cal_path}"
                    )
                else:
                    logger.debug("Calibration loaded directly from source folder, no copy needed.")

            if hasattr(self, "_update_project_info_panel"):
                self._update_project_info_panel()

            copy_note = (
                f"\n\n📋 Також збережено у папці джерела:\n{source_cal_path}"
                if copied_to_source
                else ""
            )
            self.status_bar.showMessage(f"Калібрування: {len(ids)} якорів, кадри {ids}")
            QMessageBox.information(
                self,
                "Успіх",
                f"Завантажено {len(ids)} якір(ів)!\nКадри: {ids}\n\n"
                f"{'✅ БД вже має дані пропагації.' if propagated else '⚠ Запустіть пропагацію.'}"
                f"{copy_note}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")
