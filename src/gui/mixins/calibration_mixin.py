"""GUI-міксин калібрування: діалог якорів, фіт якоря, запуск пропагації.

Фіт якоря — детермінований 6-DoF least squares по ВСІХ точках
(``GeometryTransforms.estimate_affine_lsq``), без RANSAC: для 4–8 перевірених
користувачем точок RANSAC недетермінований (різні матриці між запусками), а його
поріг інтерпретується в одиницях призначення (метрах), через що валідні точки
випадково відкидались.

Матриця pixel→map ЗАВЖДИ має від'ємний детермінант (вісь Y пікселів ↓, вісь Y
карти ↑), тому det > 0 — це не warning, а відмова: такий якір ламає глобальний
знак у графовій оптимізації.

(Попередня версія цього докстрінга описувала вибір між estimate_affine_partial
і estimate_affine, якого в коді давно немає.)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from config import get_cfg
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.calibration.multi_calibration_manager import save_layer_calibration
from src.core.layer_status import CALIBRATION_OWNER_KEY
from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.gui.dialogs.calibration_dialog import CalibrationDialog
from src.localization.matcher import FeatureMatcher
from src.security.at_rest import decrypt_to_tempfile, get_passphrase, is_encrypted, wipe_file
from src.utils.logging_utils import fmt_coord, get_logger
from src.workers.calibration_propagation_worker import CalibrationPropagationWorker

logger = get_logger(__name__)


def _materialize_keypoints_video(path: str) -> tuple[str, str | None]:
    """Повертає шлях до відеокадрів ключових точок, розшифровуючи зашифровані файли у тимчасовий файл."""
    src = Path(path)
    if not src.is_file():
        return path, None
    with open(src, "rb") as f:
        if not is_encrypted(f.read(64)):
            return path, None  # plaintext: unchanged path
    tmp = decrypt_to_tempfile(str(src), get_passphrase(), suffix=".mp4")
    logger.info("Decrypted keypoint video to a temp file for calibration")
    return tmp, tmp


class CalibrationMixin:
    # ── Calibration dialog ───────────────────────────────────────────────────

    def _layer_info(self, source_id: str) -> dict:
        """What the calibration dialog shows about the layer being calibrated."""
        settings = self.project_manager.settings if self.project_manager.is_loaded else None
        src = settings.get_source(source_id) if settings else None
        info: dict = {"num_layers": len(settings.video_sources or []) if settings else 1}
        if src is not None:
            info.update(
                area_id=src.area_id,
                description=src.description,
                video_path=src.video_path,
                database_file=src.database_file,
                calibration_file=src.calibration_file,
            )
            if src.scale_layer is not None and src.scale_layer.nominal_gsd_m_per_px:
                info["gsd_m_per_px"] = src.scale_layer.nominal_gsd_m_per_px
        return info

    def _calibration_target_ok(self) -> bool:
        """Anchors may only land in the layer the calibration dialog was opened for."""
        target = getattr(self, "_calib_target", None)
        if target is None:
            return True
        sid, calibration, database = target
        current = self._get_current_source_id()
        if sid == current and calibration is self.calibration and database is self.database:
            return True
        logger.error(
            f"Calibration target mismatch: dialog layer '{sid}', active layer '{current}' — "
            f"anchor change refused to keep layer calibrations apart"
        )
        QMessageBox.critical(
            self,
            "Інший шар",
            f"Вікно калібрування відкрите для шару «{sid}», а активним став шар «{current}».\n\n"
            f"Зміну не збережено, щоб не змішати калібрування шарів. "
            f"Закрийте вікно і відкрийте калібрування потрібного шару знову.",
        )
        return False

    @pyqtSlot()
    def on_calibrate(self):
        if self.calibration.converter.mode == "LOCAL":
            QMessageBox.information(self, "Локальна карта", "Цей шар використовує умовні X/Y. Для GPS-калібрування створіть окремий шар із відео.")
            return
        source_id = self._get_current_source_id()
        if not self.database or self.database.db_file is None:
            QMessageBox.warning(
                self,
                "Помилка",
                f"Для шару «{source_id}» ще немає бази даних.\n\n"
                f"Спочатку побудуйте її (ПКМ по шару → Побудувати базу даних).",
            )
            return
        if self._get_calibration_save_path() is None:
            QMessageBox.critical(
                self,
                "Помилка",
                f"Не вдалося визначити файл калібрування шару «{source_id}» — "
                f"перевірте calibration_file у project.json.",
            )
            return

        anchors_data = [a.to_dict() for a in self.calibration.anchors]

        # Video frame ↔ DB slot mapping parameters.
        # Without these the dialog cannot convert frame numbers and anchors
        # bind to incorrect slots.
        db_num_frames = self.database.get_num_frames()
        frame_step = int(self.database.metadata.get("frame_step", 0) or 0)
        if frame_step < 1:
            # Legacy DB without frame_step in metadata — fallback to config
            frame_step = int(get_cfg(self.config, "database.frame_step", 30))
            logger.warning(
                f"DB metadata has no 'frame_step' — falling back to config value {frame_step}. "
                f"If the DB was built with a different step, anchor frame ids may be wrong."
            )
        kp_video_path = str(Path(self.database.db_path).with_suffix("")) + "_keypoints.mp4"
        # HARDENING P1-6 SP3: the keypoint video is opened by path (decord/cv2),
        # so an encrypted one is decrypted to a temp file for the dialog's
        # lifetime and wiped as soon as it closes.
        kp_video_path, kp_tempfile = _materialize_keypoints_video(kp_video_path)

        # Bind the dialog to THIS layer's objects: every anchor change is checked
        # against them, so an anchor can never be saved into another layer.
        self._calib_target = (source_id, self.calibration, self.database)
        self._calib_dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=anchors_data,
            source_id=source_id,
            parent=self,
            db_num_frames=db_num_frames,
            frame_step=frame_step,
            keypoints_video_path=kp_video_path,
            layer_info=self._layer_info(source_id),
        )
        self._calib_dialog.anchor_added.connect(self.on_anchor_added)
        self._calib_dialog.anchor_removed.connect(self.on_anchor_removed)
        self._calib_dialog.calibration_complete.connect(self.on_run_propagation)
        try:
            self._calib_dialog.exec()
        finally:
            self._calib_dialog = None
            self._calib_target = None
            if kp_tempfile:
                wipe_file(kp_tempfile)
                logger.info("Decrypted keypoint video wiped")

    @pyqtSlot(object)
    def on_anchor_added(self, anchor_data: dict):
        # Refuse before mutating in-memory state: the save below would raise and
        # leave the anchor list out of sync with what is on disk. Viewing an
        # encrypted project's calibration stays allowed.
        if self._refuse_if_encrypted_project("Додавання якоря"):
            return
        if not self._calibration_target_ok():
            return
        source_id = self._get_current_source_id()
        cal_path = self._get_calibration_save_path()
        if self.project_manager and self.project_manager.is_loaded and not cal_path:
            QMessageBox.critical(
                self, "Помилка", f"Невідомий файл калібрування шару «{source_id}» — якір не додано."
            )
            return
        try:
            points_2d = anchor_data.get("points_2d")
            points_gps = anchor_data.get("points_gps")
            frame_id = anchor_data.get("calib_frame_id")

            if not points_2d or not points_gps or len(points_2d) < 4:
                QMessageBox.warning(self, "Помилка", "Потрібно мінімум 4 точки для якоря!")
                return

            # Initialize or switch coordinate projection mode (UTM / WEB_MERCATOR).
            mode = str(get_cfg(self.config, "projection.default_mode", "WEB_MERCATOR")).upper()
            conv = self.calibration.converter
            if not conv.is_initialized or (not self.calibration.anchors and conv.mode != mode):
                reference_gps = tuple(points_gps[0]) if mode == "UTM" else None
                self.calibration.converter = CoordinateConverter(mode, reference_gps)
                logger.info(f"Projection initialized for calibration: {mode}")

            # Frame size — needed for anchor interpolation around frame center
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

            # Deterministic LSQ fit for anchor pixel -> metric transformation.
            # Coordinate system: pixel axis Y ↓, metric Y ↑, so valid matrix ALWAYS has det < 0.
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
                # det > 0 physically impossible for pixel->map (requires flipped Y axis)
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

            # Quality thresholds validation
            rmse_threshold = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)
            max_err_threshold = get_cfg(self.config, "projection.anchor_max_error_m", 5.0)

            # Leave-one-out validation
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
                    proj_j = GeometryTransforms.apply_affine(pts_2d_np[j].reshape(1, 2), M_loo)[0]
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

            # Save results
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

            if cal_path:
                save_layer_calibration(self.calibration, cal_path, source_id)

            # Layer table / info / registry reflect the new anchor immediately.
            if hasattr(self, "_after_layer_data_changed"):
                self._after_layer_data_changed()

            # Point-by-point diagnostic logging
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
                        f"    GPS Calc: {fmt_coord(lat_c, lon_c, precision=7)} | "
                        f"Target: {fmt_coord(lat_t, lon_t, precision=7)}"
                    )

            logger.info(
                f"Anchor {frame_id} QA Summary: {best_type} | points={len(pts_2d_np)} | "
                f"RMSE={rmse_p:.3f}м | MedianErr={median_p:.3f}м | MaxErr={max_p:.3f}м"
            )

            self.status_bar.showMessage(
                f"Шар «{source_id}»: додано якір (кадр {frame_id}, RMSE: {rmse_p:.2f}м)"
            )

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
        # Refuse before remove_anchor mutates memory — the save below would
        # raise and leave the anchor list out of sync with disk.
        if self._refuse_if_encrypted_project("Видалення якоря"):
            return
        if not self._calibration_target_ok():
            return
        source_id = self._get_current_source_id()
        try:
            if self.calibration.remove_anchor(frame_id):
                if self.project_manager and self.project_manager.is_loaded:
                    cal_path = self._get_calibration_save_path()
                    if cal_path:
                        save_layer_calibration(self.calibration, cal_path, source_id)

                if hasattr(self, "_after_layer_data_changed"):
                    self._after_layer_data_changed()

                logger.info(f"Anchor {frame_id} removed from layer '{source_id}'")
                self.status_bar.showMessage(
                    f"Шар «{source_id}»: якір {frame_id} видалено. Потрібно оновити пропагацію.",
                    5000,
                )
        except Exception as e:
            logger.error(f"Failed to remove anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося видалити якір:\n{e}")

    # ── Propagation ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_build_relative_map(self):
        self.on_run_propagation(relative=True)

    @pyqtSlot()
    def on_run_propagation(self, relative=False):
        if self._refuse_if_encrypted_project("Пропагація калібрування"):
            return
        current_local = self.calibration.converter.mode == "LOCAL"
        relative = relative or current_local
        if relative and not current_local and (
            self.calibration.anchors or (self.database and self.database.is_propagated)
        ):
            QMessageBox.warning(self, "Локальна карта", "Цей шар уже має геоприв’язку. Створіть окремий шар для FPV-відео.")
            return
        if not relative and not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Додайте хоча б один якір калібрування!")
            return
        if not self.database:
            QMessageBox.warning(self, "Увага", "База даних не завантажена!")
            return
        existing = getattr(self, "propagation_worker", None)
        if existing is not None and existing.isRunning():
            return
        # Mutual exclusion with tracking: propagation overwrites HDF5.
        tw = getattr(self, "tracking_worker", None)
        if tw is not None and tw.isRunning():
            QMessageBox.warning(
                self,
                "Увага",
                "Зупиніть трекінг перед запуском пропагації — вони використовують одну базу даних.",
            )
            return

        try:
            matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося ініціалізувати матчер:\n{e}")
            return

        source_id = self._get_current_source_id()
        work_calibration = (
            MultiAnchorCalibration(CoordinateConverter("LOCAL")) if relative else self.calibration
        )
        anchor_ids = [a.frame_id for a in work_calibration.anchors]
        n_frames = self.database.get_num_frames()
        logger.info(
            f"Propagation of layer '{source_id}': {len(anchor_ids)} anchors {anchor_ids}, "
            f"{n_frames} frames"
        )
        # The report and the map check must use the layer that was propagated.
        self._propagation_target = (source_id, self.database, work_calibration)

        self._propagation_dialog = QProgressDialog(
            f"Шар «{source_id}»: пропагація GPS від {len(anchor_ids)} якорів "
            f"на {n_frames} кадрів...",
            "Скасувати",
            0,
            100,
            self,
        )
        self._propagation_dialog.setWindowTitle(f"Розповсюдження GPS — шар «{source_id}»")
        if relative:
            self._propagation_dialog.setWindowTitle("Локальна карта FPV — без GPS")
            self._propagation_dialog.setLabelText("Побудова карти з міжкадрових зв’язків…")
        self._propagation_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._propagation_dialog.setMinimumDuration(0)
        self._propagation_dialog.setValue(0)

        self.propagation_worker = CalibrationPropagationWorker(
            database=self.database,
            calibration=work_calibration,
            matcher=matcher,
            config=self.config,
        )
        self.propagation_worker.progress.connect(self.on_propagation_progress)
        self.propagation_worker.completed.connect(self.on_propagation_completed)
        self.propagation_worker.error.connect(self.on_propagation_error)
        self.propagation_worker.cancelled.connect(self.on_propagation_cancelled)
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
    def on_propagation_cancelled(self):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None
        self._propagation_target = None
        self.status_bar.showMessage("Пропагацію скасовано")
        if hasattr(self, "_after_layer_data_changed"):
            self._after_layer_data_changed()

    @pyqtSlot()
    def on_propagation_completed(self):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None

        target = getattr(self, "_propagation_target", None)
        self._propagation_target = None
        source_id, database = (
            (target[0], target[1]) if target else (self._get_current_source_id(), self.database)
        )
        if target and target[2].converter.mode == "LOCAL":
            local_cal = target[2]
            if database is self.database:
                self.calibration = local_cal
            manager = getattr(self, "calib_manager", None)
            if manager is not None:
                manager.set(source_id, local_cal)
            if database is self.database:
                path = self._get_calibration_save_path()
                if path:
                    try:
                        save_layer_calibration(local_cal, path, source_id)
                    except Exception as exc:
                        logger.warning(f"Local map is saved in HDF5; JSON save failed: {exc}")
                self.map_widget.set_coordinate_mode("LOCAL", reset=True)
            if hasattr(self, "_after_layer_data_changed"):
                self._after_layer_data_changed()
            self.status_bar.showMessage("Локальна карта 0–100 готова. Координати в умовних одиницях.")
            if database is self.database:
                self.on_verify_propagation()
            return

        # Status first: the table must show the new state even while the report is open.
        if hasattr(self, "_after_layer_data_changed"):
            self._after_layer_data_changed()

        num_frames = database.get_num_frames()
        valid_mask = database.frame_valid
        valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0

        avg_rmse = 0.0
        max_rmse = 0.0
        avg_dis = 0.0
        avg_matches = 0.0

        if valid_count > 0:
            rmse_data = getattr(database, "frame_rmse", None)
            if rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                avg_rmse = float(np.mean(valid_rmse))
                max_rmse = float(np.max(valid_rmse))

            dis_data = getattr(database, "frame_disagreement", None)
            if dis_data is not None:
                dis_valid = dis_data[valid_mask]
                if np.any(dis_valid > 0):
                    avg_dis = float(np.mean(dis_valid[dis_valid > 0]))

            matches_data = getattr(database, "frame_matches", None)
            if matches_data is not None:
                avg_matches = float(np.mean(matches_data[valid_mask]))

        rmse_thresh = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)

        # NOTE: frame_rmse from propagation is in reprojection pixels between frames, not meters
        report = (
            f"<b>Пропагація шару «{source_id}» завершена!</b><br><br>"
            f"Валідних кадрів: <b>{valid_count} / {num_frames}</b> ({valid_count / num_frames * 100:.1f}%)<br>"
            f"Середній RMSE матчингу: <b style='color:{'green' if avg_rmse < rmse_thresh * 0.5 else 'orange'}'>{avg_rmse:.3f} px</b><br>"
            f"Середній матчинг: <b>{avg_matches:.1f} точок</b><br>"
        )

        log_msg = (
            f"Пропагація шару '{source_id}' завершена. "
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
            f"Шар «{source_id}»: пропагація готова: {valid_count} к., "
            f"RMSE: {avg_rmse:.2f}px, Mat: {avg_matches:.0f}"
        )

        # Map check only for the layer that was actually propagated.
        if self.map_widget and database is self.database:
            self.on_verify_propagation()

    @pyqtSlot()
    def on_verify_propagation(self):
        """Visualizes propagation quality markers on the map."""
        if not self.database or not self.database.is_propagated:
            QMessageBox.warning(self, "Увага", "Дані пропагації не знайдено.")
            return

        try:
            self.map_widget.set_coordinate_mode(self.calibration.converter.mode)
            self.map_widget.clear_verification_markers()
            num_frames = self.database.get_num_frames()
            # Limit number of Leaflet markers to ~600 with uniform step
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

                    # Frame center
                    mx, my = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h / 2) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h / 2) + affine[1, 2],
                    )
                    lat_c, lon_c = self.calibration.converter.metric_to_gps(float(mx), float(my))

                    # Frame bottom
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
                            f"Verify Frame {i}: CENTER={lat_c:.6f},{lon_c:.6f} | "
                            f"BOTTOM={lat_b:.6f},{lon_b:.6f} | RMSE={rmse:.2f}px"
                        )

                    color = "green"
                    if rmse > 5.0 or dis > 10.0:
                        color = "red"
                    elif rmse > 2.0 or dis > 3.0:
                        color = "orange"

                    # Render frame center marker only
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
        self._propagation_target = None
        if hasattr(self, "_after_layer_data_changed"):
            self._after_layer_data_changed()
        logger.error(f"Propagation error: {error_msg}")
        QMessageBox.critical(self, "Помилка пропагації", error_msg)

    # ── Save / Load calibration ──────────────────────────────────────────────

    def _get_current_source_id(self) -> str:
        """Active layer id (DatabaseMixin._activate_source keeps it in sync)."""
        active = getattr(self, "active_source_id", None)
        if active:
            return active
        # Legacy fallback (no project / nothing activated yet): match the DB path.
        if not self.project_manager or not self.project_manager.is_loaded or not self.database:
            return "main"
        current_db = str(Path(self.database.db_path).resolve())
        project_dir = self.project_manager.project_dir
        for src_dict in self.project_manager.settings.video_sources or []:
            db_file = src_dict.get("database_file", "")
            if db_file and str((project_dir / db_file).resolve()) == current_db:
                return src_dict.get("source_id", "main")
        return "main"

    def _get_calibration_save_path(self) -> str | None:
        """calibration.json of the ACTIVE layer, or None if it cannot be determined.

        Resolved from the active layer's ``calibration_file`` — never guessed from
        the database path, and never silently redirected to the main layer's file
        (that fallback is how one layer's anchors ended up in another's file).
        """
        if not self.project_manager or not self.project_manager.is_loaded:
            return None
        source_id = self._get_current_source_id()
        settings = self.project_manager.settings
        src = settings.get_source(source_id) if settings else None
        if src is None or not src.calibration_file:
            logger.error(f"No calibration_file for active layer '{source_id}'")
            return None
        cal_path = self.project_manager.project_dir / src.calibration_file
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Calibration path: {cal_path} (layer='{source_id}')")
        return str(cal_path)

    @pyqtSlot()
    def on_save_calibration(self):
        if self._refuse_if_encrypted_project("Збереження калібрування"):
            return
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Немає даних для збереження.")
            return

        source_id = self._get_current_source_id()
        # Default path — active layer folder
        default_path = self._get_calibration_save_path() or "calibration.json"

        path, _ = QFileDialog.getSaveFileName(
            self, f"Зберегти калібрування шару «{source_id}»", default_path, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            # Stamped with the layer id: loading it into another layer warns.
            save_layer_calibration(self.calibration, path, source_id)
            n = len(self.calibration.anchors)
            self.status_bar.showMessage(
                f"Калібрування шару «{source_id}» збережено: {path} ({n} якорів)"
            )
            QMessageBox.information(
                self,
                "Збережено",
                f"Калібрування шару «{source_id}» збережено!\nЯкорів: {n}\nФайл: {path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{e}")

    def _calibration_load_issues(self, loaded: MultiAnchorCalibration, path: str, source_id: str):
        """Reasons to believe ``loaded`` belongs to another layer (empty = looks fine)."""
        issues = []
        owner = loaded.extra_metadata.get(CALIBRATION_OWNER_KEY)
        if owner and str(owner) != source_id:
            issues.append(f"Файл записаний для шару «{owner}».")

        if self.project_manager and self.project_manager.is_loaded:
            try:
                target = str(Path(path).resolve()).casefold()
                for src in self.project_manager.settings.source_configs():
                    if src.source_id == source_id or not src.calibration_file:
                        continue
                    other = self.project_manager.project_dir / src.calibration_file
                    if str(other.resolve()).casefold() == target:
                        issues.append(f"Це файл калібрування шару «{src.source_id}».")
            except OSError:
                pass

        if self.database is not None:
            n = self.database.get_num_frames()
            bad = sorted(a.frame_id for a in loaded.anchors if not 0 <= int(a.frame_id) < n)
            if n and bad:
                issues.append(
                    f"Якорі на кадрах {bad} виходять за межі БД шару «{source_id}» ({n} слотів)."
                )
        return issues

    @pyqtSlot()
    def on_load_calibration(self):
        source_id = self._get_current_source_id()
        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir)
            own = self._get_calibration_save_path()
            if own:
                default_dir = str(Path(own).parent)

        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Завантажити калібрування в шар «{source_id}»",
            default_dir,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            # Parse into a fresh object first: a wrong or broken file must not
            # wipe the anchors of the active layer.
            loaded = MultiAnchorCalibration()
            loaded.load(path)
            db_mode = getattr(getattr(self.database, "converter", None), "mode", None)
            if db_mode and (db_mode == "LOCAL") != (loaded.converter.mode == "LOCAL"):
                QMessageBox.warning(self, "Несумісні координати", "Локальну карту X/Y і GPS-калібрування не можна змішувати в одному шарі. Створіть окремий шар.")
                return

            issues = self._calibration_load_issues(loaded, path, source_id)
            if issues:
                reply = QMessageBox.warning(
                    self,
                    "Калібрування іншого шару?",
                    f"Цей файл, схоже, не належить шару «{source_id}»:\n\n"
                    + "\n".join(f"• {msg}" for msg in issues)
                    + f"\n\nВсе одно завантажити його в шар «{source_id}»?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            calib_manager = getattr(self, "calib_manager", None)
            if calib_manager is not None:
                calib_manager.set(source_id, loaded)
            self.calibration = loaded

            ids = [a.frame_id for a in self.calibration.anchors]
            propagated = self.database and self.database.is_propagated
            self.control_panel.update_status(f"Калібрування шару «{source_id}» завантажено")

            # Automatically save copy into current layer folder
            source_cal_path = self._get_calibration_save_path()
            copied_to_source = False
            # An encrypted copy is immutable: load into memory for viewing, but
            # never mirror the file into the project.
            if getattr(self.project_manager, "is_encrypted", False):
                source_cal_path = None
                logger.info("Encrypted project: calibration loaded for viewing, not persisted")
            if source_cal_path:
                norm_loaded = str(Path(path).resolve())
                norm_source = str(Path(source_cal_path).resolve())
                if norm_loaded != norm_source:
                    # Copy calibration file if loaded from external location
                    save_layer_calibration(self.calibration, source_cal_path, source_id)
                    copied_to_source = True
                    logger.info(
                        f"Calibration copied to layer '{source_id}' folder: {source_cal_path}"
                    )
                else:
                    logger.debug("Calibration loaded directly from source folder, no copy needed.")

            if hasattr(self, "_after_layer_data_changed"):
                self._after_layer_data_changed()

            copy_note = (
                f"\n\n📋 Також збережено у папці шару:\n{source_cal_path}"
                if copied_to_source
                else ""
            )
            self.status_bar.showMessage(
                f"Шар «{source_id}»: калібрування {len(ids)} якорів, кадри {ids}"
            )
            QMessageBox.information(
                self,
                "Успіх",
                f"Шар «{source_id}»: завантажено {len(ids)} якір(ів)!\nКадри: {ids}\n\n"
                f"{'✅ БД вже має дані пропагації.' if propagated else '⚠ Запустіть пропагацію.'}"
                f"{copy_note}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")
