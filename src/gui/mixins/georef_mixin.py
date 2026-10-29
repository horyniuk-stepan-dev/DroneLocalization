"""GUI flow: georeference a local (no-GPS) map from ground control points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from config import get_cfg
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox, QProgressDialog
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.calibration.multi_calibration_manager import save_layer_calibration
from src.geometry.coordinates import CoordinateConverter
from src.gui.map_features import GEOGRAPHIC, LOCAL, MapFeature
from src.utils.logging_utils import get_logger

from src.database.local_georef import apply_local_georef, georef_info, reset_local_georef
from src.geometry.georef_fit import apply as apply_matrix
from src.geometry.georef_fit import required_rotation_sign
from src.geometry.georef_auto import consensus_similarity, select_probe_frames
from src.geometry.georef_graph import GRAPH, graph_from_layer

logger = get_logger(__name__)

GEOREF_PROJECTION = "WEB_MERCATOR"
_SIDECAR = "georef_gcps.json"
_PANORAMA = Path("panoramas") / "local_map.png"
_AUTO_PREFIX = "auto_gcp_"


class GeorefMixin:
    """Needs: database, calibration, map_widget, calib_manager, status_bar."""

    def _georef_sidecar(self) -> Path:
        return Path(self.database.db_path).parent / _SIDECAR

    def _local_affines(self) -> list[np.ndarray]:
        f = self.database.db_file
        grp = f["calibration"]
        name = "local_frame_affine" if "local_frame_affine" in grp else "frame_affine"
        valid = grp["frame_valid"][:].astype(bool)
        return list(grp[name][:][valid])

    def _local_graph(self):
        """Frame graph of the local solution for the non-rigid fit, or None."""
        try:
            grp = self.database.db_file["calibration"]
            name = "local_frame_affine" if "local_frame_affine" in grp else "frame_affine"
            w = int(self.database.metadata.get("frame_width", 0))
            h = int(self.database.metadata.get("frame_height", 0))
            return graph_from_layer(grp[name][:], grp["frame_valid"][:], w, h)
        except Exception as e:
            logger.warning(f"Non-rigid georeference unavailable (no frame graph): {e}")
            return None

    @pyqtSlot()
    def on_georeference_local_map(self):
        from src.gui.dialogs.georef_dialog import GeorefDialog, load_points, save_points

        if self._refuse_if_encrypted_project("Прив’язка до GPS"):
            return
        db = self.database
        if db is None or not getattr(db, "is_propagated", False):
            QMessageBox.warning(self, "Увага", "Спочатку побудуйте локальну карту без GPS.")
            return
        info = georef_info(db)
        if self.calibration.converter.mode != "LOCAL" and info is None:
            QMessageBox.warning(
                self,
                "Прив’язка до GPS",
                "Цей шар уже має GPS-калібрування. Прив’язка за точками — лише для "
                "локальної карти X/Y.",
            )
            return
        tw = getattr(self, "tracking_worker", None)
        if tw is not None and tw.isRunning():
            QMessageBox.warning(self, "Увага", "Зупиніть трекінг — він використовує цю базу.")
            return

        sidecar = self._georef_sidecar()
        points = load_points(sidecar, self.map_widget.features.features(LOCAL))
        if not points:
            QMessageBox.information(
                self,
                "Прив’язка до GPS",
                "На локальній карті ще немає точок.\n\nОберіть інструмент 📍 на карті, "
                "поставте 2–5 точок на впізнавані об’єкти по краях карти й відкрийте "
                "це вікно знову.",
            )
            return
        with db.lock:
            sign = required_rotation_sign(self._local_affines())
            graph = self._local_graph()
        converter = CoordinateConverter(GEOREF_PROJECTION)
        dlg = GeorefDialog(
            points, converter, sign, can_reset=info is not None, graph=graph, parent=self
        )
        result = dlg.exec()
        save_points(sidecar, points)  # keep typed coordinates even on cancel
        if dlg.deleted_ids:  # deleted in the dialog -> also off the map
            store = self.map_widget.features
            if any([store.remove(fid) for fid in dlg.deleted_ids]):
                self.map_widget._features_updated()
        if result != QDialog.DialogCode.Accepted:
            return
        if dlg.reset_requested:
            self._reset_georef(info)
            return
        if dlg.fit is None:
            return

        panorama_layout = self._local_panorama_layout() if info is None else None
        fit = dlg.fit
        gcps = dlg.gcp_records()
        rmse_m = float(np.sqrt(np.mean([g["residual_m"] ** 2 for g in gcps])))
        frame_mats, extra, mapper = None, {}, None
        if fit.kind == GRAPH:
            k = float(getattr(dlg, "ground_k", 1.0))
            frame_mats = {int(f): m for f, m in zip(fit.frame_ids, fit.frame_matrices, strict=True)}
            extra = {
                "stiffness": fit.stiffness,
                "max_correction_m": fit.max_correction * k,
                "scale_range_rel": list(fit.scale_range),
                "graph_nodes": int(len(fit.frame_ids)),
            }
            if fit.loo_residuals is not None:
                extra["loo_rmse_m"] = float(np.sqrt(np.nanmean(fit.loo_residuals**2))) * k
            if fit.global_loo_residuals is not None:
                extra["global_loo_rmse_m"] = (
                    float(np.sqrt(np.nanmean(fit.global_loo_residuals**2))) * k
                )
            mapper = fit.map_points
        cal = self._georef_calibration(fit.matrix, converter, gcps, frame_mats)
        try:
            n = apply_local_georef(
                db,
                fit.matrix,
                converter,
                kind=fit.kind,
                gcps=gcps,
                rmse_m=rmse_m,
                anchors=[a.to_dict() for a in cal.anchors],
                frame_matrices=frame_mats,
                extra=extra,
            )
        except Exception as e:
            logger.error(f"Georeference failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося прив’язати карту:\n{e}")
            return

        self._install_calibration(cal)
        if info is not None:  # re-fit: bring earlier converted features back first
            self._move_features(np.asarray(info["matrix"]), converter, to_geo=False)
        if dlg.convert_features:
            self._move_features(fit.matrix, converter, to_geo=True, mapper=mapper)
        self.map_widget.set_coordinate_mode(GEOGRAPHIC, reset=True)
        if hasattr(self, "_after_layer_data_changed"):
            self._after_layer_data_changed()
        self.on_verify_propagation()
        self._overlay_georef_panorama(panorama_layout, fit.matrix, converter)
        self.status_bar.showMessage(
            f"Карту прив’язано до GPS: {n} кадрів, RMSE {rmse_m:.2f} м за {len(gcps)} точками"
        )

    # ── pieces ───────────────────────────────────────────────────────────────

    def _georef_calibration(self, M, converter, gcps, frame_mats=None) -> MultiAnchorCalibration:
        """A geographic calibration whose anchors are the georeferenced frames
        nearest to each control point (so status/staleness logic keeps working)."""
        from src.geometry.georef_fit import compose_affine

        grp = self.database.db_file["calibration"]
        name = "local_frame_affine" if "local_frame_affine" in grp else "frame_affine"
        local = grp[name][:]
        valid = grp["frame_valid"][:].astype(bool)
        w = int(self.database.metadata.get("frame_width", 0))
        h = int(self.database.metadata.get("frame_height", 0))
        ids = np.flatnonzero(valid)
        centers = local[ids] @ np.array([w / 2.0, h / 2.0, 1.0])
        cal = MultiAnchorCalibration(converter)
        cal.set_frame_size(w, h)
        chosen = []
        for g in gcps:
            fid = int(ids[np.argmin(np.linalg.norm(centers - [g["x"], g["y"]], axis=1))])
            if fid in chosen:
                continue
            chosen.append(fid)
            cal.add_anchor(
                fid,
                compose_affine((frame_mats or {}).get(fid, M), local[fid]),
                {
                    "transform_type": "local_gcp_georef",
                    "projection_mode": converter.mode,
                    "rmse_m": float(g.get("residual_m", 0.0)),
                    "points_gps": [[g["lat"], g["lon"]]],
                    "notes": f"GCP '{g['name']}' on local map ({g['x']:.3f}, {g['y']:.3f})",
                },
            )
        return cal

    def _install_calibration(self, cal):
        source_id = self._get_current_source_id()
        self.calibration = cal
        manager = getattr(self, "calib_manager", None)
        if manager is not None:
            manager.set(source_id, cal)
        path = self._get_calibration_save_path()
        if path:
            try:
                save_layer_calibration(cal, path, source_id)
            except Exception as exc:
                logger.warning(f"Georeference is in HDF5; calibration JSON save failed: {exc}")

    def _move_features(self, M, converter, *, to_geo: bool, mapper=None):
        """Carry LOCAL map objects into GPS (``mapper``: local [x, y] -> metric,
        default the matrix ``M``) or back. The original local coordinates are
        kept in the sidecar, so going back is exact even for a non-rigid fit."""
        store = self.map_widget.features
        sidecar = self._georef_sidecar()
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {"version": 1, "points": []}
        M3 = np.vstack([np.asarray(M, float), [0, 0, 1]])
        s = float(np.sqrt(abs(np.linalg.det(M3[:2, :2]))))
        to_metric = mapper if mapper is not None else (lambda xy: apply_matrix(M, xy))
        moved = []
        if to_geo:
            originals = {}
            for f in store.features(LOCAL):
                originals[f.id] = {"coords": [list(p) for p in f.coords], "radius": f.radius}
                xy = np.array([[p[1], p[0]] for p in f.coords])
                mx = to_metric(xy)
                lat, lon = converter.metric_to_gps_array(mx[:, 0], mx[:, 1])
                f.coords = [[float(a), float(b)] for a, b in zip(lat, lon, strict=True)]
                if f.radius is not None:
                    k = converter.ground_scale_factor(float(lat[0]))
                    f.radius = f.radius * s * k
                f.mode = GEOGRAPHIC
                moved.append(f.id)
            data["moved_feature_ids"] = moved
            data["moved_feature_local"] = originals
        else:
            ids = set(data.get("moved_feature_ids", []))
            originals = data.get("moved_feature_local", {})
            Minv = np.linalg.inv(M3)[:2]
            for f in store.features(GEOGRAPHIC):
                if f.id not in ids:
                    continue
                orig = originals.get(f.id)
                if orig is not None:
                    f.coords = [[float(a), float(b)] for a, b in orig["coords"]]
                    f.radius = orig.get("radius")
                    f.mode = LOCAL
                    continue
                lat = np.array([p[0] for p in f.coords])
                lon = np.array([p[1] for p in f.coords])
                mx = np.array(
                    [converter.gps_to_metric(a, b) for a, b in zip(lat, lon, strict=True)]
                )
                xy = apply_matrix(Minv, mx)
                if f.radius is not None:
                    f.radius = f.radius / (s * converter.ground_scale_factor(float(lat[0])))
                f.coords = [[float(y), float(x)] for x, y in xy]
                f.mode = LOCAL
            data["moved_feature_ids"] = []
            data["moved_feature_local"] = {}
        sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        self.map_widget._features_updated()

    def _reset_georef(self, info):
        answer = QMessageBox.question(
            self,
            "Скинути прив’язку",
            "Повернути шар до локальної карти X/Y без GPS?\nВведені координати точок збережуться.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            reset_local_georef(self.database)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося скинути прив’язку:\n{e}")
            return
        converter = CoordinateConverter(GEOREF_PROJECTION)
        self._move_features(np.asarray(info["matrix"]), converter, to_geo=False)
        self._install_calibration(MultiAnchorCalibration(CoordinateConverter("LOCAL")))
        self.map_widget.set_coordinate_mode("LOCAL", reset=True)
        if hasattr(self, "_after_layer_data_changed"):
            self._after_layer_data_changed()
        self.on_verify_propagation()
        self.status_bar.showMessage("Прив’язку скинуто: шар знову локальна карта X/Y")

    # ── panorama ─────────────────────────────────────────────────────────────

    def _local_panorama_layout(self):
        try:
            from src.geometry.local_panorama import LocalPanoramaLayout

            return LocalPanoramaLayout.from_database(self.database)
        except Exception as e:
            logger.debug(f"No local panorama layout: {e}")
            return None

    def _overlay_georef_panorama(self, layout, M, converter):
        """Re-place the existing local panorama on the satellite map, if it
        belongs to the local map that was just georeferenced."""
        if layout is None:
            return
        path = Path(self.database.db_path).parent / _PANORAMA
        meta_path = Path(str(path) + ".json")
        try:
            from src.geometry.local_panorama import load_local_panorama_metadata

            corners_yx = np.asarray(load_local_panorama_metadata(path, layout))
            xy = corners_yx[:, ::-1]
            mx = apply_matrix(M, xy)
            lat, lon = converter.metric_to_gps_array(mx[:, 0], mx[:, 1])
            corners = np.column_stack([lat, lon]).tolist()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["georef_corners_latlon"] = corners
            meta["georef_created_at"] = (georef_info(self.database) or {}).get("created_at")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            self._display_panorama(str(path), corners)
        except Exception as e:
            logger.info(f"Local panorama not re-placed after georeference: {e}")

    # ── automatic control points ─────────────────────────────────────────────

    def _georef_reference_layers(self) -> list[tuple]:
        """(source_id, label, database, calibration) of propagated GPS layers."""
        mgr = getattr(self, "db_manager", None)
        cm = getattr(self, "calib_manager", None)
        if mgr is None or cm is None:
            return []
        current = self._get_current_source_id()
        out = []
        for sid in mgr.all_source_ids:  # property
            if sid == current:
                continue
            if sid not in cm:
                continue
            rdb = mgr.get_database(sid)
            if rdb is None or not getattr(rdb, "is_propagated", False):
                continue
            try:
                cal = cm.get(sid)
            except Exception:
                continue
            if cal is None or getattr(cal.converter, "mode", "LOCAL") == "LOCAL":
                continue
            src = mgr.get_source_config(sid)
            desc = getattr(src, "description", "") if src is not None else ""
            out.append((sid, f"{sid} — {desc}" if desc else sid, rdb, cal))
        return out

    @pyqtSlot()
    def on_auto_georeference(self):
        if self._refuse_if_encrypted_project("Автоприв’язка"):
            return
        db = self.database
        if db is None or not getattr(db, "is_propagated", False):
            QMessageBox.warning(self, "Увага", "Спочатку побудуйте локальну карту без GPS.")
            return
        if self.calibration.converter.mode != "LOCAL" or georef_info(db) is not None:
            QMessageBox.warning(
                self,
                "Автоприв’язка",
                "Автоприв’язка працює з локальною картою X/Y. Якщо шар уже прив’язано, "
                "спершу скиньте прив’язку у вікні «📍 Прив’язати локальну карту до GPS».",
            )
            return
        tw = getattr(self, "tracking_worker", None)
        if tw is not None and tw.isRunning():
            QMessageBox.warning(self, "Увага", "Зупиніть трекінг — він використовує модель.")
            return
        refs = self._georef_reference_layers()
        if not refs:
            QMessageBox.information(
                self,
                "Автоприв’язка",
                "У проєкті немає опорного шару з GPS (з калібруванням і пропагацією), "
                "наприклад карти із симулятора. Додайте його й увімкніть.",
            )
            return

        source = self._source_config(self._get_current_source_id()) if hasattr(
            self, "_source_config"
        ) else None
        video_path = db.metadata.get("source_path", "") or (
            getattr(source, "video_path", "") if source else ""
        )
        if not video_path or not Path(video_path).is_file():
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Вихідне відео локальної карти", str(video_path), "Video (*.mp4 *.avi *.mkv)"
            )
            if not video_path:
                return
        frame_step = int(db.metadata.get("frame_step", 0) or 0)
        if frame_step <= 0:
            QMessageBox.warning(self, "Автоприв’язка", "У базі немає frame_step — перебудуйте її.")
            return

        with db.lock:
            graph = self._local_graph()
            sign = required_rotation_sign(self._local_affines())
        if graph is None:
            QMessageBox.warning(self, "Автоприв’язка", "Не вдалося прочитати граф локальної карти.")
            return
        budget = int(get_cfg(self.config, "propagation.auto_georef_probes", 40))
        probes = select_probe_frames(graph.centers, graph.frame_ids, budget)

        localizers = []
        for ref_sid, _, ref_db, ref_cal in refs:
            conv = ref_cal.converter
            if not getattr(conv, "is_initialized", True) and getattr(conv, "reference_gps", None):
                conv.gps_to_metric(*conv.reference_gps[:2])
            try:
                localizers.append((ref_sid, self._build_localizer(ref_db, ref_cal)))
            except Exception as e:
                logger.error(f"Auto-georef localizer for {ref_sid} failed: {e}", exc_info=True)
        if not localizers:
            QMessageBox.critical(self, "Помилка", "Не вдалося підготувати жоден локалізатор.")
            return
        ref_names = ", ".join(sid for sid, _ in localizers)

        from src.workers.auto_georef_worker import AutoGeorefWorker

        worker = AutoGeorefWorker(localizers, video_path, probes, frame_step, parent=self)
        progress = QProgressDialog(
            f"Пошук {len(probes)} кадрів у шарах: {ref_names}…", "Скасувати", 0, 100, self
        )
        progress.setWindowTitle("Автоприв’язка")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(worker.stop)

        def on_progress(pct, msg):
            progress.setValue(pct)
            progress.setLabelText(msg)

        def on_failed(msg):
            progress.close()
            QMessageBox.critical(self, "Автоприв’язка", msg)

        def on_done(records):
            progress.close()
            self._on_auto_georef_done(records, graph, sign, ref_names)

        worker.progress.connect(on_progress)
        worker.failed.connect(on_failed)
        worker.completed.connect(on_done)
        worker.finished.connect(worker.deleteLater)
        self._auto_georef_worker = worker
        self.status_bar.showMessage(
            f"Автоприв’язка: {len(probes)} кадрів × {len(localizers)} шарів ({ref_names})"
        )
        worker.start()

    def _on_auto_georef_done(self, records, graph, sign, ref_names):
        self._auto_georef_worker = None
        converter = CoordinateConverter(GEOREF_PROJECTION)
        node = {int(f): i for i, f in enumerate(graph.frame_ids)}
        cands = [r for r in records if r.get("success") and int(r["frame_id"]) in node]
        if len(cands) < 2:
            QMessageBox.warning(
                self,
                "Автоприв’язка",
                f"Перевірено {len(records)} кадрів, знайдено в шарах ({ref_names}) лише "
                f"{len(cands)}.\nЦього замало: поставте точки вручну або додайте опорний шар.",
            )
            return
        src = graph.centers[[node[int(r["frame_id"])] for r in cands]]
        dst = np.array([converter.gps_to_metric(r["lat"], r["lon"]) for r in cands])
        k = float(converter.ground_scale_factor(float(np.mean([r["lat"] for r in cands]))))
        tol_m = float(get_cfg(self.config, "propagation.auto_georef_tolerance_m", 25.0))
        drift = float(get_cfg(self.config, "propagation.auto_georef_drift_frac", 0.05))
        cons = consensus_similarity(
            src,
            dst,
            rotation_sign=sign,
            tolerance=tol_m / k,
            drift_frac=drift,
            weights=[max(float(r.get("confidence", 0.0)), 0.05) for r in cands],
        )
        self._store_auto_points(cands, src, cons)
        per_layer = {}
        for r, ok in zip(cands, cons.inliers, strict=True):
            a = per_layer.setdefault(r.get("source_id") or "?", [0, 0])
            a[0] += 1
            a[1] += int(ok)
        lines = [
            f"Перевірено кадрів: {len(records)}; знайдено: {len(cands)}; "
            f"узгоджених між собою: {cons.n_inliers}.",
            "За шарами (знайдено / узгоджено): "
            + ", ".join(f"{sid} {a[0]}/{a[1]}" for sid, a in sorted(per_layer.items())),
            "Узгоджені додано як увімкнені контрольні точки «Авто …» (зелені), "
            "неузгоджені — вимкнені (червоні, «✗»).",
            "Далі відкриється вікно прив’язки: перевірте стовпчик «Перевірка» і застосуйте.",
        ]
        lines += [f"⚠ {w}" for w in cons.warnings]
        logger.info(
            f"Auto-georef vs [{ref_names}]: probes={len(records)} found={len(cands)} "
            f"inliers={cons.n_inliers}"
        )
        QMessageBox.information(self, "Автоприв’язка", "\n".join(lines))
        self.on_georeference_local_map()

    def _store_auto_points(self, cands, src, cons):
        """Replace previous automatic points (map + sidecar) with this run's."""
        from src.gui.dialogs.georef_dialog import ControlPoint, load_points, save_points

        store = self.map_widget.features
        for f in list(store.features(LOCAL)):
            if f.id.startswith(_AUTO_PREFIX):
                store.remove(f.id)
        sidecar = self._georef_sidecar()
        points = [p for p in load_points(sidecar, []) if not p.id.startswith(_AUTO_PREFIX)]
        for r, (x, y), ok in zip(cands, src, cons.inliers, strict=True):
            fid = int(r["frame_id"])
            pid = f"{_AUTO_PREFIX}{fid}"
            name = f"Авто {fid} · {r.get('source_id') or '?'}" + ("" if ok else " ✗")
            store.upsert(
                MapFeature(
                    "point",
                    [[float(y), float(x)]],
                    mode=LOCAL,
                    name=name,
                    color="#2ecc71" if ok else "#e74c3c",
                    id=pid,
                )
            )
            points.append(
                ControlPoint(
                    pid, name, float(x), float(y), f"{r['lat']:.7f}, {r['lon']:.7f}", bool(ok)
                )
            )
        save_points(sidecar, points)
        self.map_widget._features_updated()

