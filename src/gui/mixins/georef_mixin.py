"""GUI flow: georeference a local (no-GPS) map from ground control points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QDialog, QMessageBox
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.calibration.multi_calibration_manager import save_layer_calibration
from src.geometry.coordinates import CoordinateConverter
from src.gui.map_features import GEOGRAPHIC, LOCAL
from src.utils.logging_utils import get_logger

from src.database.local_georef import apply_local_georef, georef_info, reset_local_georef
from src.geometry.georef_fit import apply as apply_matrix
from src.geometry.georef_fit import required_rotation_sign

logger = get_logger(__name__)

GEOREF_PROJECTION = "WEB_MERCATOR"
_SIDECAR = "georef_gcps.json"
_PANORAMA = Path("panoramas") / "local_map.png"


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
        converter = CoordinateConverter(GEOREF_PROJECTION)
        dlg = GeorefDialog(points, converter, sign, can_reset=info is not None, parent=self)
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
        cal = self._georef_calibration(fit.matrix, converter, gcps)
        try:
            n = apply_local_georef(
                db,
                fit.matrix,
                converter,
                kind=fit.kind,
                gcps=gcps,
                rmse_m=rmse_m,
                anchors=[a.to_dict() for a in cal.anchors],
            )
        except Exception as e:
            logger.error(f"Georeference failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося прив’язати карту:\n{e}")
            return

        self._install_calibration(cal)
        if info is not None:  # re-fit: bring earlier converted features back first
            self._move_features(np.asarray(info["matrix"]), converter, to_geo=False)
        if dlg.convert_features:
            self._move_features(fit.matrix, converter, to_geo=True)
        self.map_widget.set_coordinate_mode(GEOGRAPHIC, reset=True)
        if hasattr(self, "_after_layer_data_changed"):
            self._after_layer_data_changed()
        self.on_verify_propagation()
        self._overlay_georef_panorama(panorama_layout, fit.matrix, converter)
        self.status_bar.showMessage(
            f"Карту прив’язано до GPS: {n} кадрів, RMSE {rmse_m:.2f} м за {len(gcps)} точками"
        )

    # ── pieces ───────────────────────────────────────────────────────────────

    def _georef_calibration(self, M, converter, gcps) -> MultiAnchorCalibration:
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
                compose_affine(M, local[fid]),
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

    def _move_features(self, M, converter, *, to_geo: bool):
        store = self.map_widget.features
        sidecar = self._georef_sidecar()
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {"version": 1, "points": []}
        M3 = np.vstack([np.asarray(M, float), [0, 0, 1]])
        s = float(np.sqrt(abs(np.linalg.det(M3[:2, :2]))))
        moved = []
        if to_geo:
            for f in store.features(LOCAL):
                xy = np.array([[p[1], p[0]] for p in f.coords])
                mx = apply_matrix(M, xy)
                lat, lon = converter.metric_to_gps_array(mx[:, 0], mx[:, 1])
                f.coords = [[float(a), float(b)] for a, b in zip(lat, lon, strict=True)]
                if f.radius is not None:
                    k = converter.ground_scale_factor(float(lat[0]))
                    f.radius = f.radius * s * k
                f.mode = GEOGRAPHIC
                moved.append(f.id)
            data["moved_feature_ids"] = moved
        else:
            ids = set(data.get("moved_feature_ids", []))
            Minv = np.linalg.inv(M3)[:2]
            for f in store.features(GEOGRAPHIC):
                if f.id not in ids:
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
