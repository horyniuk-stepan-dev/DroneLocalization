import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from config import get_cfg
from src.geometry.local_panorama import LocalPanoramaLayout, load_local_panorama_metadata
from src.geometry.transformations import GeometryTransforms
from src.localization.localizer import Localizer
from src.localization.matcher import FeatureMatcher
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.workers.local_panorama_worker import LocalPanoramaWorker
from src.workers.panorama_worker import PanoramaWorker

_MAX_DISPLAY_PX = 2048
_CROP_SIZE_MAX = 800
_JPEG_QUALITY = 80


from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PanoramaMixin:
    @pyqtSlot()
    def on_generate_panorama(self):
        if getattr(self, "pano_worker", None) is not None and self.pano_worker.isRunning():
            return
        if self.calibration.converter.mode == "LOCAL":
            self._generate_local_panorama()
            return
        default_video = ""
        default_save = "panorama.jpg"

        if self.project_manager and self.project_manager.is_loaded:
            default_video = self.project_manager.settings.video_path
            default_save = str(self.project_manager.project_dir / "panoramas" / "panorama.jpg")

        video_path, _ = QFileDialog.getOpenFileName(
            self, "Відео для панорами", default_video, "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not video_path:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти панораму", default_save, "Images (*.jpg *.png)"
        )
        if not save_path:
            return

        self.pano_worker = PanoramaWorker(video_path, save_path, frame_step=20)
        self.control_panel.btn_gen_pano.setEnabled(False)
        self.pano_worker.progress.connect(self.on_db_progress)

        def on_complete(path: str):
            self.control_panel.btn_gen_pano.setEnabled(True)
            self.status_bar.showMessage(f"Панораму збережено: {path}")
            QMessageBox.information(self, "Успіх", "Панораму успішно створено!")

        def on_error(err: str):
            self.control_panel.btn_gen_pano.setEnabled(True)
            QMessageBox.critical(self, "Помилка", err)

        self.pano_worker.completed.connect(on_complete)
        self.pano_worker.error.connect(on_error)
        self.pano_worker.start()

    def _generate_local_panorama(self):
        if self.database is None or not self.database.is_propagated:
            QMessageBox.warning(
                self, "Локальна панорама", "Спочатку побудуйте локальну карту цього шару."
            )
            return
        if self._layer_ops_busy("Панорама локальної карти"):
            return
        try:
            layout = LocalPanoramaLayout.from_database(self.database)
            source_id = self._get_current_source_id()
            source = self._source_config(source_id)
            video_path = self.database.metadata.get("source_path", "") or (
                source.video_path if source else ""
            )
            if not video_path or not Path(video_path).is_file():
                video_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Вихідне відео активного шару",
                    str(video_path),
                    "Video (*.mp4 *.avi *.mkv)",
                )
                if not video_path:
                    return
            default_path = Path(self.database.db_path).parent / "panoramas" / "local_map.png"
            path, _ = QFileDialog.getSaveFileName(
                self, "Зберегти локальну панораму", str(default_path), "PNG (*.png)"
            )
            if not path:
                return
            if not path.lower().endswith(".png"):
                path += ".png"
            worker = LocalPanoramaWorker(
                video_path, path, layout, self.database.metadata.get("source_sha256", "")
            )
        except Exception as exc:
            QMessageBox.critical(self, "Локальна панорама", str(exc))
            return
        self.pano_worker = worker
        self.control_panel.btn_gen_pano.setEnabled(False)
        worker.progress.connect(self.on_db_progress)
        worker.finished.connect(lambda: self.control_panel.btn_gen_pano.setEnabled(True))

        def completed(saved_path):
            # A user may have selected another layer while rendering. Never put
            # the old layer's image on the new layer, or on a rebuilt local map.
            try:
                if (
                    self._get_current_source_id() == source_id
                    and self.database is not None
                    and self.calibration.converter.mode == "LOCAL"
                    and LocalPanoramaLayout.from_database(self.database).map_signature
                    == layout.map_signature
                ):
                    self._display_panorama(saved_path, layout.corners_yx)
                else:
                    self.status_bar.showMessage(
                        f"Панораму збережено: {saved_path}. Відкрийте її у вихідному шарі."
                    )
            except Exception as exc:
                QMessageBox.warning(self, "Відображення панорами", str(exc))

        worker.completed.connect(completed)
        worker.error.connect(
            lambda message: QMessageBox.critical(self, "Локальна панорама", message)
        )
        worker.start()

    def _display_panorama(self, path, corners=None):
        # Preserve transparent areas in a generated map mosaic.
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Не вдалося прочитати панораму")
        if corners is None:
            color = (
                img
                if img.ndim == 3 and img.shape[2] == 3
                else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if img.ndim == 3 else cv2.COLOR_GRAY2BGR)
            )
            corners = self._localize_panorama_corners(color)
            if corners is None:
                return
        h, w = img.shape[:2]
        if max(w, h) > _MAX_DISPLAY_PX:
            scale = _MAX_DISPLAY_PX / max(w, h)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        transparent = img.ndim == 3 and img.shape[2] == 4
        ext = ".png" if transparent else ".jpg"
        ok, buf = cv2.imencode(
            ext, img, [] if transparent else [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
        )
        if not ok:
            raise ValueError("Не вдалося закодувати панораму")
        mime = "png" if transparent else "jpeg"
        data_url = f"data:image/{mime};base64," + base64.b64encode(buf).decode("ascii")
        self.map_widget.set_coordinate_mode(self.calibration.converter.mode)
        self.map_widget.set_panorama_overlay(data_url, *np.asarray(corners).ravel().tolist())
        self.status_bar.showMessage("Панораму накладено на карту!")

    @pyqtSlot()
    def on_show_panorama(self):
        if not (self.calibration.is_calibrated or getattr(self.database, "is_propagated", False)):
            QMessageBox.warning(self, "Увага", "Спочатку виконайте калібрування!")
            return

        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir / "panoramas")
        if self.calibration.converter.mode == "LOCAL" and self.database is not None:
            default_dir = str(Path(self.database.db_path).parent / "panoramas")

        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть панораму", default_dir, "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if not path:
            return

        self.status_bar.showMessage("Завантаження та прив’язка панорами...")
        self.repaint()

        try:
            corners = None
            if self.calibration.converter.mode == "LOCAL" and Path(path + ".json").exists():
                layout = LocalPanoramaLayout.from_database(self.database)
                corners = load_local_panorama_metadata(path, layout)
            self._display_panorama(path, corners)

        except Exception as e:
            logger.error(f"Panorama overlay failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося накласти панораму:\n{e}")

    def _localize_panorama_corners(self, img: np.ndarray):
        """
        Localizes 4 quarter-crops, fits an affine and returns (y,x) or GPS corners.
        """
        H, W = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1])
        if coords is None:
            QMessageBox.warning(self, "Помилка", "Зображення повністю чорне.")
            return None

        x, y, w, h = cv2.boundingRect(coords)
        crop_size = min(_CROP_SIZE_MAX, min(w, h) // 2)

        # Compute distance from each pixel to the black background;
        # used to select crop centres that lie deep inside the panorama.
        dist = cv2.distanceTransform(
            cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1], cv2.DIST_L2, 5
        )

        # Define a "safe zone" for crop centres that avoids black borders;
        # if the panorama is thin, allow at least 80% of its max thickness.
        safe_dist = min(crop_size // 2, dist.max() * 0.8)
        safe_mask = dist >= safe_dist

        safe_y, safe_x = np.where(safe_mask > 0)

        if len(safe_x) == 0:
            QMessageBox.warning(self, "Error", "Panorama is too thin for analysis.")
            return None

        # Target ideal 4 corners of the bounding rect
        target_corners = [
            (x, y),  # Top-Left
            (x + w, y),  # Top-Right
            (x, y + h),  # Bottom-Left
            (x + w, y + h),  # Bottom-Right
        ]

        # Find the nearest safe point to each target corner
        centers = []
        for tx, ty in target_corners:
            dists_sq = (safe_x - tx) ** 2 + (safe_y - ty) ** 2
            best_idx = np.argmin(dists_sq)
            centers.append((safe_x[best_idx], safe_y[best_idx]))

        crops = []
        for cx, cy in centers:
            # Offset so crop centre aligns with (cx, cy), clamped to image bounds
            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)

            # Clamp to image bounds
            x1 = min(x1, W - crop_size)
            y1 = min(y1, H - crop_size)

            # Clamp in case the image is smaller than crop_size
            x1, y1 = max(0, x1), max(0, y1)

            crops.append((img[y1 : y1 + crop_size, x1 : x1 + crop_size], x1, y1))

        device = self.model_manager.device
        xf = self.model_manager.load_local_extractor()
        nv = self.model_manager.load_dinov2()

        cesp = None
        if get_cfg(self.config, "models.cesp.enabled", False):
            try:
                cesp = self.model_manager.load_cesp()
            except Exception:
                pass

        fe = FeatureExtractor(xf, nv, device=device, config=self.config, cesp_module=cesp)
        matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)
        localizer = Localizer(
            self.database,
            fe,
            matcher,
            self.calibration,
            {**self.config, "_model_manager": self.model_manager},
            ref_frame_width=int(self.database.metadata.get("frame_width", 0)),
            ref_frame_height=int(self.database.metadata.get("frame_height", 0)),
        )

        pts_pano, pts_metric = [], []
        try:
            for i, (crop, off_x, off_y) in enumerate(crops):
                self.status_bar.showMessage(f"Розпізнавання чверті {i + 1}/4...")
                self.repaint()

                localizer.reset_session()  # Crops are independent locations, not a flight.
                res = localizer.localize_frame(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if (
                    not res.get("success")
                    or not res.get("fov_polygon")
                    or res.get("fallback_mode") == "retrieval_only"
                ):
                    continue

                ch, cw = crop.shape[:2]
                for (px, py), (lat, lon) in zip(
                    [(0, 0), (cw, 0), (cw, ch), (0, ch)], res["fov_polygon"]
                ):
                    pts_pano.append((px + off_x, py + off_y))
                    pts_metric.append(self.calibration.converter.gps_to_metric(lat, lon))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(pts_pano) < 3:
            QMessageBox.warning(self, "Помилка", "Замало точок для прив'язки панорами.")
            return None

        M, _ = cv2.estimateAffine2D(
            np.array(pts_pano, dtype=np.float32),
            np.array(pts_metric, dtype=np.float32),
        )
        if M is None:
            QMessageBox.warning(self, "Помилка", "Помилка розрахунку матриці панорами.")
            return None

        corners_px = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        corners_m = GeometryTransforms.apply_affine(corners_px, M)
        return [
            self.calibration.converter.metric_to_gps(float(pt[0]), float(pt[1])) for pt in corners_m
        ]
