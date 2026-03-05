import base64
import cv2
import numpy as np
import torch
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.localization.matcher import FeatureMatcher
from src.localization.localizer import Localizer
from src.workers.panorama_worker import PanoramaWorker

_MAX_DISPLAY_PX = 2048
_CROP_SIZE_MAX  = 800
_JPEG_QUALITY   = 80


class PanoramaMixin:

    @pyqtSlot()
    def on_generate_panorama(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Відео для панорами", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not video_path:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти панораму", "panorama.jpg", "Images (*.jpg *.png)"
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

    @pyqtSlot()
    def on_show_panorama(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Спочатку виконайте калібрування!")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть панораму", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Помилка", "Не вдалося прочитати зображення.")
            return

        self.status_bar.showMessage("Аналіз панорами... (10–20 секунд)")
        self.repaint()

        try:
            corners_gps = self._localize_panorama_corners(img)
            if corners_gps is None:
                return

            H, W = img.shape[:2]
            if max(W, H) > _MAX_DISPLAY_PX:
                scale = _MAX_DISPLAY_PX / max(W, H)
                img = cv2.resize(img, (int(W * scale), int(H * scale)))

            ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
            if not ok:
                raise RuntimeError("Не вдалося закодувати зображення")

            data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
            (lat_tl, lon_tl), (lat_tr, lon_tr), (lat_br, lon_br), (lat_bl, lon_bl) = corners_gps

            self.map_widget.set_panorama_overlay(
                data_url,
                lat_tl, lon_tl, lat_tr, lon_tr,
                lat_br, lon_br, lat_bl, lon_bl,
            )
            self.status_bar.showMessage("Панораму накладено на карту!")

        except Exception as e:
            self.logger.error(f"Panorama overlay failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося накласти панораму:\n{e}")

    def _localize_panorama_corners(self, img: np.ndarray):
        """
        Localizes 4 quarter-crops of the panorama on CPU,
        fits affine matrix, returns GPS corners or None.
        """
        H, W = img.shape[:2]

        # Find valid (non-black) bounding box
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1])
        if coords is None:
            QMessageBox.warning(self, "Помилка", "Зображення повністю чорне.")
            return None

        x, y, w, h = cv2.boundingRect(coords)
        crop_size = min(_CROP_SIZE_MAX, min(w, h) // 2)

        centers = [
            (x + w // 4,     y + h // 4),
            (x + 3*w // 4,   y + h // 4),
            (x + w // 4,     y + 3*h // 4),
            (x + 3*w // 4,   y + 3*h // 4),
        ]
        crops = []
        for cx, cy in centers:
            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)
            crops.append((img[y1:y1+crop_size, x1:x1+crop_size], x1, y1))

        # Move models to CPU temporarily to free VRAM
        device = self.model_manager.device
        sp = self.model_manager.load_superpoint().to('cpu')
        nv = self.model_manager.load_dinov2().to('cpu')
        lg = self.model_manager.load_lightglue().to('cpu')

        fe       = FeatureExtractor(sp, nv, device='cpu', config=self.config)
        matcher  = FeatureMatcher(lg, device='cpu')
        localizer = Localizer(self.database, fe, matcher, self.calibration, self.config)

        pts_pano, pts_metric = [], []
        try:
            for i, (crop, off_x, off_y) in enumerate(crops):
                self.status_bar.showMessage(f"Розпізнавання чверті {i+1}/4...")
                self.repaint()

                res = localizer.localize_frame(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if not res.get('success') or 'fov_polygon' in res is False:
                    continue

                ch, cw = crop.shape[:2]
                for (px, py), (lat, lon) in zip(
                    [(0, 0), (cw, 0), (cw, ch), (0, ch)],
                    res['fov_polygon']
                ):
                    pts_pano.append((px + off_x, py + off_y))
                    pts_metric.append(CoordinateConverter.gps_to_metric(lat, lon))
        finally:
            localizer.reset_cache()
            sp.to(device); nv.to(device); lg.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(pts_pano) < 3:
            QMessageBox.warning(self, "Помилка",
                "Замало точок для прив'язки панорами.")
            return None

        M, _ = cv2.estimateAffine2D(
            np.array(pts_pano, dtype=np.float32),
            np.array(pts_metric, dtype=np.float32),
        )
        if M is None:
            QMessageBox.warning(self, "Помилка", "Помилка розрахунку матриці панорами.")
            return None

        corners_px = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
        corners_m  = GeometryTransforms.apply_affine(corners_px, M)
        return [CoordinateConverter.metric_to_gps(*pt) for pt in corners_m]
