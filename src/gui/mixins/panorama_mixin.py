import base64

import cv2
import numpy as np
import torch
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.localization.localizer import Localizer
from src.localization.matcher import FeatureMatcher
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.workers.panorama_worker import PanoramaWorker

_MAX_DISPLAY_PX = 2048
_CROP_SIZE_MAX = 800
_JPEG_QUALITY = 80


from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PanoramaMixin:
    @pyqtSlot()
    def on_generate_panorama(self):
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

    @pyqtSlot()
    def on_show_panorama(self):
        if not (self.calibration.is_calibrated or getattr(self.database, "is_propagated", False)):
            QMessageBox.warning(self, "Увага", "Спочатку виконайте калібрування!")
            return

        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir / "panoramas")

        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть панораму", default_dir, "Images (*.png *.jpg *.jpeg);;All Files (*)"
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

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
            if not ok:
                raise RuntimeError("Не вдалося закодувати зображення")

            data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
            (lat_tl, lon_tl), (lat_tr, lon_tr), (lat_br, lon_br), (lat_bl, lon_bl) = corners_gps

            self.map_widget.set_panorama_overlay(
                data_url,
                lat_tl,
                lon_tl,
                lat_tr,
                lon_tr,
                lat_br,
                lon_br,
                lat_bl,
                lon_bl,
            )
            self.status_bar.showMessage("Панораму накладено на карту!")

        except Exception as e:
            logger.error(f"Panorama overlay failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося накласти панораму:\n{e}")

    def _localize_panorama_corners(self, img: np.ndarray):
        """
        Localizes 4 quarter-crops of the panorama on CPU,
        fits affine matrix, returns GPS corners or None.
        """
        H, W = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1])
        if coords is None:
            QMessageBox.warning(self, "Помилка", "Зображення повністю чорне.")
            return None

        x, y, w, h = cv2.boundingRect(coords)
        crop_size = min(_CROP_SIZE_MAX, min(w, h) // 2)

        # Обчислюємо відстань від кожного пікселя до чорного фону
        # Це допоможе нам вибрати центри, які знаходяться глибоко всередині зображення
        dist = cv2.distanceTransform(
            cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1], cv2.DIST_L2, 5
        )

        # Визначаємо "безпечну зону", де можна брати центр кропу, щоб він (майже) не захоплював чорні краї
        # Якщо панорама тонка, беремо хоча б 80% від її максимальної товщини
        safe_dist = min(crop_size // 2, dist.max() * 0.8)
        safe_mask = dist >= safe_dist

        safe_y, safe_x = np.where(safe_mask > 0)

        if len(safe_x) == 0:
            QMessageBox.warning(self, "Помилка", "Панорама занадто тонка для аналізу.")
            return None

        # Цільові ідеальні 4 кути рамки
        target_corners = [
            (x, y),  # Top-Left
            (x + w, y),  # Top-Right
            (x, y + h),  # Bottom-Left
            (x + w, y + h),  # Bottom-Right
        ]

        # Формуємо 4 центри, знаходячи найближчу "безпечну" точку до кожного ідеального кута
        centers = []
        for tx, ty in target_corners:
            dists_sq = (safe_x - tx) ** 2 + (safe_y - ty) ** 2
            best_idx = np.argmin(dists_sq)
            centers.append((safe_x[best_idx], safe_y[best_idx]))

        crops = []
        for cx, cy in centers:
            # Зміщуємо так, щоб центр crop співпадав з cx, cy (або якомога ближче, враховуючи межі)
            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)

            # Коригуємо межі, щоб не вилізти за межі зображення
            x1 = min(x1, W - crop_size)
            y1 = min(y1, H - crop_size)

            # Якщо розмір менший за crop_size (наприклад зображення мале)
            x1, y1 = max(0, x1), max(0, y1)

            crops.append((img[y1 : y1 + crop_size, x1 : x1 + crop_size], x1, y1))

        device = self.model_manager.device
        xf = self.model_manager.load_aliked()
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
        )

        pts_pano, pts_metric = [], []
        try:
            for i, (crop, off_x, off_y) in enumerate(crops):
                self.status_bar.showMessage(f"Розпізнавання чверті {i + 1}/4...")
                self.repaint()

                res = localizer.localize_frame(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if not res.get("success") or "fov_polygon" not in res:
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
