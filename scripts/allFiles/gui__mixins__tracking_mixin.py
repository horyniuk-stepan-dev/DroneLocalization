import numpy as np
import cv2
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QFileDialog

from src.models.wrappers.feature_extractor import FeatureExtractor
from src.localization.matcher import FeatureMatcher
from src.localization.localizer import Localizer
from src.workers.tracking_worker import RealtimeTrackingWorker
from src.utils.image_utils import opencv_to_qpixmap


class TrackingMixin:

    def _build_localizer(self) -> Localizer:
        """Shared factory — used by tracking and single-image localization."""
        sp = self.model_manager.load_superpoint()
        nv = self.model_manager.load_dinov2()
        lg = self.model_manager.load_lightglue()
        fe      = FeatureExtractor(sp, nv, self.model_manager.device, config=self.config)
        matcher = FeatureMatcher(lg, self.model_manager.device)
        return Localizer(self.database, fe, matcher, self.calibration, config=self.config)

    @pyqtSlot()
    def on_start_tracking(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Виконайте калібрування GPS.")
            return
        if not self.database.is_propagated:
            reply = QMessageBox.question(
                self, "Увага",
                "Пропагація GPS ще не виконана.\nТочність буде знижена.\n\nПродовжити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        video_path, _ = QFileDialog.getOpenFileName(
            self, "Відео з дрона", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not video_path:
            return

        localizer = self._build_localizer()
        self.tracking_worker = RealtimeTrackingWorker(
            video_path, localizer,
            model_manager=self.model_manager,
            config=self.config,
        )
        self.tracking_worker.frame_ready.connect(self.on_frame_ready)
        self.tracking_worker.location_found.connect(self.on_location_found)
        self.tracking_worker.status_update.connect(self.control_panel.update_status)
        self.tracking_worker.fov_found.connect(self.map_widget.update_fov)
        self.map_widget.clear_trajectory()
        self.tracking_worker.start()
        self.status_bar.showMessage("Відстеження розпочато")

    @pyqtSlot()
    def on_stop_tracking(self):
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
            self.status_bar.showMessage("Відстеження зупинено")
            self.control_panel.update_status("Очікування")

    @pyqtSlot()
    def on_localize_image(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Виконайте калібрування GPS.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть зображення", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")
            return

        self.status_bar.showMessage("Локалізація зображення...")
        self.control_panel.update_status("Локалізація фото...")

        try:
            localizer = self._build_localizer()
            result = localizer.localize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if hasattr(self.video_widget, 'display_frame'):
                self.video_widget.display_frame(opencv_to_qpixmap(frame))

            if result.get("success"):
                lat, lon  = result["lat"], result["lon"]
                conf      = result["confidence"]
                inliers   = result.get("inliers", 0)
                anchor    = result.get("anchor_frame", "?")

                self.map_widget.update_marker(lat, lon)
                if "fov_polygon" in result:
                    self.map_widget.update_fov(result["fov_polygon"])

                self.status_bar.showMessage(
                    f"Локалізація: {lat:.6f}, {lon:.6f} | "
                    f"Впевненість: {conf:.2f} | Точок: {inliers} | Якір: {anchor}"
                )
                self.control_panel.update_status("Фото локалізовано")
                QMessageBox.information(self, "Успіх",
                    f"Координати знайдено!\n\n"
                    f"Широта: {lat:.6f}\nДовгота: {lon:.6f}\n"
                    f"Впевненість: {conf:.2f}\nТочок збігу: {inliers}\n"
                    f"Якір: кадр {anchor}")
            else:
                err = result.get('error', 'Невідома помилка')
                self.status_bar.showMessage(f"Помилка: {err}")
                self.control_panel.update_status("Помилка локалізації")
                QMessageBox.warning(self, "Помилка", f"Не вдалося знайти координати:\n{err}")

        except Exception as e:
            self.logger.error(f"Image localization error: {e}", exc_info=True)
            QMessageBox.critical(self, "Критична помилка", str(e))
            self.status_bar.showMessage("Помилка обробки")

    @pyqtSlot(np.ndarray)
    def on_frame_ready(self, frame_rgb: np.ndarray):
        if hasattr(self.video_widget, 'display_frame'):
            self.video_widget.display_frame(opencv_to_qpixmap(frame_rgb))

    @pyqtSlot(float, float, float, int)
    def on_location_found(self, lat: float, lon: float, confidence: float, inliers: int):
        self.map_widget.update_marker(lat, lon)
        self.map_widget.add_trajectory_point(lat, lon)
        self.status_bar.showMessage(
            f"Локалізація: {lat:.6f}, {lon:.6f} | "
            f"Впевненість: {confidence:.2f} | Точок: {inliers}"
        )
