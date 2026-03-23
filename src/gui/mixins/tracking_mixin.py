import numpy as np
import cv2
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication, QPushButton

from src.models.wrappers.feature_extractor import FeatureExtractor
from src.localization.matcher import FeatureMatcher
from src.localization.localizer import Localizer
from src.workers.tracking_worker import RealtimeTrackingWorker
from src.utils.image_utils import opencv_to_qpixmap


from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrackingMixin:

    def _build_localizer(self) -> Localizer:
        """Shared factory — used by tracking and single-image localization."""
        # ОНОВЛЕНО: Завантажуємо ALIKED та DINOv2
        xf = self.model_manager.load_aliked()
        nv = self.model_manager.load_dinov2()

        # Опціональне завантаження CESP для покращення DINOv2 global descriptors
        cesp = None
        if self.config.get('models', {}).get('cesp', {}).get('enabled', False):
            try:
                cesp = self.model_manager.load_cesp()
            except Exception as e:
                logger.warning(f"CESP loading failed, continuing without it: {e}")

        fe = FeatureExtractor(xf, nv, self.model_manager.device, config=self.config, cesp_module=cesp)

        # ОНОВЛЕНО: Матчер сам вирішить (Numpy для XFeat або LightGlue для SuperPoint)
        matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)

        # Передаємо model_manager у конфіг для SuperPoint+LightGlue fallback
        localizer_config = {**self.config, '_model_manager': self.model_manager}
        return Localizer(self.database, fe, matcher, self.calibration, config=localizer_config)

    def _ensure_utm_initialized(self) -> bool:
        """Перевіряє чи ініціалізована проєкція UTM, якщо ні - пробує ініціалізувати з калібрування."""
        from src.geometry.coordinates import CoordinateConverter
        if CoordinateConverter._initialized:
            return True
        
        if self.calibration and self.calibration.reference_gps:
            CoordinateConverter.gps_to_metric(self.calibration.reference_gps[0], self.calibration.reference_gps[1])
            return True
            
        QMessageBox.warning(
            self, "Помилка формату", 
            "Проєкція UTM не ініціалізована.\n\n"
            "Схоже, що база даних створена у старій версії програми, або не була завантажена GPS-прив'язка.\n"
            "Будь ласка, завантажте файл калібрування (.json) або виконайте додавання GPS-якорів наново."
        )
        return False

    @pyqtSlot()
    def on_start_tracking(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated and not (self.database and self.database.is_propagated):
            QMessageBox.warning(self, "Увага", "Виконайте калібрування GPS або завантажте базу з пропагацією.")
            return
        if not self.database.is_propagated:
            reply = QMessageBox.question(
                self, "Увага",
                "Пропагація GPS ще не виконана.\nТочність буде знижена.\n\nПродовжити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir / "test_videos")

        video_path, _ = QFileDialog.getOpenFileName(
            self, "Відео з дрона", default_dir, "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not video_path:
            return

        if not self._ensure_utm_initialized():
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
        self.tracking_worker.finished.connect(self._on_tracking_finished)

        self.map_widget.clear_trajectory()
        self._tracking_results = []  # Ініціалізуємо список результатів
        
        self.control_panel.set_tracking_enabled(False)
        self.tracking_worker.start()
        self.status_bar.showMessage("Відстеження розпочато")

    @pyqtSlot()
    def on_stop_tracking(self):
        if hasattr(self, 'tracking_worker') and self.tracking_worker and self.tracking_worker.isRunning():
            self.control_panel.update_status("Зупинка...")
            self.tracking_worker.stop()
            # НЕ чекаємо тут — finished сигнал прийде сам

    @pyqtSlot()
    def _on_tracking_finished(self):
        """Викликається коли воркер завершує роботу (сам або через зупинку)."""
        logger.info("Tracking worker finished.")
        self.control_panel.set_tracking_enabled(True)
        self.status_bar.showMessage("Відстеження зупинено")
        self.control_panel.update_status("Очікування")

    @pyqtSlot()
    def on_localize_image(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated and not (self.database and self.database.is_propagated):
            QMessageBox.warning(self, "Увага", "Виконайте калібрування GPS або завантажте базу з пропагацією.")
            return

        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir / "test_photos")

        path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть зображення", default_dir, "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")
            return
            
        if not self._ensure_utm_initialized():
            return

        self.status_bar.showMessage("Локалізація зображення...")
        self.control_panel.update_status("Локалізація фото...")

        try:
            localizer = self._build_localizer()
            result = localizer.localize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if hasattr(self.video_widget, 'display_frame'):
                self.video_widget.display_frame(opencv_to_qpixmap(frame))

            if result.get("success"):
                lat, lon = result["lat"], result["lon"]
                conf = result["confidence"]
                inliers = result.get("inliers", 0)
                anchor = result.get("matched_frame", "?")  # ВИПРАВЛЕНО КЛЮЧ

                self.map_widget.update_marker(lat, lon)
                is_fallback = result.get("fallback_mode") == "retrieval_only"
                status_prefix = "ПРИБЛИЗНА Локалізація" if is_fallback else "Локалізація"
                
                if "fov_polygon" in result and result["fov_polygon"] is not None:
                    self.map_widget.update_fov(result["fov_polygon"])

                self.status_bar.showMessage(
                    f"{status_prefix}: {lat:.6f}, {lon:.6f} | "
                    f"Впевненість: {conf:.2f} | Точок: {inliers} | Якір: {anchor}"
                )
                self.control_panel.update_status("Фото локалізовано (приблизно)" if is_fallback else "Фото локалізовано")
                msg_box = QMessageBox(self)
                msg_title = "⚓ Приблизна локалізація" if is_fallback else "⚓ Успіх"
                msg_text = (f"Знайдено ПРИБЛИЗНІ координати (retrieval-only)!\n\n" if is_fallback else f"Координати знайдено!\n\n")
                msg_text += (f"Широта: {lat:.6f}\nДовгота: {lon:.6f}\n"
                                 f"Впевненість: {conf:.2f}\nТочок збігу: {inliers}\n"
                                 f"Якір: кадр {anchor}")
                
                msg_box.setWindowTitle(msg_title)
                msg_box.setText(msg_text)
                msg_box.setIcon(QMessageBox.Icon.Information)
                
                # Default OK button
                ok_btn = msg_box.addButton(QMessageBox.StandardButton.Ok)
                
                # Custom Copy button
                copy_btn = msg_box.addButton("📋 Копіювати координати", QMessageBox.ButtonRole.ActionRole)
                
                msg_box.exec()
                
                if msg_box.clickedButton() == copy_btn:
                    cb = QApplication.clipboard()
                    cb.setText(f"{lat:.6f}, {lon:.6f}")

            else:
                err = result.get('error', 'Невідома помилка')
                self.status_bar.showMessage(f"Помилка: {err}")
                self.control_panel.update_status("Помилка локалізації")
                QMessageBox.warning(self, "Помилка", f"Не вдалося знайти координати:\n{err}")

        except Exception as e:
            logger.error(f"Image localization error: {e}", exc_info=True)
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
        
        # Зберігаємо результат для експорту
        if not hasattr(self, '_tracking_results'): self._tracking_results = []
        self._tracking_results.append({'lat': lat, 'lon': lon, 'confidence': confidence, 'inliers': inliers, 'timestamp': str(np.datetime64('now'))})
        if len(self._tracking_results) == 1: self.control_panel.btn_export.setEnabled(True)
        self.status_bar.showMessage(
            f"Локалізація: {lat:.6f}, {lon:.6f} | "
            f"Впевненість: {confidence:.2f} | Точок: {inliers}"
        )