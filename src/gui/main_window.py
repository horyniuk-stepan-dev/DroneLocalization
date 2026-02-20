import gc

import numpy as np
import torch
from PyQt6.QtWidgets import (QMainWindow, QDockWidget, QStatusBar,
                             QFileDialog, QMessageBox, QProgressDialog)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap

from src.gui.widgets.video_widget import VideoWidget
from src.gui.widgets.map_widget import MapWidget
from src.gui.widgets.control_panel import ControlPanel
from src.gui.dialogs.new_mission_dialog import NewMissionDialog
from src.gui.dialogs.calibration_dialog import CalibrationDialog

from src.models.model_manager import ModelManager
from src.workers.database_worker import DatabaseGenerationWorker
from src.workers.tracking_worker import RealtimeTrackingWorker
from src.workers.calibration_propagation_worker import CalibrationPropagationWorker

from src.database.database_loader import DatabaseLoader
# ВИПРАВЛЕНО: використовуємо MultiAnchorCalibration замість GPSCalibration
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.localization.matcher import FeatureMatcher
from src.localization.localizer import Localizer
from src.utils.logging_utils import get_logger   # ВИПРАВЛЕНО: правильний шлях імпорту
from config.config import APP_CONFIG
from utils.image_utils import opencv_to_qpixmap
from workers.panorama_overlay_worker import PanoramaOverlayWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Topometric Localizer")
        self.setGeometry(100, 100, 1600, 900)
        self.logger = get_logger('MainWindow')

        self.current_database_path = None
        self.database = None
        # ВИПРАВЛЕНО: MultiAnchorCalibration замість GPSCalibration
        self.calibration = MultiAnchorCalibration()
        self.config = APP_CONFIG
        self.model_manager = ModelManager(config=APP_CONFIG)
        self.db_worker = None
        self.tracking_worker = None
        self.propagation_worker = None
        self._propagation_dialog = None   # QProgressDialog під час пропагації

        self.init_ui()

    def init_ui(self):
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        self.control_dock = QDockWidget("Панель управління", self)
        self.control_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.control_panel = ControlPanel(self.control_dock)
        self.control_dock.setWidget(self.control_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)

        self.map_dock = QDockWidget("Інтерактивна карта", self)
        self.map_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.map_widget = MapWidget(self.map_dock)
        self.map_dock.setWidget(self.map_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.map_dock)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.create_menu_bar()
        self.connect_signals()

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('Файл')
        file_menu.addAction('Вихід', self.close)

        calibration_menu = menubar.addMenu('Калібрування')
        calibration_menu.addAction('Додати якір...', self.on_calibrate)
        calibration_menu.addAction('Завантажити калібрування...', self.on_load_calibration)
        calibration_menu.addAction('Зберегти калібрування...', self.on_save_calibration)
        calibration_menu.addSeparator()
        calibration_menu.addAction('Запустити пропагацію вручну', self.on_run_propagation)

        view_menu = menubar.addMenu('Вигляд')
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())

    def connect_signals(self):
        self.control_panel.new_mission_clicked.connect(self.on_new_mission)
        self.control_panel.load_database_clicked.connect(self.on_load_database)
        self.control_panel.start_tracking_clicked.connect(self.on_start_tracking)
        self.control_panel.stop_tracking_clicked.connect(self.on_stop_tracking)
        self.control_panel.calibrate_clicked.connect(self.on_calibrate)
        self.control_panel.load_calibration_clicked.connect(self.on_load_calibration)
        self.control_panel.generate_panorama_clicked.connect(self.on_generate_panorama)
        self.control_panel.show_panorama_clicked.connect(self.on_show_panorama)
        self.control_panel.localize_image_clicked.connect(self.on_localize_image)

    # ──────────────────────────────────────────────────────────────────
    # Калібрування
    # ──────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_calibrate(self):
        if not self.database or self.database.db_file is None:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте або створіть базу даних!")
            return

        # ВИПРАВЛЕНО: Витягуємо frame_id з нових об'єктів AnchorCalibration
        existing_ids = [anchor.frame_id for anchor in self.calibration.anchors]

        dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=existing_ids,
            parent=self
        )
        dialog.anchor_added.connect(self.on_anchor_added)
        dialog.calibration_complete.connect(self.on_run_propagation)
        dialog.exec()



    @pyqtSlot(object)
    def on_anchor_added(self, anchor_data):
        try:
            points_2d = anchor_data.get('points_2d')
            points_gps = anchor_data.get('points_gps')
            frame_id = anchor_data.get('calib_frame_id')

            if not points_2d or not points_gps or len(points_2d) < 3:
                QMessageBox.warning(self, "Помилка", "Потрібно мінімум 3 точки для створення якоря!")
                return

            # Імпортуємо необхідні модулі для математики
            from src.geometry.coordinates import CoordinateConverter
            from src.geometry.transformations import GeometryTransforms
            import numpy as np

            # Переводимо екранні пікселі у numpy масив
            pts_2d_np = np.array(points_2d, dtype=np.float32)
            pts_metric = []

            # Переводимо GPS широту/довготу у фізичні метри
            for lat, lon in points_gps:
                x, y = CoordinateConverter.gps_to_metric(lat, lon)
                pts_metric.append((x, y))

            pts_metric_np = np.array(pts_metric, dtype=np.float32)

            # Розраховуємо афінну матрицю для цього кадру
            M, inliers = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)

            if M is None:
                QMessageBox.critical(self, "Помилка",
                                     "Не вдалося обчислити математичну матрицю з цих точок. Спробуйте розставити їх ширше!")
                return

            # Додаємо вже готовий якір у нову систему
            self.calibration.add_anchor(frame_id=frame_id, affine_matrix=M)

            # Зберігаємо оновлений файл калібрування
            if self.database and self.database.db_path:
                import os
                calib_path = self.database.db_path.replace('.h5', '_calib.json')
                self.calibration.save(calib_path)

            self.status_bar.showMessage(f"Якір для кадру {frame_id} успішно створено та збережено!")

        except Exception as e:
            self.logger.error(f"Failed to add anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося додати якір:\n{e}")

    @pyqtSlot()
    def on_run_propagation(self):
        """
        Запускає CalibrationPropagationWorker — хвильова пропагація
        GPS від усіх якорів на всі кадри бази.
        """
        if not self.calibration.is_calibrated:
            QMessageBox.warning(
                self, "Увага", "Додайте хоча б один якір калібрування!"
            )
            return

        if not self.database:
            QMessageBox.warning(
                self, "Увага", "База даних не завантажена!"
            )
            return

        # Завантажуємо matcher для пропагації
        try:
            lg_model = self.model_manager.load_lightglue()
            matcher = FeatureMatcher(lg_model, self.model_manager.device)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити LightGlue:\n{e}")
            return

        n_anchors = len(self.calibration.anchors)

        # ВИПРАВЛЕНО: Збираємо ID з нових об'єктів якорів
        anchor_ids = [a.frame_id for a in self.calibration.anchors]

        n_frames = self.database.get_num_frames()

        self.logger.info(
            f"Starting propagation: {n_anchors} anchors "
            f"at frames {anchor_ids}, total {n_frames} frames"
        )

        # Прогрес-діалог
        self._propagation_dialog = QProgressDialog(
            f"Пропагація GPS від {n_anchors} якорів на {n_frames} кадрів...",
            "Скасувати", 0, 100, self
        )
        self._propagation_dialog.setWindowTitle("Розповсюдження GPS координат")
        self._propagation_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._propagation_dialog.setMinimumDuration(0)
        self._propagation_dialog.setValue(0)

        self.propagation_worker = CalibrationPropagationWorker(
            database=self.database,
            calibration=self.calibration,
            matcher=matcher,
            config=self.config
        )
        self.propagation_worker.progress.connect(self.on_propagation_progress)
        self.propagation_worker.completed.connect(self.on_propagation_completed)
        self.propagation_worker.error.connect(self.on_propagation_error)

        self._propagation_dialog.canceled.connect(self.propagation_worker.stop)

        self.propagation_worker.start()


    @pyqtSlot(int, str)
    def on_worker_progress(self, percent: int, message: str):
        """Оновлює діалогове вікно прогресу під час роботи фонових потоків"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.setValue(percent)
            self.progress_dialog.setLabelText(message)
        else:
            self.status_bar.showMessage(f"{message} ({percent}%)")

    @pyqtSlot(str)
    def on_worker_error(self, error_msg: str):
        """Обробляє помилки, що надходять від фонових потоків"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.close()

        self.logger.error(f"Worker Error: {error_msg}")
        QMessageBox.critical(self, "Помилка обробки", f"Сталася помилка:\n{error_msg}")
        self.status_bar.showMessage("Операцію перервано через помилку.")


    @pyqtSlot()
    def on_propagation_finished(self):
        self.progress_dialog.close()
        self.status_bar.showMessage("Пропагація завершена успішно. Базу оновлено.")

        # Примусово очищуємо кеш відеокарти після важких обчислень LightGlue
        self.model_manager.unload_model('lightglue')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


    @pyqtSlot(int, str)
    def on_propagation_progress(self, percent: int, message: str):
        # Робимо локальну копію, щоб уникнути конфлікту під час закриття вікна
        dialog = self._propagation_dialog
        if dialog is not None:
            try:
                # Спочатку ставимо текст, потім значення (щоб уникнути автозакриття на 100%)
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

        n_valid = int(self.database.frame_valid.sum()) if self.database.frame_valid is not None else 0
        n_total = self.database.get_num_frames()

        self.status_bar.showMessage(
            f"✅ Пропагація завершена: {n_valid}/{n_total} кадрів мають GPS координати"
        )

        # Пропонуємо зберегти калібрування
        reply = QMessageBox.question(
            self, "Пропагація завершена",
            f"GPS успішно розповсюджено на {n_valid}/{n_total} кадрів!\n\n"
            f"Зберегти файл калібрування (якорі)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.on_save_calibration()

        # Скидаємо runtime-кеш localizer якщо він запущений
        if self.tracking_worker and self.tracking_worker.isRunning():
            if hasattr(self.tracking_worker, 'localizer'):
                self.tracking_worker.localizer.reset_cache()

    @pyqtSlot(str)
    def on_propagation_error(self, error_msg: str):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None
        self.logger.error(f"Propagation error: {error_msg}")
        QMessageBox.critical(self, "Помилка пропагації", error_msg)

    # ──────────────────────────────────────────────────────────────────
    # Збереження / завантаження калібрування
    # ──────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_save_calibration(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Немає даних для збереження.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти калібрування", "calibration.json", "JSON Files (*.json)"
        )
        if save_path:
            try:
                self.calibration.save(save_path)
                n = len(self.calibration.anchors)
                self.status_bar.showMessage(f"Калібрування збережено: {save_path} ({n} якорів)")
                QMessageBox.information(
                    self, "Збережено",
                    f"Калібрування збережено!\nЯкорів: {n}\nФайл: {save_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{e}")

    @pyqtSlot()
    def on_load_calibration(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Завантажити калібрування", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_name:
            return
        try:
            self.calibration.load(file_name)
            n = len(self.calibration.anchors)

            # ВИПРАВЛЕНО: Збираємо ID з нових об'єктів якорів
            ids = [a.frame_id for a in self.calibration.anchors]

            self.status_bar.showMessage(
                f"Калібрування завантажено: {n} якорів, кадри {ids}"
            )
            self.control_panel.update_status("Калібрування завантажено")
            QMessageBox.information(
                self, "Успіх",
                f"Завантажено {n} якір(ів)!\n"
                f"Кадри: {ids}\n\n"
                f"{'✅ База вже містить дані пропагації.' if self.database and self.database.is_propagated else '⚠ Запустіть пропагацію для точних результатів.'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")
    # ──────────────────────────────────────────────────────────────────
    # Бази даних
    # ──────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_new_mission(self):
        dialog = NewMissionDialog(self)
        if dialog.exec():
            mission_data = dialog.get_mission_data()
            video_path = mission_data.get('video_path')
            if not video_path:
                QMessageBox.warning(self, "Помилка", "Будь ласка, виберіть еталонне відео.")
                return
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Зберегти базу HDF5", "", "HDF5 Files (*.h5 *.hdf5)"
            )
            if save_path:
                self.start_database_generation(video_path, save_path, mission_data)
        else:
            self.status_bar.showMessage("Створення місії скасовано")

    def start_database_generation(self, video_path, save_path, config):
        self.control_panel.btn_new_mission.setEnabled(False)
        self.control_panel.btn_load_db.setEnabled(False)
        self.control_panel.update_progress(0)

        self.db_worker = DatabaseGenerationWorker(
            video_path=video_path,
            output_path=save_path,
            model_manager=self.model_manager,
            config=APP_CONFIG
        )
        self.db_worker.progress.connect(self.on_db_progress)
        self.db_worker.completed.connect(self.on_db_completed)
        self.db_worker.error.connect(self.on_db_error)
        self.db_worker.start()

    @pyqtSlot(int, str)
    def on_db_progress(self, percent, message):
        self.control_panel.update_progress(percent)
        self.control_panel.update_status(message)

    @pyqtSlot(str)
    def on_db_completed(self, db_path):
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.current_database_path = db_path
        self.database = DatabaseLoader(db_path)
        self.control_panel.update_progress(100)
        self.control_panel.update_status("Базу успішно створено")
        self.status_bar.showMessage(f"Завантажено нову базу: {db_path}")
        QMessageBox.information(self, "Успіх", "Базу даних успішно згенеровано!")

    @pyqtSlot(str)
    def on_db_error(self, error_msg):
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.control_panel.update_progress(0)
        self.control_panel.update_status("Помилка генерації")
        QMessageBox.critical(self, "Помилка", f"Помилка під час генерації:\n{error_msg}")

    @pyqtSlot()
    def on_load_database(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Виберіть базу HDF5", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if file_name:
            self.current_database_path = file_name
            try:
                self.database = DatabaseLoader(file_name)

                # Повідомлення про стан пропагації
                if self.database.is_propagated:
                    n_valid = int(self.database.frame_valid.sum())
                    n_total = self.database.get_num_frames()
                    self.status_bar.showMessage(
                        f"База завантажена: {file_name} "
                        f"(GPS: {n_valid}/{n_total} кадрів)"
                    )
                else:
                    self.status_bar.showMessage(
                        f"База завантажена: {file_name} (без даних GPS пропагації)"
                    )

                self.control_panel.update_status("База завантажена")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити базу:\n{e}")
        else:
            self.status_bar.showMessage("Вибір бази скасовано")

    # ──────────────────────────────────────────────────────────────────
    # Відстеження
    # ──────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_start_tracking(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated:
            QMessageBox.warning(
                self, "Увага",
                "Система не відкалібрована. Виконайте калібрування GPS."
            )
            return
        if not self.database.is_propagated:
            reply = QMessageBox.question(
                self, "Увага",
                "Пропагація GPS ще не виконана.\n"
                "Точність локалізації буде знижена.\n\n"
                "Продовжити без пропагації?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        video_path, _ = QFileDialog.getOpenFileName(
            self, "Відео з дрона", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if not video_path:
            return

        sp_model = self.model_manager.load_superpoint()
        nv_model = self.model_manager.load_netvlad()
        lg_model = self.model_manager.load_lightglue()

        feature_extractor = FeatureExtractor(sp_model, nv_model, self.model_manager.device)
        matcher = FeatureMatcher(lg_model, self.model_manager.device)
        localizer = Localizer(
            self.database, feature_extractor, matcher,
            self.calibration, config=self.config
        )

        self.tracking_worker = RealtimeTrackingWorker(
            video_path, localizer,
            model_manager=self.model_manager,
            config=self.config
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
            self.tracking_worker.wait()
            self.status_bar.showMessage("Відстеження зупинено")
            self.control_panel.update_status("Очікування")

    # ──────────────────────────────────────────────────────────────────
    # Локалізація одного фото
    # ──────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_localize_image(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Виконайте калібрування GPS.")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть зображення", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not image_path:
            return

        import cv2
        from PyQt6.QtWidgets import QApplication
        from src.utils.image_utils import opencv_to_qpixmap

        frame = cv2.imread(image_path)
        if frame is None:
            QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")
            return

        self.status_bar.showMessage("Обробка зображення...")
        self.control_panel.update_status("Локалізація фото...")
        QApplication.processEvents()

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            sp_model = self.model_manager.load_superpoint()
            nv_model = self.model_manager.load_netvlad()
            lg_model = self.model_manager.load_lightglue()

            feature_extractor = FeatureExtractor(sp_model, nv_model, self.model_manager.device)
            matcher = FeatureMatcher(lg_model, self.model_manager.device)
            localizer = Localizer(
                self.database, feature_extractor, matcher,
                self.calibration, config=APP_CONFIG
            )

            result = localizer.localize_frame(frame_rgb)

            if hasattr(self.video_widget, 'display_frame'):
                self.video_widget.display_frame(opencv_to_qpixmap(frame))

            if result.get("success"):
                lat, lon = result["lat"], result["lon"]
                conf = result["confidence"]
                inliers = result.get("inliers", 0)
                anchor = result.get("anchor_frame", "?")

                self.map_widget.update_marker(lat, lon)
                if "fov_polygon" in result:
                    self.map_widget.update_fov(result["fov_polygon"])

                self.status_bar.showMessage(
                    f"Локалізація: {lat:.6f}, {lon:.6f} | "
                    f"Впевненість: {conf:.2f} | Точок: {inliers} | "
                    f"Якір: кадр {anchor}"
                )
                self.control_panel.update_status("Фото локалізовано")
                QMessageBox.information(
                    self, "Успіх",
                    f"Координати знайдено!\n\n"
                    f"Широта: {lat:.6f}\nДовгота: {lon:.6f}\n"
                    f"Впевненість: {conf:.2f}\nТочок збігу: {inliers}\n"
                    f"Використаний якір: кадр {anchor}"
                )
            else:
                err = result.get('error', 'Невідома помилка')
                self.status_bar.showMessage(f"Помилка: {err}")
                self.control_panel.update_status("Помилка локалізації")
                QMessageBox.warning(self, "Помилка", f"Не вдалося знайти координати:\n{err}")

        except Exception as e:
            self.logger.error(f"Image localization error: {e}", exc_info=True)
            QMessageBox.critical(self, "Критична помилка", str(e))
            self.status_bar.showMessage("Помилка обробки")

    # ──────────────────────────────────────────────────────────────────
    # Панорама
    # ──────────────────────────────────────────────────────────────────

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

        from src.workers.panorama_worker import PanoramaWorker
        self.pano_worker = PanoramaWorker(video_path, save_path, frame_step=20)
        self.control_panel.btn_gen_pano.setEnabled(False)
        self.pano_worker.progress.connect(self.on_db_progress)

        def on_complete(path):
            self.control_panel.btn_gen_pano.setEnabled(True)
            self.status_bar.showMessage(f"Панораму збережено: {path}")
            QMessageBox.information(self, "Успіх", "Панораму успішно створено!")

        def on_error(err):
            self.control_panel.btn_gen_pano.setEnabled(True)
            QMessageBox.critical(self, "Помилка", err)

        self.pano_worker.completed.connect(on_complete)
        self.pano_worker.error.connect(on_error)
        self.pano_worker.start()

    @pyqtSlot()
    def on_show_panorama(self):
        if not self.localizer:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте базу даних!")
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Виберіть зображення панорами", "", "Images (*.jpg *.png *.jpeg)"
        )
        if not file_name:
            return

        self.status_bar.showMessage("Розпізнавання панорами у фоновому режимі...")

        # Запускаємо процес у фоновому потоці замість блокування інтерфейсу
        self.panorama_overlay_worker = PanoramaOverlayWorker(file_name, self.localizer)
        self.panorama_overlay_worker.success.connect(self.on_panorama_overlay_success)
        self.panorama_overlay_worker.error.connect(self.on_worker_error)
        self.panorama_overlay_worker.start()

    @pyqtSlot(str, float, float, float, float, float, float, float, float)
    def on_panorama_overlay_success(self, data_url, lat_tl, lon_tl, lat_tr, lon_tr, lat_br, lon_br, lat_bl, lon_bl):
        self.map_widget.set_panorama_overlay(
            data_url, lat_tl, lon_tl, lat_tr, lon_tr, lat_br, lon_br, lat_bl, lon_bl
        )
        self.status_bar.showMessage("Панорама успішно розпізнана і накладена на карту!")

    @pyqtSlot()
    def on_show_panorama1(self):
        if not self.calibration.is_calibrated or not getattr(self.calibration, 'anchors', None):
            QMessageBox.warning(self, "Увага", "Спочатку виконайте калібрування бази!")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть панораму", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not image_path: return

        import cv2
        import base64
        import numpy as np
        import torch
        from src.geometry.coordinates import CoordinateConverter
        from src.geometry.transformations import GeometryTransforms

        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.critical(self, "Помилка", "Не вдалося прочитати зображення.")
            return

        H_pano, W_pano = img.shape[:2]
        self.logger.info(f"Панорама завантажена: {W_pano}x{H_pano} пікселів")
        self.status_bar.showMessage("Аналіз панорами в оперативній пам'яті (це може зайняти 10-20 секунд)...")
        self.repaint()

        try:
            # 1. Готуємо 3 шматки панорами
            crop_size = min(H_pano, W_pano)
            crops = []

            if W_pano > H_pano:  # Горизонтальна панорама
                crops.append((img[0:crop_size, 0:crop_size], 0, 0))
                cx = W_pano // 2 - crop_size // 2
                crops.append((img[0:crop_size, cx:cx + crop_size], cx, 0))
                crops.append((img[0:crop_size, W_pano - crop_size:W_pano], W_pano - crop_size, 0))
            else:  # Вертикальна панорама
                crops.append((img[0:crop_size, 0:crop_size], 0, 0))
                cy = H_pano // 2 - crop_size // 2
                crops.append((img[cy:cy + crop_size, 0:crop_size], 0, cy))
                crops.append((img[H_pano - crop_size:H_pano, 0:crop_size], 0, H_pano - crop_size))

            # --- ПЕРЕКИДАЄМО МОДЕЛІ В ОПЕРАТИВНУ ПАМ'ЯТЬ (CPU) ---
            from src.models.wrappers.feature_extractor import FeatureExtractor
            from src.localization.matcher import FeatureMatcher
            from src.localization.localizer import Localizer

            original_device = self.model_manager.device  # Запам'ятовуємо, де ми були (cuda)

            # Переносимо моделі на процесор (щоб використовувати всю RAM комп'ютера)
            sp_model = self.model_manager.load_superpoint().to('cpu')
            nv_model = self.model_manager.load_netvlad().to('cpu')
            lg_model = self.model_manager.load_lightglue().to('cpu')

            # Ініціалізуємо екстрактори з примусовим вказуванням 'cpu'
            fe = FeatureExtractor(sp_model, nv_model, device='cpu')
            matcher = FeatureMatcher(lg_model, device='cpu')
            localizer = Localizer(self.database, fe, matcher, self.calibration, self.config)

            pts_pano = []
            pts_metric = []

            localizer.reset_cache()

            # 2. Відправляємо ОРИГІНАЛЬНІ ВЕЛИЧЕЗНІ шматки в локалізатор
            for i, (crop_img, offset_x, offset_y) in enumerate(crops):
                self.status_bar.showMessage(f"Обробка шматка {i + 1}/3 в оперативній пам'яті...")
                self.repaint()

                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                res = localizer.localize_frame(crop_rgb)

                if res.get('success') and 'fov_polygon' in res:
                    crop_pts = [(0, 0), (crop_size, 0), (crop_size, crop_size), (0, crop_size)]
                    for (cx, cy), (lat, lon) in zip(crop_pts, res['fov_polygon']):
                        pano_x = cx + offset_x
                        pano_y = cy + offset_y
                        mx, my = CoordinateConverter.gps_to_metric(lat, lon)
                        pts_pano.append((pano_x, pano_y))
                        pts_metric.append((mx, my))

            # --- ПОВЕРТАЄМО ВСЕ НА ВІДЕОКАРТУ ---
            localizer.reset_cache()
            sp_model.to(original_device)
            nv_model.to(original_device)
            lg_model.to(original_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 3. Перевіряємо результати
            if len(pts_pano) < 3:
                QMessageBox.warning(self, "Помилка",
                                    "Не вдалося автоматично прив'язати панораму. Зображення занадто спотворене або не збігається з базою.")
                self.status_bar.clearMessage()
                return

            # 4. Будуємо ідеальну афінну матрицю
            pts_pano_np = np.array(pts_pano, dtype=np.float32)
            pts_metric_np = np.array(pts_metric, dtype=np.float32)
            M, _ = cv2.estimateAffine2D(pts_pano_np, pts_metric_np)

            if M is None:
                QMessageBox.warning(self, "Помилка", "Математична помилка при розрахунку матриці панорами.")
                return

            # 5. Знаходимо координати кутів
            corners_pano = np.array([
                [0, 0], [W_pano, 0], [W_pano, H_pano], [0, H_pano]
            ], dtype=np.float32)
            corners_metric = GeometryTransforms.apply_affine(corners_pano, M)

            lat_tl, lon_tl = CoordinateConverter.metric_to_gps(*corners_metric[0])
            lat_tr, lon_tr = CoordinateConverter.metric_to_gps(*corners_metric[1])
            lat_br, lon_br = CoordinateConverter.metric_to_gps(*corners_metric[2])
            lat_bl, lon_bl = CoordinateConverter.metric_to_gps(*corners_metric[3])

            self.logger.info(f"Авто-GPS кути панорами: TL({lat_tl:.4f}, {lon_tl:.4f}), BR({lat_br:.4f}, {lon_br:.4f})")

            # 6. Стискаємо картинку суто для ВІДОБРАЖЕННЯ на карті (щоб браузер не завис)
            max_dim = 2048
            if W_pano > max_dim or H_pano > max_dim:
                scale = max_dim / max(W_pano, H_pano)
                img = cv2.resize(img, (int(W_pano * scale), int(H_pano * scale)))

            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_string = base64.b64encode(buffer).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{b64_string}"

            self.map_widget.set_panorama_overlay(
                data_url,
                lat_tl, lon_tl,
                lat_tr, lon_tr,
                lat_br, lon_br,
                lat_bl, lon_bl
            )
            self.status_bar.showMessage("Панорама успішно розпізнана і накладена на карту!")

        except Exception as e:
            self.logger.error(f"Panorama auto-overlay failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося накласти панораму:\n{e}")

    @pyqtSlot()
    def on_show_panorama(self):
        if not self.calibration.is_calibrated or not getattr(self.calibration, 'anchors', None):
            QMessageBox.warning(self, "Увага", "Спочатку виконайте калібрування бази!")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть панораму", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not image_path: return

        import cv2
        import base64
        import numpy as np
        import torch
        from src.geometry.coordinates import CoordinateConverter
        from src.geometry.transformations import GeometryTransforms

        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.critical(self, "Помилка", "Не вдалося прочитати зображення.")
            return

        H_pano, W_pano = img.shape[:2]
        self.logger.info(f"Панорама завантажена: {W_pano}x{H_pano} пікселів")
        self.status_bar.showMessage("Аналіз панорами в оперативній пам'яті (це може зайняти 10-20 секунд)...")
        self.repaint()

        try:
            # 1. Знаходимо реальні межі зображення (ігноруємо чорні краї)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)

            if coords is None:
                QMessageBox.warning(self, "Помилка", "Зображення повністю чорне.")
                return

            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(coords)
            self.logger.info(f"Корисна зона панорами: x={x_bb}, y={y_bb}, w={w_bb}, h={h_bb}")

            # 2. Розраховуємо центри 4-х чвертей (щоб точно не зачепити краї)
            centers = [
                (x_bb + w_bb // 4, y_bb + h_bb // 4),  # Верхня ліва чверть
                (x_bb + 3 * w_bb // 4, y_bb + h_bb // 4),  # Верхня права чверть
                (x_bb + w_bb // 4, y_bb + 3 * h_bb // 4),  # Нижня ліва чверть
                (x_bb + 3 * w_bb // 4, y_bb + 3 * h_bb // 4)  # Нижня права чверть
            ]

            # Беремо безпечний розмір шматка (максимум 800 пікселів, щоб працювало швидко)
            crop_size = min(800, min(w_bb, h_bb) // 2)

            crops = []
            for cx, cy in centers:
                x1 = max(0, cx - crop_size // 2)
                y1 = max(0, cy - crop_size // 2)
                x2 = min(W_pano, x1 + crop_size)
                y2 = min(H_pano, y1 + crop_size)

                crop_img = img[y1:y2, x1:x2]
                crops.append((crop_img, x1, y1))

            # --- ПЕРЕКИДАЄМО МОДЕЛІ В ОПЕРАТИВНУ ПАМ'ЯТЬ (CPU) ---
            from src.models.wrappers.feature_extractor import FeatureExtractor
            from src.localization.matcher import FeatureMatcher
            from src.localization.localizer import Localizer

            original_device = self.model_manager.device  # Запам'ятовуємо, де ми були (cuda)

            # Переносимо моделі на процесор (використовуємо RAM)
            sp_model = self.model_manager.load_superpoint().to('cpu')
            nv_model = self.model_manager.load_netvlad().to('cpu')
            lg_model = self.model_manager.load_lightglue().to('cpu')

            # Ініціалізуємо екстрактори з примусовим вказуванням 'cpu'
            fe = FeatureExtractor(sp_model, nv_model, device='cpu')
            matcher = FeatureMatcher(lg_model, device='cpu')
            localizer = Localizer(self.database, fe, matcher, self.calibration, self.config)

            pts_pano = []
            pts_metric = []

            localizer.reset_cache()

            # 3. Відправляємо 4 шматки в локалізатор
            for i, (crop_img, offset_x, offset_y) in enumerate(crops):
                self.status_bar.showMessage(f"Розпізнавання чверті {i + 1}/4 в оперативній пам'яті...")
                self.repaint()

                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                res = localizer.localize_frame(crop_rgb)

                if res.get('success') and 'fov_polygon' in res:
                    h_c, w_c = crop_img.shape[:2]
                    crop_pts = [(0, 0), (w_c, 0), (w_c, h_c), (0, h_c)]

                    for (cx_pt, cy_pt), (lat, lon) in zip(crop_pts, res['fov_polygon']):
                        pano_x = cx_pt + offset_x
                        pano_y = cy_pt + offset_y
                        mx, my = CoordinateConverter.gps_to_metric(lat, lon)

                        pts_pano.append((pano_x, pano_y))
                        pts_metric.append((mx, my))

            # --- ПОВЕРТАЄМО ВСЕ НА ВІДЕОКАРТУ ---
            localizer.reset_cache()
            sp_model.to(original_device)
            nv_model.to(original_device)
            lg_model.to(original_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4. Перевіряємо результати
            if len(pts_pano) < 3:
                QMessageBox.warning(self, "Помилка",
                                    "Не вдалося автоматично прив'язати панораму. Знайдено замало точок.")
                self.status_bar.clearMessage()
                return

            # 5. Будуємо ідеальну афінну матрицю
            pts_pano_np = np.array(pts_pano, dtype=np.float32)
            pts_metric_np = np.array(pts_metric, dtype=np.float32)
            M, _ = cv2.estimateAffine2D(pts_pano_np, pts_metric_np)

            if M is None:
                QMessageBox.warning(self, "Помилка", "Математична помилка при розрахунку матриці панорами.")
                return

            # 6. Знаходимо координати кутів ВСІЄЇ панорами
            corners_pano = np.array([
                [0, 0], [W_pano, 0], [W_pano, H_pano], [0, H_pano]
            ], dtype=np.float32)
            corners_metric = GeometryTransforms.apply_affine(corners_pano, M)

            lat_tl, lon_tl = CoordinateConverter.metric_to_gps(*corners_metric[0])
            lat_tr, lon_tr = CoordinateConverter.metric_to_gps(*corners_metric[1])
            lat_br, lon_br = CoordinateConverter.metric_to_gps(*corners_metric[2])
            lat_bl, lon_bl = CoordinateConverter.metric_to_gps(*corners_metric[3])

            self.logger.info(f"Авто-GPS кути панорами: TL({lat_tl:.4f}, {lon_tl:.4f}), BR({lat_br:.4f}, {lon_br:.4f})")

            # 7. Стискаємо картинку суто для ВІДОБРАЖЕННЯ на веб-карті (щоб браузер не завис)
            max_dim = 2048
            if W_pano > max_dim or H_pano > max_dim:
                scale = max_dim / max(W_pano, H_pano)
                img = cv2.resize(img, (int(W_pano * scale), int(H_pano * scale)))

            _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_string = base64.b64encode(buffer).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{b64_string}"

            self.map_widget.set_panorama_overlay(
                data_url,
                lat_tl, lon_tl,
                lat_tr, lon_tr,
                lat_br, lon_br,
                lat_bl, lon_bl
            )
            self.status_bar.showMessage("Панорама успішно розпізнана і накладена на карту!")

        except Exception as e:
            self.logger.error(f"Panorama auto-overlay failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося накласти панораму:\n{e}")
    # ──────────────────────────────────────────────────────────────────
    # Слоти відео/локалізації
    # ──────────────────────────────────────────────────────────────────

    @pyqtSlot(np.ndarray)
    def on_frame_ready(self, frame_rgb):
        # Конвертуємо безпечно у головному потоці
        if hasattr(self.video_widget, 'display_frame'):
            pixmap = opencv_to_qpixmap(frame_rgb)
            self.video_widget.display_frame(pixmap)

    @pyqtSlot(float, float, float, int)
    def on_location_found(self, lat, lon, confidence, inliers):
        self.map_widget.update_marker(lat, lon)
        self.map_widget.add_trajectory_point(lat, lon)
        self.status_bar.showMessage(
            f"Локалізація: {lat:.6f}, {lon:.6f} | "
            f"Впевненість: {confidence:.2f} | Точок: {inliers}"
        )