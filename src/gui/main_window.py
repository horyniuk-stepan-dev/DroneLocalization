from PyQt6.QtWidgets import QMainWindow, QDockWidget, QStatusBar, QFileDialog, QMessageBox
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

from src.database.database_loader import DatabaseLoader
from src.calibration.gps_calibration import GPSCalibration
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.localization.matcher import FeatureMatcher
from src.localization.localizer import Localizer
from utils.logging_utils import get_logger
from config.config import APP_CONFIG

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Topometric Localizer")
        self.setGeometry(100, 100, 1600, 900)
        self.logger = get_logger('MainWindow')

        self.current_database_path = None
        self.database = None
        self.calibration = GPSCalibration()
        self.config = APP_CONFIG
        self.model_manager = ModelManager(config = APP_CONFIG)
        self.db_worker = None
        self.tracking_worker = None

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
        mission_menu = menubar.addMenu('Місія')
        view_menu = menubar.addMenu('Вигляд')
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())

    def connect_signals(self):
        self.control_panel.new_mission_clicked.connect(self.on_new_mission)
        self.control_panel.load_database_clicked.connect(self.on_load_database)
        self.control_panel.start_tracking_clicked.connect(self.on_start_tracking)
        self.control_panel.stop_tracking_clicked.connect(self.on_stop_tracking)
        self.control_panel.calibrate_clicked.connect(self.on_calibrate)
        self.control_panel.localize_image_clicked.connect(self.on_localize_image)
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
                self, "Зберегти базу даних HDF5", "", "HDF5 Files (*.h5 *.hdf5)"
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
        QMessageBox.critical(self, "Помилка", f"Сталася помилка під час створення бази:\n{error_msg}")

    @pyqtSlot()
    def on_load_database(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть базу даних HDF5",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if file_name:
            self.current_database_path = file_name
            try:
                self.database = DatabaseLoader(file_name)
                self.status_bar.showMessage(f"Вибрано базу даних: {file_name}")
                self.control_panel.update_status("База завантажена")
            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити базу: {str(e)}")
        else:
            self.status_bar.showMessage("Вибір бази даних скасовано")

    @pyqtSlot()
    def on_start_tracking(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте базу даних HDF5 для початку відстеження!")
            return

        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Система не відкалібрована. Будь ласка, виконайте калібрування GPS.")
            return

        video_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть відео з дрона для відстеження", "", "Video Files (*.mp4 *.avi *.mkv)"
        )

        if not video_path:
            return

        sp_model = self.model_manager.load_superpoint()
        nv_model = self.model_manager.load_netvlad()
        lg_model = self.model_manager.load_lightglue()

        feature_extractor = FeatureExtractor(sp_model, nv_model, self.model_manager.device)
        matcher = FeatureMatcher(lg_model, self.model_manager.device)

        localizer = Localizer(self.database, feature_extractor, matcher, self.calibration)

        self.tracking_worker = RealtimeTrackingWorker(video_path, localizer)
        self.tracking_worker.frame_ready.connect(self.on_frame_ready)
        self.tracking_worker.location_found.connect(self.on_location_found)
        self.tracking_worker.status_update.connect(self.control_panel.update_status)

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

    @pyqtSlot()
    def on_localize_image(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте базу даних HDF5!")
            return

        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Система не відкалібрована. Виконайте калібрування GPS.")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self, "Виберіть зображення для локалізації", "", "Images (*.png *.jpg *.jpeg)"
        )

        if not image_path:
            return

        import cv2
        from PyQt6.QtWidgets import QApplication
        from config.config import APP_CONFIG
        from src.utils.image_utils import opencv_to_qpixmap

        frame = cv2.imread(image_path)
        if frame is None:
            QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")
            return

        self.status_bar.showMessage("Завантаження моделей та обробка зображення...")
        self.control_panel.update_status("Локалізація фото...")
        QApplication.processEvents()  # Оновлюємо інтерфейс перед важкими обчисленнями

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Завантажуємо моделі
            sp_model = self.model_manager.load_superpoint()
            nv_model = self.model_manager.load_netvlad()
            lg_model = self.model_manager.load_lightglue()

            feature_extractor = FeatureExtractor(sp_model, nv_model, self.model_manager.device)
            matcher = FeatureMatcher(lg_model, self.model_manager.device)

            localizer = Localizer(self.database, feature_extractor, matcher, self.calibration, config=APP_CONFIG)

            # Шукаємо координати
            result = localizer.localize_frame(frame_rgb)

            # Відображаємо фото на екрані
            pixmap = opencv_to_qpixmap(frame)
            if hasattr(self.video_widget, 'display_frame'):
                self.video_widget.display_frame(pixmap)

            if result.get("success"):
                lat = result["lat"]
                lon = result["lon"]
                conf = result["confidence"]
                inliers = result.get("inliers", 0)

                self.map_widget.update_marker(lat, lon)
                self.status_bar.showMessage(
                    f"Локалізація: {lat:.6f}, {lon:.6f} | Впевненість: {conf:.2f} | Точок: {inliers}")
                self.control_panel.update_status("Фото успішно локалізовано")

                QMessageBox.information(
                    self,
                    "Успіх",
                    f"Координати знайдено!\n\nШирота: {lat:.6f}\nДовгота: {lon:.6f}\nВпевненість: {conf:.2f}\nТочок збігу: {inliers}"
                )
            else:
                err = result.get('error', 'Невідома помилка')
                self.status_bar.showMessage(f"Помилка: {err}")
                self.control_panel.update_status("Помилка локалізації")
                QMessageBox.warning(self, "Помилка", f"Не вдалося знайти координати:\n{err}")

        except Exception as e:
            self.logger.error(f"Помилка локалізації зображення: {e}", exc_info=True)
            QMessageBox.critical(self, "Критична помилка", f"Сталася помилка:\n{str(e)}")
            self.status_bar.showMessage("Помилка обробки")

    @pyqtSlot()
    def on_calibrate(self):
        if not self.current_database_path:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте базу даних HDF5 для калібрування!")
            return
        dialog = CalibrationDialog(database_path=self.current_database_path, parent=self)
        dialog.calibration_complete.connect(self.on_calibration_complete)
        dialog.exec()

    @pyqtSlot(object)
    def on_calibration_complete(self, calibration_data):
        try:
            result = self.calibration.calibrate(
                calibration_data['points_2d'],
                calibration_data['points_gps']
            )
            self.status_bar.showMessage(f"Калібрування успішне! Похибка: {result['rmse_meters']:.2f} м.")
            QMessageBox.information(self, "Успіх",
                                    f"Калібрування збережено.\nСередня похибка: {result['rmse_meters']:.2f} метрів.")
        except Exception as e:
            QMessageBox.critical(self, "Помилка калібрування", str(e))

    @pyqtSlot(QPixmap)
    def on_frame_ready(self, pixmap):
        if hasattr(self.video_widget, 'display_frame'):
            self.video_widget.display_frame(pixmap)

    @pyqtSlot(float, float, float, int)
    def on_location_found(self, lat, lon, confidence, inliers):
        self.map_widget.update_marker(lat, lon)
        self.map_widget.add_trajectory_point(lat, lon)
        self.status_bar.showMessage(
            f"Локалізація: {lat:.6f}, {lon:.6f} | Впевненість: {confidence:.2f} | Точок: {inliers}")