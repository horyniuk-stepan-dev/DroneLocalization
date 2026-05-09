import sys
import time
import signal
from pathlib import Path

from PyQt6.QtCore import QCoreApplication

from config.config import APP_CONFIG, APP_SETTINGS
from src.calibration.multi_calibration_manager import MultiCalibrationManager
from src.core.project import ProjectManager
from src.database.database_loader import DatabaseLoader
from src.database.multi_database_manager import MultiDatabaseManager
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.localization.localizer import Localizer
from src.localization.matcher import FeatureMatcher
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.models.model_manager import ModelManager
from src.workers.tracking_worker import RealtimeTrackingWorker
from src.network.coordinates_broker import CoordinatesBroker
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HeadlessRunner:
    """Керує запуском системи без GUI (консольний режим)."""

    def __init__(self, project_dir: str, video_source: str):
        self.project_dir = Path(project_dir)
        self.video_source = video_source
        self.app = QCoreApplication.instance()
        if not self.app:
            self.app = QCoreApplication(sys.argv)
            
        self.model_manager = ModelManager(config=APP_CONFIG)
        self.calibration = MultiAnchorCalibration()
        self.database = None
        self.tracking_worker = None

        # Мультиджерельна підтримка
        self.db_manager = None
        self.calib_manager = None
        
        # Вмикаємо network api примусово для headless
        APP_SETTINGS.network_api.enabled = True
        self.coordinates_broker = CoordinatesBroker(config=APP_SETTINGS.network_api)

    def _setup_project(self):
        """Завантажує БД та калібрування з проекту."""
        logger.info(f"Loading project from {self.project_dir}")

        # Завантажуємо проект через ProjectManager для підтримки multi-source
        pm = ProjectManager()
        if pm.load_project(str(self.project_dir)):
            sources = pm.settings.get_enabled_sources()
            is_multi = len(sources) > 1 or any(s.source_id != "main" for s in sources)

            if is_multi and len(sources) > 0:
                # Мультиджерельний режим
                self.db_manager = MultiDatabaseManager(
                    sources, self.project_dir, config=APP_CONFIG.model_dump()
                )
                self.calib_manager = MultiCalibrationManager()
                self.calib_manager.load_all(sources, self.project_dir)

                first_id = self.db_manager.all_source_ids[0] if self.db_manager.all_source_ids else None
                if first_id:
                    self.database = self.db_manager.get_database(first_id)
                    self.calibration = self.calib_manager.get(first_id)
                else:
                    raise RuntimeError("Multi-source project: no database loaded")

                logger.info(
                    f"Multi-source project: {self.db_manager.num_databases} databases, "
                    f"sources={self.db_manager.all_source_ids}"
                )
            else:
                # Single-source mode
                db_path = self.project_dir / "database.h5"
                if not db_path.exists():
                    raise FileNotFoundError(f"Database not found at {db_path}")
                self.database = DatabaseLoader(str(db_path))

                calib_path = self.project_dir / "calibration.json"
                if calib_path.exists():
                    self.calibration.load(str(calib_path))
        else:
            # Fallback: прямий шлях (legacy)
            db_path = self.project_dir / "database.h5"
            calib_path = self.project_dir / "calibration.json"
            
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found at {db_path}")
                
            self.database = DatabaseLoader(str(db_path))
            if calib_path.exists():
                self.calibration.load(str(calib_path))

        if not self.database.is_propagated:
            logger.warning("Database is not propagated! Precision will be degraded.")
            
        if not self.calibration.converter.is_initialized:
            ref_gps = self.calibration.converter.reference_gps
            if self.calibration.is_calibrated and ref_gps:
                self.calibration.converter.gps_to_metric(ref_gps[0], ref_gps[1])
            else:
                raise ValueError("UTM projection is not initialized and calibration is missing.")

    def _build_localizer(self) -> Localizer:
        xf = self.model_manager.load_local_extractor()
        nv = self.model_manager.load_dinov2()

        cesp = None
        if APP_CONFIG.models.cesp.enabled:
            try:
                cesp = self.model_manager.load_cesp()
            except Exception as e:
                logger.warning(f"CESP loading failed: {e}")

        fe = FeatureExtractor(
            xf, nv, self.model_manager.device, config=APP_CONFIG, cesp_module=cesp
        )

        matcher = FeatureMatcher(model_manager=self.model_manager, config=APP_CONFIG)
        localizer_config = {**APP_CONFIG.model_dump(), "_model_manager": self.model_manager}
        
        return Localizer(
            self.database, fe, matcher, self.calibration, config=localizer_config,
            ref_frame_width=int(self.database.metadata.get("frame_width", 0)),
            ref_frame_height=int(self.database.metadata.get("frame_height", 0)),
            db_manager=self.db_manager,
            calib_manager=self.calib_manager,
        )

    def run(self):
        """Головний цикл Headless-режиму."""
        logger.info("Starting Headless Localization System")
        try:
            self._setup_project()
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            sys.exit(1)

        localizer = self._build_localizer()
        
        self.tracking_worker = RealtimeTrackingWorker(
            self.video_source,
            localizer,
            model_manager=self.model_manager,
            config=APP_CONFIG,
        )
        
        # Підключаємо брокер координат
        self.tracking_worker.location_found.connect(self.coordinates_broker.on_location_found)
        self.tracking_worker.objects_gps_updated.connect(self.coordinates_broker.on_objects_gps_updated)
        
        def on_tracking_finished():
            logger.info("Tracking finished.")
            self.coordinates_broker.set_tracking_active(False)
            self.app.quit()
            
        self.tracking_worker.finished.connect(on_tracking_finished)
        
        def signal_handler(sig, frame):
            logger.info("\nCaught interrupt signal, stopping gracefully...")
            if self.tracking_worker and self.tracking_worker.isRunning():
                self.tracking_worker.stop()
            else:
                self.app.quit()
                
        signal.signal(signal.SIGINT, signal_handler)
        
        self.coordinates_broker.set_tracking_active(True)
        self.tracking_worker.start()
        
        logger.info("System is running. Press Ctrl+C to stop.")
        self.app.exec()
        
        # Очищення
        self.coordinates_broker.stop()
        logger.info("Headless runner exited gracefully.")
