

# ================================================================================
# File: __init__.py
# ================================================================================
"""Drone Topometric Localization System"""

__version__ = "1.0.0"
__author__ = "Drone Localization Team"


# ================================================================================
# File: calibration\multi_anchor_calibration.py
# ================================================================================
try:
    import orjson as _json_lib
    _USE_ORJSON = True
except ImportError:
    import json as _json_lib
    _USE_ORJSON = False
from datetime import datetime
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AnchorCalibration:
    """
    Одна точка прив'язки GPS — конкретний кадр з афінною матрицею та повними QA-метриками.
    """

    def __init__(
        self, frame_id: int, affine_matrix: np.ndarray, qa_data: dict[str, Any] | None = None
    ):
        self.frame_id = frame_id
        self.affine_matrix = affine_matrix
        self.update_qa(qa_data or {})

    def update_qa(self, qa_data: dict[str, Any]) -> None:
        """Оновлює QA метрики якоря без перестворення об'єкта."""
        self.qa_data = qa_data

        # Основні метрики якості
        self.rmse_m = float(self.qa_data.get("rmse_m", 0.0))
        self.median_err_m = float(self.qa_data.get("median_err_m", 0.0))
        self.max_err_m = float(self.qa_data.get("max_err_m", 0.0))
        self.inliers_count = int(self.qa_data.get("inliers_count", 0))

        # Дані точок
        self.points_2d = self.qa_data.get("points_2d", [])  # [[x,y], ...]
        self.points_gps = self.qa_data.get("points_gps", [])  # [[lat,lon], ...]
        self.points_metric = self.qa_data.get("points_metric", [])  # [[mx,my], ...]

        # Метадані та UX
        self.transform_type = self.qa_data.get("transform_type", "unknown")
        self.projection_mode = self.qa_data.get("projection_mode", "WEB_MERCATOR")
        self.created_at = self.qa_data.get("created_at", datetime.now().isoformat())
        self.updated_at = self.qa_data.get("updated_at", self.created_at)
        self.notes = self.qa_data.get("notes", "")
        self.quality_flag = self.qa_data.get("quality_flag", "normal")  # 'normal', 'warning', 'bad'

    def pixel_to_metric(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[x, y]], dtype=np.float32)
        result = GeometryTransforms.apply_affine(pt, self.affine_matrix)[0]
        return float(result[0]), float(result[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "affine_matrix": self.affine_matrix.tolist(),
            "qa_data": {
                "rmse_m": self.rmse_m,
                "median_err_m": self.median_err_m,
                "max_err_m": self.max_err_m,
                "inliers_count": self.inliers_count,
                "points_2d": self.points_2d,
                "points_gps": self.points_gps,
                "points_metric": self.points_metric,
                "transform_type": self.transform_type,
                "projection_mode": self.projection_mode,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "notes": self.notes,
                "quality_flag": self.quality_flag,
            },
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AnchorCalibration":
        # Підтримка зовсім старих форматів без qa_data
        qa = data.get("qa_data", {})

        # Якщо це старий формат v1.0/v2.0, де деякі поля були плоскими
        if not qa and "rmse_m" in data:
            qa = {
                "rmse_m": data.get("rmse_m"),
                "max_err_m": data.get("max_err_m"),
                "num_points": data.get("num_points"),
                "transform_type": data.get("transform_type"),
                "created_at": data.get("created_at"),
            }

        return AnchorCalibration(
            frame_id=int(data["frame_id"]),
            affine_matrix=np.array(data["affine_matrix"], dtype=np.float32),
            qa_data=qa,
        )


class MultiAnchorCalibration:
    """Менеджер декількох якорів калібрування з підтримкою версіонування та проєкцій"""

    VERSION: str = "2.2"

    def __init__(self, converter: CoordinateConverter | None = None) -> None:
        self.anchors: list[AnchorCalibration] = []
        self.converter = converter or CoordinateConverter("WEB_MERCATOR")
        self._interp: PchipInterpolator | None = None  # кешований інтерполятор

    def _rebuild_interpolators(self) -> None:
        """Перебудовує PCHIP-інтерполятор. Викликати після кожної зміни anchors."""
        if len(self.anchors) < 2:
            self._interp = None
            return

        ids = np.array([a.frame_id for a in self.anchors], dtype=np.float64)
        matrices = np.stack([a.affine_matrix.ravel() for a in self.anchors])  # (N, 6)
        # PchipInterpolator обробляє multi-column масиви нативно
        self._interp = PchipInterpolator(ids, matrices, extrapolate=True)

    @property
    def is_calibrated(self) -> bool:
        return len(self.anchors) > 0

    def add_anchor(
        self, frame_id: int, affine_matrix: np.ndarray, qa_data: dict[str, Any] | None = None
    ) -> None:
        existing = next((a for a in self.anchors if a.frame_id == frame_id), None)
        if existing:
            existing.affine_matrix = affine_matrix
            if qa_data:
                existing.update_qa(qa_data)
                existing.updated_at = datetime.now().isoformat()
            logger.info(f"Updated anchor for frame {frame_id}")
        else:
            self.anchors.append(AnchorCalibration(frame_id, affine_matrix, qa_data))
            self.anchors.sort(key=lambda a: a.frame_id)
            logger.info(
                f"Added new anchor for frame {frame_id}. Total anchors: {len(self.anchors)}"
            )
        self._rebuild_interpolators()

    def get_anchor(self, frame_id: int) -> AnchorCalibration | None:
        return next((a for a in self.anchors if a.frame_id == frame_id), None)

    def remove_anchor(self, frame_id: int) -> bool:
        initial_len = len(self.anchors)
        self.anchors = [a for a in self.anchors if a.frame_id != frame_id]
        success = len(self.anchors) < initial_len
        if success:
            self._rebuild_interpolators()
            logger.info(f"Removed anchor for frame {frame_id}")
        return success

    def get_metric_position(self, frame_id: int, x: float, y: float) -> tuple[float, float] | None:
        if not self.is_calibrated:
            return None

        # Якщо якір один — екстраполяція неможлива, повертаємо його координати
        if len(self.anchors) == 1:
            return self.anchors[0].pixel_to_metric(x, y)

        # PCHIP: інтерполяція/екстраполяція матриці → застосування до точки
        if self._interp is not None:
            flat = self._interp(float(frame_id))  # (6,) float64
            if flat is not None and not np.any(np.isnan(flat)):
                M = flat.reshape(2, 3).astype(np.float32)
                pt = np.array([[x, y]], dtype=np.float32)
                result = GeometryTransforms.apply_affine(pt, M)[0]
                return float(result[0]), float(result[1])

        # Fallback — лінійна інтерполяція
        for i in range(len(self.anchors) - 1):
            a1, a2 = self.anchors[i], self.anchors[i + 1]
            if a1.frame_id <= frame_id <= a2.frame_id:
                dist_1 = abs(frame_id - a1.frame_id)
                dist_2 = abs(frame_id - a2.frame_id)
                total = dist_1 + dist_2
                if total == 0:
                    return a1.pixel_to_metric(x, y)
                w2 = dist_1 / total
                m1 = a1.pixel_to_metric(x, y)
                m2 = a2.pixel_to_metric(x, y)
                return m1[0] * (1 - w2) + m2[0] * w2, m1[1] * (1 - w2) + m2[1] * w2
        return None

    def save(self, path: str) -> None:
        """Збереження якорів та метаданих проєкції у JSON."""
        data = {
            "version": self.VERSION,
            "projection": self.converter.export_metadata(),
            "anchors": [a.to_dict() for a in self.anchors],
        }

        if _USE_ORJSON:
            raw = _json_lib.dumps(
                data,
                option=_json_lib.OPT_INDENT_2 | getattr(_json_lib, "OPT_NON_STR_KEYS", 0),
            )
            with open(path, "wb") as f:
                f.write(raw)
        else:
            with open(path, "w", encoding="utf-8") as f:
                _json_lib.dump(data, f, indent=2, ensure_ascii=False)
        logger.success(
            f"MultiAnchorCalibration saved: {path} (v{self.VERSION}, {len(self.anchors)} anchors)"
        )

    def load(self, path: str) -> None:
        logger.info(f"Loading MultiAnchorCalibration from: {path}")
        with open(path, "rb") as f:
            content = f.read()

        if _USE_ORJSON:
            data = _json_lib.loads(content)
        else:
            data = _json_lib.loads(content.decode("utf-8"))

        self.anchors.clear()
        version = data.get("version", "1.0")

        # 1. Відновлення проєкції
        if "projection" in data:
            self.converter = CoordinateConverter.from_metadata(data["projection"])
        elif "reference_gps" in data and data["reference_gps"] is not None:
            # Fallback для v2.0
            self.converter = CoordinateConverter("UTM", tuple(data["reference_gps"]))
        else:
            # Fallback для v1.0 або відсутніх даних
            logger.warning(
                "No projection metadata found in calibration file. Defaulting to WEB_MERCATOR fallback."
            )
            self.converter = CoordinateConverter("WEB_MERCATOR")

        # 2. Завантаження якорів
        if version == "1.0" and "affine_matrix" in data and "calib_frame_id" in data:
            # Старий формат (один якір)
            anchor = AnchorCalibration(
                frame_id=int(data.get("calib_frame_id", 0)),
                affine_matrix=np.array(data["affine_matrix"], dtype=np.float32),
            )
            self.anchors.append(anchor)
        elif "anchors" in data:
            # Новій формат (список якорів)
            for item in data["anchors"]:
                self.anchors.append(AnchorCalibration.from_dict(item))

        self.anchors.sort(key=lambda a: a.frame_id)
        self._rebuild_interpolators()
        logger.success(f"Loaded {len(self.anchors)} anchors (file version: {version})")

# ================================================================================
# File: calibration\__init__.py
# ================================================================================
"""Calibration module"""


# ================================================================================
# File: core\export_results.py
# ================================================================================
import csv
from datetime import datetime
from typing import Any

import geojson

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResultExporter:
    """Експорт результатів локалізації у різні формати."""

    @staticmethod
    def export_csv(results: list[dict[str, Any]], output_path: str) -> None:
        """
        Експорт у CSV файл.

        Args:
            results: список словників з ключами:
                frame_id, lat, lon, confidence, timestamp, matched_frame, inliers
            output_path: шлях до вихідного файлу
        """
        if not results:
            logger.warning("No results to export")
            return

        fieldnames = [
            "frame_id",
            "timestamp",
            "lat",
            "lon",
            "confidence",
            "matched_frame",
            "inliers",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        logger.success(f"Exported {len(results)} results to CSV: {output_path}")

    @staticmethod
    def export_geojson(results: list[dict[str, Any]], output_path: str) -> None:
        """Експорт у GeoJSON (для GIS-систем). Додає точки та полігони FOV."""
        features = []
        for r in results:
            if "lat" not in r or "lon" not in r:
                continue

            # 1. Point feature (trajectory)
            point = geojson.Feature(
                geometry=geojson.Point((r["lon"], r["lat"])),
                properties={
                    "type": "trajectory_point",
                    "frame_id": r.get("frame_id"),
                    "confidence": r.get("confidence"),
                    "timestamp": r.get("timestamp"),
                    "matched_frame": r.get("matched_frame"),
                },
            )
            features.append(point)

            # 2. Polygon feature (FOV) - якщо є дані
            fov = r.get("fov_polygon")
            if fov and len(fov) >= 3:
                # GeoJSON Polygon coordinates must be a list of rings,
                # each ring is a list of [lon, lat] points, first and last must be same.
                coords = [[lon, lat] for lat, lon in fov]
                # Close the polygon
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                polygon = geojson.Feature(
                    geometry=geojson.Polygon([coords]),
                    properties={
                        "type": "fov_polygon",
                        "frame_id": r.get("frame_id"),
                        "confidence": r.get("confidence"),
                    },
                )
                features.append(polygon)

        feature_collection = geojson.FeatureCollection(
            features, properties={"exported_at": datetime.now().isoformat()}
        )

        with open(output_path, "w", encoding="utf-8") as f:
            geojson.dump(feature_collection, f, indent=2, ensure_ascii=False)

        logger.success(f"Exported {len(features)} features to GeoJSON: {output_path}")

    @staticmethod
    def export_kml(
        results: list[dict[str, Any]], output_path: str, name: str = "Drone Track"
    ) -> None:
        """Експорт у KML (для Google Earth)."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "<Document>",
            f"  <name>{name}</name>",
            f"  <description>Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}</description>",
        ]

        # Стиль маркера
        lines.extend(
            [
                '  <Style id="dronePoint">',
                "    <IconStyle>",
                "      <scale>0.6</scale>",
                "      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/airports.png</href></Icon>",
                "    </IconStyle>",
                "  </Style>",
            ]
        )

        # Точки
        for r in results:
            if "lat" not in r or "lon" not in r:
                continue
            conf = r.get("confidence", 0)
            fid = r.get("frame_id", "?")
            lines.extend(
                [
                    "  <Placemark>",
                    f"    <name>Frame {fid}</name>",
                    f"    <description>Confidence: {conf:.2f}</description>",
                    "    <styleUrl>#dronePoint</styleUrl>",
                    "    <Point>",
                    f"      <coordinates>{r['lon']},{r['lat']},0</coordinates>",
                    "    </Point>",
                    "  </Placemark>",
                ]
            )

        # Трек (лінія)
        coords_str = " ".join(
            f"{r['lon']},{r['lat']},0" for r in results if "lat" in r and "lon" in r
        )
        if coords_str:
            lines.extend(
                [
                    "  <Placemark>",
                    f"    <name>{name} - Path</name>",
                    "    <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>",
                    "    <LineString>",
                    "      <tessellate>1</tessellate>",
                    f"      <coordinates>{coords_str}</coordinates>",
                    "    </LineString>",
                    "  </Placemark>",
                ]
            )

        lines.extend(["</Document>", "</kml>"])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.success(f"Exported {len(results)} points to KML: {output_path}")


# ================================================================================
# File: core\project.py
# ================================================================================
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectSettings:
    """Stores metadata about a Drone Localization Project"""

    project_name: str
    created_at: str
    video_path: str

    # Internal relative paths
    database_filename: str = "database.h5"
    calibration_filename: str = "calibration.json"

    # Optional mission parameters inherited from NewMissionDialog
    altitude_m: float = 100.0
    focal_length_mm: float = 13.2
    sensor_width_mm: float = 8.8
    image_width_px: int = 4000

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class ProjectManager:
    """
    Manages the workspace directory, saving/loading project configuration,
    and resolving absolute paths to project files.
    """

    def __init__(self):
        self.project_dir: Path | None = None
        self.settings: ProjectSettings | None = None

    @property
    def is_loaded(self) -> bool:
        return self.project_dir is not None and self.settings is not None

    @property
    def project_name(self) -> str:
        return self.settings.project_name if self.settings else "No Project"

    @property
    def database_path(self) -> str | None:
        if not self.is_loaded:
            return None
        return str(self.project_dir / self.settings.database_filename)

    @property
    def calibration_path(self) -> str | None:
        if not self.is_loaded:
            return None
        return str(self.project_dir / self.settings.calibration_filename)

    def create_project(self, workspace_dir: str, mission_data: dict) -> bool:
        """
        Creates a new project folder inside `workspace_dir` using the mission name.
        Saves the project.json.
        """
        try:
            name = mission_data.get("mission_name", "Untitled_Mission")
            # Create a safe folder name
            safe_folder_name = "".join([c if c.isalnum() else "_" for c in name])
            self.project_dir = Path(workspace_dir) / safe_folder_name

            # Ensure the directory exists
            self.project_dir.mkdir(parents=True, exist_ok=True)

            # Create subfolders for organization
            (self.project_dir / "panoramas").mkdir(exist_ok=True)
            (self.project_dir / "test_photos").mkdir(exist_ok=True)
            (self.project_dir / "test_videos").mkdir(exist_ok=True)

            self.settings = ProjectSettings(
                project_name=name,
                created_at=datetime.now().isoformat(),
                video_path=mission_data.get("video_path", ""),
                altitude_m=mission_data.get("altitude_m", 100.0),
                focal_length_mm=mission_data.get("focal_length_mm", 13.2),
                sensor_width_mm=mission_data.get("sensor_width_mm", 8.8),
                image_width_px=mission_data.get("image_width_px", 4000),
            )

            self.save_project()
            logger.info(f"Project created successfully at: {self.project_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to create project: {e}", exc_info=True)
            self.project_dir = None
            self.settings = None
            return False

    def load_project(self, project_dir: str) -> bool:
        """Loads an existing project from the given directory path."""
        try:
            dir_path = Path(project_dir)
            if not dir_path.is_dir():
                logger.error(f"Project directory does not exist: {project_dir}")
                return False

            json_file = dir_path / "project.json"
            if not json_file.exists():
                logger.error(f"Valid project not found. Missing project.json in: {project_dir}")
                return False

            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            self.settings = ProjectSettings.from_dict(data)
            self.project_dir = dir_path

            logger.info(f"Project loaded successfully: {self.settings.project_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load project: {e}", exc_info=True)
            self.project_dir = None
            self.settings = None
            return False

    def save_project(self) -> bool:
        """Saves current ProjectSettings to project.json"""
        if not self.is_loaded:
            return False

        try:
            json_file = self.project_dir / "project.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save project settings: {e}", exc_info=True)
            return False


# ================================================================================
# File: core\project_registry.py
# ================================================================================
import json
from datetime import datetime
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProjectRegistry:
    """
    Централізований реєстр усіх проєктів.
    Зберігає шляхи та метадані в JSON файлі у домашній директорії.
    """

    def __init__(self):
        self._registry_dir = Path.home() / ".drone_localizer"
        self._registry_path = self._registry_dir / "projects.json"
        self._projects: list[dict] = []
        self._load()

    def _load(self):
        """Завантажити реєстр з диска."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._projects = data.get("projects", [])
                logger.debug(f"ProjectRegistry loaded: {len(self._projects)} projects")
            except Exception as e:
                logger.warning(f"Failed to load project registry: {e}")
                self._projects = []
        else:
            self._projects = []

    def _save(self):
        """Зберегти реєстр на диск."""
        try:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump({"projects": self._projects}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save project registry: {e}")

    def _find_index(self, project_dir: str) -> int:
        """Знайти індекс проєкту за шляхом."""
        norm = str(Path(project_dir).resolve())
        for i, p in enumerate(self._projects):
            if str(Path(p["path"]).resolve()) == norm:
                return i
        return -1

    def register(self, project_dir: str, name: str, video_path: str = ""):
        """Додати або оновити проєкт у реєстрі."""
        idx = self._find_index(project_dir)
        now = datetime.now().isoformat()

        entry = {
            "name": name,
            "path": str(Path(project_dir).resolve()),
            "video_path": video_path,
            "created_at": now,
            "last_opened": now,
            "has_database": (Path(project_dir) / "database.h5").exists(),
            "has_calibration": (Path(project_dir) / "calibration.json").exists(),
        }

        if idx >= 0:
            # Зберігаємо оригінальну дату створення
            entry["created_at"] = self._projects[idx].get("created_at", now)
            self._projects[idx] = entry
        else:
            self._projects.append(entry)

        self._save()
        logger.info(f"Project registered: {name} at {project_dir}")

    def unregister(self, project_dir: str):
        """Видалити проєкт з реєстру (файли НЕ видаляються)."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            removed = self._projects.pop(idx)
            self._save()
            logger.info(f"Project unregistered: {removed['name']}")

    def update_last_opened(self, project_dir: str):
        """Оновити дату останнього відкриття."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            self._projects[idx]["last_opened"] = datetime.now().isoformat()
            self._save()

    def refresh_status(self, project_dir: str):
        """Оновити статус наявності БД та калібрації."""
        idx = self._find_index(project_dir)
        if idx >= 0:
            p = Path(project_dir)
            self._projects[idx]["has_database"] = (p / "database.h5").exists()
            self._projects[idx]["has_calibration"] = (p / "calibration.json").exists()
            self._save()

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Повернути останні відкриті проєкти (відсортовані за датою)."""
        valid = [p for p in self._projects if Path(p["path"]).is_dir()]
        valid.sort(key=lambda p: p.get("last_opened", ""), reverse=True)
        return valid[:limit]

    def get_all(self) -> list[dict]:
        """Повернути всі зареєстровані проєкти."""
        return list(self._projects)


# ================================================================================
# File: database\database_builder.py
# ================================================================================
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import h5py
import numpy as np
import torch

from config.config import get_cfg
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.models.wrappers.masking_strategy import create_masking_strategy
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseBuilder:
    """Builds HDF5 topometric database from reference video using XFeat & DINOv2"""

    def __init__(self, output_path, matcher=None, config=None):
        self.output_path = output_path
        self.config = config or {}
        self.matcher = matcher
        db_cfg = self.config.get("database", {})
        self.descriptor_dim = get_cfg(self.config, "dinov2.descriptor_dim", 384)
        self.prefetch_size = get_cfg(self.config, "database.prefetch_queue_size", 32)
        self.kp_scale_cfg = get_cfg(self.config, "database.keypoint_video_scale", 0.5)
        self.db_file = None

        logger.info(f"DatabaseBuilder initialized with output: {output_path}")
        if self.matcher:
            logger.info("Using provided FeatureMatcher for inter-frame poses")
        logger.info(f"DINOv2 descriptor dimension: {self.descriptor_dim}")

    def build_from_video(
        self,
        video_path: str,
        model_manager,
        progress_callback=None,
        save_keypoint_video: bool = True,
    ):
        """
        Process video and build database.
        """
        logger.info(f"Starting database build from video: {video_path}")

        # Читаємо налаштування з конфігу (з дефолтом)
        frame_step = get_cfg(self.config, "database.frame_step", 3)
        if frame_step < 1:
            frame_step = 1

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Не вдалося відкрити відео: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Обчислюємо скільки кадрів РЕАЛЬНО буде оброблено
        num_frames = (total_frames + frame_step - 1) // frame_step
        effective_fps = original_fps / frame_step

        if num_frames <= 0:
            cap.release()
            logger.error(
                f"Invalid frame count ({num_frames}). Video might be corrupted or uses unsupported codec."
            )
            raise ValueError(
                "OpenCV не зміг розпізнати відео. Файл пошкоджений або використовує непідтримуваний кодек. "
                "Спробуйте переконвертувати відео у стандартний MP4 (H.264)."
            )

        logger.info(
            f"Video properties: {width}x{height}, {total_frames} total frames, {original_fps:.2f} FPS"
        )
        logger.info(
            f"Processing with step={frame_step} -> {num_frames} frames to process ({effective_fps:.2f} effective FPS)"
        )

        # Ініціалізуємо запис відео з keypoints
        kp_video_path = None
        kp_writer = None
        kp_scale = 1.0  # ЗАВЖДИ 1.0, щоб координати відео і бази HDF5 збігалися (виправлення багу масштабу)
        if save_keypoint_video:
            try:
                kp_width = int(width * kp_scale)
                kp_height = int(height * kp_scale)

                kp_video_path = str(Path(self.output_path).with_suffix("")) + "_keypoints.mp4"

                # Attempt H.264 (avc1)
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                kp_writer = cv2.VideoWriter(
                    kp_video_path, fourcc, effective_fps, (kp_width, kp_height)
                )

                if not kp_writer or not kp_writer.isOpened():
                    logger.warning("H.264 (avc1) codec not available, falling back to mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    kp_writer = cv2.VideoWriter(
                        kp_video_path, fourcc, effective_fps, (kp_width, kp_height)
                    )

                if kp_writer and kp_writer.isOpened():
                    logger.info(
                        f"Keypoint video will be saved to: {kp_video_path} at {kp_width}x{kp_height} (Scale: {kp_scale})"
                    )
                else:
                    logger.warning(
                        "Failed to initialize any compatible video codec, keypoint video disabled"
                    )
                    kp_writer = None
            except Exception as e:
                logger.warning(f"VideoWriter initialization crashed: {e}")
                kp_writer = None

        # Initialize neural network wrappers
        logger.info("Loading neural network models...")
        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, model_manager.device)

        # MaskingStrategy — абстракція над YOLO/none/інші стратегії (конфіг: preprocessing.masking_strategy)
        masking_strategy_name = get_cfg(self.config, "preprocessing.masking_strategy", "yolo")
        logger.info(f"Loading masking strategy: {masking_strategy_name}")
        masking_strategy = create_masking_strategy(
            masking_strategy_name, model_manager, model_manager.device
        )

        local_model = model_manager.load_aliked()
        nv_model = model_manager.load_dinov2()

        cesp = None
        if get_cfg(self.config, "models.cesp.enabled", False):
            try:
                cesp = model_manager.load_cesp()
            except Exception:
                logger.warning("CESP loading failed during DB build, continuing without it")

        feature_extractor = FeatureExtractor(
            local_model, nv_model, model_manager.device, config=self.config, cesp_module=cesp
        )
        logger.success("All models loaded successfully")

        # Fix 10: Dynamic descriptor dimension detection to avoid broadcast errors
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                free_mb, total_mb = torch.cuda.mem_get_info()
                logger.info(f"VRAM before dimension detection: {free_mb / (1024**2):.1f}MB free")

            logger.info("Detecting descriptor dimension...")
            if hasattr(nv_model, "embed_dim"):
                self.descriptor_dim = int(nv_model.embed_dim)
            else:
                # Use a small dummy tensor directly to save VRAM
                with torch.no_grad():
                    dino_size = get_cfg(self.config, "dinov2.input_size", 336)
                    dummy_input = torch.zeros((1, 3, dino_size, dino_size)).to(model_manager.device)
                    # Use the same logic as FeatureExtractor
                    if cesp is not None:
                        features = nv_model.forward_features(dummy_input)
                        patch_tokens = features["x_norm_patchtokens"]
                        h_patches, w_patches = dino_size // 14, dino_size // 14
                        dummy_out = cesp(patch_tokens, h_patches, w_patches)[0]
                        self.descriptor_dim = int(dummy_out.shape[0])
                    else:
                        dummy_out = nv_model(dummy_input)[0]
                        self.descriptor_dim = int(dummy_out.shape[0])

            logger.info(f"Detected global descriptor dimension: {self.descriptor_dim}")
        except Exception as e:
            import traceback

            logger.warning(f"Failed to detect descriptor dimension: {e}\n{traceback.format_exc()}")
            logger.warning(f"Using default dimension: {self.descriptor_dim}")

        # Create empty database structure
        logger.info("Creating HDF5 database structure...")
        self.create_hdf5_structure(num_frames, width, height)

        current_pose = np.eye(3, dtype=np.float32)
        prev_features = None

        # cuDNN benchmark conditionally (Fix 5)

        if torch.cuda.is_available():
            model_type = get_cfg(self.config, "localization.fallback_extractor", "aliked")
            if model_type in ("xfeat", "aliked"):  # CNN-based types
                torch.backends.cudnn.benchmark = True
                logger.info(f"cuDNN benchmark ENABLED for {model_type}")

        # Increased prefetch queue (Fix 5)
        frame_queue = Queue(maxsize=self.prefetch_size)

        def prefetch_frames():
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                if i % frame_step != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                db_index = i // frame_step
                frame_queue.put((db_index, (frame, frame_rgb)))

            frame_queue.put((-1, None))

        prefetch_thread = Thread(target=prefetch_frames, daemon=True)
        prefetch_thread.start()

        try:
            self.db_file = h5py.File(self.output_path, "a")
            logger.info(f"Opened HDF5 file for writing: {self.output_path}")

            prev_features = None
            db_index = 0
            frame_index_map = []
            use_keyframe_selection = get_cfg(self.config, "database.keyframe_min_translation_px", 0.0) > 0
            while True:
                idx, data = frame_queue.get()
                if idx == -1 or data is None:
                    break

                frame, frame_rgb = data
                i = idx

                # Sequential processing (restored original logic)
                static_mask, detections = yolo_wrapper.detect_and_mask(frame_rgb)
                features = feature_extractor.extract_features(frame_rgb, static_mask)
                features["coords_2d"] = features["keypoints"]

                if kp_writer is not None:
                    kp_frame = self._draw_keypoints_frame(
                        frame, features["keypoints"], static_mask, i, num_frames
                    )
                    if kp_scale != 1.0:
                        kp_width = int(width * kp_scale)
                        kp_height = int(height * kp_scale)
                        kp_frame = cv2.resize(
                            kp_frame, (kp_width, kp_height), interpolation=cv2.INTER_AREA
                        )
                    kp_writer.write(kp_frame)

                if i == 0 or prev_features is None:
                    current_pose = np.eye(3, dtype=np.float64)
                    save_this_frame = True
                else:
                    H_step = self._compute_inter_frame_H(prev_features, features)
                    if H_step is not None:
                        current_pose = current_pose @ H_step.astype(np.float64)
                        if use_keyframe_selection:
                            save_this_frame = self._is_significant_motion(H_step)
                        else:
                            save_this_frame = True
                    else:
                        logger.warning(
                            f"Frame {i}: inter-frame match failed, reusing previous pose"
                        )
                        save_this_frame = True

                prev_features = features

                # ЗАВЖДИ зберігаємо pose для повного ланцюга пропагації,
                # навіть якщо кадр не є keyframe (пропущений через малий рух).
                self.db_file["global_descriptors"]["frame_poses"][i] = current_pose

                if not save_this_frame:
                    progress_percent = int((i + 1) / num_frames * 100)
                    if progress_callback:
                        progress_callback(progress_percent)
                    continue

                frame_index_map.append(i)
                self.save_frame_data(db_index, features, current_pose)
                db_index += 1

                progress_percent = int((i + 1) / num_frames * 100)
                if progress_callback:
                    progress_callback(progress_percent)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{num_frames} frames ({progress_percent}%)")

            # Save the index mapping
            if "metadata" in self.db_file:
                self.db_file["metadata"].attrs["actual_num_frames"] = db_index
                self.db_file["metadata"].create_dataset(
                    "frame_index_map", data=np.array(frame_index_map, dtype=np.int32)
                )

        except Exception as e:
            logger.error(f"Error during database building: {e}")
            raise
        finally:
            prefetch_thread.join(timeout=5)
            if kp_writer is not None:
                kp_writer.release()
            if self.db_file:
                self.db_file.close()
            cap.release()

        logger.success(f"Database build completed successfully: {self.output_path}")

    def _is_significant_motion(self, H: np.ndarray) -> bool:
        """Повертає True якщо гомографія H відповідає значному руху."""
        min_t = get_cfg(self.config, "database.keyframe_min_translation_px", 15.0)
        min_r = get_cfg(self.config, "database.keyframe_min_rotation_deg", 1.5)

        # Розміри беремо з конфігу — всі кадри нормалізуються до target_size перед обробкою
        target_w = get_cfg(self.config, "preprocessing.target_width", 1920)
        target_h = get_cfg(self.config, "preprocessing.target_height", 1080)
        cx, cy = target_w / 2.0, target_h / 2.0
        p_src = np.array([cx, cy, 1.0], dtype=np.float64)
        p_dst = H.astype(np.float64) @ p_src
        if p_dst[2] != 0:
            p_dst /= p_dst[2]
        translation = float(np.linalg.norm(p_dst[:2] - np.array([cx, cy])))

        if translation >= min_t:
            return True

        A = H[:2, :2].astype(np.float64)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return True
        angle_rad = np.arctan2(A[1, 0], A[0, 0])
        angle_deg = abs(np.degrees(angle_rad))
        return angle_deg >= min_r

    def _draw_keypoints_frame(
        self,
        frame_bgr: np.ndarray,
        keypoints: np.ndarray,
        static_mask: np.ndarray,
        frame_id: int,
        total_frames: int,
    ) -> np.ndarray:
        vis = frame_bgr.copy()

        if static_mask is not None:
            dynamic_zone = static_mask == 0
            if dynamic_zone.any():
                overlay = vis.copy()
                overlay[dynamic_zone] = (0, 0, 200)
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        for x, y in keypoints:
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(vis, (cx, cy), radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(vis, (cx, cy), radius=4, color=(0, 180, 0), thickness=1)

        info_lines = [
            f"Frame: {frame_id:05d} / {total_frames:05d}",
            f"Keypoints: {len(keypoints)}",
            f"Dynamic mask: {'YES' if static_mask is not None else 'NO'}",
        ]
        panel_h = len(info_lines) * 28 + 14
        cv2.rectangle(vis, (0, 0), (340, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (0, 0), (340, panel_h), (80, 80, 80), 1)

        for idx, line in enumerate(info_lines):
            cv2.putText(
                vis,
                line,
                (8, 22 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        legend_y = vis.shape[0] - 10
        cv2.circle(vis, (12, legend_y - 4), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            "XFeat keypoint",
            (22, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(vis, (200, legend_y - 10), (218, legend_y + 2), (0, 0, 200), -1)
        cv2.putText(
            vis,
            "YOLO dynamic zone",
            (224, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1,
            cv2.LINE_AA,
        )

        return vis

    def _compute_inter_frame_H(self, fa: dict, fb: dict) -> np.ndarray | None:
        """
        Обчислює H(fb → fa): гомографію з поточного кадру в попередній.
        """
        min_matches = get_cfg(self.config, "database.inter_frame_min_matches", 15)
        ransac_thresh = get_cfg(self.config, "database.inter_frame_ransac_thresh", 3.0)

        if self.matcher is None:
            from src.localization.matcher import FeatureMatcher

            self.matcher = FeatureMatcher(config=self.config)

        mkpts_a, mkpts_b = self.matcher.match(fa, fb)

        if len(mkpts_a) < min_matches:
            return None

        from src.geometry.transformations import GeometryTransforms

        H, mask = GeometryTransforms.estimate_homography(
            mkpts_a, mkpts_b, ransac_threshold=ransac_thresh
        )

        if H is None or int(np.sum(mask)) < min_matches:
            return None

        return H.astype(np.float32)

    def create_hdf5_structure(self, num_frames: int, width: int, height: int):
        """Create optimal HDF5 hierarchy with compression and pre-allocated arrays"""
        compression = get_cfg(self.config, "database.hdf5_compression", "lzf")
        chunk_f = get_cfg(self.config, "database.hdf5_chunk_frames", 64)
        logger.info(
            f"Creating HDF5 structure for {num_frames} frames "
            f"(compression={compression}, chunks={chunk_f})"
        )

        with h5py.File(self.output_path, "w") as f:
            # Global descriptors
            g1 = f.create_group("global_descriptors")
            g1.create_dataset(
                "descriptors",
                shape=(num_frames, self.descriptor_dim),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), self.descriptor_dim),
            )
            g1.create_dataset(
                "frame_poses",
                shape=(num_frames, 3, 3),
                dtype="float64",
                compression=compression,
                chunks=(min(chunk_f, num_frames), 3, 3),
            )

            # Local features (pre-allocated arrays)
            max_kp = get_cfg(self.config, "models.aliked.max_keypoints", 4096)
            fallback_extractor = get_cfg(self.config, "localization.fallback_extractor", "aliked")

            # Determine feature dimension
            feature_dim = 128
            if fallback_extractor == "superpoint":
                feature_dim = 256
            elif fallback_extractor == "xfeat":
                feature_dim = 64

            g2 = f.create_group("local_features")
            g2.attrs["frame_width"] = width
            g2.attrs["frame_height"] = height
            g2.create_dataset(
                "keypoints",
                shape=(num_frames, max_kp, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kp, 2),
                fillvalue=0.0,
            )
            g2.create_dataset(
                "descriptors",
                shape=(num_frames, max_kp, feature_dim),
                dtype="float16",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kp, feature_dim),
                fillvalue=0.0,
            )
            g2.create_dataset(
                "coords_2d",
                shape=(num_frames, max_kp, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kp, 2),
                fillvalue=0.0,
            )
            g2.create_dataset(
                "num_kp",
                shape=(num_frames,),
                dtype="int32",
                compression=compression,
                chunks=(min(num_frames, 4096),),
                fillvalue=0,
            )

            g3 = f.create_group("metadata")
            g3.attrs["num_frames"] = num_frames
            g3.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            g3.attrs["frame_width"] = width
            g3.attrs["frame_height"] = height
            g3.attrs["descriptor_dim"] = self.descriptor_dim

        logger.success("HDF5 structure created successfully")

    def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
        """Save extracted data for a single frame into pre-allocated arrays"""
        self.db_file["global_descriptors"]["descriptors"][frame_id] = features["global_desc"]
        self.db_file["global_descriptors"]["frame_poses"][frame_id] = pose_2d

        num = len(features["keypoints"])
        
        # Write only up to 'num' elements.
        self.db_file["local_features"]["keypoints"][frame_id, :num, :] = features["keypoints"]
        # Automatic conversion to float16 since the dataset is created with float16
        self.db_file["local_features"]["descriptors"][frame_id, :num, :] = features["descriptors"]
        self.db_file["local_features"]["coords_2d"][frame_id, :num, :] = features["coords_2d"]
        self.db_file["local_features"]["num_kp"][frame_id] = num

# ================================================================================
# File: database\database_loader.py
# ================================================================================
import json
from typing import Any

import h5py
import numpy as np

from src.geometry.coordinates import CoordinateConverter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """Loads and manages access to the HDF5 topometric database (XFeat + DINOv2)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_file: h5py.File | None = None
        self.global_descriptors: np.ndarray | None = None
        self.frame_poses: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.converter: CoordinateConverter | None = None

        # Дані пропагації калібрування (заповнюються після калібрування)
        self.frame_affine: np.ndarray | None = None  # (N, 2, 3) — Metric Affine Matrices
        self.frame_valid: np.ndarray | None = None  # (N,)      — True якщо кадр має GPS
        self.frame_rmse: np.ndarray | None = None  # (N,)      — RMSE кожного кадру
        self.frame_disagreement: np.ndarray | None = None  # (N,)   — Розбіжність між гілками
        self.frame_matches: np.ndarray | None = None  # (N,)      — Кількість точок (inliers)

        # Каш для методів (заміна lru_cache для уникнення B019)
        self._size_cache: dict[int, tuple[int, int]] = {}
        self._feature_cache: dict[int, dict[str, np.ndarray]] = {}

        logger.info(f"Initializing DatabaseLoader with path: {db_path}")
        self._load_hot_data()

    def _load_hot_data(self) -> None:
        """Load global descriptors (DINOv2), poses and propagation data into RAM"""
        logger.info("Loading hot data into RAM...")

        try:
            self.db_file = h5py.File(self.db_path, "r")

            self.global_descriptors = self.db_file["global_descriptors"]["descriptors"][:]
            self.frame_poses = self.db_file["global_descriptors"]["frame_poses"][:]

            logger.info(f"Loaded global descriptors: shape {self.global_descriptors.shape}")
            logger.info(f"Loaded frame poses: shape {self.frame_poses.shape}")

            for key, value in self.db_file["metadata"].attrs.items():
                self.metadata[key] = value
                logger.debug(f"Metadata - {key}: {value}")

            if "actual_num_frames" in self.metadata:
                actual_num = int(self.metadata["actual_num_frames"])
                # DO NOT SLICE with actual_num_frames! The arrays are sized to num_frames exactly,
                # and are indexed by absolute visual frame_id!
            
            if "frame_index_map" in self.db_file["metadata"]:
                self.frame_index_map = self.db_file["metadata"]["frame_index_map"][:]
            else:
                self.frame_index_map = np.arange(len(self.global_descriptors))

            # Завантажуємо дані пропагації якщо є
            self._load_propagation_data()

            logger.success("Hot data loaded successfully into RAM")

        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            raise

    def _load_propagation_data(self) -> None:
        if self.db_file is None or "calibration" not in self.db_file:
            logger.info("No propagation data in database (not calibrated yet)")
            self.frame_affine = None
            self.frame_valid = None
            return
        try:
            grp = self.db_file["calibration"]

            # 1. Відновлення проєкції (пріоритет)
            if "projection_json" in grp.attrs:
                try:
                    meta = json.loads(grp.attrs["projection_json"])
                    self.converter = CoordinateConverter.from_metadata(meta)
                    logger.success(f"Projection restored from HDF5: {meta['mode']}")
                except Exception as e:
                    logger.warning(f"Could not load projection metadata: {e}")
            elif "reference_gps" in grp.attrs:
                # Fallback для v2.0 (UTM)
                try:
                    ref_gps = json.loads(grp.attrs["reference_gps"])
                    self.converter = CoordinateConverter("UTM", tuple(ref_gps))
                    logger.success(f"UTM auto-initialized from legacy reference GPS: {ref_gps}")
                except Exception as e:
                    logger.warning(f"Could not init UTM from legacy attr: {e}")
            else:
                # Fallback для v1.0 (WebMercator)
                logger.info("No projection metadata found. Defaulting to WEB_MERCATOR fallback.")
                self.converter = CoordinateConverter("WEB_MERCATOR")

            # 2. Завантаження датасетів
            if "frame_affine" in grp:
                self.frame_affine = grp["frame_affine"][:]
                self.frame_valid = grp["frame_valid"][:].astype(bool)

                # Метрики якості (QA)
                self.frame_rmse = grp["frame_rmse"][:] if "frame_rmse" in grp else None
                self.frame_disagreement = (
                    grp["frame_disagreement"][:] if "frame_disagreement" in grp else None
                )
                self.frame_matches = grp["frame_matches"][:] if "frame_matches" in grp else None

                valid_count = int(np.sum(self.frame_valid))
                logger.success(f"Propagation data loaded: {valid_count} frames valid")
            else:
                logger.warning("Found calibration group but no frame_affine dataset.")
                self.frame_affine = None
                self.frame_valid = None
        except Exception as e:
            logger.error(f"Failed to load propagation data: {e}")
            self.frame_affine = None
            self.frame_valid = None

    @property
    def is_propagated(self) -> bool:
        return self.frame_affine is not None

    def get_frame_affine(self, frame_id: int) -> np.ndarray | None:
        """Повертає афінну матрицю для конкретного кадру"""
        if not self.is_propagated or self.frame_affine is None or self.frame_valid is None:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.frame_affine[frame_id]

    def get_frame_size(self, frame_id: int) -> tuple[int, int]:
        """Повертає (height, width) для вказаного кадру"""
        if frame_id in self._size_cache:
            return self._size_cache[frame_id]

        if self.db_file is None:
            return 1080, 1920

        # Схема v2: розміри збережені один раз в local_features.attrs
        schema = self.metadata.get("hdf5_schema", "v1")
        if schema == "v2" and "local_features" in self.db_file:
            lf_attrs = self.db_file["local_features"].attrs
            h = int(lf_attrs.get("frame_height", self.metadata.get("frame_height", 1080)))
            w = int(lf_attrs.get("frame_width", self.metadata.get("frame_width", 1920)))
            self._size_cache[frame_id] = (h, w)
            return h, w

        # Схема v1: fallback — спочатку metadata, потім група кадру
        h = self.metadata.get("frame_height") or self.metadata.get("height") or 1080
        w = self.metadata.get("frame_width") or self.metadata.get("width") or 1920
        group_name = f"local_features/frame_{frame_id}"
        if group_name in self.db_file:
            g = self.db_file[group_name]
            if "height" in g.attrs and "width" in g.attrs:
                h, w = int(g.attrs["height"]), int(g.attrs["width"])

        res = (int(h), int(w))
        self._size_cache[frame_id] = res
        return res

    def get_local_features(self, frame_id: int) -> dict[str, np.ndarray]:
        """Повертає локальні ознаки XFeat для вказаного кадру"""
        if frame_id in self._feature_cache:
            return self._feature_cache[frame_id]

        if self.db_file is None:
            raise RuntimeError("Database not opened")

        schema = self.metadata.get("hdf5_schema", "v1")
        g = self.db_file["local_features"]

        if schema == "v2":
            n = int(g["kp_counts"][frame_id])
            if n == 0:
                raise ValueError(f"Кадр {frame_id} не має keypoints (kp_count=0).")
            res = {
                "keypoints": g["keypoints"][frame_id, :n],
                "descriptors": g["descriptors"][frame_id, :n].astype(np.float32),  # float16→32
                "coords_2d": g["coords_2d"][frame_id, :n],
            }
        else:
            # Стара схема v1 — зворотня сумісність
            if f"frame_{frame_id}" in g:
                old_g = g[f"frame_{frame_id}"]
                res = {
                    "keypoints": old_g["keypoints"][:],
                    "descriptors": old_g["descriptors"][:],
                    "coords_2d": old_g["coords_2d"][:],
                }
            else:
                # v1 pre-allocated без груп
                num = int(g["num_kp"][frame_id])
                res = {
                    "keypoints": g["keypoints"][frame_id, :num],
                    "descriptors": g["descriptors"][frame_id, :num].astype(np.float32),
                    "coords_2d": g["coords_2d"][frame_id, :num],
                }

        # Обмежуємо розмір кешу (аналог lru_cache з maxsize=200)
        if len(self._feature_cache) > 200:
            self._feature_cache.pop(next(iter(self._feature_cache)))

        self._feature_cache[frame_id] = res
        return res

    def get_num_frames(self) -> int:
        """Повертає кількість pre-allocated слотів у БД (індексація за абсолютним frame_id)."""
        return int(self.metadata.get("num_frames", 0))

    def close(self) -> None:
        if self.db_file is not None:
            self.db_file.close()
            self.db_file = None
            logger.info("Database file closed")

        # Очищення кешу при закритті БД
        self._size_cache.clear()
        self._feature_cache.clear()

# ================================================================================
# File: database\__init__.py
# ================================================================================
"""Database module"""


# ================================================================================
# File: geometry\coordinates.py
# ================================================================================
import math
from typing import Any

from pyproj import CRS, Transformer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoordinateConverter:
    """Детермінована конвертація координат (WebMercator або UTM) на основі екземпляра."""

    def __init__(
        self, mode: str = "WEB_MERCATOR", reference_gps: tuple[float, float] | None = None
    ):
        self._mode = mode.upper()
        self._reference_gps = reference_gps
        self._transformer_to_metric: Transformer | None = None
        self._transformer_to_gps: Transformer | None = None
        self._initialized = False

        if self._mode == "WEB_MERCATOR":
            self._initialize_projection(0.0, 0.0)
        elif self._reference_gps:
            self._initialize_projection(*self._reference_gps)

    def _initialize_projection(self, lat: float, lon: float) -> None:
        wgs84_crs = CRS("EPSG:4326")

        if self._mode == "UTM":
            if self._reference_gps is None:
                self._reference_gps = (lat, lon)
                logger.warning(f"Auto-initializing UTM reference from point: {self._reference_gps}")

            ref_lat, ref_lon = self._reference_gps
            zone_number = int((ref_lon + 180) / 6) + 1
            target_crs = CRS(proj="utm", zone=zone_number, ellps="WGS84")
            logger.info(
                f"Initialized UTM projection for zone {zone_number} based on ({ref_lat:.4f}, {ref_lon:.4f})"
            )
        else:
            target_crs = CRS("EPSG:3857")
            logger.info("Initialized WEB_MERCATOR projection (EPSG:3857)")

        self._transformer_to_metric = Transformer.from_crs(wgs84_crs, target_crs, always_xy=True)
        self._transformer_to_gps = Transformer.from_crs(target_crs, wgs84_crs, always_xy=True)
        self._initialized = True

    def gps_to_metric(self, lat: float, lon: float) -> tuple[float, float]:
        if not self._initialized:
            if self._mode == "WEB_MERCATOR":
                self._initialize_projection(lat, lon)
            else:
                raise RuntimeError(
                    "CoordinateConverter (UTM) must be initialized with reference_gps."
                )

        if self._transformer_to_metric is None:
            raise RuntimeError("Transformer not initialized.")

        x, y = self._transformer_to_metric.transform(lon, lat)
        return float(x), float(y)

    def metric_to_gps(self, x: float, y: float) -> tuple[float, float]:
        if not self._initialized:
            if self._mode == "WEB_MERCATOR":
                self._initialize_projection(0.0, 0.0)
            else:
                raise RuntimeError("CoordinateConverter is not initialized.")

        if self._transformer_to_gps is None:
            raise RuntimeError("Transformer not initialized.")

        lon, lat = self._transformer_to_gps.transform(x, y)
        return float(lat), float(lon)

    def export_metadata(self) -> dict[str, Any]:
        """Експорт налаштувань для серіалізації."""
        return {"mode": self._mode, "reference_gps": self._reference_gps}

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> "CoordinateConverter":
        """Створення конвертера з метаданих."""
        if not meta:
            return cls("WEB_MERCATOR")
        mode = meta.get("mode", "WEB_MERCATOR")
        ref = meta.get("reference_gps")
        return cls(mode, tuple(ref) if ref else None)

    @staticmethod
    def haversine_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
        """Розрахунок фізичної відстані між двома GPS точками в метрах."""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000  # Радіус Землі

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Глобальний екземпляр для зворотної сумісності (тимчасово)
DEFAULT_CONVERTER = CoordinateConverter("WEB_MERCATOR")


# ================================================================================
# File: geometry\transformations.py
# ================================================================================
import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GeometryTransforms:
    """Geometric transformations for localization with robust estimation (MAGSAC++)"""

    @staticmethod
    def is_matrix_valid(
        M: np.ndarray,
        is_homography: bool = False,
        min_scale: float = 0.001,
        max_scale: float = 100.0,
        max_shear: float = 0.8,
    ) -> bool:
        """
        Check if the transformation matrix is physically realistic for drone imagery.

        Args:
            M: Transformation matrix (2x3 for Affine or 3x3 for Homography)
            is_homography: True if M is a 3x3 Homography matrix
            min_scale: Minimum allowed scale factor
            max_scale: Maximum allowed scale factor
            max_shear: Maximum allowed shear (dot product of normalized basis vectors)
        """
        if M is None:
            return False

        try:
            if is_homography:
                # For Homography, we care about the affine part for stability checks
                if M.shape != (3, 3):
                    return False
                # Normalize by M[2,2] if possible
                if abs(M[2, 2]) < 1e-9:
                    return False
                M = M / M[2, 2]

                # Check perspective components (should be very small for top-down drone imagery)
                # If these are large, the corners will fly off to infinity
                if abs(M[2, 0]) > 0.005 or abs(M[2, 1]) > 0.005:
                    logger.debug(
                        f"Matrix invalid: Extreme perspective warp ({M[2, 0]:.5f}, {M[2, 1]:.5f})"
                    )
                    return False

                A = M[:2, :2]
                det = np.linalg.det(A)
            else:
                if M.shape != (2, 3):
                    return False
                A = M[:2, :2]
                det = np.linalg.det(A)

            # 1. Determinant must be non-zero (prevent degenerate matrices)
            if abs(det) < 1e-9:
                logger.debug(
                    f"Matrix invalid: Degenerate matrix with determinant near zero ({det})"
                )
                return False
            # Allow negative determinant since mapping Image Y (down) to Map Y (up) requires reflection!

            # 2. Extract scale and shear from basis vectors
            u = A[:, 0]
            v = A[:, 1]
            scale_u = np.linalg.norm(u)
            scale_v = np.linalg.norm(v)

            # Check scale bounds (drone altitude/zoom sanity)
            if not (min_scale < scale_u < max_scale and min_scale < scale_v < max_scale):
                logger.debug(f"Matrix invalid: Scale out of bounds ({scale_u:.2f}, {scale_v:.2f})")
                return False

            # 3. Check Aspect Ratio (should be close to 1.0 for drone imagery)
            aspect_ratio = scale_u / (scale_v + 1e-9)
            if not (0.5 < aspect_ratio < 2.0):
                logger.debug(
                    f"Matrix invalid: Extreme aspect ratio distortion ({aspect_ratio:.2f})"
                )
                return False

            # 4. Check Shear (cos of angle between basis vectors)
            shear = abs(np.dot(u, v) / (scale_u * scale_v + 1e-9))
            if shear > max_shear:
                logger.debug(f"Matrix invalid: Extreme shear detected ({shear:.2f} > {max_shear})")
                return False

            # 5. Check Rotation Stability (avoid 180-degree flips if not expected, though drones can rotate)
            # For now, we allow any rotation as drones can turn, but we could cap it if we have IMU data.

            return True

        except Exception as e:
            logger.error(f"Error during matrix validation: {e}")
            return False

    @staticmethod
    def estimate_homography(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 3.0,
        max_iters: int = 2000,
        confidence: float = 0.99,
        fallback_to_affine: bool = True,
    ):
        """
        Estimate Homography using MAGSAC++ with validation and optional fallback.
        """
        if len(src_pts) < 4:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        # Use USAC_MAGSAC for better outlier rejection and accuracy (OpenCV 4.5+)
        H, mask = cv2.findHomography(
            src_pts_cv,
            dst_pts_cv,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )

        # Validate Homography
        if not GeometryTransforms.is_matrix_valid(H, is_homography=True):
            if fallback_to_affine:
                logger.warning("Homography invalid/degenerate, falling back to Partial Affine")
                return GeometryTransforms.estimate_affine_partial(
                    src_pts, dst_pts, ransac_threshold
                )
            return None, None

        return H, mask

    @staticmethod
    def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        if H is None or len(points) == 0:
            return points
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.perspectiveTransform(points_cv, H)
        return transformed_pts_cv.reshape(-1, 2)

    @staticmethod
    def estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray, ransac_threshold: float = 3.0):
        """Compute full Affine transformation (6 DoF) using MAGSAC++"""
        if len(src_pts) < 3:
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffine2D(
            src_pts_cv, dst_pts_cv, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

        if not GeometryTransforms.is_matrix_valid(M, is_homography=False):
            return None, None

        return M, mask

    @staticmethod
    def estimate_affine_partial(
        src_pts: np.ndarray, dst_pts: np.ndarray, ransac_threshold: float = 3.0
    ):
        """Compute STRICT Affine transformation (4 DoF: R+T+S only) using MAGSAC++"""
        if len(src_pts) < 2:  # Partial needs only 2 points minimum
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffinePartial2D(
            src_pts_cv, dst_pts_cv, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

        if not GeometryTransforms.is_matrix_valid(M, is_homography=False):
            return None, None

        return M, mask

    @staticmethod
    def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
        if M is None or len(points) == 0:
            return points
        points_cv = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_pts_cv = cv2.transform(points_cv, M)
        return transformed_pts_cv.reshape(-1, 2)


# ================================================================================
# File: geometry\__init__.py
# ================================================================================
"""Geometry module"""


# ================================================================================
# File: gui\main_window.py
# ================================================================================
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QStatusBar

from config.config import APP_CONFIG
from src.calibration.multi_anchor_calibration import MultiAnchorCalibration
from src.core.project import ProjectManager
from src.database.database_loader import DatabaseLoader
from src.gui.mixins import CalibrationMixin, DatabaseMixin, PanoramaMixin, TrackingMixin
from src.gui.widgets.control_panel import ControlPanel
from src.gui.widgets.map_widget import MapWidget
from src.gui.widgets.video_widget import VideoWidget
from src.models.model_manager import ModelManager
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MainWindow(CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Topometric Localizer")
        self.setGeometry(100, 100, 1600, 900)

        self.config = APP_CONFIG
        self.model_manager = ModelManager(config=APP_CONFIG)
        self.project_manager = ProjectManager()
        self.database: DatabaseLoader | None = None
        self.calibration = MultiAnchorCalibration()

        # Workers
        self.db_worker = None
        self.tracking_worker = None
        self.propagation_worker = None
        self.pano_worker = None
        self._propagation_dialog = None

        self._init_ui()

    def _init_ui(self):
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

        self._create_menu_bar()
        self._connect_signals()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")
        file_menu.addAction("Вихід", self.close)

        calib_menu = menubar.addMenu("Калібрування")
        calib_menu.addAction("Додати якір...", self.on_calibrate)
        calib_menu.addAction("Завантажити калібрування...", self.on_load_calibration)
        calib_menu.addAction("Зберегти калібрування...", self.on_save_calibration)
        calib_menu.addSeparator()
        calib_menu.addAction("Запустити пропагацію вручну", self.on_run_propagation)
        calib_menu.addAction("Перевірити пропагацію на карті", self.on_verify_propagation)

        view_menu = menubar.addMenu("Вигляд")
        view_menu.addAction(self.control_dock.toggleViewAction())
        view_menu.addAction(self.map_dock.toggleViewAction())

    def _connect_signals(self):
        cp = self.control_panel
        cp.new_mission_clicked.connect(self.on_new_mission)
        cp.load_database_clicked.connect(self.on_load_database)
        cp.rebuild_database_clicked.connect(self.on_rebuild_database)
        cp.start_tracking_clicked.connect(self.on_start_tracking)
        cp.stop_tracking_clicked.connect(self.on_stop_tracking)
        cp.calibrate_clicked.connect(self.on_calibrate)
        cp.load_calibration_clicked.connect(self.on_load_calibration)
        cp.generate_panorama_clicked.connect(self.on_generate_panorama)
        cp.show_panorama_clicked.connect(self.on_show_panorama)
        cp.localize_image_clicked.connect(self.on_localize_image)
        cp.verify_propagation_clicked.connect(self.on_verify_propagation)
        cp.clear_map_clicked.connect(self.map_widget.clear_trajectory)
        cp.export_results_clicked.connect(self.on_export_results)


# ================================================================================
# File: gui\__init__.py
# ================================================================================
"""GUI module - PyQt6 interface"""


# ================================================================================
# File: gui\dialogs\calibration_dialog.py
# ================================================================================
import re

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from src.gui.widgets.video_widget import VideoWidget
from src.utils.image_utils import opencv_to_qpixmap

_UNKNOWN_FRAME_COUNT = 99999  # fallback when codec doesn't report frame count


class CalibrationDialog(QDialog):
    """
    Multi-anchor GPS calibration dialog.

    Workflow:
      1. Load video / image from database
      2. Navigate to target frame (slider)
      3. Click landmarks → enter GPS coordinates
      4. "Add anchor" — anchor saved, points cleared
      5. Repeat for other frames (first / middle / last)
      6. "Done" — triggers full-database propagation
    """

    anchor_added = pyqtSignal(object)  # dict: {points_2d, points_gps, calib_frame_id}
    anchor_removed = pyqtSignal(int)  # frame_id
    anchor_confirmed = pyqtSignal(int)  # frame_id actually saved (from MainWindow)
    calibration_complete = pyqtSignal()

    def __init__(self, database_path: str, existing_anchors=None, parent=None):
        super().__init__(parent)
        self.database_path = database_path
        self.existing_anchors = list(existing_anchors or [])

        self.points_2d = []
        self.points_gps = []
        self.current_2d_point = None
        self.cap = None
        self.last_slider_value = 0
        self._is_video = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self.is_playing = False

        self.setWindowTitle("GPS Калібрування — Мульти-якірний режим")
        self.resize(1200, 800)
        self._init_ui()
        self._refresh_anchors_list()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        self.setAcceptDrops(True)  # На майбутнє, якщо знадобиться

        # Left panel — video
        left = QVBoxLayout()

        self.btn_load_frame = QPushButton("📂  Завантажити відео / зображення")
        self.btn_load_frame.setStyleSheet("padding: 8px; font-weight: bold;")
        self.btn_load_frame.clicked.connect(self.load_frame)

        self.video_widget = VideoWidget()
        self.video_widget.frame_clicked.connect(self.on_video_clicked)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)

        player_row = QHBoxLayout()
        self.btn_step_back = QPushButton("◀◀")
        self.btn_play = QPushButton("▶")
        self.btn_step = QPushButton("▶▶")
        self.lbl_frame_info = QLabel("Кадр: 0 / 0")
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame_info.setStyleSheet("color: #555; font-size: 12px;")

        for btn in [self.btn_step_back, self.btn_play, self.btn_step]:
            btn.setFixedWidth(48)
            btn.setEnabled(False)
            player_row.addWidget(btn)
        player_row.addWidget(self.lbl_frame_info, stretch=1)

        self.btn_step_back.clicked.connect(self.step_backward)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_step.clicked.connect(self.step_forward)

        left.addWidget(self.btn_load_frame)
        left.addWidget(self.video_widget, stretch=1)
        left.addWidget(self.slider)
        left.addLayout(player_row)

        # Right panel — controls
        right = QVBoxLayout()
        right.setSpacing(6)

        anchors_group = QGroupBox("Додані якорі")
        anchors_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        ag = QVBoxLayout(anchors_group)
        self.anchors_list = QListWidget()
        self.anchors_list.setMaximumHeight(130)  # Трохи збільшимо
        self.anchors_list.setStyleSheet("font-size: 11px;")
        self.anchors_list.itemClicked.connect(self.on_anchor_selected)

        btn_row = QHBoxLayout()
        self.btn_delete_anchor = QPushButton("🗑 Видалити якір")
        self.btn_delete_anchor.setStyleSheet("color: #b71c1c;")
        self.btn_delete_anchor.setEnabled(False)
        self.btn_delete_anchor.clicked.connect(self.delete_selected_anchor)
        btn_row.addWidget(self.btn_delete_anchor)

        hint = QLabel("💡 Рекомендовано: перший кадр → середина → останній")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        ag.addWidget(self.anchors_list)
        ag.addLayout(btn_row)
        ag.addWidget(hint)

        frame_group = QGroupBox("ID кадру в базі даних")
        frame_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        fg = QHBoxLayout(frame_group)
        self.spinbox_frame_id = QSpinBox()
        self.spinbox_frame_id.setRange(0, _UNKNOWN_FRAME_COUNT)
        self.spinbox_frame_id.setValue(0)
        self.spinbox_frame_id.setToolTip(
            "При роботі з відео — заповнюється автоматично зі слайдера."
        )
        fg.addWidget(QLabel("Кадр №:"))
        fg.addWidget(self.spinbox_frame_id)

        self.lbl_frame_id_warning = QLabel("")
        self.lbl_frame_id_warning.setStyleSheet("color: #e65100; font-size: 11px;")
        self.lbl_frame_id_warning.setWordWrap(True)

        pts_group = QGroupBox("Точки прив'язки (для поточного якоря)")
        pts_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pts = QVBoxLayout(pts_group)

        self.lbl_selected_px = QLabel("Клікніть на орієнтир у відео ↖")
        self.lbl_selected_px.setStyleSheet(
            "font-weight: bold; color: #1565C0; padding: 4px;"
            "background: #E3F2FD; border-radius: 4px;"
        )

        combined_row = QHBoxLayout()
        self.input_combined = QLineEdit()
        self.input_combined.setPlaceholderText("Вставте координати: 47.820343, 34.927702")
        self.input_combined.textChanged.connect(self.parse_combined_gps)
        combined_row.addWidget(QLabel("Разом:"))
        combined_row.addWidget(self.input_combined)

        coords_row = QHBoxLayout()
        self.input_lat = QLineEdit()
        self.input_lat.setPlaceholderText("Широта")
        self.input_lon = QLineEdit()
        self.input_lon.setPlaceholderText("Довгота")
        coords_row.addWidget(QLabel("Lat:"))
        coords_row.addWidget(self.input_lat)
        coords_row.addWidget(QLabel("Lon:"))
        coords_row.addWidget(self.input_lon)

        self.btn_add_point = QPushButton("➕  Додати точку")
        self.btn_add_point.clicked.connect(self.add_point_pair)

        self.points_list = QListWidget()
        self.points_list.setMaximumHeight(110)
        self.points_list.setStyleSheet("font-size: 11px;")

        self.btn_clear_points = QPushButton("🗑  Очистити поточні точки")
        self.btn_clear_points.setStyleSheet("color: #b71c1c; font-size: 11px;")
        self.btn_clear_points.clicked.connect(self.clear_current_points)

        pts.addWidget(self.lbl_selected_px)
        pts.addLayout(combined_row)
        pts.addLayout(coords_row)
        pts.addWidget(self.btn_add_point)
        pts.addWidget(self.points_list)
        pts.addWidget(self.btn_clear_points)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ccc;")

        self.btn_add_anchor = QPushButton("⚓  Додати якір для цього кадру")
        self.btn_add_anchor.setStyleSheet(
            "background:#1565C0; color:white; font-weight:bold; padding:11px; font-size:13px;"
        )
        self.btn_add_anchor.clicked.connect(self.add_anchor)

        self.lbl_status = QLabel("Додайте мінімум 1 якір щоб продовжити")
        self.lbl_status.setStyleSheet("color:#666; font-size:11px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setWordWrap(True)

        self.btn_done = QPushButton("✅  Готово — запустити пропагацію по всій базі")
        self.btn_done.setStyleSheet(
            "background:#2e7d32; color:white; font-weight:bold; padding:11px; font-size:13px;"
        )
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self.finish_calibration)

        self.btn_cancel = QPushButton("Скасувати")
        self.btn_cancel.setStyleSheet("color:#555; padding:7px;")
        self.btn_cancel.clicked.connect(self.reject)

        right.addWidget(anchors_group)
        right.addWidget(frame_group)
        right.addWidget(self.lbl_frame_id_warning)
        right.addWidget(pts_group, stretch=1)
        right.addWidget(sep)
        right.addWidget(self.btn_add_anchor)
        right.addWidget(self.lbl_status)
        right.addWidget(self.btn_done)
        right.addWidget(self.btn_cancel)

        main_layout.addLayout(left, stretch=2)
        main_layout.addLayout(right, stretch=1)

    # ── Parsing GPS ──────────────────────────────────────────────────────────

    def parse_combined_gps(self, text: str):
        matches = re.findall(r"-?\d+\.\d+", text)
        if len(matches) >= 2:
            self.input_lat.setText(matches[0])
            self.input_lon.setText(matches[1])

    # ── Anchor list ──────────────────────────────────────────────────────────

    def _refresh_anchors_list(self):
        self.anchors_list.clear()
        if not self.existing_anchors:
            item = QListWidgetItem("  (поки немає якорів)")
            item.setForeground(QColor("#aaa"))
            self.anchors_list.addItem(item)
        else:
            # Сортуємо за frame_id
            sorted_anchors = sorted(self.existing_anchors, key=lambda a: a.get("frame_id", 0))
            for i, anchor in enumerate(sorted_anchors):
                fid = anchor.get("frame_id", 0)
                n_pts = len(anchor.get("qa_data", {}).get("points_2d", []))
                rmse = anchor.get("qa_data", {}).get("rmse_m", 0.0)

                label = f"⚓ Кадр {fid} | {n_pts} точок"
                if n_pts > 0:
                    label += f" | RMSE: {rmse:.2f}м"

                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, fid)
                item.setForeground(QColor("#1565C0"))
                self.anchors_list.addItem(item)

        has = bool(self.existing_anchors)
        self.btn_done.setEnabled(has)
        self.btn_delete_anchor.setEnabled(False)  # Очищуємо вибір

        if has:
            self.lbl_status.setText(
                f"Додано якорів: {len(self.existing_anchors)}. "
                "Оберіть якір для редагування або додайте новий."
            )
            self.lbl_status.setStyleSheet("color:#2e7d32; font-size:11px;")
        else:
            self.lbl_status.setText("Додайте мінімум 1 якір щоб продовжити")
            self.lbl_status.setStyleSheet("color:#666; font-size:11px;")

    def on_anchor_selected(self, item):
        frame_id = item.data(Qt.ItemDataRole.UserRole)
        if frame_id is None:
            return

        if self.points_2d or self.current_2d_point:
            reply = QMessageBox.question(
                self,
                "Увага",
                "У вас є незбережені точки. Перейти до іншого кадру?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        anchor = next((a for a in self.existing_anchors if a.get("frame_id") == frame_id), None)
        if not anchor:
            return

        self.btn_delete_anchor.setEnabled(True)

        # Перехід на кадр
        if self._is_video:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_id)
            self.slider.blockSignals(False)
            self._jump_to_frame(frame_id)

        # Завантаження точок
        self.clear_current_points()
        qa = anchor.get("qa_data", {})
        pts_2d = qa.get("points_2d", [])
        pts_gps = qa.get("points_gps", [])

        if pts_2d and pts_gps:
            self.points_2d = [tuple(p) for p in pts_2d]
            self.points_gps = [tuple(p) for p in pts_gps]
            for i, (p2d, pgps) in enumerate(zip(self.points_2d, self.points_gps)):
                self.points_list.addItem(
                    f"  {i + 1}. ({p2d[0]}, {p2d[1]}) → {pgps[0]:.5f}, {pgps[1]:.5f}"
                )
            self._redraw_points()
        else:
            self.lbl_status.setText(f"⚠ У кадру {frame_id} немає точок для редагування.")
            self.lbl_status.setStyleSheet("color:#b71c1c; font-size:11px;")

    def delete_selected_anchor(self):
        curr = self.anchors_list.currentItem()
        if not curr:
            return

        frame_id = curr.data(Qt.ItemDataRole.UserRole)
        if frame_id is None:
            return

        reply = QMessageBox.question(
            self,
            "Видалення",
            f"Видалити якір для кадру {frame_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        # Видаляємо локально
        self.existing_anchors = [a for a in self.existing_anchors if a.get("frame_id") != frame_id]
        self._refresh_anchors_list()
        self.clear_current_points()

        # Сигнал у MainWindow
        self.anchor_removed.emit(frame_id)

    def _jump_to_frame(self, frame_id: int):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.last_slider_value = frame_id
            self.spinbox_frame_id.setValue(frame_id)
            self.video_widget.display_frame(opencv_to_qpixmap(frame))
            self.lbl_frame_info.setText(f"Кадр: {frame_id} / {self.slider.maximum()}")

    def on_anchor_confirmed(self, frame_id: int):
        """Called by MainWindow after affine matrix is successfully computed."""
        # Отримуємо оновлені дані з MainWindow (опціонально, але MainWindow і так оновлює calibration в пам'яті)
        # У MIXIN ми вже додали якір в self.calibration. Тут ми просто хочемо оновити UI діалогу.

        # Перевіримо, чи це оновлення існуючого
        existing = next((a for a in self.existing_anchors if a.get("frame_id") == frame_id), None)

        # Оскільки діалог не має прямого доступу до об'єкта calibration з MainWindow,
        # нам треба або передати сюди новий dict, абоMainWindow сам оновить список при наступному відкритті.
        # Але ми хочемо оновити список ЗАРАЗ.

        # ХАК: оскільки ми щойно додали/оновили якір, але діалог не знає НОВІ точки (він знає лише ті, що в self.points_2d),
        # ми можемо "імітувати" оновлення об'єкта в списку діалогу.
        new_data = {
            "frame_id": frame_id,
            "qa_data": {
                "points_2d": list(self.points_2d),
                "points_gps": list(self.points_gps),
                "rmse_m": 0.0,  # Буде оновлено при наступному відкритті або якщо MainWindow надішле повний об'єкт
            },
        }

        if existing:
            idx = self.existing_anchors.index(existing)
            self.existing_anchors[idx] = new_data
        else:
            self.existing_anchors.append(new_data)

        self._refresh_anchors_list()
        self.clear_current_points()  # ОЧИЩАЄМО після збереження за запитом користувача

        QMessageBox.information(
            self,
            "⚓ Якір додано",
            f"Якір для кадру {frame_id} успішно збережено!",
        )

    # ── Video loading ────────────────────────────────────────────────────────

    def load_frame(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Відео або зображення",
            "",
            "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mkv *.mov);;"
            "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mkv *.mov)",
        )
        if not path:
            return

        self.clear_current_points()

        if path.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            self._load_video(path)
        else:
            self._load_image(path)

    def _load_video(self, path: str):
        self._is_video = True
        if self.cap:
            self.cap.release()

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити:\n{path}")
            return

        self.cap = cap
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = _UNKNOWN_FRAME_COUNT  # unknown length codec

        self.slider.blockSignals(True)
        self.slider.setEnabled(True)
        self.slider.setRange(0, total - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        for btn in [self.btn_play, self.btn_step_back, self.btn_step]:
            btn.setEnabled(True)

        self.spinbox_frame_id.setMaximum(total - 1)
        self.spinbox_frame_id.setValue(0)
        self.lbl_frame_id_warning.setText("")
        self.last_slider_value = 0
        self.on_slider_changed(0)

    def _load_image(self, path: str):
        self._is_video = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.slider.setEnabled(False)
        for btn in [self.btn_play, self.btn_step_back, self.btn_step]:
            btn.setEnabled(False)

        self.lbl_frame_info.setText("Статичне зображення")
        self.lbl_frame_id_warning.setText(
            "⚠ Статичне зображення: вкажіть вручну ID кадру з відео бази даних."
        )

        with open(path, "rb") as f:
            raw = bytearray(f.read())
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self.video_widget.display_frame(opencv_to_qpixmap(frame))
        else:
            QMessageBox.warning(self, "Помилка", "Не вдалося прочитати зображення.")

    # ── Playback ─────────────────────────────────────────────────────────────

    def toggle_playback(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(1000 / fps) if fps > 0 else 33)
            self.btn_play.setText("⏸")
            self.is_playing = True

    def play_next_frame(self):
        if not (self.cap and self.cap.isOpened()):
            return
        ret, frame = self.cap.read()
        if not ret:
            self.toggle_playback()
            return
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.last_slider_value = cur
        self.slider.blockSignals(True)
        self.slider.setValue(cur)
        self.slider.blockSignals(False)
        self.spinbox_frame_id.setValue(cur)
        self.video_widget.display_frame(opencv_to_qpixmap(frame))
        self.lbl_frame_info.setText(f"Кадр: {cur} / {self.slider.maximum()}")

    def step_forward(self):
        if self.is_playing:
            self.toggle_playback()
        self.play_next_frame()

    def step_backward(self):
        if self.is_playing:
            self.toggle_playback()
        if not (self.cap and self.cap.isOpened()):
            return
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
        self.play_next_frame()

    def on_slider_changed(self, value: int):
        if self.is_playing:
            return

        if not (self.cap and self.cap.isOpened()):
            return

        if value == self.last_slider_value:
            return

        if self.points_2d or self.current_2d_point:
            self.slider.blockSignals(True)
            self.slider.setValue(self.last_slider_value)
            self.slider.blockSignals(False)

            reply = QMessageBox.question(
                self,
                "Увага",
                "Зміна кадру очистить незбережені точки. Продовжити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

            self.clear_current_points()

            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)

        self.last_slider_value = value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        ret, frame = self.cap.read()

        if not ret:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, (value / fps) * 1000.0)
                ret, frame = self.cap.read()

        if ret and frame is not None:
            self.spinbox_frame_id.setValue(value)
            self.video_widget.display_frame(opencv_to_qpixmap(frame))
            self.lbl_frame_info.setText(f"Кадр: {value} / {self.slider.maximum()}")

    # ── Points ───────────────────────────────────────────────────────────────

    def on_video_clicked(self, x: int, y: int):
        self.current_2d_point = (x, y)
        self.lbl_selected_px.setText(f"✔ Обрано піксель: ({x}, {y})")
        self._redraw_points()

    def add_point_pair(self):
        if not self.current_2d_point:
            QMessageBox.warning(self, "Помилка", "Спочатку клікніть на орієнтир у відео!")
            return
        try:
            lat = float(self.input_lat.text().strip())
            lon = float(self.input_lon.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Помилка", "Введіть числові координати.")
            return

        self.points_2d.append(self.current_2d_point)
        self.points_gps.append((lat, lon))
        n = len(self.points_2d)
        self.points_list.addItem(
            f"  {n}. ({self.current_2d_point[0]}, {self.current_2d_point[1]})"
            f"  →  {lat:.5f}, {lon:.5f}"
        )
        self.current_2d_point = None
        self.lbl_selected_px.setText("Клікніть на наступний орієнтир")
        self.input_combined.clear()
        self.input_lat.clear()
        self.input_lon.clear()
        self._redraw_points()

    def clear_current_points(self):
        self.points_2d.clear()
        self.points_gps.clear()
        self.current_2d_point = None
        self.points_list.clear()
        self.input_combined.clear()
        self.video_widget.clear_overlays()
        self.lbl_selected_px.setText("Клікніть на орієнтир у відео ↖")

    def _redraw_points(self):
        self.video_widget.clear_overlays()
        for i, pt in enumerate(self.points_2d):
            self.video_widget.draw_numbered_point(pt[0], pt[1], str(i + 1), QColor(0, 200, 0))
        if self.current_2d_point:
            self.video_widget.draw_numbered_point(
                self.current_2d_point[0], self.current_2d_point[1], "?", QColor(255, 80, 0)
            )

    # ── Add anchor ───────────────────────────────────────────────────────────

    def add_anchor(self):
        if len(self.points_2d) < 4:
            QMessageBox.warning(
                self,
                "Увага",
                "Потрібно мінімум 4 точки для якоря!\n\n"
                "💡 Рекомендовано: 5-6 точок, розставлених широко\n"
                "по всьому кадру для найкращої точності.",
            )
            return

        frame_id = self.spinbox_frame_id.value()

        if frame_id in self.existing_anchors:
            reply = QMessageBox.question(
                self,
                "Якір існує",
                f"Якір для кадру {frame_id} вже є. Замінити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.anchor_added.emit(
            {
                "points_2d": list(self.points_2d),
                "points_gps": list(self.points_gps),
                "calib_frame_id": frame_id,
            }
        )

    # ── Finish ───────────────────────────────────────────────────────────────

    def finish_calibration(self):
        if not self.existing_anchors:
            QMessageBox.warning(self, "Увага", "Додайте хоча б один якір!")
            return

        if self.points_2d:
            reply = QMessageBox.question(
                self,
                "Незбережені точки",
                f"У вас {len(self.points_2d)} незбережених точок для кадру "
                f"{self.spinbox_frame_id.value()}.\n"
                f"Додати їх як якір перед завершенням?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.add_anchor()
                return
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self.calibration_complete.emit()
        self.accept()

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.is_playing:
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)


# ================================================================================
# File: gui\dialogs\new_mission_dialog.py
# ================================================================================
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NewMissionDialog(QDialog):
    """Dialog for creating a new localization mission."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Створення нової місії")
        self.setMinimumWidth(420)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.mission_name_edit = QLineEdit()
        self.mission_name_edit.setPlaceholderText("Введіть назву проєкту")
        form.addRow("Назва проєкту:", self.mission_name_edit)

        workspace_row = QHBoxLayout()
        self.workspace_path_edit = QLineEdit()
        self.workspace_path_edit.setReadOnly(True)
        self.workspace_path_edit.setPlaceholderText("Шлях до робочої папки не вибрано")
        btn_browse_workspace = QPushButton("Огляд...")
        btn_browse_workspace.clicked.connect(self._browse_workspace)
        workspace_row.addWidget(self.workspace_path_edit)
        workspace_row.addWidget(btn_browse_workspace)
        form.addRow("Робоча папка (Workspace):", workspace_row)

        video_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Шлях до відео не вибрано")
        btn_browse = QPushButton("Огляд...")
        btn_browse.clicked.connect(self._browse_video)
        video_row.addWidget(self.video_path_edit)
        video_row.addWidget(btn_browse)
        form.addRow("Еталонне відео:", video_row)

        # Camera parameters — used for GSD / FOV calculations in DatabaseBuilder
        self.altitude_spinbox = QDoubleSpinBox()
        self.altitude_spinbox.setRange(10.0, 5000.0)
        self.altitude_spinbox.setValue(100.0)
        self.altitude_spinbox.setSuffix(" м")
        form.addRow("Висота польоту:", self.altitude_spinbox)

        self.focal_length_spinbox = QDoubleSpinBox()
        self.focal_length_spinbox.setRange(1.0, 100.0)
        self.focal_length_spinbox.setValue(13.2)
        self.focal_length_spinbox.setSuffix(" мм")
        form.addRow("Фокусна відстань:", self.focal_length_spinbox)

        self.sensor_width_spinbox = QDoubleSpinBox()
        self.sensor_width_spinbox.setRange(1.0, 50.0)
        self.sensor_width_spinbox.setValue(8.8)
        self.sensor_width_spinbox.setSuffix(" мм")
        form.addRow("Ширина сенсора:", self.sensor_width_spinbox)

        self.image_width_spinbox = QSpinBox()
        self.image_width_spinbox.setRange(640, 8000)
        self.image_width_spinbox.setValue(4000)
        self.image_width_spinbox.setSuffix(" px")
        form.addRow("Ширина зображення:", self.image_width_spinbox)

        layout.addLayout(form)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ── Slots ────────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "Виберіть робочу папку (Workspace)", "")
        if path:
            self.workspace_path_edit.setText(path)

    @pyqtSlot()
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть еталонне відео",
            "",
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    @pyqtSlot()
    def _on_accept(self):
        if not self.mission_name_edit.text().strip():
            QMessageBox.warning(self, "Помилка", "Введіть назву проєкту!")
            self.mission_name_edit.setFocus()
            return
        if not self.workspace_path_edit.text():
            QMessageBox.warning(self, "Помилка", "Виберіть робочу папку (Workspace)!")
            return
        if not self.video_path_edit.text():
            QMessageBox.warning(self, "Помилка", "Виберіть еталонне відео!")
            return
        self.accept()

    # ── Data ─────────────────────────────────────────────────────────────────

    def get_mission_data(self) -> dict:
        data = {
            "mission_name": self.mission_name_edit.text().strip(),
            "workspace_dir": self.workspace_path_edit.text(),
            "video_path": self.video_path_edit.text(),
            "altitude_m": self.altitude_spinbox.value(),
            "focal_length_mm": self.focal_length_spinbox.value(),
            "sensor_width_mm": self.sensor_width_spinbox.value(),
            "image_width_px": self.image_width_spinbox.value(),
        }
        logger.info(
            f"Mission: '{data['mission_name']}' in '{data['workspace_dir']}' | "
            f"video={data['video_path']} | alt={data['altitude_m']}m"
        )
        return data


# ================================================================================
# File: gui\dialogs\open_project_dialog.py
# ================================================================================
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.core.project_registry import ProjectRegistry
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OpenProjectDialog(QDialog):
    """
    Діалог вибору проєкту зі списку нещодавніх.
    Замінює голий QFileDialog.getExistingDirectory.
    """

    def __init__(self, registry: ProjectRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.selected_path: str | None = None

        self.setWindowTitle("Відкрити проєкт")
        self.setMinimumSize(600, 450)
        self._init_ui()
        self._populate_list()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Пошук
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Пошук за назвою проєкту...")
        self.search_input.textChanged.connect(self._on_search)
        search_row.addWidget(self.search_input)
        layout.addLayout(search_row)

        # Список проєктів
        self.project_list = QListWidget()
        self.project_list.setAlternatingRowColors(True)
        self.project_list.setStyleSheet(
            "QListWidget { font-size: 13px; }"
            "QListWidget::item { padding: 8px 6px; }"
            "QListWidget::item:selected { background: #1565C0; color: white; }"
        )
        self.project_list.itemDoubleClicked.connect(self._on_double_click)
        self.project_list.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.project_list, stretch=1)

        # Preview панель
        self.preview_group = QGroupBox("Деталі проєкту")
        preview_layout = QVBoxLayout(self.preview_group)
        self.lbl_preview = QLabel("Виберіть проєкт зі списку")
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color: #666; font-size: 12px;")
        preview_layout.addWidget(self.lbl_preview)
        layout.addWidget(self.preview_group)

        # Кнопки
        buttons_row = QHBoxLayout()

        self.btn_browse = QPushButton("📂 Інша папка...")
        self.btn_browse.setToolTip("Відкрити проєкт з довільної папки")
        self.btn_browse.clicked.connect(self._on_browse)

        self.btn_remove = QPushButton("🗑 Видалити зі списку")
        self.btn_remove.setToolTip("Видаляє лише зі списку, файли залишаються")
        self.btn_remove.setStyleSheet("color: #b71c1c;")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove)

        self.btn_open = QPushButton("✅ Відкрити")
        self.btn_open.setStyleSheet(
            "background: #1565C0; color: white; font-weight: bold; padding: 8px 20px;"
        )
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._on_open)

        self.btn_cancel = QPushButton("Скасувати")
        self.btn_cancel.clicked.connect(self.reject)

        buttons_row.addWidget(self.btn_browse)
        buttons_row.addWidget(self.btn_remove)
        buttons_row.addStretch()
        buttons_row.addWidget(self.btn_cancel)
        buttons_row.addWidget(self.btn_open)
        layout.addLayout(buttons_row)

    def _populate_list(self, filter_text: str = ""):
        """Заповнити список проєктів."""
        self.project_list.clear()
        projects = self.registry.get_recent(limit=50)

        for proj in projects:
            name = proj.get("name", "Без назви")
            if filter_text and filter_text.lower() not in name.lower():
                continue

            # Статус-іконки
            has_db = proj.get("has_database", False)
            has_cal = proj.get("has_calibration", False)
            status = ""
            if has_db and has_cal:
                status = "✅"
            elif has_db:
                status = "⚠️ без калібрування"
            else:
                status = "❌ без бази"

            # Формат дати
            last = proj.get("last_opened", "")
            try:
                dt = datetime.fromisoformat(last)
                date_str = dt.strftime("%d.%m.%Y %H:%M")
            except (ValueError, TypeError):
                date_str = "—"

            item_text = f"{status}  {name}   [останній: {date_str}]"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, proj)

            # Позначаємо недоступні проєкти
            if not Path(proj["path"]).is_dir():
                item.setForeground(QColor("#aaa"))
                item.setToolTip("⚠ Папка проєкту не знайдена")

            self.project_list.addItem(item)

        if self.project_list.count() == 0:
            item = QListWidgetItem("    (немає проєктів — створіть новий або відкрийте папку)")
            item.setForeground(QColor("#999"))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.project_list.addItem(item)

    def _on_search(self, text: str):
        self._populate_list(filter_text=text)

    def _on_selection_changed(self, current: QListWidgetItem, _previous):
        if current is None:
            self.btn_open.setEnabled(False)
            self.btn_remove.setEnabled(False)
            self.lbl_preview.setText("Виберіть проєкт зі списку")
            return

        proj = current.data(Qt.ItemDataRole.UserRole)
        if proj is None:
            self.btn_open.setEnabled(False)
            self.btn_remove.setEnabled(False)
            return

        self.btn_open.setEnabled(True)
        self.btn_remove.setEnabled(True)

        # Preview
        path = proj.get("path", "")
        video = proj.get("video_path", "—")
        created = proj.get("created_at", "—")
        try:
            created = datetime.fromisoformat(created).strftime("%d.%m.%Y %H:%M")
        except (ValueError, TypeError):
            pass

        db_size = "—"
        db_path = Path(path) / "database.h5"
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            db_size = f"{size_mb:.1f} MB"

        self.lbl_preview.setText(
            f"<b>Назва:</b> {proj.get('name', '—')}<br>"
            f"<b>Шлях:</b> {path}<br>"
            f"<b>Відео:</b> {Path(video).name if video else '—'}<br>"
            f"<b>Створено:</b> {created}<br>"
            f"<b>База даних:</b> {'✅ ' + db_size if proj.get('has_database') else '❌ відсутня'}<br>"
            f"<b>Калібрація:</b> {'✅ є' if proj.get('has_calibration') else '❌ відсутня'}"
        )

    def _on_double_click(self, item: QListWidgetItem):
        proj = item.data(Qt.ItemDataRole.UserRole)
        if proj and Path(proj["path"]).is_dir():
            self.selected_path = proj["path"]
            self.accept()

    def _on_open(self):
        current = self.project_list.currentItem()
        if current:
            proj = current.data(Qt.ItemDataRole.UserRole)
            if proj:
                if not Path(proj["path"]).is_dir():
                    QMessageBox.warning(
                        self, "Помилка", f"Папка проєкту не знайдена:\n{proj['path']}"
                    )
                    return
                self.selected_path = proj["path"]
                self.accept()

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Виберіть папку проєкту", "")
        if path:
            self.selected_path = path
            self.accept()

    def _on_remove(self):
        current = self.project_list.currentItem()
        if not current:
            return
        proj = current.data(Qt.ItemDataRole.UserRole)
        if not proj:
            return

        reply = QMessageBox.question(
            self,
            "Підтвердження",
            f"Видалити «{proj['name']}» зі списку?\n\nФайли проєкту НЕ будуть видалені.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.registry.unregister(proj["path"])
            self._populate_list(self.search_input.text())

    def get_selected_path(self) -> str | None:
        return self.selected_path


# ================================================================================
# File: gui\dialogs\__init__.py
# ================================================================================
"""Dialogs module"""


# ================================================================================
# File: gui\mixins\calibration_mixin.py
# ================================================================================
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from config.config import get_cfg
from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.gui.dialogs.calibration_dialog import CalibrationDialog
from src.localization.matcher import FeatureMatcher
from src.utils.logging_utils import get_logger
from src.workers.calibration_propagation_worker import CalibrationPropagationWorker

logger = get_logger(__name__)


class CalibrationMixin:
    # ── Calibration dialog ───────────────────────────────────────────────────

    @pyqtSlot()
    def on_calibrate(self):
        if not self.database or self.database.db_file is None:
            QMessageBox.warning(self, "Помилка", "Спочатку завантажте або створіть базу даних!")
            return

        # Тепер передаємо повні словники якорів для редагування
        anchors_data = [a.to_dict() for a in self.calibration.anchors]

        self._calib_dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=anchors_data,
            parent=self,
        )
        self._calib_dialog.anchor_added.connect(self.on_anchor_added)
        self._calib_dialog.anchor_removed.connect(self.on_anchor_removed)  # Новий сигнал
        self._calib_dialog.calibration_complete.connect(self.on_run_propagation)
        self._calib_dialog.exec()

        # Очищуємо посилання після закриття діалогу
        self._calib_dialog = None

    @pyqtSlot(object)
    def on_anchor_added(self, anchor_data: dict):
        try:
            points_2d = anchor_data.get("points_2d")
            points_gps = anchor_data.get("points_gps")
            frame_id = anchor_data.get("calib_frame_id")

            if not points_2d or not points_gps or len(points_2d) < 4:
                QMessageBox.warning(self, "Помилка", "Потрібно мінімум 4 точки для якоря!")
                return

            # Налаштування проєкції, якщо вона ще не ініціалізована
            if not self.calibration.converter._initialized:
                # Використовуємо налаштування з конфігу або за замовчуванням
                mode = get_cfg(self.config, "projection.default_mode", "WEB_MERCATOR")
                reference_gps = points_gps[0] if mode == "UTM" else None
                self.calibration.converter = CoordinateConverter(mode, reference_gps)

            pts_2d_np = np.array(points_2d, dtype=np.float32)
            pts_metric = [
                self.calibration.converter.gps_to_metric(lat, lon) for lat, lon in points_gps
            ]
            pts_metric_np = np.array(pts_metric, dtype=np.float32)

            # 1. Спроба обчислити різні типи трансформацій
            M_partial, _ = GeometryTransforms.estimate_affine_partial(pts_2d_np, pts_metric_np)

            best_M = M_partial
            best_type = "affine_partial"  # 4-DoF (Scale, Rotate, Translate)

            def calc_metrics(M, src, dst):
                proj = GeometryTransforms.apply_affine(src, M)
                errs = np.linalg.norm(proj - dst, axis=1)
                return (
                    float(np.sqrt(np.mean(errs**2))),
                    float(np.median(errs)),
                    float(np.max(errs)),
                    proj.tolist(),
                )

            rmse_p, median_p, max_p, proj_p = calc_metrics(M_partial, pts_2d_np, pts_metric_np)

            # Якщо точок >= 5, пробуємо повний Affine (6 DoF)
            if len(pts_2d_np) >= 5:
                M_full, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
                if M_full is not None:
                    rmse_f, median_f, max_f, proj_f = calc_metrics(M_full, pts_2d_np, pts_metric_np)
                    # Вибираємо повний Affine тільки якщо він дає суттєве покращення (>15%)
                    if rmse_f < rmse_p * 0.85:
                        best_M = M_full
                        best_type = "affine_full"
                        rmse_p, median_p, max_p, proj_p = rmse_f, median_f, max_f, proj_f
                        logger.info(
                            f"Selected full affine for anchor {frame_id} (RMSE: {rmse_f:.2f}m)"
                        )

            if best_M is None:
                QMessageBox.critical(
                    self,
                    "Помилка",
                    "Не вдалося обчислити матрицю. Спробуйте іншу комбінацію точок.",
                )
                return

            # 2. Перевірка порогів якості з конфігу
            rmse_threshold = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)
            max_err_threshold = get_cfg(self.config, "projection.anchor_max_error_m", 5.0)

            # 3. QA Діалог підтвердження
            from datetime import datetime

            severity_color = "green"
            if rmse_p > rmse_threshold:
                severity_color = "red"
            elif rmse_p > rmse_threshold * 0.7:
                severity_color = "orange"

            qa_summary = (
                f"<b>Метрики якості для якоря (кадр {frame_id}):</b><br><br>"
                f"Трансформація: <code style='color:blue'>{best_type}</code><br>"
                f"Кількість точок: <b>{len(pts_2d_np)}</b><br>"
                f"RMSE: <b style='color:{severity_color}'>{rmse_p:.2f} м</b> (поріг: {rmse_threshold}м)<br>"
                f"Медіанна похибка: <b>{median_p:.2f} м</b><br>"
                f"Макс. похибка: <b>{max_p:.2f} м</b> (поріг: {max_err_threshold}м)<br>"
            )

            if rmse_p > rmse_threshold or max_p > max_err_threshold:
                qa_summary += "<br><span style='color:red'>⚠ Увага: Якість прив'язки нижча за рекомендовану!</span>"
                reply = QMessageBox.warning(
                    self,
                    "Якість калібрування",
                    qa_summary + "<br><br>Зберегти цей якір попри високу похибку?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            else:
                logger.success(f"Anchor {frame_id} QA passed: RMSE={rmse_p:.2f}m")

            # 4. Збереження результатів
            qa_data = {
                "rmse_m": rmse_p,
                "median_err_m": median_p,
                "max_err_m": max_p,
                "inliers_count": len(pts_2d_np),
                "transform_type": best_type,
                "projection_mode": self.calibration.converter._mode,
                "created_at": datetime.now().isoformat(),
                "points_2d": points_2d,
                "points_gps": points_gps,
                "points_metric": pts_metric,
            }

            self.calibration.add_anchor(frame_id=frame_id, affine_matrix=best_M, qa_data=qa_data)

            if self.project_manager and self.project_manager.is_loaded:
                self.calibration.save(self.project_manager.calibration_path)

            # ДІАГНОСТИКА: Детальний лог точок (тепер на рівні INFO)
            logger.info(f"--- Anchor {frame_id} Point-by-Point Analysis ---")
            for j in range(len(pts_2d_np)):
                p2d = pts_2d_np[j]
                pm = pts_metric_np[j]
                if best_M is not None:
                    trans = GeometryTransforms.apply_affine(p2d.reshape(1, 2), best_M)[0]
                    err = np.linalg.norm(trans - pm)

                    # Перевіряємо зворотну конвертацію для візуалізації зсуву в градусах
                    lat_c, lon_c = self.calibration.converter.metric_to_gps(
                        float(trans[0]), float(trans[1])
                    )
                    lat_t, lon_t = points_gps[j][0], points_gps[j][1]

                    dist_err = CoordinateConverter.haversine_distance(
                        (lat_c, lon_c), (lat_t, lon_t)
                    )

                    logger.info(
                        f"  Pt {j}: px={p2d} -> err={err:.3f}м ({dist_err:.3f}м по Хаверсину)"
                    )
                    logger.debug(
                        f"    GPS Calc: ({lat_c:.7f}, {lon_c:.7f}) | Target: ({lat_t:.7f}, {lon_t:.7f})"
                    )

            logger.info(
                f"Anchor {frame_id} QA Summary: {best_type} | points={len(pts_2d_np)} | "
                f"RMSE={rmse_p:.3f}м | MedianErr={median_p:.3f}м | MaxErr={max_p:.3f}м"
            )

            self.status_bar.showMessage(f"Додано якір (кадр {frame_id}, RMSE: {rmse_p:.2f}м)")

            if hasattr(self, "_calib_dialog") and self._calib_dialog is not None:
                self._calib_dialog.on_anchor_confirmed(frame_id)

        except Exception as e:
            logger.error(f"Failed to add anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося додати якір:\n{e}")

    @pyqtSlot(int)
    def on_anchor_removed(self, frame_id: int):
        """Видалення якоря та маркування пропагації застарілою."""
        try:
            if self.calibration.remove_anchor(frame_id):
                if self.project_manager and self.project_manager.is_loaded:
                    self.calibration.save(self.project_manager.calibration_path)

                logger.info(f"Anchor {frame_id} removed from project")
                self.status_bar.showMessage(
                    f"Якір {frame_id} видалено. Потрібно оновити пропагацію.", 5000
                )
        except Exception as e:
            logger.error(f"Failed to remove anchor: {e}", exc_info=True)
            QMessageBox.critical(self, "Помилка", f"Не вдалося видалити якір:\n{e}")

    # ── Propagation ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_run_propagation(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Додайте хоча б один якір калібрування!")
            return
        if not self.database:
            QMessageBox.warning(self, "Увага", "База даних не завантажена!")
            return

        try:
            # ОНОВЛЕНО: Ініціалізуємо FeatureMatcher, він автоматично підлаштується
            # під розмірність дескрипторів (64 для XFeat) та використає Numpy L2
            matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося ініціалізувати матчер:\n{e}")
            return

        anchor_ids = [a.frame_id for a in self.calibration.anchors]
        n_frames = self.database.get_num_frames()
        logger.info(f"Propagation: {len(anchor_ids)} anchors {anchor_ids}, {n_frames} frames")

        self._propagation_dialog = QProgressDialog(
            f"Пропагація GPS від {len(anchor_ids)} якорів на {n_frames} кадрів...",
            "Скасувати",
            0,
            100,
            self,
        )
        self._propagation_dialog.setWindowTitle("Розповсюдження GPS координат")
        self._propagation_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._propagation_dialog.setMinimumDuration(0)
        self._propagation_dialog.setValue(0)

        self.propagation_worker = CalibrationPropagationWorker(
            database=self.database,
            calibration=self.calibration,
            matcher=matcher,
            config=self.config,
        )
        self.propagation_worker.progress.connect(self.on_propagation_progress)
        self.propagation_worker.completed.connect(self.on_propagation_completed)
        self.propagation_worker.error.connect(self.on_propagation_error)
        self._propagation_dialog.canceled.connect(self.propagation_worker.stop)
        self.propagation_worker.start()

    @pyqtSlot(int, str)
    def on_propagation_progress(self, percent: int, message: str):
        dialog = self._propagation_dialog
        if dialog is not None:
            try:
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

        # Обчислюємо статистику якості
        num_frames = self.database.get_num_frames()
        valid_mask = self.database.frame_valid
        valid_count = int(np.sum(valid_mask)) if valid_mask is not None else 0

        avg_rmse = 0.0
        max_rmse = 0.0
        avg_dis = 0.0
        avg_matches = 0.0

        if valid_count > 0:
            rmse_data = getattr(self.database, "frame_rmse", None)
            if rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                avg_rmse = float(np.mean(valid_rmse))
                max_rmse = float(np.max(valid_rmse))

            dis_data = getattr(self.database, "frame_disagreement", None)
            if dis_data is not None:
                dis_valid = dis_data[valid_mask]
                if np.any(dis_valid > 0):
                    avg_dis = float(np.mean(dis_valid[dis_valid > 0]))

            matches_data = getattr(self.database, "frame_matches", None)
            if matches_data is not None:
                avg_matches = float(np.mean(matches_data[valid_mask]))

        rmse_thresh = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)

        report = (
            f"<b>Пропагація завершена!</b><br><br>"
            f"Валідних кадрів: <b>{valid_count} / {num_frames}</b> ({valid_count / num_frames * 100:.1f}%)<br>"
            f"Середній RMSE (grid): <b style='color:{'green' if avg_rmse < rmse_thresh * 0.5 else 'orange'}'>{avg_rmse:.3f} м</b><br>"
            f"Середній матчинг: <b>{avg_matches:.1f} точок</b><br>"
        )
        if avg_dis > 0:
            report += f"Середня розбіжність (drift): <b style='color:{'red' if avg_dis > 5.0 else 'green'}'>{avg_dis:.3f} м</b><br>"

        if avg_rmse > rmse_thresh or avg_dis > 5.0:
            report += "<br><span style='color:red'>⚠ Увага: Якість у деяких сегментах може бути нестабільною.</span>"
        else:
            report += "<br><span style='color:green'>✅ Результати стабільні. Можна починати локалізацію.</span>"

        QMessageBox.information(self, "Пропагація", report)
        self.status_bar.showMessage(
            f"Пропагація готова: {valid_count} к., RMSE: {avg_rmse:.2f}м, Mat: {avg_matches:.0f}"
        )

        if self.map_widget:
            self.on_verify_propagation()

    @pyqtSlot()
    def on_verify_propagation(self):
        """Візуалізація та звіт якості пропагації на мапі"""
        if not self.database or not self.database.is_propagated:
            QMessageBox.warning(self, "Увага", "Дані пропагації не знайдено.")
            return

        try:
            self.map_widget.clear_verification_markers()
            num_frames = self.database.get_num_frames()
            step = max(1, num_frames // 30)

            # Отримуємо дані один раз
            rmse_data = getattr(self.database, "frame_rmse", None)
            dis_data = getattr(self.database, "frame_disagreement", None)
            matches_data = getattr(self.database, "frame_matches", None)
            valid_mask = getattr(self.database, "frame_valid", None)

            points_to_show = []
            _diag_done = False
            for i in range(0, num_frames, step):
                affine = self.database.get_frame_affine(i)
                if affine is not None:
                    w = self.database.metadata.get("frame_width", 1920)
                    h = self.database.metadata.get("frame_height", 1080)

                    # ДІАГНОСТИКА: лише для першого кадру
                    if not _diag_done:
                        _diag_done = True
                        logger.warning(f"=== VERIFY DIAG frame={i} ===")
                        logger.warning(f"  frame_width={w}, frame_height={h}")
                        logger.warning(f"  affine=\n{affine}")
                        for lbl, px, py in [
                            ("corner0", 0, 0),
                            ("center", w / 2, h / 2),
                            ("corner2", w, h),
                        ]:
                            mx_d = affine[0, 0] * px + affine[0, 1] * py + affine[0, 2]
                            my_d = affine[1, 0] * px + affine[1, 1] * py + affine[1, 2]
                            lat_d, lon_d = self.calibration.converter.metric_to_gps(
                                float(mx_d), float(my_d)
                            )
                            logger.warning(
                                f"  {lbl}({px},{py}) -> metric({mx_d:.1f},{my_d:.1f}) -> GPS({lat_d:.6f},{lon_d:.6f})"
                            )

                    # 1. Центр кадру
                    mx, my = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h / 2) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h / 2) + affine[1, 2],
                    )
                    lat_c, lon_c = self.calibration.converter.metric_to_gps(float(mx), float(my))

                    # 2. Низ центру (часто там дорога)
                    mx_b, my_b = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h * 0.75) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h * 0.75) + affine[1, 2],
                    )
                    lat_b, lon_b = self.calibration.converter.metric_to_gps(
                        float(mx_b), float(my_b)
                    )

                    rmse = (
                        float(rmse_data[i]) if rmse_data is not None and i < len(rmse_data) else 0.0
                    )
                    dis = float(dis_data[i]) if dis_data is not None and i < len(dis_data) else 0.0
                    matches = (
                        int(matches_data[i])
                        if matches_data is not None and i < len(matches_data)
                        else 0
                    )

                    # Лог для вибраного кадру (кожен 3-й з тих що показуємо)
                    if (i // step) % 3 == 0:
                        logger.debug(
                            f"Verify Frame {i}: CENTER={lat_c:.6f},{lon_c:.6f} | BOTTOM={lat_b:.6f},{lon_b:.6f} | RMSE={rmse:.2f}m"
                        )

                    # Колір базується на комбінації факторів
                    color = "green"
                    if rmse > 5.0 or dis > 10.0:
                        color = "red"
                    elif rmse > 2.0 or dis > 3.0:
                        color = "orange"

                    # 3. Крайні точки (для візуалізації перекосу/масштабу)
                    # МАЛЮЄМО НЕ ВЕСЬ КАДР (1920x1080 - це величезна площа на карті!),
                    # а лише центральні 20% екрану, щоб прямокутник на карті не здавався гіпер-великим.
                    cx_px, cy_px = w / 2, h / 2
                    dw, dh = w * 0.1, h * 0.1  # 10% в кожну сторону від центру
                    pts_px = [
                        (cx_px - dw, cy_px - dh),
                        (cx_px + dw, cy_px - dh),
                        (cx_px + dw, cy_px + dh),
                        (cx_px - dw, cy_px + dh),
                    ]
                    for idx_p, (px, py) in enumerate(pts_px):
                        mx_p, my_p = (
                            affine[0, 0] * px + affine[0, 1] * py + affine[0, 2],
                            affine[1, 0] * px + affine[1, 1] * py + affine[1, 2],
                        )
                        lat_p, lon_p = self.calibration.converter.metric_to_gps(
                            float(mx_p), float(my_p)
                        )
                        points_to_show.append(
                            {
                                "lat": float(lat_p),
                                "lon": float(lon_p),
                                "label": f"Кадр {i} Корнер {idx_p}",
                                "color": "gray",
                            }
                        )

                    # Додаємо дві основні точки
                    points_to_show.append(
                        {
                            "lat": float(lat_c),
                            "lon": float(lon_c),
                            "label": f"Кадр {i} (Центр) | RMSE:{rmse:.1f}м | Mat:{matches}",
                            "color": color,
                        }
                    )
                    points_to_show.append(
                        {
                            "lat": float(lat_b),
                            "lon": float(lon_b),
                            "label": f"Кадр {i} (Низ) | RMSE:{rmse:.1f}м",
                            "color": "blue",
                        }
                    )

            if points_to_show:
                self.map_widget.show_verification_markers(points_to_show)

            if valid_mask is not None and rmse_data is not None:
                valid_rmse = rmse_data[valid_mask]
                if len(valid_rmse) > 0:
                    avg_rmse = float(np.mean(valid_rmse))
                    self.status_bar.showMessage(f"Пропагація: Середній RMSE = {avg_rmse:.3f} м")

        except Exception as e:
            logger.error(f"Error in on_verify_propagation: {e}", exc_info=True)
            self.status_bar.showMessage("Помилка візуалізації якості")

    @pyqtSlot(str)
    def on_propagation_error(self, error_msg: str):
        if self._propagation_dialog:
            self._propagation_dialog.close()
            self._propagation_dialog = None
        logger.error(f"Propagation error: {error_msg}")
        QMessageBox.critical(self, "Помилка пропагації", error_msg)

    # ── Save / Load calibration ──────────────────────────────────────────────

    @pyqtSlot()
    def on_save_calibration(self):
        if not self.calibration.is_calibrated:
            QMessageBox.warning(self, "Увага", "Немає даних для збереження.")
            return

        default_path = "calibration.json"
        if self.project_manager and self.project_manager.is_loaded:
            default_path = str(self.project_manager.project_dir / default_path)

        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти калібрування", default_path, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            self.calibration.save(path)
            n = len(self.calibration.anchors)
            self.status_bar.showMessage(f"Калібрування збережено: {path} ({n} якорів)")
            QMessageBox.information(
                self, "Збережено", f"Калібрування збережено!\nЯкорів: {n}\nФайл: {path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{e}")

    @pyqtSlot()
    def on_load_calibration(self):
        default_dir = ""
        if self.project_manager and self.project_manager.is_loaded:
            default_dir = str(self.project_manager.project_dir)

        path, _ = QFileDialog.getOpenFileName(
            self, "Завантажити калібрування", default_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self.calibration.load(path)
            ids = [a.frame_id for a in self.calibration.anchors]
            propagated = self.database and self.database.is_propagated
            self.status_bar.showMessage(f"Калібрування: {len(ids)} якорів, кадри {ids}")
            self.control_panel.update_status("Калібрування завантажено")
            QMessageBox.information(
                self,
                "Успіх",
                f"Завантажено {len(ids)} якір(ів)!\nКадри: {ids}\n\n"
                f"{'✅ БД вже має дані пропагації.' if propagated else '⚠ Запустіть пропагацію.'}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити:\n{e}")


# ================================================================================
# File: gui\mixins\database_mixin.py
# ================================================================================
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.core.export_results import ResultExporter
from src.core.project_registry import ProjectRegistry
from src.database.database_loader import DatabaseLoader
from src.gui.dialogs.new_mission_dialog import NewMissionDialog
from src.gui.dialogs.open_project_dialog import OpenProjectDialog
from src.utils.logging_utils import get_logger
from src.workers.database_worker import DatabaseGenerationWorker

logger = get_logger(__name__)


class DatabaseMixin:
    # ── Реєстр проєктів (ініціалізується один раз) ───────────────────────────

    def _get_registry(self) -> ProjectRegistry:
        if not hasattr(self, "_project_registry"):
            self._project_registry = ProjectRegistry()
        return self._project_registry

    # ── Нова місія ────────────────────────────────────────────────────────────

    @pyqtSlot()
    def on_new_mission(self):
        dialog = NewMissionDialog(self)
        if not dialog.exec():
            return

        mission_data = dialog.get_mission_data()
        workspace_dir = mission_data.get("workspace_dir")
        video_path = mission_data.get("video_path")

        if not workspace_dir or not video_path:
            return

        # Створюємо структуру проєкту
        if not self.project_manager.create_project(workspace_dir, mission_data):
            QMessageBox.critical(self, "Помилка", "Не вдалося створити проєкт!")
            return

        # Реєструємо в реєстрі
        self._get_registry().register(
            project_dir=str(self.project_manager.project_dir),
            name=self.project_manager.project_name,
            video_path=video_path,
        )

        self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")
        self._start_database_generation(video_path, self.project_manager.database_path)

    # ── Генерація бази ────────────────────────────────────────────────────────

    def _start_database_generation(self, video_path: str, save_path: str):
        from src.geometry.coordinates import CoordinateConverter

        self.calibration.converter = CoordinateConverter("WEB_MERCATOR")

        self.control_panel.btn_new_mission.setEnabled(False)
        self.control_panel.btn_load_db.setEnabled(False)
        self.control_panel.update_progress(0)
        self.control_panel.set_db_generation_running(True)

        # CRITICAL: Close and release the database file handle before overwriting/truncating it
        if hasattr(self, "database") and self.database:
            try:
                self.database.close()
                logger.info("Current database closed before starting new generation.")
            except Exception as e:
                logger.warning(f"Could not close database: {e}")
        self.database = None

        self.db_worker = DatabaseGenerationWorker(
            video_path=video_path,
            output_path=save_path,
            model_manager=self.model_manager,
            config=self.config,
        )
        self.db_worker.progress.connect(self.on_db_progress)
        self.db_worker.completed.connect(self.on_db_completed)
        self.db_worker.error.connect(self.on_db_error)
        self.db_worker.cancelled.connect(self.on_db_cancelled)

        # Connect stop button
        self.control_panel.stop_db_generation_clicked.connect(self.on_stop_db_generation)

        self.db_worker.start()

    @pyqtSlot()
    def on_stop_db_generation(self):
        if hasattr(self, "db_worker") and self.db_worker and self.db_worker.isRunning():
            self.control_panel.update_status("Зупинка... (чекаємо завершення кадру)")
            self.db_worker.stop()

    @pyqtSlot(int, str)
    def on_db_progress(self, percent: int, message: str):
        self.control_panel.update_progress(percent)
        self.control_panel.update_status(message)

    @pyqtSlot(str)
    def on_db_completed(self, db_path: str):
        self.control_panel.set_db_generation_running(False)
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.current_database_path = db_path

        if self.database:
            self.database.close()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.database = DatabaseLoader(db_path)
        finally:
            QApplication.restoreOverrideCursor()
        self.control_panel.update_progress(100)
        self.control_panel.update_status("Базу успішно створено")
        self.status_bar.showMessage(
            f"Проєкт: {self.project_manager.project_name} | База: {db_path}"
        )

        # Оновити реєстр та інфо-панель
        if self.project_manager.is_loaded:
            self._get_registry().refresh_status(str(self.project_manager.project_dir))
        self._update_project_info_panel()

        QMessageBox.information(self, "Успіх", "Проєкт та базу даних успішно згенеровано!")

    @pyqtSlot(str)
    def on_db_error(self, error_msg: str):
        self.control_panel.set_db_generation_running(False)
        self.control_panel.btn_new_mission.setEnabled(True)
        self.control_panel.btn_load_db.setEnabled(True)
        self.control_panel.update_progress(0)
        self.control_panel.update_status("Помилка генерації")
        QMessageBox.critical(self, "Помилка", f"Помилка генерації:\n{error_msg}")

    @pyqtSlot()
    def on_db_cancelled(self):
        self.control_panel.set_db_generation_running(False)
        self.control_panel.update_status("Генерацію скасовано користувачем")
        self.control_panel.update_progress(0)

    # ── Відкриття проєкту ─────────────────────────────────────────────────────

    @pyqtSlot()
    def on_load_database(self):
        dialog = OpenProjectDialog(self._get_registry(), parent=self)
        if not dialog.exec():
            self.status_bar.showMessage("Вибір проєкту скасовано")
            return

        path = dialog.get_selected_path()
        if not path:
            return

        self._open_project(path)

    def _open_project(self, path: str):
        """Завантажити проєкт за шляхом (використовується і для recent menu)."""
        if not self.project_manager.load_project(path):
            QMessageBox.critical(self, "Помилка", "Обрана папка не є валідним проєктом!")
            return

        from src.geometry.coordinates import CoordinateConverter

        self.calibration.converter = CoordinateConverter("WEB_MERCATOR")

        try:
            db_path = self.project_manager.database_path

            # НОВЕ: Перевірка наявності бази даних
            if not Path(db_path).exists():
                video_path = self.project_manager.settings.video_path
                reply = QMessageBox.question(
                    self,
                    "База даних відсутня",
                    f"Проєкт '{self.project_manager.project_name}' не має згенерованої бази даних.\n\n"
                    f"Згенерувати базу зараз з відео:\n{Path(video_path).name}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.setWindowTitle(
                        f"Drone Topometric Localizer - {self.project_manager.project_name}"
                    )
                    self._start_database_generation(video_path, db_path)
                    return
                else:
                    self.status_bar.showMessage("Завантаження скасовано: відсутня база даних")
                    return

            if self.database:
                self.database.close()

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.database = DatabaseLoader(db_path)
            finally:
                QApplication.restoreOverrideCursor()
            self.setWindowTitle(f"Drone Topometric Localizer - {self.project_manager.project_name}")

            # Оновити реєстр (завжди викликаємо register для збереження нових проєктів)
            registry = self._get_registry()
            registry.register(
                project_dir=str(self.project_manager.project_dir),
                name=self.project_manager.project_name,
                video_path=self.project_manager.settings.video_path
                if self.project_manager.settings
                else "",
            )

            # Завантажити калібрацію якщо є
            calib_path = self.project_manager.calibration_path
            if calib_path and Path(calib_path).exists():
                self.calibration.load(calib_path)

            if self.database.is_propagated:
                n_valid = int(self.database.frame_valid.sum())
                n_total = self.database.get_num_frames()
                self.status_bar.showMessage(
                    f"Проєкт: {self.project_manager.project_name} (GPS: {n_valid}/{n_total} кадрів)"
                )
            else:
                self.status_bar.showMessage(
                    f"Проєкт: {self.project_manager.project_name} (без GPS пропагації)"
                )
            self.control_panel.update_status("Проєкт завантажено")
            self._update_project_info_panel()

        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося завантажити базу проєкту:\n{e}")

    # ── Перевірка пропагації ─────────────────────────────────────────────────

    @pyqtSlot()
    def on_verify_propagation(self):
        if not self.database or not self.database.is_propagated:
            QMessageBox.warning(
                self, "Увага", "Дані пропагації відсутні або проєкт не завантажено!"
            )
            return

        import numpy as np

        num_frames = self.database.get_num_frames()
        frame_valid = self.database.frame_valid
        frame_affine = self.database.frame_affine

        # Отримуємо розміри кадру з метаданих
        width = self.database.metadata.get("frame_width", 1920)
        height = self.database.metadata.get("frame_height", 1080)

        # Центр кадру в пікселях
        center_px = np.array([[width / 2, height / 2]], dtype=np.float32)

        points_to_show = []

        # Збираємо тільки валідні кадри (з кроком 5 для продуктивності на карті)
        step = max(1, num_frames // 200)  # Максимум ~200 точок щоб не гальмував біндер

        for i in range(0, num_frames, step):
            if frame_valid[i]:
                # Приміняємо афінну матрицю (2x3)
                M = frame_affine[i]
                # Metric = M * [x, y, 1]^T
                metric_x = M[0, 0] * center_px[0, 0] + M[0, 1] * center_px[0, 1] + M[0, 2]
                metric_y = M[1, 0] * center_px[0, 0] + M[1, 1] * center_px[0, 1] + M[1, 2]

                lat, lon = self.calibration.converter.metric_to_gps(
                    float(metric_x), float(metric_y)
                )
                points_to_show.append({"lat": float(lat), "lon": float(lon), "label": str(i)})

        if not points_to_show:
            QMessageBox.information(
                self, "Інформація", "Не знайдено жодного кадру з валідними координатами."
            )
            return

        self.map_widget.show_verification_markers(points_to_show)
        self.status_bar.showMessage(f"Відображено {len(points_to_show)} точок перевірки на карті.")

    # ── Перегенерація бази ────────────────────────────────────────────────────

    @pyqtSlot()
    def on_rebuild_database(self):
        if not self.project_manager.is_loaded:
            QMessageBox.warning(self, "Увага", "Спочатку завантажте проєкт!")
            return

        video_path = self.project_manager.settings.video_path
        if not video_path or not Path(video_path).exists():
            QMessageBox.warning(
                self,
                "Увага",
                f"Відео проєкту не знайдено:\n{video_path}\n\n"
                "Перевірте шлях до відео у налаштуваннях проєкту.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Перегенерація бази",
            f"Базу даних буде перезаписано!\n\n"
            f"Відео: {Path(video_path).name}\n"
            f"Калібрація буде збережена.\n\n"
            f"Продовжити?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Зберігаємо калібрацію перед перегенерацією
        if self.calibration.is_calibrated:
            calib_path = self.project_manager.calibration_path
            if calib_path:
                self.calibration.save(calib_path)
                logger.info(f"Calibration saved before rebuild: {calib_path}")

        self._start_database_generation(video_path, self.project_manager.database_path)

    # ── Експорт результатів ───────────────────────────────────────────────────

    @pyqtSlot()
    def on_export_results(self):
        if not hasattr(self, "_tracking_results") or not self._tracking_results:
            QMessageBox.warning(
                self, "Увага", "Немає результатів для експорту!\n\nСпочатку виконайте відстеження."
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Експорт результатів",
            "tracking_results",
            "CSV (*.csv);;GeoJSON (*.geojson);;KML (*.kml)",
        )
        if not path:
            return

        try:
            if path.endswith(".csv") or "CSV" in selected_filter:
                if not path.endswith(".csv"):
                    path += ".csv"
                ResultExporter.export_csv(self._tracking_results, path)
            elif path.endswith(".geojson") or "GeoJSON" in selected_filter:
                if not path.endswith(".geojson"):
                    path += ".geojson"
                ResultExporter.export_geojson(self._tracking_results, path)
            elif path.endswith(".kml") or "KML" in selected_filter:
                if not path.endswith(".kml"):
                    path += ".kml"
                name = (
                    self.project_manager.project_name
                    if self.project_manager.is_loaded
                    else "Drone Track"
                )
                ResultExporter.export_kml(self._tracking_results, path, name=name)

            self.status_bar.showMessage(f"Результати експортовано: {path}")
            QMessageBox.information(
                self, "Успіх", f"Експортовано {len(self._tracking_results)} точок\n\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Помилка експорту:\n{e}")

    # ── Інфо-панель ───────────────────────────────────────────────────────────

    def _update_project_info_panel(self):
        """Оновити інформаційну панель проєкту у control_panel."""
        if not self.project_manager.is_loaded:
            self.control_panel.update_project_info()
            return

        num_frames = self.database.get_num_frames() if self.database else None
        num_anchors = len(self.calibration.anchors) if self.calibration else None
        num_propagated = None
        db_size_mb = None

        if self.database and self.database.is_propagated:
            num_propagated = int(self.database.frame_valid.sum())

        db_path = self.project_manager.database_path
        if db_path and Path(db_path).exists():
            db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)

        self.control_panel.update_project_info(
            project_name=self.project_manager.project_name,
            video_path=self.project_manager.settings.video_path
            if self.project_manager.settings
            else None,
            num_frames=num_frames,
            num_anchors=num_anchors,
            num_propagated=num_propagated,
            db_size_mb=db_size_mb,
        )


# ================================================================================
# File: gui\mixins\panorama_mixin.py
# ================================================================================
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
                if not res.get("success") or "fov_polygon" in res is False:
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


# ================================================================================
# File: gui\mixins\tracking_mixin.py
# ================================================================================
import cv2
import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from config.config import get_cfg
from src.localization.localizer import Localizer
from src.localization.matcher import FeatureMatcher
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.utils.image_utils import opencv_to_qpixmap
from src.utils.logging_utils import get_logger
from src.workers.tracking_worker import RealtimeTrackingWorker

logger = get_logger(__name__)


class TrackingMixin:
    def _build_localizer(self) -> Localizer:
        """Shared factory — used by tracking and single-image localization."""
        # ОНОВЛЕНО: Завантажуємо ALIKED та DINOv2
        xf = self.model_manager.load_aliked()
        nv = self.model_manager.load_dinov2()

        # Опціональне завантаження CESP для покращення DINOv2 global descriptors
        cesp = None
        if get_cfg(self.config, "models.cesp.enabled", False):
            try:
                cesp = self.model_manager.load_cesp()
            except Exception as e:
                logger.warning(f"CESP loading failed, continuing without it: {e}")

        fe = FeatureExtractor(
            xf, nv, self.model_manager.device, config=self.config, cesp_module=cesp
        )

        # ОНОВЛЕНО: Матчер сам вирішить (Numpy для XFeat або LightGlue для SuperPoint)
        matcher = FeatureMatcher(model_manager=self.model_manager, config=self.config)

        # Передаємо model_manager у конфіг для SuperPoint+LightGlue fallback
        localizer_config = {**self.config, "_model_manager": self.model_manager}
        return Localizer(self.database, fe, matcher, self.calibration, config=localizer_config)

    def _ensure_utm_initialized(self) -> bool:
        """Перевіряє чи ініціалізована проєкція UTM, якщо ні - пробує ініціалізувати з калібрування."""

        if self.calibration.converter._initialized:
            return True

        if self.calibration and self.calibration.reference_gps:
            self.calibration.converter.gps_to_metric(
                self.calibration.reference_gps[0], self.calibration.reference_gps[1]
            )
            return True

        QMessageBox.warning(
            self,
            "Помилка формату",
            "Проєкція UTM не ініціалізована.\n\n"
            "Схоже, що база даних створена у старій версії програми, або не була завантажена GPS-прив'язка.\n"
            "Будь ласка, завантажте файл калібрування (.json) або виконайте додавання GPS-якорів наново.",
        )
        return False

    @pyqtSlot()
    def on_start_tracking(self):
        if not self.database:
            QMessageBox.warning(self, "Увага", "Завантажте базу даних HDF5!")
            return
        if not self.calibration.is_calibrated and not (
            self.database and self.database.is_propagated
        ):
            QMessageBox.warning(
                self, "Увага", "Виконайте калібрування GPS або завантажте базу з пропагацією."
            )
            return
        if not self.database.is_propagated:
            reply = QMessageBox.question(
                self,
                "Увага",
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
            video_path,
            localizer,
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
        if (
            hasattr(self, "tracking_worker")
            and self.tracking_worker
            and self.tracking_worker.isRunning()
        ):
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
        if not self.calibration.is_calibrated and not (
            self.database and self.database.is_propagated
        ):
            QMessageBox.warning(
                self, "Увага", "Виконайте калібрування GPS або завантажте базу з пропагацією."
            )
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

            if hasattr(self.video_widget, "display_frame"):
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
                self.control_panel.update_status(
                    "Фото локалізовано (приблизно)" if is_fallback else "Фото локалізовано"
                )
                msg_box = QMessageBox(self)
                msg_title = "⚓ Приблизна локалізація" if is_fallback else "⚓ Успіх"
                msg_text = (
                    "Знайдено ПРИБЛИЗНІ координати (retrieval-only)!\n\n"
                    if is_fallback
                    else "Координати знайдено!\n\n"
                )
                msg_text += (
                    f"Широта: {lat:.6f}\nДовгота: {lon:.6f}\n"
                    f"Впевненість: {conf:.2f}\nТочок збігу: {inliers}\n"
                    f"Якір: кадр {anchor}"
                )

                msg_box.setWindowTitle(msg_title)
                msg_box.setText(msg_text)
                msg_box.setIcon(QMessageBox.Icon.Information)

                # Default OK button
                ok_btn = msg_box.addButton(QMessageBox.StandardButton.Ok)

                # Custom Copy button
                copy_btn = msg_box.addButton(
                    "📋 Копіювати координати", QMessageBox.ButtonRole.ActionRole
                )

                msg_box.exec()

                if msg_box.clickedButton() == copy_btn:
                    cb = QApplication.clipboard()
                    cb.setText(f"{lat:.6f}, {lon:.6f}")

            else:
                err = result.get("error", "Невідома помилка")
                self.status_bar.showMessage(f"Помилка: {err}")
                self.control_panel.update_status("Помилка локалізації")
                QMessageBox.warning(self, "Помилка", f"Не вдалося знайти координати:\n{err}")

        except Exception as e:
            logger.error(f"Image localization error: {e}", exc_info=True)
            QMessageBox.critical(self, "Критична помилка", str(e))
            self.status_bar.showMessage("Помилка обробки")

    @pyqtSlot(np.ndarray)
    def on_frame_ready(self, frame_rgb: np.ndarray):
        if hasattr(self.video_widget, "display_frame"):
            self.video_widget.display_frame(opencv_to_qpixmap(frame_rgb))

    @pyqtSlot(float, float, float, int)
    def on_location_found(self, lat: float, lon: float, confidence: float, inliers: int):
        self.map_widget.update_marker(lat, lon)
        self.map_widget.add_trajectory_point(lat, lon)

        # Зберігаємо результат для експорту
        if not hasattr(self, "_tracking_results"):
            self._tracking_results = []
        self._tracking_results.append(
            {
                "lat": lat,
                "lon": lon,
                "confidence": confidence,
                "inliers": inliers,
                "timestamp": str(np.datetime64("now")),
            }
        )
        if len(self._tracking_results) == 1:
            self.control_panel.btn_export.setEnabled(True)
        self.status_bar.showMessage(
            f"Локалізація: {lat:.6f}, {lon:.6f} | Впевненість: {confidence:.2f} | Точок: {inliers}"
        )


# ================================================================================
# File: gui\mixins\__init__.py
# ================================================================================
from .calibration_mixin import CalibrationMixin
from .database_mixin import DatabaseMixin
from .panorama_mixin import PanoramaMixin
from .tracking_mixin import TrackingMixin

__all__ = ["CalibrationMixin", "DatabaseMixin", "TrackingMixin", "PanoramaMixin"]


# ================================================================================
# File: gui\widgets\control_panel.py
# ================================================================================
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ControlPanel(QWidget):
    """Mission control sidebar — emits signals, holds no business logic."""

    new_mission_clicked = pyqtSignal()
    load_database_clicked = pyqtSignal()
    rebuild_database_clicked = pyqtSignal()
    start_tracking_clicked = pyqtSignal()
    stop_tracking_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    load_calibration_clicked = pyqtSignal()
    localize_image_clicked = pyqtSignal()
    generate_panorama_clicked = pyqtSignal()
    show_panorama_clicked = pyqtSignal()
    export_results_clicked = pyqtSignal()
    verify_propagation_clicked = pyqtSignal()
    clear_map_clicked = pyqtSignal()
    stop_db_generation_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.set_tracking_enabled(True)  # correct initial state on startup

    # ── UI ───────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Project group
        db_group = QGroupBox("Управління проєктом")
        db_layout = QVBoxLayout(db_group)

        self.btn_new_mission = QPushButton("Створити новий проєкт")
        self.btn_load_db = QPushButton("Відкрити проєкт")
        self.btn_rebuild_db = QPushButton("🔄 Перегенерувати базу")
        self.btn_rebuild_db.setToolTip("Перебудовує базу даних з оригінального відео проєкту")
        self.btn_rebuild_db.setEnabled(False)
        self.btn_gen_pano = QPushButton("Згенерувати панораму з відео")
        self.btn_show_pano = QPushButton("Накласти панораму на карту")

        self.btn_new_mission.clicked.connect(self.new_mission_clicked)
        self.btn_load_db.clicked.connect(self.load_database_clicked)
        self.btn_rebuild_db.clicked.connect(self.rebuild_database_clicked)
        self.btn_gen_pano.clicked.connect(self.generate_panorama_clicked)
        self.btn_show_pano.clicked.connect(self.show_panorama_clicked)

        self.btn_stop_db = QPushButton("⏹  Зупинити генерацію БД")
        self.btn_stop_db.setStyleSheet(
            "background:#c62828; color:white; font-weight:bold; padding:7px;"
        )
        self.btn_stop_db.setVisible(False)
        self.btn_stop_db.clicked.connect(self.stop_db_generation_clicked)

        for btn in [
            self.btn_new_mission,
            self.btn_load_db,
            self.btn_rebuild_db,
            self.btn_gen_pano,
            self.btn_show_pano,
            self.btn_stop_db,
        ]:
            db_layout.addWidget(btn)

        # Calibration group
        calib_group = QGroupBox("Калібрування GPS")
        calib_layout = QVBoxLayout(calib_group)

        self.btn_calibrate = QPushButton("Виконати калібрування (Video → Map)")
        self.btn_load_calibrate = QPushButton("Завантажити калібрування (JSON)")
        self.btn_verify_propagation = QPushButton("🔍 Перевірити пропагацію на карті")
        self.btn_verify_propagation.setToolTip(
            "Відображає центри всіх кадрів з обчисленими координатами на карті"
        )
        self.btn_clear_map = QPushButton("🗑 Очистити карту")
        self.btn_clear_map.setToolTip("Видалити траєкторію, панораму та маркери з карти")

        self.btn_calibrate.clicked.connect(self.calibrate_clicked)
        self.btn_load_calibrate.clicked.connect(self.load_calibration_clicked)
        self.btn_verify_propagation.clicked.connect(self.verify_propagation_clicked)
        self.btn_clear_map.clicked.connect(self.clear_map_clicked)

        calib_layout.addWidget(self.btn_calibrate)
        calib_layout.addWidget(self.btn_load_calibrate)
        calib_layout.addWidget(self.btn_verify_propagation)
        calib_layout.addWidget(self.btn_clear_map)

        # Localization group
        track_group = QGroupBox("Локалізація")
        track_layout = QVBoxLayout(track_group)

        self.btn_start_tracking = QPushButton("▶  Почати відстеження")
        self.btn_start_tracking.setStyleSheet(
            "background:#2e7d32; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_stop_tracking = QPushButton("■  Зупинити відстеження")
        self.btn_stop_tracking.setStyleSheet(
            "background:#c62828; color:white; font-weight:bold; padding:8px;"
        )
        self.btn_localize_image = QPushButton("🔍  Локалізувати одне фото")

        self.btn_start_tracking.clicked.connect(self.start_tracking_clicked)
        self.btn_stop_tracking.clicked.connect(self.stop_tracking_clicked)
        self.btn_localize_image.clicked.connect(self.localize_image_clicked)

        track_layout.addWidget(self.btn_start_tracking)
        track_layout.addWidget(self.btn_stop_tracking)
        track_layout.addWidget(self.btn_localize_image)

        # Export group
        export_group = QGroupBox("Результати")
        export_layout = QVBoxLayout(export_group)
        self.btn_export = QPushButton("📊 Експорт результатів")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results_clicked)
        export_layout.addWidget(self.btn_export)

        # Project info group
        self.info_group = QGroupBox("Інформація про проєкт")
        info_layout = QVBoxLayout(self.info_group)
        self.lbl_project_info = QLabel("Проєкт не завантажено")
        self.lbl_project_info.setWordWrap(True)
        self.lbl_project_info.setStyleSheet("font-size: 11px; color: #333;")
        info_layout.addWidget(self.lbl_project_info)

        # Status group
        status_group = QGroupBox("Статус системи")
        status_layout = QVBoxLayout(status_group)

        self.lbl_status = QLabel("Очікування команди...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-style:italic; color:#333; margin-bottom:6px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)

        for group in [
            db_group,
            calib_group,
            track_group,
            export_group,
            self.info_group,
            status_group,
        ]:
            layout.addWidget(group)

    # ── Public API ───────────────────────────────────────────────────────────

    def update_status(self, message: str):
        self.lbl_status.setText(message)
        logger.debug(f"Status: {message}")

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_db_generation_running(self, is_running: bool):
        """is_running=True: показати кнопку Stop, заблокувати решту кнопок проєкту."""
        self.btn_stop_db.setVisible(is_running)
        self.btn_new_mission.setEnabled(not is_running)
        self.btn_load_db.setEnabled(not is_running)
        self.btn_rebuild_db.setEnabled(not is_running)

    def set_tracking_enabled(self, enabled: bool):
        """
        enabled=True  → idle state   (Start active, Stop disabled)
        enabled=False → running state (Start disabled, Stop active)
        """
        self.btn_start_tracking.setEnabled(enabled)
        self.btn_stop_tracking.setEnabled(not enabled)

        # Disable DB/calibration ops during tracking to prevent GPU OOM
        for btn in [
            self.btn_new_mission,
            self.btn_load_db,
            self.btn_rebuild_db,
            self.btn_calibrate,
            self.btn_load_calibrate,
            self.btn_verify_propagation,
            self.btn_clear_map,
            self.btn_localize_image,
            self.btn_gen_pano,
            self.btn_export,
        ]:
            btn.setEnabled(enabled)

    def update_project_info(
        self,
        project_name: str = None,
        video_path: str = None,
        num_frames: int = None,
        num_anchors: int = None,
        num_propagated: int = None,
        db_size_mb: float = None,
    ):
        """Оновити інформаційну панель проєкту."""
        if project_name is None:
            self.lbl_project_info.setText("Проєкт не завантажено")
            self.lbl_project_info.setStyleSheet("font-size: 11px; color: #222;")
            self.btn_rebuild_db.setEnabled(False)
            return

        from pathlib import Path

        lines = [f"▶ <b>{project_name}</b>"]
        if video_path:
            lines.append(f"🎥 {Path(video_path).name}")
        if num_frames is not None:
            db_info = f"🗃 Кадрів: {num_frames}"
            if db_size_mb is not None:
                db_info += f" ({db_size_mb:.1f} MB)"
            lines.append(db_info)
        if num_anchors is not None:
            lines.append(f"⚓ Якорів: {num_anchors}")
        if num_propagated is not None and num_frames is not None:
            lines.append(f"📍 GPS: {num_propagated}/{num_frames} кадрів")

        self.lbl_project_info.setText("<br>".join(lines))
        self.lbl_project_info.setStyleSheet("font-size: 11px; color: #000;")
        self.btn_rebuild_db.setEnabled(True)


# ================================================================================
# File: gui\widgets\map_widget.py
# ================================================================================
import json
from pathlib import Path

from PyQt6.QtCore import QObject, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Resolved once at import time — safe for both dev and frozen builds
_MAP_PATH = Path(__file__).resolve().parent.parent / "resources" / "maps" / "map_template.html"


class MapBridge(QObject):
    """
    Qt↔JavaScript signal bus via QWebChannel.
    Signals here are consumed by map_template.html — not connected to Python slots.
    """

    updateMarkerSignal = pyqtSignal(float, float)
    addTrajectorySignal = pyqtSignal(float, float)
    clearTrajectorySignal = pyqtSignal()

    # 8 floats: TL, TR, BR, BL corners (lat, lon each)
    updateFOVSignal = pyqtSignal(float, float, float, float, float, float, float, float)

    # data_url (base64 JPEG) + 8 corner coords
    setPanoramaSignal = pyqtSignal(str, float, float, float, float, float, float, float, float)

    # Verification markers (JSON string of points)
    showVerificationMarkersSignal = pyqtSignal(str)
    clearVerificationMarkersSignal = pyqtSignal()


class MapWidget(QWebEngineView):
    """Interactive map widget backed by Leaflet via QWebChannel."""

    def __init__(self, parent=None):
        super().__init__(parent)

        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        self.bridge = MapBridge()
        self._channel = QWebChannel()
        self._channel.registerObject("mapBridge", self.bridge)
        self.page().setWebChannel(self._channel)

        self._load_map()

    # ── Map loading ──────────────────────────────────────────────────────────

    def _load_map(self):
        if _MAP_PATH.exists():
            self.setUrl(QUrl.fromLocalFile(str(_MAP_PATH)))
            logger.info(f"Map loaded: {_MAP_PATH}")
        else:
            logger.error(f"Map template not found: {_MAP_PATH}")
            self.setHtml(
                f"""
                <html><body style='font-family:Arial;padding:20px'>
                <h2 style='color:red'>Помилка: Файл карти не знайдено!</h2>
                <p>Очікуваний шлях:</p>
                <code style='background:#eee;padding:5px'>{_MAP_PATH}</code>
                </body></html>
            """
            )

    # ── Public API ───────────────────────────────────────────────────────────

    @pyqtSlot(float, float)
    def update_marker(self, lat: float, lon: float):
        self.bridge.updateMarkerSignal.emit(lat, lon)

    @pyqtSlot(float, float)
    def add_trajectory_point(self, lat: float, lon: float):
        self.bridge.addTrajectorySignal.emit(lat, lon)

    @pyqtSlot()
    def clear_trajectory(self):
        self.bridge.clearTrajectorySignal.emit()

    @pyqtSlot(list)
    def update_fov(self, fov: list):
        """
        Accepts FOV as a Python list: [(lat0,lon0), (lat1,lon1), (lat2,lon2), (lat3,lon3)]
        """
        if not fov or len(fov) != 4:
            logger.warning(f"update_fov: expected 4 points, got {len(fov) if fov else 0}")
            return

        try:
            self.bridge.updateFOVSignal.emit(
                float(fov[0][0]),
                float(fov[0][1]),
                float(fov[1][0]),
                float(fov[1][1]),
                float(fov[2][0]),
                float(fov[2][1]),
                float(fov[3][0]),
                float(fov[3][1]),
            )
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"update_fov: malformed point data: {e}")

    @pyqtSlot(str, float, float, float, float, float, float, float, float)
    def set_panorama_overlay(
        self,
        data_url: str,
        lat_tl: float,
        lon_tl: float,
        lat_tr: float,
        lon_tr: float,
        lat_br: float,
        lon_br: float,
        lat_bl: float,
        lon_bl: float,
    ):
        self.bridge.setPanoramaSignal.emit(
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

    @pyqtSlot(list)
    def show_verification_markers(self, points: list):
        """
        Accepts points as list of dicts [{'lat': float, 'lon': float, 'label': str}]
        """
        self.bridge.showVerificationMarkersSignal.emit(json.dumps(points))

    @pyqtSlot()
    def clear_verification_markers(self):
        self.bridge.clearVerificationMarkersSignal.emit()


# ================================================================================
# File: gui\widgets\video_widget.py
# ================================================================================
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoWidget(QGraphicsView):
    """Displays video frames with optional overlay annotations (calibration points)."""

    frame_clicked = pyqtSignal(int, int)  # pixel coords in image space

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._video_item = QGraphicsPixmapItem()
        self._overlay_items: list = []
        self._scene.addItem(self._video_item)

    # ── Display ──────────────────────────────────────────────────────────────

    def display_frame(self, pixmap: QPixmap):
        self._video_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._video_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        logger.debug(f"Frame: {pixmap.width()}×{pixmap.height()}")

    # ── Overlays ─────────────────────────────────────────────────────────────

    def _dpr(self) -> float:
        """Device pixel ratio поточного pixmap (1.0 на 100% DPI, 2.0 на 200%)."""
        pm = self._video_item.pixmap()
        return pm.devicePixelRatio() if pm and not pm.isNull() else 1.0

    def draw_numbered_point(self, x: int, y: int, label: str, color: QColor):
        """Draw a filled circle with a label at (x, y) in ACTUAL image pixel coordinates."""
        # Конвертуємо з фактичних пікселів у логічні координати сцени
        dpr = self._dpr()
        lx, ly = x / dpr, y / dpr

        pen = QPen(color, 2)
        brush = QBrush(color)
        radius = 8

        ellipse = self._scene.addEllipse(
            lx - radius, ly - radius, radius * 2, radius * 2, pen, brush
        )
        self._overlay_items.append(ellipse)

        text = self._scene.addText(label)
        text.setDefaultTextColor(QColor(255, 255, 255))

        font = text.font()
        font.setBold(True)
        # Scale text relative to logical image width
        img_width = self._video_item.boundingRect().width()
        font.setPixelSize(max(12, int(img_width) // 80))
        text.setFont(font)
        text.setPos(lx + radius + 2, ly - radius - font.pixelSize())

        self._overlay_items.append(text)

    def clear_overlays(self):
        for item in self._overlay_items:
            self._scene.removeItem(item)
            item.setParentItem(None)  # break Qt ownership before Python GC
        self._overlay_items.clear()

    # ── Events ───────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._video_item.pixmap().isNull():
            super().mousePressEvent(event)
            return

        # Map viewport → scene → item
        scene_pos = self.mapToScene(event.pos())
        item_pos = self._video_item.mapFromScene(scene_pos)

        if self._video_item.contains(item_pos):
            pm = self._video_item.pixmap()
            pm_dpr = self._dpr()
            br = self._video_item.boundingRect()

            # Screen devicePixelRatio — може відрізнятися від pixmap dpr
            screen = self.screen()
            screen_dpr = screen.devicePixelRatio() if screen else 1.0

            # Якщо boundingRect ≠ pixmap size, масштабуємо вручну
            if br.width() > 0 and br.height() > 0:
                scale_x = pm.width() / br.width()
                scale_y = pm.height() / br.height()
            else:
                scale_x = scale_y = 1.0

            # Множимо на pm_dpr (Device Pixel Ratio), оскільки Qt на High-DPI
            # повертає "логічні" координати (напр. 1280 замість 1920).
            # Нам потрібні ФІЗИЧНІ пікселі зображення для метчингу бази даних.
            actual_x = int(item_pos.x() * scale_x * pm_dpr)
            actual_y = int(item_pos.y() * scale_y * pm_dpr)

            logger.warning(
                f"CLICK DIAG: "
                f"event=({event.pos().x()},{event.pos().y()}) "
                f"scene=({scene_pos.x():.0f},{scene_pos.y():.0f}) "
                f"item=({item_pos.x():.0f},{item_pos.y():.0f}) "
                f"actual=({actual_x},{actual_y}) "
                f"pm_dpr={pm_dpr} screen_dpr={screen_dpr} "
                f"pixmap={pm.width()}x{pm.height()} "
                f"bRect={br.width():.0f}x{br.height():.0f} "
                f"viewport={self.viewport().width()}x{self.viewport().height()} "
                f"sceneRect={self.sceneRect().width():.0f}x{self.sceneRect().height():.0f}"
            )
            self.frame_clicked.emit(actual_x, actual_y)

        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._video_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ================================================================================
# File: gui\widgets\__init__.py
# ================================================================================
"""Custom Qt widgets"""


# ================================================================================
# File: localization\localizer.py
# ================================================================================
import numpy as np

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Localizer:
    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        # Дефолти синхронізовані з APP_CONFIG через get_cfg()
        self.min_matches = get_cfg(self.config, "localization.min_matches", 12)
        self.ransac_thresh = get_cfg(self.config, "localization.ransac_threshold", 3.0)
        self.enable_auto_rotation = get_cfg(self.config, "localization.auto_rotation", True)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=get_cfg(self.config, "tracking.kalman_process_noise", 2.0),
            measurement_noise=get_cfg(self.config, "tracking.kalman_measurement_noise", 5.0),
            dt=1.0,
        )
        self.outlier_detector = OutlierDetector(
            window_size=get_cfg(self.config, "tracking.outlier_window", 10),
            threshold_std=get_cfg(self.config, "tracking.outlier_threshold_std", 25.0),
            max_speed_mps=get_cfg(self.config, "tracking.max_speed_mps", 60.0),
            max_consecutive=get_cfg(self.config, "tracking.max_consecutive_outliers", 5),
        )

        # Створюємо FastRetrieval один раз — нормалізація дескрипторів відбувається лише тут
        self.retriever = FastRetrieval(self.database.global_descriptors)

        # Fallback: SuperPoint+LightGlue для складних сцен
        self.model_manager = self.config.get("_model_manager", None)
        self.fallback_enabled = get_cfg(self.config, "localization.enable_lightglue_fallback", True)
        self.min_inliers_for_accept = get_cfg(self.config, "localization.min_inliers_accept", 10)
        self.retrieval_top_k = get_cfg(self.config, "localization.retrieval_top_k", 8)
        self.early_stop_inliers = get_cfg(self.config, "localization.early_stop_inliers", 30)

    def localize_frame(
        self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0
    ) -> dict:
        height, width = query_frame.shape[:2]

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]

        best_global_score = -1.0
        best_global_angle = 0
        best_global_candidates = []
        best_query_features = None

        top_k = self.retrieval_top_k

        # 1. Екстракція ознак для всіх дозволених кутів обертання та вибір найкращого ракурсу
        for angle in angles_to_try:
            k = angle // 90
            rotated_frame = np.rot90(query_frame, k=k).copy()

            # Витягуємо ТІЛЬКИ глобальний дескриптор DINOv2 для швидкого пошуку ракурсу
            global_desc = self.feature_extractor.extract_global_descriptor(rotated_frame)

            # Шукаємо кандидатів за допомогою DINOv2
            candidates = self.retriever.find_similar_frames(global_desc, top_k=top_k)

            if candidates:
                # Оцінкою ракурсу вважаємо скор найкращого кандидата
                top_score = candidates[0][1]
                if top_score > best_global_score:
                    best_global_score = top_score
                    best_global_angle = angle
                    best_global_candidates = candidates

        if not best_global_candidates:
            return {
                "success": False,
                "error": "No candidates found via global descriptor (DINOv2) in any rotation",
            }

        logger.info(
            f"Selected best rotation {best_global_angle}° with global score {best_global_score:.3f}"
        )

        # 1.5. Локальна екстракція (XFeat) ТІЛЬКИ для НАЙКРАЩОГО ракурсу
        k = best_global_angle // 90
        best_rotated_frame = np.rot90(query_frame, k=k).copy()
        best_rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None

        # Обчислюємо ключові точки лише один раз для обраного кута!
        best_query_features = self.feature_extractor.extract_local_features(
            best_rotated_frame, static_mask=best_rotated_mask
        )

        best_inliers = 0
        best_candidate_id = -1
        best_H_query_to_ref = None
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0
        best_rmse = 999.0

        early_stop = self.early_stop_inliers

        # 2. Локальний пошук (XFeat) ТІЛЬКИ для найкращого знайденого ракурсу
        for candidate_id, score in best_global_candidates:
            ref_features = self.database.get_local_features(candidate_id)

            mkpts_q, mkpts_r = self.matcher.match(best_query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                # Використовуємо Homography (8 DoF)
                H_eval, mask = GeometryTransforms.estimate_homography(
                    mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                )

                if H_eval is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(np.sum(inlier_mask))
                    pts_q_in = mkpts_q[inlier_mask]
                    pts_r_in = mkpts_r[inlier_mask]

                    # Розрахунок RMSE для оцінки якості геометрії
                    pts_q_transformed = GeometryTransforms.apply_homography(pts_q_in, H_eval)
                    rmse = float(
                        np.sqrt(np.mean(np.sum((pts_q_transformed - pts_r_in) ** 2, axis=1)))
                    )

                    if inliers > best_inliers and inliers >= self.min_matches:
                        best_inliers = inliers
                        best_candidate_id = candidate_id
                        best_H_query_to_ref = H_eval
                        best_mkpts_q_inliers = pts_q_in
                        best_mkpts_r_inliers = pts_r_in
                        best_total_matches = len(mkpts_q)
                        best_rmse = rmse
                        logger.debug(
                            f"Homography for {candidate_id}: {inliers} inliers, RMSE: {rmse:.2f}"
                        )

            if best_inliers >= early_stop:
                logger.info(
                    f"Early stop triggered with {best_inliers} inliers on candidate {best_candidate_id}"
                )
                break

        # Оскільки LightGlue (ALIKED) тепер основний метод, окремий fallback не потрібен

        if (
            best_inliers < self.min_matches
            or best_mkpts_r_inliers is None
            or best_H_query_to_ref is None
        ):
            # Спробуємо фоллбек перед тим як повертати помилку.
            # Якщо ми не знайшли жодного відповідного кадру через Matching, беремо топ-1 з retrieval.
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"Feature matching failed ({best_inliers} inliers), using retrieval-only fallback for frame {target_id} (score {best_global_score:.3f})"
                )
                return fallback_res
            return {
                "success": False,
                "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})",
            }

        # 3. Отримуємо матрицю знайденого кадру з бази
        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"No propagated calibration for frame {target_id}, using retrieval-only fallback"
                )
                return fallback_res
            return {"success": False, "error": "No propagated calibration"}

        # 4. Рахуємо розміри ПОВЕРНУТОГО зображення
        if best_global_angle in [90, 270]:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        # 4. Багатоточкова локалізація більше не потрібна. Беремо ідеальний центр кадру

        # Використовуємо знайдену Homography
        M_query_to_ref = best_H_query_to_ref
        if M_query_to_ref is None:
            return {"success": False, "error": "Failed to compute transform"}

        # 5. Трансформуємо центральну точку: Query -> Reference (через Homography) -> Metric (через Affine)
        center_query = np.array([[rot_width / 2.0, rot_height / 2.0]], dtype=np.float32)
        pts_in_ref = GeometryTransforms.apply_homography(center_query, M_query_to_ref)
        if pts_in_ref is None or len(pts_in_ref) == 0:
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"Homography transform failure, using retrieval-only fallback for frame {target_id} (score {best_global_score:.3f})"
                )
                return fallback_res
            return {
                "success": False,
                "error": "Coordinate transformation error (homography failed)",
            }

        pts_metric = GeometryTransforms.apply_affine(pts_in_ref, affine_ref)

        # Оскільки ми взяли одну центральну точку, просто беремо її координати
        mx = float(pts_metric[0, 0])
        my = float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float32)

        # 6. Перевіряємо чи нова точка — аномалія (стрибок координат)
        if self.outlier_detector.is_outlier(metric_pt, dt):
            # Ми все одно логуємо спробу, але кажемо, що це аутлаєр
            logger.warning(
                f"Outlier filtered at frame {best_candidate_id}: jump from previous trajectory"
            )
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # Оновлення Калмана (фільтрація шумів)
        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt)
        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )

        # Зсув для корекції FOV через фільтрацію
        dx, dy = filtered_pt[0] - metric_pt[0], filtered_pt[1] - metric_pt[1]

        # 7. Розрахунок поля зору (FOV)
        # Користувач очікує бачити повне покриття камери (від 0 до rot_width).
        # Проектуємо повний кадр
        corners = np.array(
            [[0, 0], [rot_width, 0], [rot_width, rot_height], [0, rot_height]], dtype=np.float32
        )

        ref_corners = GeometryTransforms.apply_homography(corners, M_query_to_ref)

        # Захист від перспективного "вибуху" гомографії (якщо кластер ALIKED занадто локальний)
        is_exploded = False
        if ref_corners is not None:
            max_coord = np.max(np.abs(ref_corners))
            if max_coord > 50000:  # якщо кут відлетів далі ніж на 50к пікселів
                is_exploded = True

        if is_exploded and best_mkpts_q_inliers is not None and len(best_mkpts_q_inliers) > 0:
            logger.warning(
                "Homography exploded the FOV (perspective distortion)! Falling back to inliers bounding box."
            )
            pts = best_mkpts_q_inliers
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            pad_x, pad_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
            safe_corners = np.array(
                [
                    [max(0, min_x - pad_x), max(0, min_y - pad_y)],
                    [min(rot_width, max_x + pad_x), max(0, min_y - pad_y)],
                    [min(rot_width, max_x + pad_x), min(rot_height, max_y + pad_y)],
                    [max(0, min_x - pad_x), min(rot_height, max_y + pad_y)],
                ],
                dtype=np.float32,
            )
            ref_corners = GeometryTransforms.apply_homography(safe_corners, M_query_to_ref)
            original_poly_px = safe_corners
        else:
            original_poly_px = corners
            logger.debug("FOV projected using full frame Homography matrix.")

        # --- ДІАГНОСТИЧНІ ЛОГИ МАКСИМАЛЬНОГО РІВНЯ ---
        logger.info(f"--- FOV DIAGNOSTICS FOR FRAME {best_candidate_id} ---")
        w_px = np.linalg.norm(original_poly_px[0] - original_poly_px[1])
        h_px = np.linalg.norm(original_poly_px[1] - original_poly_px[2])
        logger.info(f"[1] Original FOV in Query image: {w_px:.1f} x {h_px:.1f} pixels")

        if ref_corners is not None:
            w_ref = np.linalg.norm(ref_corners[0] - ref_corners[1])
            h_ref = np.linalg.norm(ref_corners[1] - ref_corners[2])
            logger.info(
                f"[2] FOV mapped to Reference via Homography: {w_ref:.1f} x {h_ref:.1f} pixels"
            )

        gps_corners = []
        if ref_corners is not None:
            metric_corners = GeometryTransforms.apply_affine(ref_corners, affine_ref)
            if metric_corners is not None:
                # Діагностика розмірів FOV в метрах
                fov_w = np.linalg.norm(metric_corners[1] - metric_corners[0])
                fov_h = np.linalg.norm(metric_corners[3] - metric_corners[0])
                logger.info(
                    f"[3] FOV mapped to Web Mercator Metric space: {fov_w:.1f}m x {fov_h:.1f}m"
                )
                logger.debug(
                    f"FOV dimensions: {fov_w:.1f}m x {fov_h:.1f}m | "
                    f"Center metric: ({mx:.1f}, {my:.1f}) | "
                    f"Filtered: ({filtered_pt[0]:.1f}, {filtered_pt[1]:.1f})"
                )
                for cx, cy in metric_corners:
                    try:
                        clat, clon = self.calibration.converter.metric_to_gps(
                            float(cx + dx), float(cy + dy)
                        )
                        gps_corners.append((clat, clon))
                    except Exception:
                        pass

        confidence = self._compute_confidence(
            best_candidate_id, best_inliers, best_total_matches, best_rmse
        )

        # ДІАГНОСТИКА
        logger.debug(
            f"Localize Frame {best_candidate_id}: Center transformed via Homography (8 DoF)"
        )
        logger.debug(f"Sample Center METRIC: ({mx:.1f}, {my:.1f})")

        logger.success(
            f"Localized ({lat:.6f}, {lon:.6f}) | frame={best_candidate_id} | "
            f"metric=({mx:.1f}, {my:.1f}) | inliers={best_inliers} | conf={confidence:.2f}"
        )

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": int(best_candidate_id),
            "inliers": int(best_inliers),
            "fov_polygon": gps_corners,
            "sample_spread_m": 0.0,
        }

    def _compute_confidence(
        self, best_candidate_id: int, best_inliers: int, total_matches: int, rmse_val: float
    ) -> float:
        """Обчислює впевненість на основі QA бази даних (RMSE, Disagreement) та кількості інлаєрів."""
        # Налаштування з конфігу
        max_inliers = get_cfg(self.config, "localization.confidence.confidence_max_inliers", 80)
        rmse_norm = get_cfg(self.config, "localization.confidence.rmse_norm_m", 10.0)
        diag_norm = get_cfg(self.config, "localization.confidence.disagreement_norm_m", 5.0)
        w_inlier = get_cfg(self.config, "localization.confidence.inlier_weight", 0.7)
        w_stability = get_cfg(self.config, "localization.confidence.stability_weight", 0.3)

        # 1. Показник інлаєрів (0-1)
        inlier_score = min(1.0, best_inliers / max_inliers)

        # 2. Показник стабільності бази (на основі QA метрик)
        rmse = (
            self.database.frame_rmse[best_candidate_id]
            if self.database.frame_rmse is not None
            else 0.0
        )
        disagreement = (
            self.database.frame_disagreement[best_candidate_id]
            if self.database.frame_disagreement is not None
            else 0.0
        )

        stability_score = 1.0 - (
            min(rmse, rmse_norm) / rmse_norm * 0.5 + min(disagreement, diag_norm) / diag_norm * 0.5
        )
        stability_score = float(np.clip(stability_score, 0.0, 1.0))

        # 3. Показник поточної відповідності (ПЕР-ФРЕЙМ)
        # a) Inlier ratio
        ratio_score = float(best_inliers / (total_matches + 1e-6))
        # b) RMSE score (1.0 if RMSE=0, 0.5 if RMSE=thresh)
        rmse_score_val = 1.0 / (1.0 + (rmse_val / (self.ransac_thresh + 1e-6)))

        match_score = ratio_score * 0.5 + rmse_score_val * 0.5

        # 4. Комбінована оцінка
        # (QA бази * 0.3) + (Кількість інлаєрів * 0.4) + (Якість відповідності * 0.3)
        final_conf = stability_score * 0.3 + inlier_score * 0.4 + match_score * 0.3

        return float(np.clip(final_conf, 0.05, 1.0))

    def _localize_by_reference_frame(self, frame_id: int, score: float) -> dict:
        """Приблизна локалізація за центром опорного кадру (retrieval-only fallback)"""
        if frame_id == -1:
            return None

        threshold = get_cfg(self.config, "localization.retrieval_only_min_score", 0.90)
        if score < threshold:
            return None

        affine_ref = self.database.get_frame_affine(frame_id)
        if affine_ref is None:
            return None

        ref_h, ref_w = self.database.get_frame_size(frame_id)
        # Центр кадру в системі координат БД
        center_ref = np.array([[ref_w / 2, ref_h / 2]], dtype=np.float32)
        metric_pt = GeometryTransforms.apply_affine(center_ref, affine_ref)[0]

        lat, lon = self.calibration.converter.metric_to_gps(metric_pt[0], metric_pt[1])

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": 0.3,  # Низький confidence для retrieval-only
            "inliers": 0,
            "matched_frame": frame_id,
            "fallback_mode": "retrieval_only",
            "global_score": score,
            "fov_polygon": None,
        }


# ================================================================================
# File: localization\matcher.py
# ================================================================================
import faiss
import numpy as np
import torch

from config.config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FastRetrieval:
    """Fast candidate search using DINOv2 global descriptors (optimized with FAISS)"""

    def __init__(self, global_descriptors: np.ndarray):
        logger.info(
            f"Initializing FastRetrieval with {len(global_descriptors)} descriptors using FAISS"
        )
        self.dim = global_descriptors.shape[1]

        # Inner Product index (для косинусної схожості нормалізованих векторів)
        self.index = faiss.IndexFlatIP(self.dim)

        # Нормалізуємо і додаємо в індекс
        normed = self.normalize_vectors(global_descriptors)
        self.index.add(normed.astype(np.float32))

        logger.success(f"FAISS index built with {self.index.ntotal} vectors")

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def find_similar_frames(self, query_desc: np.ndarray, top_k: int = 5) -> list:
        # Підготовка запиту
        q = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        q = q.astype(np.float32)

        if q.ndim == 1:
            q = q[None]

        # Пошук у FAISS
        scores, ids = self.index.search(q, top_k)

        # Повертаємо список (id, score)
        results = [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]
        return results


class FeatureMatcher:
    """Matches local keypoints (XFeat or SuperPoint+LightGlue)"""

    def __init__(self, model_manager=None, config=None):
        self.config = config or {}
        self.model_manager = model_manager
        self.ratio_threshold = get_cfg(self.config, "localization.ratio_threshold", 0.95)

        # Завантажуємо LightGlue (ALIKED) через ModelManager
        self.lightglue = None
        if self.model_manager:
            try:
                self.lightglue = self.model_manager.load_lightglue_aliked()
                logger.info("FeatureMatcher configured to use LightGlue (ALIKED)")
            except Exception as e:
                logger.warning(f"Failed to load LightGlue ALIKED: {e}. Falling back to Numpy L2.")
        else:
            logger.info("FeatureMatcher configured to use fast Numpy L2 matching")

    def match(self, query_features: dict, ref_features: dict) -> tuple:
        """
        Dynamically routes to LightGlue (for 256-dim SuperPoint)
        or Fast L2 Matcher (for 64-dim XFeat / 128-dim ALIKED).
        """
        desc_dim = (
            query_features["descriptors"].shape[1] if len(query_features["descriptors"]) > 0 else 0
        )

        # Якщо є LightGlue і розмірність дескриптора 128 (ALIKED)
        if self.lightglue is not None and desc_dim == 128:
            return self._lightglue_match(query_features, ref_features)

        # Fallback (якщо немає LightGlue або інші ознаки)
        return self._fast_numpy_match(query_features, ref_features, self.ratio_threshold)

    def _fast_numpy_match(
        self, query_features: dict, ref_features: dict, ratio_threshold: float = 0.80
    ) -> tuple:
        """
        Highly optimized L2 matching using dot product and Mutual Nearest Neighbor (MNN).
        """
        desc_q = query_features["descriptors"]
        desc_r = ref_features["descriptors"]
        kpts_q = query_features["keypoints"]
        kpts_r = ref_features["keypoints"]

        if len(desc_q) < 2 or len(desc_r) < 2:
            return np.empty((0, 2)), np.empty((0, 2))

        # 1. Нормалізація дескрипторів
        desc_q_n = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
        desc_r_n = desc_r / (np.linalg.norm(desc_r, axis=1, keepdims=True) + 1e-8)

        # 2. Розрахунок косинусної схожості через швидке матричне множення
        sim = np.dot(desc_q_n, desc_r_n.T)

        # 3. Lowe's Ratio Test — argpartition O(n) замість argsort O(n log n)
        # Потрібні лише top-2 для ratio test
        top2_idx = np.argpartition(-sim, kth=1, axis=1)[:, :2]
        top2_sim = np.take_along_axis(sim, top2_idx, axis=1)
        # Сортуємо лише 2 елементи щоб best >= second_best
        order = np.argsort(-top2_sim, axis=1)
        top2_idx = np.take_along_axis(top2_idx, order, axis=1)
        top2_sim = np.take_along_axis(top2_sim, order, axis=1)

        best_sim = top2_sim[:, 0]
        second_best_sim = top2_sim[:, 1]
        best_matches_indices = top2_idx[:, 0]

        # Переводимо схожість у L2-відстань: D = sqrt(2 - 2*sim)
        best_dist = np.sqrt(np.clip(2.0 - 2.0 * best_sim, 0, None))
        second_best_dist = np.sqrt(np.clip(2.0 - 2.0 * second_best_sim, 0, None))

        valid_ratio = (best_dist / (second_best_dist + 1e-8)) < ratio_threshold

        # 4. Mutual Nearest Neighbor (MNN) check
        reverse_best_indices = np.argmax(sim, axis=0)
        is_mutual = reverse_best_indices[best_matches_indices] == np.arange(len(desc_q))

        valid_matches = valid_ratio & is_mutual

        mkpts_q = kpts_q[valid_matches]
        mkpts_r = kpts_r[best_matches_indices[valid_matches]]

        return mkpts_q, mkpts_r

    def _lightglue_match(self, query_features: dict, ref_features: dict) -> tuple:
        """Matches features using Neural LightGlue Matcher"""
        try:
            if len(query_features["keypoints"]) == 0 or len(ref_features["keypoints"]) == 0:
                logger.warning("Empty keypoints array provided to LightGlue.")
                return np.empty((0, 2)), np.empty((0, 2))

            device = next(self.lightglue.parameters()).device

            # Підготовка тензорів для LightGlue
            data = {
                "image0": {
                    "keypoints": torch.from_numpy(query_features["keypoints"])
                    .float()[None]
                    .to(device),
                    "descriptors": torch.from_numpy(query_features["descriptors"])
                    .float()[None]
                    .to(device),
                },
                "image1": {
                    "keypoints": torch.from_numpy(ref_features["keypoints"])
                    .float()[None]
                    .to(device),
                    "descriptors": torch.from_numpy(ref_features["descriptors"])
                    .float()[None]
                    .to(device),
                },
            }

            with torch.no_grad():
                res = self.lightglue(data)

            matches = res["matches"][0].cpu().numpy()

            if len(matches) == 0:
                return np.empty((0, 2)), np.empty((0, 2))

            m_q = matches[:, 0]
            m_r = matches[:, 1]

            mkpts_q = query_features["keypoints"][m_q]
            mkpts_r = ref_features["keypoints"][m_r]

            return mkpts_q, mkpts_r

        except Exception as e:
            logger.error(f"LightGlue match failed: {e}")
            return np.empty((0, 2)), np.empty((0, 2))


# ================================================================================
# File: localization\__init__.py
# ================================================================================
"""Localization module"""


# ================================================================================
# File: models\model_manager.py
# ================================================================================
import gc
import time
from contextlib import contextmanager

import torch

from config.config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, config=None, device="cuda"):
        self.config = config or {}

        use_cuda = get_cfg(self.config, "models.use_cuda", True)
        if not use_cuda:
            logger.info("CUDA force disabled in configuration")

        self.device = (
            device if (use_cuda and torch.cuda.is_available() and device == "cuda") else "cpu"
        )
        self.models = {}
        self.model_usage = {}

        # Конфігурація VRAM
        self.max_vram_ratio = get_cfg(self.config, "models.vram_management.max_vram_ratio", 0.8)
        self.default_vram_required = get_cfg(
            self.config, "models.vram_management.default_required_mb", 2000.0
        )

        logger.info(f"ModelManager initialized with device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )

    def get_available_vram_mb(self) -> float:
        if self.device == "cpu":
            return float("inf")
        free_mem, total_mem = torch.cuda.mem_get_info()
        available_mb = free_mem / (1024 * 1024)
        return available_mb

    def _ensure_vram_available(self, required_mb: float | None = None):
        if self.device == "cpu":
            return

        req = required_mb if required_mb is not None else self.default_vram_required

        while self.get_available_vram_mb() < req and self.models:
            least_used = min(self.model_usage.items(), key=lambda x: x[1])[0]
            logger.warning(f"VRAM insufficient. Unloading least used model: {least_used}")
            self.unload_model(least_used)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()

    def load_yolo(self):
        name = "yolo"
        if name not in self.models:
            model_path = get_cfg(self.config, "models.yolo.model_path", "yolo11x-seg.pt")
            vram_req = get_cfg(self.config, "models.yolo.vram_required_mb", 1200.0)

            logger.info(f"Loading YOLO model: {model_path}...")
            self._ensure_vram_available(vram_req)
            try:
                from ultralytics import YOLO

                model = YOLO(model_path)
                model.to(self.device)
                self.models[name] = model
                logger.success(f"YOLO model {model_path} loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_xfeat(self):
        name = "xfeat"
        if name not in self.models:
            repo = get_cfg(self.config, "models.xfeat.hub_repo", "verlab/accelerated_features")
            model_name = get_cfg(self.config, "models.xfeat.hub_model", "XFeat")
            top_k = get_cfg(self.config, "models.xfeat.top_k", 2048)
            vram_req = get_cfg(self.config, "models.xfeat.vram_required_mb", 300.0)

            logger.info(f"Loading XFeat model ({repo}/{model_name})...")
            self._ensure_vram_available(vram_req)
            try:
                model = torch.hub.load(repo, model_name, pretrained=True, top_k=top_k)
                # FIX: XFeat hardcodes self.dev='cuda' if available, causing crashes if we move to CPU
                if hasattr(model, "dev"):
                    model.dev = torch.device(self.device)
                model = model.eval().to(self.device)
                self.models[name] = model
                logger.success(f"XFeat loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load XFeat: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_superpoint(self):
        name = "superpoint"
        if name not in self.models:
            vram_req = get_cfg(self.config, "models.superpoint.vram_required_mb", 500.0)

            logger.info("Loading SuperPoint model (for LightGlue compatibility)...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import SuperPoint

                sp_config = {
                    "nms_radius": get_cfg(self.config, "models.superpoint.nms_radius", 4),
                    "max_num_keypoints": get_cfg(
                        self.config, "models.superpoint.max_keypoints", 4096
                    ),
                }
                model = SuperPoint(**sp_config).eval().to(self.device)
                self.models[name] = model
                logger.success("SuperPoint model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SuperPoint model: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_lightglue(self):
        name = "lightglue"
        if name not in self.models:
            vram_req = get_cfg(self.config, "models.lightglue.vram_required_mb", 1000.0)

            logger.info("Loading LightGlue model...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import LightGlue

                lg_config = {
                    "depth_confidence": get_cfg(
                        self.config, "models.lightglue.depth_confidence", -1
                    ),
                    "width_confidence": get_cfg(
                        self.config, "models.lightglue.width_confidence", -1
                    ),
                }
                model = LightGlue(features="superpoint", **lg_config).eval().to(self.device)
                self.models[name] = model
                logger.success("LightGlue model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LightGlue: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_dinov2(self):
        name = "dinov2"
        if name not in self.models:
            repo = get_cfg(self.config, "models.dinov2.hub_repo", "facebookresearch/dinov2")
            model_name = get_cfg(self.config, "models.dinov2.hub_model", "dinov2_vitl14")
            vram_req = get_cfg(self.config, "models.dinov2.vram_required_mb", 1600.0)

            logger.info(f"Loading DINOv2 ({model_name}) model...")
            self._ensure_vram_available(vram_req)
            try:
                model = torch.hub.load(repo, model_name)
                model = model.eval().to(self.device)
                self.models[name] = model
                logger.success(f"DINOv2 model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_aliked(self):
        """Завантажує ALIKED extractor (128-dim, lightglue-compatible)"""
        name = "aliked"
        if name not in self.models:
            vram_req = get_cfg(self.config, "models.aliked.vram_required_mb", 400.0)
            max_keypoints = get_cfg(self.config, "models.aliked.max_keypoints", 4096)

            logger.info(f"Loading ALIKED model (max_keypoints={max_keypoints})...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import ALIKED

                model = ALIKED(max_num_keypoints=max_keypoints).eval().to(self.device)
                self.models[name] = model
                logger.success(f"ALIKED loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load ALIKED: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_lightglue_aliked(self):
        """Завантажує LightGlue з вагами для ALIKED (128-dim)"""
        name = "lightglue_aliked"
        if name not in self.models:
            vram_req = get_cfg(self.config, "models.lightglue.vram_required_mb", 1000.0)

            logger.info("Loading LightGlue (ALIKED weights)...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import LightGlue

                model = (
                    LightGlue(
                        features="aliked",
                        depth_confidence=get_cfg(
                            self.config, "models.lightglue.depth_confidence", -1
                        ),
                        width_confidence=get_cfg(
                            self.config, "models.lightglue.width_confidence", -1
                        ),
                    )
                    .eval()
                    .to(self.device)
                )
                self.models[name] = model
                logger.success("LightGlue (ALIKED) loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LightGlue (ALIKED): {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_cesp(self):
        """Завантажує CESP модуль для покращення DINOv2 global descriptors"""
        name = "cesp"
        if name not in self.models:
            logger.info("Loading CESP module...")
            try:
                from src.models.wrappers.cesp_module import CESP

                scales = get_cfg(self.config, "models.cesp.scales", [1, 2, 4])
                cesp = CESP(dim=1024, scales=tuple(scales))

                # Завантаження pretrained ваг (якщо є)
                weights_path = get_cfg(self.config, "models.cesp.weights_path", None)
                if weights_path:
                    cesp.load_state_dict(torch.load(weights_path, map_location=self.device))
                    logger.success(f"CESP pretrained weights loaded from {weights_path}")
                else:
                    logger.warning("CESP initialized WITHOUT pretrained weights (random init)")

                cesp = cesp.eval().to(self.device)
                self.models[name] = cesp
                logger.success("CESP module loaded")
            except Exception as e:
                logger.error(f"Failed to load CESP: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            logger.info(f"Unloading model: {model_name}")
            del self.models[model_name]
            del self.model_usage[model_name]
            if self.device != "cpu":
                torch.cuda.empty_cache()
                gc.collect()

    @contextmanager
    def inference_context(self):
        try:
            with torch.no_grad():
                yield
        finally:
            if self.device != "cpu":
                torch.cuda.empty_cache()


# ================================================================================
# File: models\__init__.py
# ================================================================================
"""Neural network models module"""


# ================================================================================
# File: models\wrappers\aliked_wrapper.py
# ================================================================================
import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ALIKEDWrapper:
    """ALIKED feature extractor для LightGlue fallback.

    ALIKED видає 128-dim дескриптори (vs SuperPoint 256-dim).
    LightGlue має офіційні pretrained ваги для ALIKED.
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> dict:
        """Екстракція ALIKED features з тензору зображення (lightglue format)."""
        return self.model.extract(image_tensor)

    @torch.no_grad()
    def extract_from_numpy(self, image_rgb: np.ndarray, static_mask: np.ndarray = None) -> dict:
        """Екстракція з numpy RGB зображення + фільтрація за YOLO маскою.

        Returns:
            dict з ключами: keypoints (1, K, 2), descriptors (1, K, 128)
        """
        from lightglue.utils import numpy_image_to_torch

        tensor = numpy_image_to_torch(image_rgb).to(self.device)
        features = self.model.extract(tensor)

        # Фільтрація за маскою динамічних об'єктів
        if static_mask is not None and "keypoints" in features:
            kpts = features["keypoints"][0].cpu().numpy()
            if len(kpts) > 0:
                ix = np.round(kpts[:, 0]).astype(np.intp)
                iy = np.round(kpts[:, 1]).astype(np.intp)
                h, w = static_mask.shape[:2]
                in_bounds = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
                valid = np.zeros(len(kpts), dtype=bool)
                valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128

                if valid.any():
                    valid_t = torch.from_numpy(valid).to(self.device)
                    filtered = {
                        "keypoints": features["keypoints"][:, valid_t],
                        "descriptors": features["descriptors"][:, valid_t],
                    }
                    # Зберігаємо keypoint_scores якщо є
                    if "keypoint_scores" in features and features["keypoint_scores"] is not None:
                        filtered["keypoint_scores"] = features["keypoint_scores"][:, valid_t]
                    features = filtered
                    logger.debug(
                        f"ALIKED: {int(valid.sum())}/{len(kpts)} keypoints after mask filter"
                    )

        return features


# ================================================================================
# File: models\wrappers\cesp_module.py
# ================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class CESP(nn.Module):
    """Cross-Enhancement Spatial Pyramid для DINOv2 patch tokens.

    IEEE RA-L 2025: "DINOv2-based UAV Visual Self-localization"
    Покращує multi-scale сприйняття для aerial imagery.

    Вхід: patch_tokens (B, N, D) з DINOv2
    Вихід: enhanced_descriptor (B, D) — L2-нормалізований

    Примітка: потребує навчання на парах UAV↔satellite зображень.
    Без навчених ваг повертає усереднення multi-scale features (random projection).
    """

    def __init__(self, dim: int = 1024, scales: tuple = (1, 2, 4)):
        super().__init__()
        self.dim = dim
        self.scales = scales

        # Проекційні шари для кожного масштабу піраміди
        self.projectors = nn.ModuleList([nn.Linear(dim, dim) for _ in scales])

        # Фінальне злиття (N_scales * dim → dim)
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patch_tokens: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) — patch tokens з DINOv2 (без CLS)
            h_patches: кількість патчів по висоті (для 336×336 + patch_size=14 → 24)
            w_patches: кількість патчів по ширині

        Returns:
            enhanced: (B, D) — L2-нормалізований глобальний дескриптор
        """
        B, N, D = patch_tokens.shape
        # Reshape до 2D просторової сітки: (B, D, H, W)
        x = patch_tokens.reshape(B, h_patches, w_patches, D).permute(0, 3, 1, 2)

        scale_features = []
        for scale, proj in zip(self.scales, self.projectors):
            if scale == 1:
                # Глобальне усереднення всіх патчів
                pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, D)
            else:
                # Spatial Pyramid: розбити на scale×scale регіонів → усереднити
                pooled = F.adaptive_avg_pool2d(x, scale)  # (B, D, scale, scale)
                pooled = pooled.flatten(2).mean(dim=2)  # (B, D)
            scale_features.append(proj(pooled))

        # Cross-Enhancement: конкатенація + fusion
        multi_scale = torch.cat(scale_features, dim=1)  # (B, N_scales*D)
        enhanced = self.fusion(multi_scale)  # (B, D)
        enhanced = F.normalize(enhanced, p=2, dim=1)  # L2 нормалізація

        return enhanced


# ================================================================================
# File: models\wrappers\feature_extractor.py
# ================================================================================
import contextlib
import numpy as np
import torch
import torchvision.transforms as T

from config.config import get_cfg
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (ALIKED + DINOv2 [+ CESP])"""

    def __init__(self, local_model, global_model, device="cuda", config=None, cesp_module=None):
        self.local_model = local_model  # ALIKED
        self.global_model = global_model  # DINOv2
        self.device = device
        self.config = config or {}
        self.preprocessor = ImagePreprocessor(config)
        self.cesp_module = cesp_module  # Опціональний CESP для покращення global descriptors

        # Трансформації для DINOv2 (ImageNet стандарти)
        dino_size = get_cfg(self.config, "dinov2.input_size", 336)
        self.dino_size = dino_size
        self.dinov2_transform = T.Compose(
            [
                T.Resize((dino_size, dino_size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.use_half = (
            device == "cuda"
            and torch.cuda.is_available()
            and get_cfg(self.config, "models.performance.fp16_enabled", True)
        )
        self.amp_dtype = torch.float16 if self.use_half else torch.float32
        
        if self.use_half:
            logger.info("FP16 mixed precision ENABLED for inference")

        cesp_status = "with CESP" if cesp_module is not None else "without CESP"
        logger.info(
            f"FeatureExtractor initialized with ALIKED and DINOv2 ({cesp_status}) on {device}"
        )
        
        if device == "cuda":
            self.stream_global = torch.cuda.Stream()
            self.stream_local = torch.cuda.Stream()
        else:
            self.stream_global = None
            self.stream_local = None

    @torch.no_grad()
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        logger.debug("Extracting global descriptor with DINOv2...")
        dino_tensor = torch.from_numpy(image).float().div_(255.0)
        dino_tensor = dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_tensor)

        if self.cesp_module is not None:
            # CESP mode: отримуємо patch tokens замість CLS
            with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.use_half):
                features = self.global_model.forward_features(dino_input)
                patch_tokens = features["x_norm_patchtokens"].float()

            h_patches = self.dino_size // 14
            w_patches = self.dino_size // 14
            global_desc = self.cesp_module(patch_tokens, h_patches, w_patches)[0].cpu().numpy()
        else:
            # Стандартний mode: CLS token
            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_half):
                global_desc = self.global_model(dino_input)[0].float().cpu().numpy()

        return global_desc

    @torch.no_grad()
    def extract_local_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting local features (ALIKED) from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка зображення для ALIKED (LightGlue format)
        rgb_tensor = torch.from_numpy(enhanced_image).float().div_(255.0)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        # ALIKED очікує словник зі списком/тензором 'image'
        input_dict = {"image": rgb_tensor}

        # ALIKED behaves unstably and yields NaNs inside AMP autocast. Always run it in FP32!
        with contextlib.nullcontext():
            aliked_out = self.local_model(input_dict)

        # LightGlue ALIKED wrapper повертає батч: (1, N, 2) та (1, N, 128)
        keypoints = aliked_out["keypoints"][0].cpu().numpy()
        descriptors = aliked_out["descriptors"][0].cpu().numpy()

        # Фільтрація точок за маскою динамічних об'єктів (YOLO)
        if static_mask is not None and len(keypoints) > 0:
            # Vectorized YOLO mask filtering
            ix = np.round(keypoints[:, 0]).astype(np.intp)
            iy = np.round(keypoints[:, 1]).astype(np.intp)
            in_bounds = (
                (iy >= 0) & (iy < static_mask.shape[0]) & (ix >= 0) & (ix < static_mask.shape[1])
            )
            valid = np.zeros(len(keypoints), dtype=bool)
            valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128

            if valid.any():
                keypoints = keypoints[valid]
                descriptors = descriptors[valid]
            else:
                logger.warning("All keypoints filtered out by YOLO mask!")

        return {"keypoints": keypoints, "descriptors": descriptors, "coords_2d": keypoints.copy()}

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        local_feats = self.extract_local_features(image, static_mask)
        global_desc = self.extract_global_descriptor(image)
        local_feats["global_desc"] = global_desc

        # logger.success(
        #     f"Extracted {len(local_feats['keypoints'])} ALIKED keypoints, global DINOv2 desc dim {len(global_desc)}"
        # )
        return local_feats

    @torch.no_grad()
    def extract_features_batch(self, images: list[np.ndarray], static_masks: list[np.ndarray]) -> list[dict]:
        """
        Extracts features for a batch of images using CUDA streams for parallel execution.
        """
        B = len(images)
        if B == 0:
            return []

        # 1. Prepare DINOv2 Tensor
        dino_tensors = []
        for img in images:
            rgb = torch.from_numpy(img).float().div_(255.0)
            dino_tensors.append(rgb.permute(2, 0, 1))
        dino_batch = torch.stack(dino_tensors).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_batch)

        # 2. Prepare ALIKED Tensor
        prep_images = [self.preprocessor.preprocess(img) for img in images]
        aliked_tensors = []
        for p_img in prep_images:
            rgb = torch.from_numpy(p_img).float().div_(255.0)
            aliked_tensors.append(rgb.permute(2, 0, 1))
        aliked_batch = torch.stack(aliked_tensors).to(self.device, non_blocking=True)
        input_dict = {"image": aliked_batch}

        stream_global = self.stream_global if self.device == "cuda" else None
        stream_local = self.stream_local if self.device == "cuda" else None

        global_descs = None
        aliked_out = None

        # PARALLEL EXECUTION
        context_global = torch.cuda.stream(stream_global) if stream_global else contextlib.nullcontext()
        with context_global:
            if self.cesp_module is not None:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_half):
                    features = self.global_model.forward_features(dino_input)
                    patch_tokens = features["x_norm_patchtokens"].float()
                h_p, w_p = self.dino_size // 14, self.dino_size // 14
                out_global = self.cesp_module(patch_tokens, h_p, w_p)
            else:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_half):
                    out_global = self.global_model(dino_input).float()

        out_kpts = []
        out_descs = []
        context_local = torch.cuda.stream(stream_local) if stream_local else contextlib.nullcontext()
        with context_local:
            for b in range(B):
                single_img = aliked_batch[b:b+1]  # shape (1, 3, H, W)
                input_dict = {"image": single_img}
                # ALIKED behaves unstably and yields NaNs inside AMP autocast. Always run it in FP32!
                aliked_out = self.local_model(input_dict)
                out_kpts.append(aliked_out["keypoints"][0].float())
                out_descs.append(aliked_out["descriptors"][0].float())

        if self.device == "cuda":
            torch.cuda.synchronize()

        global_descs = out_global.cpu().numpy()
        keypoints_batch = [kp.cpu().numpy() for kp in out_kpts]
        descriptors_batch = [desc.cpu().numpy() for desc in out_descs]

        # Assembly
        results = []
        for i in range(B):
            kp = keypoints_batch[i]
            desc = descriptors_batch[i]
            mask = static_masks[i]
            gd = global_descs[i]

            if mask is not None and len(kp) > 0:
                ix = np.round(kp[:, 0]).astype(np.intp)
                iy = np.round(kp[:, 1]).astype(np.intp)
                in_bounds = (iy >= 0) & (iy < mask.shape[0]) & (ix >= 0) & (ix < mask.shape[1])
                valid = np.zeros(len(kp), dtype=bool)
                valid[in_bounds] = mask[iy[in_bounds], ix[in_bounds]] > 128

                if valid.any():
                    kp = kp[valid]
                    desc = desc[valid]
                else:
                    kp = np.empty((0, 2), dtype=np.float32)
                    desc = np.empty((0, 128), dtype=np.float32)
            
            results.append({
                "keypoints": kp,
                "descriptors": desc,
                "coords_2d": kp.copy(),
                "global_desc": gd
            })

        return results


# ================================================================================
# File: models\wrappers\masking_strategy.py
# ================================================================================
# src/models/wrappers/masking_strategy.py
#
# Поліморфний інтерфейс маскування динамічних об'єктів (Strategy Pattern).
# Дозволяє підміняти реалізацію (YOLO, EfficientViT-SAM, none) через конфіг.

from abc import ABC, abstractmethod

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MaskingStrategy(ABC):
    """Абстрактний інтерфейс для стратегій маскування динамічних об'єктів."""

    @abstractmethod
    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Повертає бінарну маску: 255 = статичний фон, 0 = динамічний об'єкт.

        Args:
            frame_rgb: RGB зображення (H, W, 3), uint8

        Returns:
            Бінарна маска (H, W), uint8: 255 = статика, 0 = динаміка
        """

    @abstractmethod
    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Батчева обробка кадрів.

        Args:
            frames_rgb: список RGB зображень

        Returns:
            Список бінарних масок (одна на кадр)
        """


class YOLOMaskingStrategy(MaskingStrategy):
    """Стратегія маскування через YOLO сегментацію.

    Делегує обробку існуючому YOLOWrapper, зберігаючи всю логіку
    micro-batching, over-masking та фільтрації за класами.
    """

    def __init__(self, yolo_wrapper):
        """
        Args:
            yolo_wrapper: екземпляр YOLOWrapper (вже ініціалізований)
        """
        self._wrapper = yolo_wrapper
        logger.info("YOLOMaskingStrategy initialized")

    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        static_mask, _detections = self._wrapper.detect_and_mask(frame_rgb)
        return static_mask

    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        results = self._wrapper.detect_and_mask_batch(frames_rgb)
        return [static_mask for static_mask, _detections in results]


class NoMaskingStrategy(MaskingStrategy):
    """Заглушка без маскування — повертає повністю білу маску.

    Використовується для тестів та режиму без YOLO.
    """

    def __init__(self):
        logger.info("NoMaskingStrategy initialized (masking disabled)")

    def get_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

    def get_mask_batch(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        return [self.get_mask(f) for f in frames_rgb]


def create_masking_strategy(
    strategy_name: str,
    model_manager=None,
    device: str = "cuda",
) -> MaskingStrategy:
    """Фабрика стратегій маскування.

    Args:
        strategy_name: назва стратегії з конфігу ("yolo" | "none")
        model_manager: ModelManager для завантаження моделей
        device: пристрій для інференсу ("cuda" | "cpu")

    Returns:
        Екземпляр MaskingStrategy
    """
    if strategy_name == "yolo":
        if model_manager is None:
            raise ValueError("model_manager is required for YOLO masking strategy")
        from src.models.wrappers.yolo_wrapper import YOLOWrapper

        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, device)
        return YOLOMaskingStrategy(yolo_wrapper)

    if strategy_name == "none":
        return NoMaskingStrategy()

    raise ValueError(f"Unknown masking strategy: '{strategy_name}'. Supported: 'yolo', 'none'")


# ================================================================================
# File: models\wrappers\yolo_wrapper.py
# ================================================================================
import cv2
import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class YOLOWrapper:
    """Wrapper for YOLOv11 segmentation (compatible with YOLOv8 API)"""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        # Класи COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        self.dynamic_classes = {0, 1, 2, 3, 5, 7}

        # FP16 для YOLO — прискорює інференс на ~40%
        # Ultralytics керує FP16 через параметр half=True при виклику
        self.use_half = device == "cuda" and torch.cuda.is_available()

    @torch.no_grad()
    def detect_and_mask(self, image: np.ndarray) -> tuple:
        """
        Detect objects and create static mask (single image).
        Делегує до batch-методу для уникнення дублювання логіки.

        Returns:
            static_mask: Binary mask of static areas (255 for static, 0 for dynamic)
            detections: List of detection dicts
        """
        results = self.detect_and_mask_batch([image])
        if not results:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255, []
        return results[0]

    @torch.no_grad()
    def detect_and_mask_batch(self, images: list[np.ndarray]) -> list[tuple]:
        """
        Detect objects and create static masks for a batch of images.
        Повертає list[(static_mask, detections)] того самого порядку.
        """
        if not images:
            return []

        # YOLO expects list of images or a 4D tensor (B, H, W, 3). Ultralytics natively handles list of ndarrays.
        # half=True для FP16 інференсу
        # conf=0.50 відкидає слабкі передбачення
        results = self.model(images, verbose=False, half=self.use_half, conf=0.50)

        output = []
        MAX_SINGLE_MASK_RATIO = 0.40
        MAX_COMBINED_MASK_RATIO = 0.70

        for result, image in zip(results, images):
            height, width = image.shape[:2]
            static_mask = np.ones((height, width), dtype=np.uint8) * 255
            detections = []
            total_pixels = height * width

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                dynamic_mask_indices = [
                    i for i, cls in enumerate(classes) if cls in self.dynamic_classes
                ]

                for i, (cls, conf, box) in enumerate(zip(classes, confidences, boxes)):
                    detections.append(
                        {"class_id": int(cls), "confidence": float(conf), "bbox": box[:4].tolist()}
                    )

                if dynamic_mask_indices:
                    combined_dynamic = np.zeros((height, width), dtype=np.float32)
                    for idx in dynamic_mask_indices:
                        mask_resized = cv2.resize(
                            masks[idx], (width, height), interpolation=cv2.INTER_NEAREST
                        )
                        mask_area = np.sum(mask_resized > 0.5)
                        if mask_area / total_pixels > MAX_SINGLE_MASK_RATIO:
                            continue
                        combined_dynamic = np.maximum(combined_dynamic, mask_resized)

                    combined_area = np.sum(combined_dynamic > 0.5)
                    if combined_area / total_pixels < MAX_COMBINED_MASK_RATIO:
                        static_mask[combined_dynamic > 0.5] = 0
                    else:
                        logger.warning(
                            f"YOLO OVER-MASKING DETECTED ({combined_area / total_pixels:.2%}). "
                            "Frame preserved."
                        )

            output.append((static_mask, detections))

        return output


# ================================================================================
# File: models\wrappers\__init__.py
# ================================================================================
"""Model wrappers module"""


# ================================================================================
# File: tracking\kalman_filter.py
# ================================================================================
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrajectoryFilter:
    """Kalman filter for GPS trajectory smoothing optimized for high speeds"""

    def __init__(self, process_noise=2.0, measurement_noise=5.0, dt=1.0):
        # Filter state: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Збільшений шум процесу та зменшений шум вимірювання
        # дозволяють фільтру швидше реагувати на зміни курсу на високих швидкостях
        self.process_noise = process_noise
        self.is_initialized = False

        logger.info("Initializing Kalman filter for high-speed trajectory smoothing")
        logger.info(
            f"Parameters: process_noise={process_noise}, measurement_noise={measurement_noise}, dt={dt}"
        )

        self.kf.P *= 1000.0

        self.kf.F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        self.kf.R = np.array([[measurement_noise, 0.0], [0.0, measurement_noise]])

        self._update_matrices_for_dt(dt)

    def _update_matrices_for_dt(self, dt: float):
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        q_var = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        self.kf.Q = np.zeros((4, 4))
        self.kf.Q[0:2, 0:2] = q_var
        self.kf.Q[2:4, 2:4] = q_var

    def update(self, measurement: tuple, dt: float = 1.0) -> tuple:
        z = np.array([[measurement[0]], [measurement[1]]])

        if not self.is_initialized:
            self.kf.x = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])
            self.is_initialized = True
            logger.info(f"Kalman filter initialized: ({measurement[0]:.2f}, {measurement[1]:.2f})")
            return measurement

        dt = max(0.01, min(dt, 5.0))
        self._update_matrices_for_dt(dt)

        self.kf.predict()
        self.kf.update(z)

        filtered_x = float(self.kf.x[0, 0])
        filtered_y = float(self.kf.x[1, 0])

        return filtered_x, filtered_y


# ================================================================================
# File: tracking\outlier_detector.py
# ================================================================================
from collections import deque

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history"""

    def __init__(self, window_size=10, threshold_std=3.0, max_speed_mps=1000.0, max_consecutive=5):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self._consecutive_outliers = 0
        self._max_consecutive = max_consecutive

        logger.info("Initializing OutlierDetector")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}"
        )

    def add_position(self, position: tuple):
        self.window.append(np.array(position, dtype=np.float32))
        self._consecutive_outliers = 0  # Скидаємо лічильник — позиція прийнята

    def is_outlier(self, new_position: tuple, dt: float = 1.0) -> bool:
        if len(self.window) < 3:
            return False

        new_pos_np = np.array(new_position, dtype=np.float32)
        last_pos = self.window[-1]

        # 1. Перевірка максимально допустимої швидкості
        distance = float(np.linalg.norm(new_pos_np - last_pos))
        instantaneous_speed = distance / max(dt, 0.01)

        is_speed_outlier = instantaneous_speed > self.max_speed_mps

        # 2. Статистичний Z-score тест
        history = list(self.window)
        distances = [np.linalg.norm(history[i] - history[i - 1]) for i in range(1, len(history))]

        mean_dist = np.mean(distances)
        std_dist = max(np.std(distances), 1.0)

        z_score = abs(distance - mean_dist) / std_dist
        is_zscore_outlier = z_score > self.threshold_std and abs(distance - mean_dist) > 50.0

        if is_speed_outlier or is_zscore_outlier:
            self._consecutive_outliers += 1

            # Якщо забагато підряд — дрон реально перемістився, скидаємо вікно
            if self._consecutive_outliers >= self._max_consecutive:
                logger.warning(
                    f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                    f"accepting new position as legitimate movement"
                )
                self.window.clear()
                self._consecutive_outliers = 0
                return False  # Приймаємо нову позицію

            if is_speed_outlier:
                logger.warning(
                    f"OUTLIER DETECTED: Speed too high ({instantaneous_speed:.2f} m/s > {self.max_speed_mps} m/s)"
                )
            else:
                logger.warning(
                    f"OUTLIER DETECTED: Z-score {z_score:.2f} > {self.threshold_std}, distance {distance:.0f}m"
                )
            return True

        self._consecutive_outliers = 0
        return False


# ================================================================================
# File: tracking\__init__.py
# ================================================================================
"""Tracking module"""


# ================================================================================
# File: utils\image_preprocessor.py
# ================================================================================
import cv2
import numpy as np

from config.config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config or {}
        # Ініціалізуємо алгоритм локального контрасту CLAHE
        # clipLimit=3.0 дає сильне витягування тіней, tileGridSize=(8,8) - розмір блоку
        clip = get_cfg(self.config, "preprocessing.clahe_clip_limit", 3.0)
        tile_cfg = get_cfg(self.config, "preprocessing.clahe_tile_grid", [8, 8])
        # Підтримуємо і список [8, 8] і число 8
        tile = tuple(tile_cfg) if isinstance(tile_cfg, list) else (tile_cfg, tile_cfg)
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        logger.info("ImagePreprocessor initialized with CLAHE (Local Contrast Enhancement)")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        # 1. Переводимо RGB в колірний простір LAB, щоб відділити яскравість від кольору
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # 2. Застосовуємо CLAHE виключно до каналу яскравості (L)
        l_clahe = self.clahe.apply(l_channel)

        # 3. Збираємо канали назад і повертаємо в RGB
        merged_lab = cv2.merge((l_clahe, a, b))
        enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

        return enhanced_rgb


# ================================================================================
# File: utils\image_utils.py
# ================================================================================
import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def opencv_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """Перетворення зображення OpenCV (BGR) у QPixmap (RGB) для PyQt6"""
    if cv_image is None or cv_image.size == 0:
        return QPixmap()

    if len(cv_image.shape) == 3:
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # ВИПРАВЛЕНО: використовуємо .copy() щоб гарантувати неперервність
        # буфера в пам'яті та захистити від його знищення GC раніше QPixmap.
        # Без copy() буфер numpy може стати недійсним до відображення → segfault.
        cv_rgb = np.ascontiguousarray(cv_rgb)
        q_img = QImage(cv_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Копіюємо в QPixmap одразу, поки cv_rgb ще існує в цьому scope
        return QPixmap.fromImage(q_img.copy())

    elif len(cv_image.shape) == 2:
        height, width = cv_image.shape
        bytes_per_line = width

        gray = np.ascontiguousarray(cv_image)
        q_img = QImage(gray.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(q_img.copy())

    return QPixmap()


def qpixmap_to_opencv(pixmap: QPixmap) -> np.ndarray:
    """Перетворення QPixmap (RGB) у масив OpenCV (BGR)"""
    q_img = pixmap.toImage()
    q_img = q_img.convertToFormat(QImage.Format.Format_RGB888)

    width = q_img.width()
    height = q_img.height()

    ptr = q_img.bits()
    ptr.setsize(height * width * 3)

    # ВИПРАВЛЕНО: робимо copy() щоб масив numpy не залежав від буфера
    # QImage, який може бути знищений після виходу з функції
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)).copy()

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ================================================================================
# File: utils\logging_utils.py
# ================================================================================
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """Налаштування системи логування для всієї програми."""
    logger.remove()

    # Standart output (pretty console)
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard file output (text)
    logger.add(
        str(log_path),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    # JSON sink for structured logging/metrics
    json_path = log_path.with_name("metrics.jsonl")
    logger.add(
        str(json_path),
        level=log_level,
        serialize=True,
        rotation="10 MB",
        retention="14 days",
    )


def get_logger(name: str | None = None) -> Any:
    """Отримання екземпляра логера."""
    if name:
        return logger.bind(name=name)
    return logger


# ================================================================================
# File: utils\__init__.py
# ================================================================================
"""Utilities module"""


# ================================================================================
# File: workers\calibration_propagation_worker.py
# ================================================================================
import json

import cv2
import h5py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Хвильова пропагація на основі візуального матчингу.
    Генерує фінальну метричну афінну матрицю (2x3) для кожного кадру в базі.
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, database, calibration, matcher, config=None):
        super().__init__()
        self.database = database
        self.calibration = calibration
        self.matcher = matcher
        self.config = config or {}
        self._is_running = True

        self.min_matches = get_cfg(self.config, "localization.min_matches", 15)
        self.ransac_thresh = get_cfg(self.config, "localization.ransac_threshold", 3.0)

        self.frame_w = self.database.metadata.get("frame_width", 1920)
        self.frame_h = self.database.metadata.get("frame_height", 1080)

        # Базова сітка точок для точної апроксимації афінної матриці (4x4)
        grid_x = np.linspace(0, self.frame_w, 4)
        grid_y = np.linspace(0, self.frame_h, 4)

        gx, gy = np.meshgrid(grid_x, grid_y)
        self.grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._propagate()
        except Exception as e:
            logger.error(f"Propagation failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        all_anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)
        anchors = [a for a in all_anchors if a.frame_id < num_frames]
        
        if len(anchors) < len(all_anchors):
            logger.warning(f"Filtered {len(all_anchors) - len(anchors)} out-of-bounds anchors (DB has {num_frames} frames).")
            
        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        logger.info(
            f"Starting visual wave propagation for {num_frames} frames using {len(anchors)} anchors"
        )

        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.zeros(num_frames, dtype=bool)

        # QA metrics
        frame_rmse = np.zeros(num_frames, dtype=np.float32)
        frame_disagreement = np.zeros(num_frames, dtype=np.float32)
        frame_matches = np.zeros(num_frames, dtype=np.int32)

        # Оптимізація A: Batch prefetch всіх фіч у RAM
        self.progress.emit(0, "Передзавантаження фіч у RAM...")
        self._all_features = {}
        for i in range(num_frames):
            if not self._is_running:
                return
            try:
                self._all_features[i] = self.database.get_local_features(i)
            except Exception:
                pass
            if i % 1000 == 0:
                self.progress.emit(int(i / num_frames * 10), f"Prefetch: {i}/{num_frames}")
        logger.info(f"Prefetched features for {len(self._all_features)} frames")

        anchor_features = {}
        for anchor in anchors:
            try:
                feat = self._all_features.get(anchor.frame_id)
                if feat is not None:
                    anchor_features[anchor.frame_id] = feat

                frame_affine[anchor.frame_id] = anchor.affine_matrix
                frame_valid[anchor.frame_id] = True
                frame_rmse[anchor.frame_id] = getattr(anchor, "rmse_m", 0.0)
                frame_matches[anchor.frame_id] = getattr(anchor, "inliers_count", 0)
            except Exception as e:
                self.error.emit(f"Не вдалося ініціалізувати якір {anchor.frame_id}: {e}")
                return

        segments = self._build_segments(anchors, num_frames)
        total_segments = len(segments)

        # Fix 7: Паралельна propagation по незалежних сегментах
        from concurrent.futures import ThreadPoolExecutor, as_completed

        between_segments = [s for s in segments if s["type"] == "between"]
        tail_segments = [s for s in segments if s["type"] == "tail"]

        logger.info(f"Parallel processing of {len(between_segments)} segments...")
        max_workers = get_cfg(self.config, "models.performance.propagation_max_workers", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_segment,
                    seg,
                    anchor_features,
                    frame_affine,
                    frame_valid,
                    frame_rmse,
                    frame_disagreement,
                    frame_matches,
                    i,
                    total_segments,
                    num_frames,
                ): seg
                for i, seg in enumerate(between_segments)
            }
            for future in as_completed(futures):
                if not self._is_running:
                    break
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Segment processing failed: {e}")

        # Хвости зазвичай дешевші і можуть бути послідовними або теж паралельними
        for i, seg in enumerate(tail_segments):
            if not self._is_running:
                break
            self._process_segment(
                seg,
                anchor_features,
                frame_affine,
                frame_valid,
                frame_rmse,
                frame_disagreement,
                frame_matches,
                len(between_segments) + i,
                total_segments,
                num_frames,
            )

        valid_count = int(np.sum(frame_valid))
        self.progress.emit(90, "Збереження результатів у HDF5...")
        self._save_to_hdf5(
            frame_affine, frame_valid, frame_rmse, frame_disagreement, frame_matches, anchors
        )

        self.progress.emit(100, f"Готово! {valid_count}/{num_frames} кадрів отримали координати.")
        self.completed.emit()

    def _build_segments(self, anchors: list, num_frames: int) -> list:
        segments = []
        if anchors[0].frame_id > 0:
            segments.append(
                {
                    "type": "tail",
                    "frames": list(range(anchors[0].frame_id - 1, -1, -1)),
                    "anchor": anchors[0],
                }
            )

        for i in range(len(anchors) - 1):
            left_anchor = anchors[i]
            right_anchor = anchors[i + 1]
            segments.append(
                {
                    "type": "between",
                    "left_anchor": left_anchor,
                    "right_anchor": right_anchor,
                    "frames": list(range(left_anchor.frame_id + 1, right_anchor.frame_id)),
                }
            )

        if anchors[-1].frame_id < num_frames - 1:
            segments.append(
                {
                    "type": "tail",
                    "frames": list(range(anchors[-1].frame_id + 1, num_frames)),
                    "anchor": anchors[-1],
                }
            )

        # Валідація: сегменти НЕ повинні перетинатися — це гарантує
        # thread-safety при паралельному записі в numpy-масиви через ThreadPoolExecutor
        all_frame_ids = set()
        for seg in segments:
            seg_frames = set(seg["frames"])
            overlap = all_frame_ids & seg_frames
            assert not overlap, (
                f"CRITICAL: Segment overlap detected on frames {overlap}! "
                f"This would cause race conditions in ThreadPoolExecutor."
            )
            all_frame_ids |= seg_frames

        return segments

    def _process_segment(
        self,
        segment,
        anchor_features,
        frame_affine,
        frame_valid,
        frame_rmse,
        frame_disagreement,
        frame_matches,
        seg_idx,
        total_segments,
        num_frames,
    ):
        if segment["type"] == "tail":
            anchor = segment["anchor"]
            frames = segment["frames"]
            self._wave_from_anchor(
                frames=frames,
                anchor=anchor,
                anchor_feat=anchor_features[anchor.frame_id],
                frame_affine=frame_affine,
                frame_valid=frame_valid,
                frame_rmse=frame_rmse,
                frame_matches=frame_matches,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames,
            )
        elif segment["type"] == "between":
            left_anchor = segment["left_anchor"]
            right_anchor = segment["right_anchor"]
            frames = segment["frames"]

            h_left_res = self._build_homography_chain(
                frames, left_anchor, anchor_features[left_anchor.frame_id]
            )
            frames_reversed = list(reversed(frames))
            h_right_res = self._build_homography_chain(
                frames_reversed, right_anchor, anchor_features[right_anchor.frame_id]
            )

            total_frames_in_seg = len(frames)
            for local_idx, frame_id in enumerate(frames):
                if not self._is_running:
                    return

                if local_idx % 10 == 0:
                    prog = int(
                        (
                            seg_idx / total_segments
                            + local_idx / (total_frames_in_seg * total_segments)
                        )
                        * 90
                    )
                    self.progress.emit(prog, f"Блендінг: кадр {frame_id}/{num_frames}...")

                H_to_left_info = h_left_res.get(frame_id)
                H_to_right_info = h_right_res.get(frame_id)

                metric_pts_left = None
                n_left = 0
                if H_to_left_info:
                    metric_pts_left = self._project_to_metric(H_to_left_info["H"], left_anchor)
                    n_left = H_to_left_info["matches"]

                metric_pts_right = None
                n_right = 0
                if H_to_right_info:
                    metric_pts_right = self._project_to_metric(H_to_right_info["H"], right_anchor)
                    n_right = H_to_right_info["matches"]

                final_metric_pts = None
                disagreement = 0.0

                if metric_pts_left is not None and metric_pts_right is not None:
                    disagreement = np.mean(
                        np.linalg.norm(metric_pts_left - metric_pts_right, axis=1)
                    )

                    dist_to_left = abs(frame_id - left_anchor.frame_id)
                    dist_to_right = abs(frame_id - right_anchor.frame_id)
                    weight_left = dist_to_right / (dist_to_left + dist_to_right)
                    weight_right = 1.0 - weight_left
                    final_metric_pts = (
                        metric_pts_left * weight_left + metric_pts_right * weight_right
                    )
                    frame_matches[frame_id] = int((n_left + n_right) / 2)
                elif metric_pts_left is not None:
                    final_metric_pts = metric_pts_left
                    frame_matches[frame_id] = n_left
                elif metric_pts_right is not None:
                    final_metric_pts = metric_pts_right
                    frame_matches[frame_id] = n_right

                if final_metric_pts is not None:
                    M, _ = cv2.estimateAffine2D(self.grid_points, final_metric_pts)
                    if M is not None:
                        frame_affine[frame_id] = M
                        frame_valid[frame_id] = True
                        proj = GeometryTransforms.apply_affine(self.grid_points, M)
                        rmse = np.sqrt(
                            np.mean(np.linalg.norm(proj - final_metric_pts, axis=1) ** 2)
                        )
                        frame_rmse[frame_id] = rmse
                        frame_disagreement[frame_id] = disagreement

    def _wave_from_anchor(
        self,
        frames,
        anchor,
        anchor_feat,
        frame_affine,
        frame_valid,
        frame_rmse,
        frame_matches,
        seg_idx,
        total_segments,
        num_frames,
    ):
        h_chain = self._build_homography_chain(frames, anchor, anchor_feat)
        total_frames_in_seg = len(frames)

        for local_idx, frame_id in enumerate(frames):
            if not self._is_running:
                return

            if local_idx % 20 == 0:
                prog = int(
                    (seg_idx / total_segments + local_idx / (total_frames_in_seg * total_segments))
                    * 90
                )
                self.progress.emit(
                    prog, f"Хвиля від {anchor.frame_id}: кадр {frame_id}/{num_frames}..."
                )

            info = h_chain.get(frame_id)
            if info:
                metric_pts = self._project_to_metric(info["H"], anchor)
                if metric_pts is not None:
                    M, _ = cv2.estimateAffine2D(self.grid_points, metric_pts)
                    if M is not None:
                        frame_affine[frame_id] = M
                        frame_valid[frame_id] = True
                        proj = GeometryTransforms.apply_affine(self.grid_points, M)
                        rmse = np.sqrt(np.mean(np.linalg.norm(proj - metric_pts, axis=1) ** 2))
                        frame_rmse[frame_id] = rmse
                        frame_matches[frame_id] = info["matches"]

    def _build_homography_chain(self, frames, anchor, anchor_feat):
        """
        Fix 4: Використання збережених frame_poses для миттєвої пропагації.
        O(1) замість O(N) GPU-викликів матчингу.
        """
        result = {anchor.frame_id: {"H": np.eye(3, dtype=np.float32), "matches": 100}}
        anchor_id = anchor.frame_id

        try:
            pose_anchor = self.database.frame_poses[anchor_id].astype(np.float64)
            if np.abs(np.linalg.det(pose_anchor)) > 1e-9:
                inv_pose_anchor = np.linalg.inv(pose_anchor)

                for frame_id in frames:
                    if not self._is_running:
                        break
                    try:
                        pose_frame = self.database.frame_poses[frame_id].astype(np.float64)
                        # Пропускаємо кадри з вироженою позою (не збережені / zeros)
                        if np.abs(np.linalg.det(pose_frame)) < 1e-9:
                            continue
                        # H від frame до anchor через збережені pose-ланцюги
                        H = (inv_pose_anchor @ pose_frame).astype(np.float32)
                        result[frame_id] = {"H": H, "matches": 50}  # 50 = estimated
                    except Exception:
                        continue
                return result
        except Exception as e:
            logger.warning(
                f"Failed to use saved poses for anchor {anchor_id}, falling back to visual: {e}"
            )

        # Fallback до візуальної пропагації (тільки якщо поз немає)
        h_cache = {anchor_id: np.eye(3, dtype=np.float32)}
        matches_cache = {anchor_id: 100}
        prev_features = anchor_feat
        prev_frame_id = anchor_id

        for frame_id in frames:
            if not self._is_running:
                break
            try:
                curr_features = self._all_features.get(frame_id)
                if curr_features is None:
                    continue

                res = self._match_pair_with_count(curr_features, prev_features)
                if res is None:
                    continue
                H_curr_to_prev, n_matches = res

                H_prev_to_anchor = h_cache[prev_frame_id]
                H_curr_to_anchor = (
                    H_prev_to_anchor.astype(np.float64) @ H_curr_to_prev.astype(np.float64)
                ).astype(np.float32)

                h_cache[frame_id] = H_curr_to_anchor
                matches_cache[frame_id] = n_matches
                result[frame_id] = {"H": H_curr_to_anchor, "matches": n_matches}

                prev_features = curr_features
                prev_frame_id = frame_id
            except Exception:
                continue
        return result

    def _match_pair_with_count(self, features_a: dict, features_b: dict) -> tuple | None:
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None
            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
            )
            if H is None:
                return None
            inliers = int(np.sum(mask))
            if inliers < self.min_matches:
                return None
            return H, inliers
        except Exception:
            return None

    def _project_to_metric(self, H_to_anchor, anchor):
        pts_in_anchor = GeometryTransforms.apply_homography(self.grid_points, H_to_anchor)
        if pts_in_anchor is None or len(pts_in_anchor) != len(self.grid_points):
            return None
        metric_pts = GeometryTransforms.apply_affine(pts_in_anchor, anchor.affine_matrix)
        return metric_pts

    def _save_to_hdf5(
        self, frame_affine, frame_valid, frame_rmse, frame_disagreement, frame_matches, anchors
    ):
        db_path = self.database.db_path
        self.database.close()
        try:
            with h5py.File(db_path, "a") as f:
                if "calibration" in f:
                    del f["calibration"]
                grp = f.create_group("calibration")

                # Дані та метадані версії 2.1
                grp.attrs["version"] = "2.1"
                grp.attrs["num_anchors"] = len(anchors)
                grp.attrs["anchors_json"] = json.dumps(
                    [a.to_dict() for a in anchors], ensure_ascii=False
                )
                grp.attrs["projection_json"] = json.dumps(
                    self.calibration.converter.export_metadata()
                )

                # Датасети
                grp.create_dataset("frame_affine", data=frame_affine, compression="gzip")
                grp.create_dataset(
                    "frame_valid", data=frame_valid.astype(np.uint8), compression="gzip"
                )
                grp.create_dataset("frame_rmse", data=frame_rmse, compression="gzip")
                grp.create_dataset(
                    "frame_disagreement", data=frame_disagreement, compression="gzip"
                )
                grp.create_dataset("frame_matches", data=frame_matches, compression="gzip")

            logger.success(
                f"Successful propagation saved to HDF5 (rev 2.1, {len(anchors)} anchors)"
            )
        finally:
            self.database._load_hot_data()

# ================================================================================
# File: workers\database_worker.py
# ================================================================================
from PyQt6.QtCore import QThread, pyqtSignal

from src.database.database_builder import DatabaseBuilder
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseGenerationWorker(QThread):
    """Фоновий потік для генерації HDF5 бази даних (XFeat + DINOv2)"""

    progress = pyqtSignal(int, str)
    frame_processed = pyqtSignal(int)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, video_path: str, output_path: str, model_manager, config=None):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_manager = model_manager
        self.config = config or {}
        self._is_running = True

        logger.info("DatabaseGenerationWorker initialized")
        logger.info(f"Video: {video_path}")
        logger.info(f"Output: {output_path}")

    def run(self):
        logger.info("DatabaseGenerationWorker thread started")

        try:
            self.progress.emit(0, "Ініціалізація бази даних (XFeat + DINOv2)...")
            logger.info("Initializing database builder...")

            builder = DatabaseBuilder(
                output_path=self.output_path,
                config=self.config,
            )

            def update_progress(percent: int):
                if not self._is_running:
                    logger.warning("Database generation interrupted by user")
                    raise InterruptedError("Обробку скасовано користувачем")
                self.progress.emit(percent, f"Обробка кадрів... {percent}%")

            logger.info("Starting video processing...")
            builder.build_from_video(
                video_path=self.video_path,
                model_manager=self.model_manager,
                progress_callback=update_progress,
            )

            if self._is_running:
                self.progress.emit(100, "Базу даних успішно створено!")
                logger.success(f"Database generation completed: {self.output_path}")
                self.completed.emit(self.output_path)

        except InterruptedError as e:
            logger.warning(f"Database generation interrupted: {e}")
            self.cancelled.emit()
        except Exception as e:
            logger.error(f"Database generation failed: {e}", exc_info=True)
            self.error.emit(f"Критична помилка: {str(e)}")

    def stop(self):
        logger.info("Stopping DatabaseGenerationWorker...")
        self._is_running = False


# ================================================================================
# File: workers\panorama_overlay_worker.py
# ================================================================================
import base64

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PanoramaOverlayWorker(QThread):
    """Фоновий потік для локалізації та підготовки панорами до відображення на карті"""

    success = pyqtSignal(str, float, float, float, float, float, float, float, float)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, localizer):
        super().__init__()
        self.image_path = image_path
        self.localizer = localizer

    def run(self):
        try:
            logger.info(f"Starting background panorama overlay for {self.image_path}")
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError("Не вдалося прочитати файл зображення панорами")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loc_result = self.localizer.localize_frame(img_rgb)

            if not loc_result.get("success"):
                raise RuntimeError(loc_result.get("error", "Не вдалося локалізувати панораму"))

            fov = loc_result.get("fov_polygon")
            if not fov or len(fov) != 4:
                raise RuntimeError("Локалізатор не повернув коректні кути (FOV) для панорами")

            h, w = img.shape[:2]
            scale = 1.0
            if w > 4000:
                scale = 4000.0 / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_string = base64.b64encode(buffer).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64_string}"

            self.success.emit(
                data_url,
                fov[0][0],
                fov[0][1],
                fov[1][0],
                fov[1][1],
                fov[2][0],
                fov[2][1],
                fov[3][0],
                fov[3][1],
            )

            logger.success("Panorama successfully processed in background")

        except Exception as e:
            logger.error(f"Panorama overlay worker failed: {e}")
            self.error.emit(str(e))


# ================================================================================
# File: workers\panorama_worker.py
# ================================================================================
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PanoramaWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, frame_step: int = 30):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.frame_step = frame_step
        self._is_running = True

    def run(self):
        logger.info(f"Starting panorama generation from: {self.video_path}")
        try:
            self.progress.emit(0, "Відкриття відео...")

            # Використовуємо FFmpeg для надійності
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    raise ValueError("Не вдалося відкрити відеофайл")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_stitch = []
            frame_count = 0

            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.frame_step == 0:
                    # Зменшуємо кадр для економії пам'яті (4K→Full HD)
                    h, w = frame.shape[:2]
                    if w > 1920:
                        scale = 1920.0 / w
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    frames_to_stitch.append(frame)

                frame_count += 1

                if frame_count % 30 == 0:
                    prog = int((frame_count / total_frames) * 50)  # Перші 50% прогресу - зчитування
                    self.progress.emit(prog, f"Збирання кадрів: {len(frames_to_stitch)} шт.")

            cap.release()

            if not self._is_running:
                return

            self.progress.emit(50, "Зшивання панорами (це може зайняти час)...")
            logger.info(f"Stitching {len(frames_to_stitch)} frames...")

            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            status, panorama = stitcher.stitch(frames_to_stitch)

            if status == cv2.Stitcher_OK:
                cv2.imwrite(self.output_path, panorama)
                self.progress.emit(100, "Панораму збережено!")
                self.completed.emit(self.output_path)
            else:
                raise ValueError(
                    f"Помилка зшивання (Код OpenCV: {status}). Спробуйте змінити крок кадрів."
                )

        except Exception as e:
            logger.error(f"Panorama generation failed: {e}")
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False


# ================================================================================
# File: workers\tracking_worker.py
# ================================================================================
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config.config import get_cfg
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealtimeTrackingWorker(QThread):
    """Real-time localization worker thread (Optimized for XFeat + YOLO11)"""

    frame_ready = pyqtSignal(np.ndarray)
    location_found = pyqtSignal(float, float, float, int)
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    fov_found = pyqtSignal(list)

    def __init__(self, video_source: str, localizer, model_manager=None, config=None):
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.model_manager = model_manager
        self.config = config or {}
        self._stop_event = threading.Event()

        # Скільки кадрів розпізнавати за одну секунду ВІДЕО.
        # 1.0 = 1 кадр в секунду; 2.0 = кожні 0.5 секунд; 0.5 = кожні 2 секунди відео.
        # Ти можеш змінити це число прямо тут для тестів:
        self.process_fps = get_cfg(self.config, "tracking.process_fps", 1.0)

    def run(self):
        # Fix 6: Pre-warm fallback моделей при старті трекінгу
        threading.Thread(target=self._prewarm_fallback_models, daemon=True).start()

        logger.info(f"Starting tracking from source: {self.video_source}")

        yolo_wrapper = None
        if self.model_manager:
            try:
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
                logger.success("YOLO loaded for dynamic object masking in tracking loop")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                self.error.emit(f"YOLO load error: {e}")
                return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.video_source}")
            return

        # Визначаємо натуральну швидкість відео (зазвичай 30 FPS)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        # Інтервал обробки у секундах відео (наприклад, 1.0 / 2.0 = 0.5 секунд)
        process_interval_sec = 1.0 / self.process_fps if self.process_fps > 0 else 1.0

        # Ставимо від'ємний час, щоб гарантовано обробити найперший кадр
        last_process_video_time = -process_interval_sec
        last_localization_real_time = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                self.status_update.emit("Відеопотік завершено.")
                break

            # Отримуємо поточний час САМОГО ВІДЕО у секундах (не залежить від швидкості комп'ютера)
            current_video_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # Fallback: деякі кодеки повертають 0 — рахуємо за номером кадру
            if current_video_time_sec <= 0:
                current_video_time_sec = cap.get(cv2.CAP_PROP_POS_FRAMES) * frame_duration_sec

            # 1. Завжди відправляємо кадр в GUI для плавного відтворення
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame_rgb)

            # 2. Локалізація (спрацьовує тільки якщо відео пройшло заданий інтервал)
            if current_video_time_sec - last_process_video_time >= process_interval_sec:
                start_process = time.time()

                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                # Розраховуємо реальний dt для фільтра Калмана
                current_real_time = time.time()
                calculated_dt = current_real_time - last_localization_real_time

                # БЛОК TRY-EXCEPT для запобігання "зависанню на першому кадрі"
                try:
                    loc_result = self.localizer.localize_frame(
                        frame_rgb, static_mask=static_mask, dt=calculated_dt
                    )
                except Exception as e:
                    logger.error(f"Localization exception: {e}", exc_info=True)
                    loc_result = {"success": False, "error": str(e)}

                if loc_result.get("success"):
                    self.location_found.emit(
                        loc_result["lat"],
                        loc_result["lon"],
                        loc_result["confidence"],
                        loc_result["inliers"],
                    )
                    if "fov_polygon" in loc_result and loc_result["fov_polygon"] is not None:
                        self.fov_found.emit(loc_result["fov_polygon"])

                    if loc_result.get("fallback_mode") == "retrieval_only":
                        self.status_update.emit(
                            f"Приблизно (Схожість: {loc_result.get('global_score', 0):.2f}, Кадр: {loc_result['matched_frame']})"
                        )
                    else:
                        self.status_update.emit(
                            f"Знайдено (Inliers: {loc_result['inliers']}, Кадр: {loc_result['matched_frame']})"
                        )
                else:
                    self.status_update.emit(
                        f"Втрата: {loc_result.get('error', 'Невідома помилка')}"
                    )

                # Оновлюємо завжди — інакше dt накопичується і Kalman робить стрибок
                last_localization_real_time = current_real_time

                last_process_video_time = current_video_time_sec

                # Рахуємо швидкість самого алгоритму
                process_duration = time.time() - start_process
                self.fps_updated.emit(1.0 / process_duration if process_duration > 0 else 0)

            # 3. Синхронізація відтворення: щоб відео не "пролітало" за секунду,
            # змушуємо потік почекати, імітуючи реальну швидкість відео (1x)
            elapsed_in_loop = time.time() - loop_start
            sleep_time = frame_duration_sec - elapsed_in_loop
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

        cap.release()
        logger.info("Tracking worker thread finished cleanly.")

    def _prewarm_fallback_models(self):
        """Завантажує важкі моделі фоллбеку заздалегідь."""
        try:
            if not self.model_manager:
                return

            fallback = get_cfg(self.config, "localization.fallback_extractor", "aliked")
            logger.info(f"Pre-warming fallback models ({fallback})...")

            if fallback == "aliked":
                self.model_manager.load_aliked()
                self.model_manager.load_lightglue_aliked()
            else:
                self.model_manager.load_superpoint()
                self.model_manager.load_lightglue()

            logger.success("Fallback models pre-warmed successfully")
        except Exception as e:
            logger.warning(f"Fallback pre-warm failed: {e}")

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._stop_event.set()
        self.wait(5000)  # чекаємо максимум 5 секунд


# ================================================================================
# File: workers\__init__.py
# ================================================================================
"""Worker threads module"""
