

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


# Єдине джерело — src.geometry.affine_utils (усунення дублювання)
from src.geometry.affine_utils import (
    compose_affine as _compose_affine,
)
from src.geometry.affine_utils import (
    decompose_affine as _decompose_affine,
)
from src.geometry.affine_utils import (
    unwrap_angles as _unwrap_angles,
)


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
        """
        Перебудовує PCHIP-інтерполятори на основі ДЕКОМПОЗИЦІЇ афінних матриць.

        Замість поелементної інтерполяції raw-компонентів матриці (що порушує
        жорсткість обертання і призводить до артефактів зсуву/shear), кожна
        матриця розкладається на: (tx, ty, scale, angle).  Кути розгортаються
        через np.unwrap для уникнення стрибків через ±π.  Для кожного з 4
        скалярних каналів будується окремий PchipInterpolator.  При запиті
        (_get_interpolated_matrix) компоненти відновлюються назад у валідну
        афінну матрицю через _compose_affine.
        """
        if len(self.anchors) < 2:
            self._interp = None
            logger.debug(f"Interpolator not built: need ≥2 anchors, have {len(self.anchors)}")
            return

        ids = np.array([a.frame_id for a in self.anchors], dtype=np.float64)
        decomposed = np.array(
            [_decompose_affine(a.affine_matrix) for a in self.anchors],
            dtype=np.float64,
        )  # shape (N, 4): tx, ty, scale, angle

        # Розгортаємо кути для коректної інтерполяції через межу ±π
        decomposed[:, 3] = _unwrap_angles(decomposed[:, 3])

        # Один багатоколонковий PCHIP-інтерполятор для всіх 4 каналів
        self._interp = PchipInterpolator(ids, decomposed, extrapolate=True)

    def _get_interpolated_matrix(self, frame_id: float) -> np.ndarray | None:
        """Повертає інтерпольовану афінну матрицю 2x3 для заданого frame_id."""
        if self._interp is None:
            return None
        components = self._interp(frame_id)  # (4,): tx, ty, scale, angle
        if components is None or np.any(np.isnan(components)):
            return None
        tx, ty, scale, angle = components
        # Захист від вироджених значень масштабу
        scale = float(np.clip(scale, 1e-6, 1e6))
        return _compose_affine(float(tx), float(ty), scale, float(angle))

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

        # Decomposition-based PCHIP: інтерполяція через tx/ty/scale/angle
        if self._interp is not None:
            M = self._get_interpolated_matrix(float(frame_id))
            if M is not None:
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
            logger.error(
                f"Failed to create project: {e} | "
                f"workspace_dir={workspace_dir}, mission_name={mission_data.get('mission_name', '?')}. "
                f"Check disk permissions and available space.",
                exc_info=True,
            )
            self.project_dir = None
            self.settings = None
            return False

    def load_project(self, project_dir: str) -> bool:
        """Loads an existing project from the given directory path."""
        try:
            dir_path = Path(project_dir)
            if not dir_path.is_dir():
                logger.error(
                    f"Project directory does not exist: {project_dir}. "
                    f"It may have been moved or deleted."
                )
                return False

            json_file = dir_path / "project.json"
            if not json_file.exists():
                logger.error(
                    f"Missing project.json in: {project_dir}. "
                    f"This directory may not be a valid project or project.json was deleted. "
                    f"Available files: {[f.name for f in dir_path.iterdir() if f.is_file()][:10]}"
                )
                return False

            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            self.settings = ProjectSettings.from_dict(data)
            self.project_dir = dir_path

            logger.info(f"Project loaded successfully: {self.settings.project_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to load project: {e} | dir={project_dir}. "
                f"The project.json file may be corrupted or have invalid format.",
                exc_info=True,
            )
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
            logger.error(
                f"Failed to save project settings: {e} | "
                f"path={self.project_dir / 'project.json'}. "
                f"Check disk permissions and available space.",
                exc_info=True,
            )
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
                logger.warning(
                    f"Failed to load project registry: {e} | "
                    f"path={self._registry_path}. "
                    f"Registry will be reset. This is safe but recent project history will be lost."
                )
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
            logger.error(
                f"Failed to save project registry: {e} | "
                f"path={self._registry_path}. "
                f"Check disk permissions for {self._registry_dir}.",
                exc_info=True,
            )

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
import gc
import traceback
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import h5py
import numpy as np
import torch

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FeatureMatcher
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.models.wrappers.masking_strategy import create_masking_strategy
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

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

        use_decord = get_cfg(self.config, "database.use_decord", True)
        vr = None
        cap = None

        if use_decord:
            try:
                import decord

                decord.bridge.set_bridge("numpy")
                # FFMPEG multi-threaded CPU decode is usually the most stable fallback
                # GPU decode requires custom decord builds on Windows
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                logger.info("Decord VideoReader initialized successfully.")
            except ImportError:
                logger.warning("decord not installed, falling back to cv2.VideoCapture")
                use_decord = False
            except Exception as e:
                logger.warning(
                    f"Failed to initialize decord VideoReader: {e}. Falling back to cv2.VideoCapture"
                )
                use_decord = False

        if not use_decord:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(
                    f"Failed to open video: {video_path}. "
                    f"Check that the file exists and uses a supported codec (H.264/H.265 recommended)."
                )
                raise ValueError(f"Не вдалося відкрити відео: {video_path}")

        if use_decord:
            total_frames = len(vr)
            # Sample first frame to get dims
            h, w, c = vr.get_batch([0]).shape[1:]
            width, height = int(w), int(h)
            original_fps = vr.get_avg_fps()
        else:
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

        # Ініціалізуємо стратегію маскування (YOLO / none / ...)
        masking_strategy_name = get_cfg(self.config, "preprocessing.masking_strategy", "yolo")
        logger.info(f"Loading masking strategy: {masking_strategy_name}")
        masking_strategy = create_masking_strategy(
            masking_strategy_name, model_manager, model_manager.device
        )

        local_ext_type = get_cfg(self.config, "localization.fallback_extractor", "aliked")
        if local_ext_type == "xfeat":
            local_model = model_manager.load_xfeat()
        else:
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
            logger.warning(
                f"Failed to detect descriptor dimension: {e}\n{traceback.format_exc()}"
                f"Falling back to configured default: {self.descriptor_dim}"
            )
            logger.warning(f"Using default dimension: {self.descriptor_dim}")

        # Detect local descriptor dimension
        try:
            with torch.no_grad():
                dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                dummy_feats = feature_extractor.extract_features(dummy_img)
                if len(dummy_feats["descriptors"]) > 0:
                    self.local_descriptor_dim = dummy_feats["descriptors"].shape[-1]
                else:
                    # Fallback default dims
                    self.local_descriptor_dim = 64 if local_ext_type == "xfeat" else 128
            logger.info(f"Detected local descriptor dimension: {self.local_descriptor_dim}")
        except Exception as e:
            logger.warning(f"Failed to detect local feature dimension: {e}. Using 128 as fallback.")
            self.local_descriptor_dim = 128

        # Create empty database structure
        logger.info("Creating HDF5 database structure...")
        self.create_hdf5_structure(num_frames, width, height)

        current_pose = np.eye(3, dtype=np.float32)
        prev_features = None

        # Adaptive Keyframe Selection (П4)
        saved_count = 0  # лічильник РЕАЛЬНО записаних кадрів
        frame_index_map: list[int] = []  # список збережених frame_id
        use_keyframe_selection = (
            get_cfg(self.config, "database.keyframe_min_translation_px", 0.0) > 0
        )
        if use_keyframe_selection:
            logger.info(
                f"Adaptive keyframe selection ENABLED "
                f"(min_translation={get_cfg(self.config, 'database.keyframe_min_translation_px', 15.0)}px, "
                f"min_rotation={get_cfg(self.config, 'database.keyframe_min_rotation_deg', 1.5)}°)"
            )

        # cuDNN benchmark conditionally (Fix 5)

        if torch.cuda.is_available():
            model_type = get_cfg(self.config, "localization.fallback_extractor", "aliked")
            if model_type in ("xfeat", "aliked"):  # CNN-based types
                torch.backends.cudnn.benchmark = True
                logger.info(f"cuDNN benchmark ENABLED for {model_type}")

        # Increased prefetch queue (Fix 5)
        frame_queue = Queue(maxsize=self.prefetch_size)

        def prefetch_frames():
            if use_decord:
                # Decord provides batched read
                batch_size = get_cfg(self.config, "database.decode_batch_size", 32)
                indices = list(range(0, total_frames, frame_step))
                for chunk_start in range(0, len(indices), batch_size):
                    chunk_indices = indices[chunk_start : chunk_start + batch_size]

                    with Telemetry.profile("video_read"):
                        # Decord returns RGB (B, H, W, C)
                        frames_rgb = vr.get_batch(chunk_indices).asnumpy()

                    for i, frame_rgb in enumerate(frames_rgb):
                        orig_frame_idx = chunk_indices[i] // frame_step
                        with Telemetry.profile("rgb_to_bgr"):
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        frame_queue.put((orig_frame_idx, (frame_bgr, frame_rgb)))
            else:
                for i in range(total_frames):
                    with Telemetry.profile("video_read"):
                        ret, frame = cap.read()
                    if not ret:
                        break

                    if i % frame_step != 0:
                        continue

                    with Telemetry.profile("bgr_to_rgb"):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    orig_frame_idx = i // frame_step
                    frame_queue.put((orig_frame_idx, (frame, frame_rgb)))

            frame_queue.put((-1, None))

        prefetch_thread = Thread(target=prefetch_frames, daemon=True)
        prefetch_thread.start()

        try:
            self.db_file = h5py.File(self.output_path, "a")
            logger.info(f"Opened HDF5 file for writing: {self.output_path}")

            # YOLO micro-batching (П8)
            yolo_batch_size = get_cfg(self.config, "database.yolo_batch_size", 2)
            if yolo_batch_size > 1:
                logger.info(f"YOLO micro-batching ENABLED (batch_size={yolo_batch_size})")
            pending_frames: list[tuple] = []  # буфер (idx, frame, frame_rgb)

            def _flush_mask_batch(batch: list) -> list:
                """Обробляє батч через MaskingStrategy, повертає (idx, frame, frame_rgb, static_mask)."""
                images_rgb = [b[2] for b in batch]
                with Telemetry.profile("yolo"):
                    masks_list = masking_strategy.get_mask_batch(images_rgb)
                return [(b[0], b[1], b[2], m) for b, m in zip(batch, masks_list)]

            def _process_single_frame(
                p_idx,
                p_frame,
                p_frame_rgb,
                p_static_mask,
                current_pose,
                prev_features,
                saved_count,
                frame_index_map,
            ):
                """Обробляє один кадр після YOLO: feature extraction, pose, keyframe selection."""
                features = feature_extractor.extract_features(p_frame_rgb, p_static_mask)
                features["coords_2d"] = features["keypoints"]

                if kp_writer is not None:
                    kp_frame = self._draw_keypoints_frame(
                        p_frame, features["keypoints"], p_static_mask, p_idx, num_frames
                    )
                    if kp_scale != 1.0:
                        kp_w = int(width * kp_scale)
                        kp_h = int(height * kp_scale)
                        kp_frame = cv2.resize(kp_frame, (kp_w, kp_h), interpolation=cv2.INTER_AREA)
                    kp_writer.write(kp_frame)

                if p_idx == 0 or prev_features is None:
                    current_pose = np.eye(3, dtype=np.float64)
                    save_this_frame = True
                else:
                    H_step = self._compute_inter_frame_H(prev_features, features)
                    if H_step is not None:
                        current_pose = current_pose @ H_step.astype(np.float64)
                        if use_keyframe_selection:
                            save_this_frame = self._is_significant_motion(H_step, width, height)
                        else:
                            save_this_frame = True
                    else:
                        logger.warning(
                            f"Frame {p_idx}: inter-frame match failed, reusing previous pose"
                        )
                        save_this_frame = (
                            True  # Or False? Usually better to keep it if tracking fails
                        )

                prev_features = features

                # ЗАВЖДИ зберігаємо pose для повного ланцюга пропагації,
                # навіть якщо кадр не є keyframe (пропущений через малий рух).
                # Без цього frame_poses[frame_id] = zeros → пропагація ламається.
                if self.db_file:
                    self.db_file["global_descriptors"]["frame_poses"][p_idx] = current_pose

                if save_this_frame:
                    frame_index_map.append(p_idx)
                    # Зберігаємо за ОРИГІНАЛЬНИМ індексом p_idx, а не послідовним
                    # Це зберігає frame_id ↔ slot identity для калібрування/пропагації
                    self.save_frame_data(p_idx, features, current_pose)
                    saved_count += 1

                    if saved_count % 100 == 0:
                        progress_pct = int((p_idx + 1) / num_frames * 100)
                        logger.info(
                            f"Saved {saved_count} keyframes from {p_idx + 1}/{num_frames} processed "
                            f"({progress_pct}%)"
                        )

                progress_percent = int((p_idx + 1) / num_frames * 100)
                if progress_callback:
                    progress_callback(progress_percent)

                return current_pose, prev_features, saved_count

            while True:
                idx, data = frame_queue.get()

                if idx != -1 and data is not None:
                    frame, frame_rgb = data
                    pending_frames.append((idx, frame, frame_rgb))
                    if len(pending_frames) < yolo_batch_size:
                        continue  # накопичуємо батч

                # Якщо EOF або батч повний — обробляємо все накопичене
                if not pending_frames:
                    break

                processed = _flush_mask_batch(pending_frames)
                pending_frames = []

                for p_idx, p_frame, p_frame_rgb, p_static_mask in processed:
                    current_pose, prev_features, saved_count = _process_single_frame(
                        p_idx,
                        p_frame,
                        p_frame_rgb,
                        p_static_mask,
                        current_pose,
                        prev_features,
                        saved_count,
                        frame_index_map,
                    )

                if idx == -1:
                    break

        except Exception as e:
            logger.error(
                f"Error during database building: {e} | "
                f"video={video_path}, output={self.output_path}, "
                f"processed_frames={saved_count}",
                exc_info=True,
            )
            raise
        finally:
            # Зберігаємо frame_index_map і actual_num_frames у metadata
            if self.db_file and saved_count > 0:
                try:
                    meta = self.db_file["metadata"]
                    meta.attrs["actual_num_frames"] = saved_count
                    if "frame_index_map" not in meta:
                        meta.create_dataset(
                            "frame_index_map",
                            data=np.array(frame_index_map, dtype=np.int32),
                        )
                    if use_keyframe_selection:
                        logger.info(
                            f"Keyframe selection: {saved_count}/{num_frames} frames saved "
                            f"({100 - saved_count / num_frames * 100:.1f}% reduction)"
                        )
                except Exception as e:
                    logger.warning(f"Could not save frame_index_map: {e}")

            prefetch_thread.join(timeout=5)
            if kp_writer is not None:
                kp_writer.release()
            if self.db_file:
                self.db_file.close()
            if cap is not None:
                cap.release()

        logger.success(f"Database build completed successfully: {self.output_path}")

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
            self.matcher = FeatureMatcher(config=self.config)

        mkpts_a, mkpts_b = self.matcher.match(fa, fb)

        if len(mkpts_a) < min_matches:
            return None

        H, mask = GeometryTransforms.estimate_homography(
            mkpts_a, mkpts_b, ransac_threshold=ransac_thresh
        )

        if H is None or int(np.sum(mask)) < min_matches:
            return None

        return H.astype(np.float32)

    def _is_significant_motion(self, H: np.ndarray, frame_w: int, frame_h: int) -> bool:
        """
        Повертає True якщо гомографія H відповідає значному руху.
        H: (3,3) float32 — матриця з frame_b до frame_a.
        """
        min_t = get_cfg(self.config, "database.keyframe_min_translation_px", 15.0)
        min_r = get_cfg(self.config, "database.keyframe_min_rotation_deg", 1.5)

        # Трансляція: зсув центру кадру через H
        cx, cy = frame_w / 2.0, frame_h / 2.0
        p_src = np.array([cx, cy, 1.0], dtype=np.float64)
        p_dst = H.astype(np.float64) @ p_src
        p_dst /= p_dst[2]
        translation = np.linalg.norm(p_dst[:2] - np.array([cx, cy]))

        if translation >= min_t:
            return True

        # Кут: з лінійної частини H (2×2 зліва вгорі)
        A = H[:2, :2].astype(np.float64)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return True  # вироджена матриця → вважаємо рухом
        angle_rad = np.arctan2(A[1, 0], A[0, 0])
        angle_deg = abs(np.degrees(angle_rad))
        return angle_deg >= min_r

    def create_hdf5_structure(self, num_frames: int, width: int, height: int):
        """Create optimal HDF5 hierarchy with pre-allocated chunked arrays (schema v2)"""
        compression = get_cfg(self.config, "database.hdf5_compression", "lzf")
        chunk_f = get_cfg(self.config, "database.hdf5_chunk_frames", 64)
        max_kps = get_cfg(self.config, "database.max_keypoints_stored", 2048)
        local_desc_dim = getattr(self, "local_descriptor_dim", 128)

        logger.info(
            f"Creating HDF5 v2 structure for {num_frames} frames "
            f"(compression={compression}, chunks={chunk_f}, max_kps={max_kps})"
        )

        with h5py.File(self.output_path, "w", libver="latest") as f:
            # --- global_descriptors: chunked ---
            g1 = f.create_group("global_descriptors")
            g1.create_dataset(
                "descriptors",
                shape=(num_frames, self.descriptor_dim),
                maxshape=(None, self.descriptor_dim),
                dtype="float32",
                compression=compression,
                chunks=(min(256, num_frames), self.descriptor_dim),
            )
            g1.create_dataset(
                "frame_poses",
                shape=(num_frames, 3, 3),
                maxshape=(None, 3, 3),
                dtype="float64",
                compression=compression,
                chunks=(min(256, num_frames), 3, 3),
            )

            # --- local_features: PRE-ALLOCATED chunked arrays (НОВА СХЕМА v2) ---
            lf = f.create_group("local_features")
            lf.create_dataset(
                "keypoints",
                shape=(num_frames, max_kps, 2),
                maxshape=(None, max_kps, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, 2),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "descriptors",
                shape=(num_frames, max_kps, local_desc_dim),
                maxshape=(None, max_kps, local_desc_dim),
                dtype="float16",  # float16: -50% розміру (П2)
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, local_desc_dim),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "coords_2d",
                shape=(num_frames, max_kps, 2),
                maxshape=(None, max_kps, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, 2),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "kp_counts",  # скільки keypoints у кожному кадрі
                shape=(num_frames,),
                maxshape=(None,),
                dtype="int16",
                compression=compression,
                chunks=(min(num_frames, 4096),),
                fillvalue=0,
            )
            # Розміри кадру — зберігаємо ОДИН РАЗ у групі
            lf.attrs["frame_width"] = width
            lf.attrs["frame_height"] = height

            g3 = f.create_group("metadata")
            g3.attrs["num_frames"] = num_frames
            g3.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            g3.attrs["frame_width"] = width
            g3.attrs["frame_height"] = height
            g3.attrs["descriptor_dim"] = self.descriptor_dim
            g3.attrs["hdf5_schema"] = "v2"  # версія схеми для зворотної сумісності
            g3.attrs["max_keypoints"] = max_kps

            # Enable SWMR mode for parallel reading while writing
            f.swmr_mode = True

        logger.success("HDF5 v2 structure created successfully in SWMR mode")

    def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
        """Save extracted data for a single frame via slice assignment (schema v2)"""
        with Telemetry.profile("hdf5_write"):
            # global — без змін
            self.db_file["global_descriptors"]["descriptors"][frame_id] = features["global_desc"]
            self.db_file["global_descriptors"]["frame_poses"][frame_id] = pose_2d

            # local — slice assignment замість create_group + create_dataset
            kps = features["keypoints"]
            descs = features["descriptors"]
            c2d = features["coords_2d"]

            max_kps = self.db_file["local_features"]["keypoints"].shape[1]
            n = min(len(kps), max_kps)

            lf = self.db_file["local_features"]
            lf["keypoints"][frame_id, :n] = kps[:n]
            lf["descriptors"][frame_id, :n] = descs[:n].astype("float16")
            lf["coords_2d"][frame_id, :n] = c2d[:n]
            lf["kp_counts"][frame_id] = n


# ================================================================================
# File: database\database_loader.py
# ================================================================================
import json
from collections import OrderedDict
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
        self._feature_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        logger.info(f"Initializing DatabaseLoader | path={db_path}")
        self._load_hot_data()

    def _load_hot_data(self) -> None:
        """Load global descriptors (DINOv2), poses and propagation data into RAM"""
        logger.info(f"Loading hot data into RAM from: {self.db_path}")

        try:
            self.db_file = h5py.File(self.db_path, "r")
            logger.debug(f"HDF5 file opened | top-level groups: {list(self.db_file.keys())}")

            if "global_descriptors" not in self.db_file:
                raise KeyError(
                    f"HDF5 file is missing 'global_descriptors' group. "
                    f"Available groups: {list(self.db_file.keys())}. "
                    f"The database file may be corrupted or was created with an incompatible version."
                )

            self.global_descriptors = self.db_file["global_descriptors"]["descriptors"][:]
            self.frame_poses = self.db_file["global_descriptors"]["frame_poses"][:]

            logger.info(
                f"Loaded global descriptors: shape={self.global_descriptors.shape}, "
                f"dtype={self.global_descriptors.dtype}, "
                f"mem={self.global_descriptors.nbytes / 1024**2:.1f} MB"
            )
            logger.info(f"Loaded frame poses: shape={self.frame_poses.shape}")

            for key, value in self.db_file["metadata"].attrs.items():
                self.metadata[key] = value
                logger.debug(f"Metadata — {key}: {value}")

            if "actual_num_frames" in self.metadata:
                actual_num = int(self.metadata["actual_num_frames"])
                total_slots = len(self.global_descriptors)
                logger.info(
                    f"Database contains {actual_num} actual frames in {total_slots} pre-allocated slots"
                )
                # DO NOT SLICE with actual_num_frames! The arrays are sized to num_frames exactly,
                # and are indexed by absolute visual frame_id!

            if "frame_index_map" in self.db_file["metadata"]:
                self.frame_index_map = self.db_file["metadata"]["frame_index_map"][:]
                logger.debug(f"Frame index map loaded: {len(self.frame_index_map)} entries")
            else:
                self.frame_index_map = np.arange(len(self.global_descriptors))
                logger.debug("No frame_index_map found — using sequential indices")

            # Завантажуємо дані пропагації якщо є
            self._load_propagation_data()

            logger.success(
                f"Hot data loaded successfully | "
                f"{len(self.global_descriptors)} frames, "
                f"descriptor_dim={self.global_descriptors.shape[1]}, "
                f"propagated={'yes' if self.is_propagated else 'no'}"
            )

        except KeyError as e:
            logger.error(
                f"Database structure error: {e} | path={self.db_path}. "
                f"This usually means the HDF5 file is incomplete or was created with a different version."
            )
            raise
        except OSError as e:
            logger.error(
                f"Cannot open database file: {e} | path={self.db_path}. "
                f"Check that the file exists and is not locked by another process."
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error loading database: {e} | path={self.db_path}", exc_info=True
            )
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
                    logger.warning(
                        f"Could not load projection metadata: {e}. "
                        f"Raw value: {grp.attrs.get('projection_json', '<missing>')}. "
                        f"Falling back to default projection."
                    )
            elif "reference_gps" in grp.attrs:
                # Fallback для v2.0 (UTM)
                try:
                    ref_gps = json.loads(grp.attrs["reference_gps"])
                    self.converter = CoordinateConverter("UTM", tuple(ref_gps))
                    logger.success(f"UTM auto-initialized from legacy reference GPS: {ref_gps}")
                except Exception as e:
                    logger.warning(
                        f"Could not init UTM from legacy attribute: {e}. "
                        f"Raw reference_gps value: {grp.attrs.get('reference_gps', '<missing>')}. "
                        f"Defaulting to WEB_MERCATOR."
                    )
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
            logger.error(
                f"Failed to load propagation data: {e} | db_path={self.db_path}. "
                f"Calibration data may be corrupted — recalibration recommended.",
                exc_info=True,
            )
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

        # Нова схема v2: розміри збережені один раз в local_features.attrs
        schema = self.metadata.get("hdf5_schema", "v1")
        if schema == "v2" and "local_features" in self.db_file:
            lf_attrs = self.db_file["local_features"].attrs
            h = int(lf_attrs.get("frame_height", self.metadata.get("frame_height", 1080)))
            w = int(lf_attrs.get("frame_width", self.metadata.get("frame_width", 1920)))
            self._size_cache[frame_id] = (h, w)
            return h, w

        # Стара схема v1: fallback — читаємо з групи кадру (зворотня сумісність)
        group_name = f"local_features/frame_{frame_id}"
        if group_name in self.db_file:
            g = self.db_file[group_name]
            if "height" in g.attrs and "width" in g.attrs:
                h, w = int(g.attrs["height"]), int(g.attrs["width"])
            else:
                h = self.metadata.get("frame_height") or self.metadata.get("height") or 1080
                w = self.metadata.get("frame_width") or self.metadata.get("width") or 1920

        res = (int(h), int(w))
        self._size_cache[frame_id] = res
        return res

    def get_local_features(self, frame_id: int) -> dict[str, np.ndarray]:
        """Повертає локальні ознаки для вказаного кадру (сумісно з v1 і v2)"""
        if frame_id in self._feature_cache:
            self._feature_cache.move_to_end(frame_id)
            return self._feature_cache[frame_id]

        if self.db_file is None:
            raise RuntimeError("Database not opened")

        schema = self.metadata.get("hdf5_schema", "v1")
        if schema == "v2":
            lf = self.db_file["local_features"]
            n = int(lf["kp_counts"][frame_id])
            if n == 0:
                raise ValueError(f"Кадр {frame_id} не має keypoints (kp_count=0).")
            res = {
                "keypoints": lf["keypoints"][frame_id, :n],
                "descriptors": lf["descriptors"][frame_id, :n].astype("float32"),  # float16→32
                "coords_2d": lf["coords_2d"][frame_id, :n],
            }
        else:
            # Стара схема v1 — зворотня сумісність
            group_name = f"local_features/frame_{frame_id}"
            if group_name not in self.db_file:
                raise ValueError(f"Кадр {frame_id} не знайдено у базі даних.")
            g = self.db_file[group_name]
            res = {
                "keypoints": g["keypoints"][:],
                "descriptors": g["descriptors"][:],
                "coords_2d": g["coords_2d"][:],
            }

        # LRU-витіснення
        if len(self._feature_cache) >= 200:
            self._feature_cache.popitem(last=False)

        self._feature_cache[frame_id] = res
        return res

    def get_num_frames(self) -> int:
        """Повертає кількість кадрів у БД (pre-allocated slots для v2)."""
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
# File: geometry\affine_utils.py
# ================================================================================
"""
Утиліти декомпозиції/складання ізотропних афінних матриць.

Єдине джерело істини для decompose/compose — використовується в:
  - src.calibration.multi_anchor_calibration
  - src.workers.calibration_propagation_worker (графова оптимізація)
  - src.geometry.pose_graph_optimizer
"""

import numpy as np


def decompose_affine(M: np.ndarray) -> tuple[float, float, float, float]:
    """
    Розкладає афінну матрицю 2x3 на компоненти:
    (tx, ty, scale, angle_rad).

    Для афінної матриці вигляду:
        [s*cos(a)  -s*sin(a)  tx]
        [s*sin(a)   s*cos(a)  ty]
    scale = sqrt(det(R_part)), angle = atan2(M[1,0], M[0,0]).
    При наявності шуму (незначний зсув / анізотропний масштаб)
    беремо ізотропне наближення через норму першого стовпця.
    """
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    s_x = float(np.linalg.norm(M[:2, 0]))
    s_y = float(np.linalg.norm(M[:2, 1]))
    scale = (s_x + s_y) * 0.5
    if scale < 1e-9:
        scale = 1e-9
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, scale, angle


def compose_affine(tx: float, ty: float, scale: float, angle: float) -> np.ndarray:
    """Збирає афінну матрицю 2x3 з компонентів перенесення, масштабу та кута (рад)."""
    c = np.cos(angle) * scale
    s = np.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Розгортає масив кутів (рад) для уникнення стрибків ±π при інтерполяції."""
    return np.unwrap(angles)


def decompose_affine_5dof(M: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Розкладає афінну матрицю 2x3 на 5 компонентів для збереження анізотропії:
    (tx, ty, sx, sy, angle_rad).
    """
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    sx = float(np.linalg.norm(M[:2, 0]))
    sy = float(np.linalg.norm(M[:2, 1]))
    angle = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, sx, sy, angle


def compose_affine_5dof(tx: float, ty: float, sx: float, sy: float, angle: float) -> np.ndarray:
    """
    Збирає афінну матрицю 2x3 з незалежними масштабами X та Y.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c * sx, -s * sy, tx], [s * sx, c * sy, ty]], dtype=np.float64)


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

    @property
    def is_initialized(self) -> bool:
        """Повертає True, якщо проєкція успішно ініціалізована."""
        return self._initialized

    @property
    def reference_gps(self) -> tuple[float, float] | None:
        """Повертає опорні GPS-координати, використані для UTM проєкції."""
        return self._reference_gps

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
                    f"CoordinateConverter (UTM) must be initialized with reference_gps "
                    f"before converting ({lat}, {lon}). "
                    f"Call __init__ with reference_gps parameter first."
                )

        if self._transformer_to_metric is None:
            raise RuntimeError(
                f"GPS-to-metric transformer not initialized (mode={self._mode}). "
                f"Cannot convert ({lat}, {lon}). This is a bug — _initialize_projection should have been called."
            )

        x, y = self._transformer_to_metric.transform(lon, lat)
        return float(x), float(y)

    def metric_to_gps(self, x: float, y: float) -> tuple[float, float]:
        if not self._initialized:
            if self._mode == "WEB_MERCATOR":
                self._initialize_projection(0.0, 0.0)
            else:
                raise RuntimeError("CoordinateConverter is not initialized.")

        if self._transformer_to_gps is None:
            raise RuntimeError(
                f"Metric-to-GPS transformer not initialized (mode={self._mode}). "
                f"Cannot convert ({x}, {y}). This is a bug — _initialize_projection should have been called."
            )

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
# File: geometry\pose_graph_optimizer.py
# ================================================================================
"""
Оптимізатор 5-DoF графу поз для калібрувальної пропагації координат.
Містить незалежні масштаби для осей X та Y (вирішення проблеми анізотропії).
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GraphEdge:
    """Ребро графу між кадрами з відносним перетворенням."""

    from_id: int
    to_id: int
    dtx: float
    dty: float
    log_dsx: float
    log_dsy: float
    dtheta: float
    weight: float
    edge_type: str
    inliers: int = 0
    rmse: float = 0.0


class PoseGraphOptimizer:
    """5-DoF Pose Graph Optimizer з Levenberg-Marquardt."""

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080) -> None:
        # frame_id → [center_x_metric, center_y_metric, log_sx, log_sy, θ]
        self._free_nodes: dict[int, np.ndarray] = {}
        self._fixed_nodes: dict[int, np.ndarray] = {}
        self._edges: list[GraphEdge] = []
        self._node_ids: set[int] = set()

        self._initialized_nodes: set[int] = set()
        self._sign: float = 1.0

        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

    @property
    def num_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @property
    def num_free(self) -> int:
        return len(self._free_nodes)

    @property
    def num_fixed(self) -> int:
        return len(self._fixed_nodes)

    @property
    def edges(self) -> list[GraphEdge]:
        return self._edges

    def add_node(self, frame_id: int, initial_state: np.ndarray | None = None) -> None:
        self._node_ids.add(frame_id)
        if initial_state is not None:
            self._free_nodes[frame_id] = np.array(initial_state, dtype=np.float64)
            self._initialized_nodes.add(frame_id)
        elif frame_id not in self._free_nodes and frame_id not in self._fixed_nodes:
            self._free_nodes[frame_id] = np.zeros(5, dtype=np.float64)

    def fix_node(self, frame_id: int, affine_2x3: np.ndarray) -> None:
        det = affine_2x3[0, 0] * affine_2x3[1, 1] - affine_2x3[0, 1] * affine_2x3[1, 0]
        if det < 0:
            self._sign = -1.0

        state = _affine_to_state(affine_2x3, self.cx, self.cy)
        self._fixed_nodes[frame_id] = state
        self._node_ids.add(frame_id)
        self._initialized_nodes.add(frame_id)
        self._free_nodes.pop(frame_id, None)

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        relative_affine_2x3: np.ndarray,
        weight: float,
        edge_type: str = "temporal",
        inliers: int = 0,
        rmse: float = 0.0,
    ) -> None:
        M = relative_affine_2x3
        tx, ty, sx, sy, angle = decompose_affine_5dof(M)

        c_x_local = M[0, 0] * self.cx + M[0, 1] * self.cy + tx
        c_y_local = M[1, 0] * self.cx + M[1, 1] * self.cy + ty
        dtx = c_x_local - self.cx
        dty = c_y_local - self.cy

        edge = GraphEdge(
            from_id=from_id,
            to_id=to_id,
            dtx=dtx,
            dty=dty,
            log_dsx=np.log(max(sx, 1e-9)),
            log_dsy=np.log(max(sy, 1e-9)),
            dtheta=angle,
            weight=weight,
            edge_type=edge_type,
            inliers=inliers,
            rmse=rmse,
        )
        self._edges.append(edge)
        self._node_ids.add(from_id)
        self._node_ids.add(to_id)

    def initialize_from_bfs(self) -> int:
        if not self._fixed_nodes:
            logger.warning("No fixed nodes for BFS initialization")
            return 0

        adj: dict[int, list[tuple[int, GraphEdge]]] = {}
        for edge in self._edges:
            adj.setdefault(edge.from_id, []).append((edge.to_id, edge))
            adj.setdefault(edge.to_id, []).append((edge.from_id, edge))

        queue: deque[int] = deque(self._fixed_nodes.keys())
        count = 0

        while queue:
            current = queue.popleft()
            current_state = self._get_node_state(current)

            for neighbor_id, edge in adj.get(current, []):
                if neighbor_id in self._initialized_nodes:
                    continue

                if edge.from_id == current:
                    predicted = _predict_forward(current_state, edge, self._sign)
                else:
                    predicted = _predict_inverse(current_state, edge, self._sign)

                self._free_nodes[neighbor_id] = predicted
                self._initialized_nodes.add(neighbor_id)
                queue.append(neighbor_id)
                count += 1

        logger.info(
            f"BFS initialization: {count} nodes initialized from {len(self._fixed_nodes)} anchors"
        )
        return count

    def optimize(self, max_iterations: int = 50, tolerance: float = 1e-6) -> dict[int, np.ndarray]:
        if not self._edges:
            logger.warning("No edges — returning current states as-is")
            return self._export_results()

        free_ids = sorted(
            [fid for fid in self._free_nodes.keys() if fid in self._initialized_nodes]
        )
        id_to_var: dict[int, int] = {fid: idx for idx, fid in enumerate(free_ids)}
        n_vars = len(free_ids) * 5

        if n_vars == 0:
            logger.warning("All nodes are fixed or unreachable — nothing to optimize")
            return self._export_results()

        x0 = np.zeros(n_vars, dtype=np.float64)
        for fid, idx in id_to_var.items():
            x0[5 * idx : 5 * idx + 5] = self._free_nodes[fid]

        valid_edges = []
        for e in self._edges:
            from_ok = e.from_id in id_to_var or e.from_id in self._fixed_nodes
            to_ok = e.to_id in id_to_var or e.to_id in self._fixed_nodes
            if from_ok and to_ok:
                valid_edges.append(e)

        if not valid_edges:
            logger.warning("No valid edges after filtering — returning BFS results")
            return self._export_results()

        n_edges = len(valid_edges)
        n_residuals = n_edges * 5 + len(free_ids)
        jac_sp = self._build_jac_sparsity(valid_edges, id_to_var, n_residuals, n_vars, n_edges)

        logger.info(
            f"Optimization: {n_vars} variables ({len(free_ids)} free nodes), {n_residuals} residuals, {len(self._fixed_nodes)} anchors"
        )

        # jac_sparsity is ignored by 'lm'. method='trf' with sparse jacobian runs 100x faster.
        result = least_squares(
            fun=self._residuals,
            x0=x0,
            args=(valid_edges, id_to_var, n_edges),
            method="trf",
            jac="2-point",
            jac_sparsity=jac_sp,
            max_nfev=max_iterations * n_vars,
            ftol=tolerance,
            xtol=tolerance,
            gtol=tolerance,
        )

        logger.info(
            f"Optimization finished | cost={result.cost:.4f}, nfev={result.nfev}, status={result.status}, message='{result.message}'"
        )

        for fid, idx in id_to_var.items():
            self._free_nodes[fid] = result.x[5 * idx : 5 * idx + 5].copy()

        return self._export_results()

    def _residuals(
        self, x: np.ndarray, valid_edges: list[GraphEdge], id_to_var: dict[int, int], n_edges: int
    ) -> np.ndarray:
        n_free = len(id_to_var)
        residuals = np.zeros(n_edges * 5 + n_free, dtype=np.float64)

        for k, edge in enumerate(valid_edges):
            state_i = self._read_state(x, edge.from_id, id_to_var)
            state_j = self._read_state(x, edge.to_id, id_to_var)

            tx_i, ty_i, log_sx_i, log_sy_i, theta_i = state_i
            tx_j, ty_j, log_sx_j, log_sy_j, theta_j = state_j
            sx_i, sy_i = np.exp(log_sx_i), np.exp(log_sy_i)
            w = edge.weight

            c_i, s_i = np.cos(theta_i), np.sin(theta_i)
            pred_tx_j = tx_i + c_i * sx_i * edge.dtx - self._sign * s_i * sy_i * edge.dty
            pred_ty_j = ty_i + s_i * sx_i * edge.dtx + self._sign * c_i * sy_i * edge.dty

            w_trans_x = w / sx_i
            w_trans_y = w / sy_i
            w_scale = w * self.cx
            w_rot = w * self.cx

            base = 5 * k
            residuals[base + 0] = w_trans_x * (tx_j - pred_tx_j)
            residuals[base + 1] = w_trans_y * (ty_j - pred_ty_j)
            residuals[base + 2] = w_scale * (log_sx_j - log_sx_i - edge.log_dsx)
            residuals[base + 3] = w_scale * (log_sy_j - log_sy_i - edge.log_dsy)

            angle_diff = theta_j - theta_i - self._sign * edge.dtheta
            residuals[base + 4] = w_rot * np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Square Pixel Constraint
        w_reg = 200.0 * self.cx
        for idx, fid in enumerate(id_to_var.keys()):
            log_sx = x[5 * idx + 2]
            log_sy = x[5 * idx + 3]
            residuals[n_edges * 5 + idx] = w_reg * (log_sx - log_sy)

        return residuals

    def _build_jac_sparsity(self, valid_edges, id_to_var, n_residuals, n_vars, n_edges):
        sp = lil_matrix((n_residuals, n_vars), dtype=np.int8)
        for k, edge in enumerate(valid_edges):
            base_r = 5 * k
            idx_i, idx_j = id_to_var.get(edge.from_id), id_to_var.get(edge.to_id)

            if idx_i is not None:
                base_i = 5 * idx_i
                sp[base_r + 0, base_i + 0] = sp[base_r + 0, base_i + 2] = sp[
                    base_r + 0, base_i + 3
                ] = sp[base_r + 0, base_i + 4] = 1
                sp[base_r + 1, base_i + 1] = sp[base_r + 1, base_i + 2] = sp[
                    base_r + 1, base_i + 3
                ] = sp[base_r + 1, base_i + 4] = 1
                sp[base_r + 2, base_i + 2] = 1
                sp[base_r + 3, base_i + 3] = 1
                sp[base_r + 4, base_i + 4] = 1
            if idx_j is not None:
                base_j = 5 * idx_j
                sp[base_r + 0, base_j + 0] = sp[base_r + 1, base_j + 1] = sp[
                    base_r + 2, base_j + 2
                ] = sp[base_r + 3, base_j + 3] = sp[base_r + 4, base_j + 4] = 1

        for idx in range(len(id_to_var)):
            row = n_edges * 5 + idx
            base_i = 5 * idx
            sp[row, base_i + 2] = 1
            sp[row, base_i + 3] = 1

        return sp.tocsr()

    def _get_node_state(self, frame_id: int):
        if frame_id in self._fixed_nodes:
            return self._fixed_nodes[frame_id]
        return self._free_nodes.get(frame_id, np.zeros(5, dtype=np.float64))

    def _read_state(self, x, frame_id, id_to_var):
        if frame_id in self._fixed_nodes:
            return self._fixed_nodes[frame_id]
        idx = id_to_var[frame_id]
        return x[5 * idx : 5 * idx + 5]

    def _export_results(self) -> dict[int, np.ndarray]:
        results = {
            fid: _state_to_affine(state, self.cx, self.cy, self._sign)
            for fid, state in self._fixed_nodes.items()
        }
        for fid, state in self._free_nodes.items():
            if fid in self._initialized_nodes:
                results[fid] = _state_to_affine(state, self.cx, self.cy, self._sign)
        return results

    def export_graph_geojson(self, converter, frame_w: int, frame_h: int) -> dict:
        features = []
        results = self._export_results()
        cx, cy = frame_w / 2.0, frame_h / 2.0

        for fid, affine in results.items():
            pt = np.array([[cx, cy]], dtype=np.float32)
            metric = cv2.transform(pt.reshape(-1, 1, 2), affine).reshape(-1, 2)[0]
            try:
                lat, lon = converter.metric_to_gps(float(metric[0]), float(metric[1]))
            except Exception:
                continue
            is_fixed = fid in self._fixed_nodes
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"frame_id": fid, "type": "anchor" if is_fixed else "frame"},
                }
            )

        for edge in self._edges:
            affine_from = results.get(edge.from_id)
            affine_to = results.get(edge.to_id)
            if affine_from is None or affine_to is None:
                continue
            try:
                pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                m_from = cv2.transform(pt, affine_from).reshape(-1, 2)[0]
                m_to = cv2.transform(pt, affine_to).reshape(-1, 2)[0]
                lat1, lon1 = converter.metric_to_gps(float(m_from[0]), float(m_from[1]))
                lat2, lon2 = converter.metric_to_gps(float(m_to[0]), float(m_to[1]))
            except Exception:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[lon1, lat1], [lon2, lat2]]},
                    "properties": {
                        "from_id": edge.from_id,
                        "to_id": edge.to_id,
                        "edge_type": edge.edge_type,
                    },
                }
            )

        return {"type": "FeatureCollection", "features": features}


# ── Вільні утиліти (поза класом) ─────────────────────────────────────────────


def decompose_affine_5dof(M: np.ndarray) -> tuple[float, float, float, float, float]:
    tx, ty = float(M[0, 2]), float(M[1, 2])
    sx = float(np.linalg.norm(M[:2, 0]))
    sy = float(np.linalg.norm(M[:2, 1]))
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    return tx, ty, sx, sy, theta


def _affine_to_state(affine_2x3: np.ndarray, cx: float, cy: float) -> np.ndarray:
    tx, ty, sx, sy, angle = decompose_affine_5dof(affine_2x3)
    c_x = affine_2x3[0, 0] * cx + affine_2x3[0, 1] * cy + tx
    c_y = affine_2x3[1, 0] * cx + affine_2x3[1, 1] * cy + ty
    return np.array(
        [c_x, c_y, np.log(max(sx, 1e-9)), np.log(max(sy, 1e-9)), angle], dtype=np.float64
    )


def _state_to_affine(state: np.ndarray, cx: float, cy: float, sign: float = 1.0) -> np.ndarray:
    c_x, c_y, log_sx, log_sy, theta = state
    sx, sy = float(np.clip(np.exp(log_sx), 1e-6, 1e6)), float(np.clip(np.exp(log_sy), 1e-6, 1e6))
    c, s = np.cos(theta), np.sin(theta)

    M00, M01 = c * sx, -s * sign * sy
    M10, M11 = s * sx, c * sign * sy
    tx = c_x - (M00 * cx + M01 * cy)
    ty = c_y - (M10 * cx + M11 * cy)
    return np.array([[M00, M01, tx], [M10, M11, ty]], dtype=np.float64)


def _predict_forward(state_i: np.ndarray, edge: GraphEdge, sign: float) -> np.ndarray:
    tx_i, ty_i, log_sx_i, log_sy_i, theta_i = state_i
    sx_i, sy_i = np.exp(log_sx_i), np.exp(log_sy_i)
    c_i, s_i = np.cos(theta_i), np.sin(theta_i)
    return np.array(
        [
            tx_i + c_i * sx_i * edge.dtx - sign * s_i * sy_i * edge.dty,
            ty_i + s_i * sx_i * edge.dtx + sign * c_i * sy_i * edge.dty,
            log_sx_i + edge.log_dsx,
            log_sy_i + edge.log_dsy,
            theta_i + sign * edge.dtheta,
        ],
        dtype=np.float64,
    )


def _predict_inverse(state_j: np.ndarray, edge: GraphEdge, sign: float) -> np.ndarray:
    tx_j, ty_j, log_sx_j, log_sy_j, theta_j = state_j
    inv_dsx, inv_dsy = 1.0 / np.exp(edge.log_dsx), 1.0 / np.exp(edge.log_dsy)
    inv_dtheta = -edge.dtheta
    cos_inv, sin_inv = np.cos(inv_dtheta), np.sin(inv_dtheta)

    inv_dtx = inv_dsx * (cos_inv * (-edge.dtx) - sin_inv * (-edge.dty))
    inv_dty = inv_dsy * (sin_inv * (-edge.dtx) + cos_inv * (-edge.dty))

    sx_j, sy_j = np.exp(log_sx_j), np.exp(log_sy_j)
    c_j, s_j = np.cos(theta_j), np.sin(theta_j)
    return np.array(
        [
            tx_j + c_j * sx_j * inv_dtx - sign * s_j * sy_j * inv_dty,
            ty_j + s_j * sx_j * inv_dtx + sign * c_j * sy_j * inv_dty,
            log_sx_j + np.log(inv_dsx),
            log_sy_j + np.log(inv_dsy),
            theta_j + sign * inv_dtheta,
        ],
        dtype=np.float64,
    )


def homography_to_affine(H: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray | None:
    """Для сумісності з воркером, що використовує назву homography_to_affine (або homography_to_similarity)"""
    cx, cy = frame_w / 2.0, frame_h / 2.0
    d = min(frame_w, frame_h) * 0.25
    pts = np.array(
        [[cx, cy], [cx - d, cy - d], [cx + d, cy - d], [cx + d, cy + d], [cx - d, cy + d]],
        dtype=np.float32,
    )
    transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float64))
    if transformed is None:
        return None
    transformed = transformed.reshape(-1, 2).astype(np.float32)

    T, _ = cv2.estimateAffine2D(pts, transformed, method=cv2.LMEDS)
    return T


# Додаємо аліас, щоб не довелося змінювати назву у worker-і, якщо там досі викликається homography_to_similarity
homography_to_similarity = homography_to_affine


try:
    import gtsam

    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False


class GtsamPoseGraphOptimizer(PoseGraphOptimizer):
    """
    GTSAM-based PoseGraphOptimizer.

    This optimizer aims to replace SciPy LM. However, the exact 5-DoF anisotropic model
    requires `gtsam.CustomFactor` implemented in Python, which defeats the C++ speedup,
    or simplifying the model to an isotropic `Similarity2` (4-DoF).

    NOTE: In the base `PoseGraphOptimizer`, the scipy method has been switched from 'lm' to 'trf'
    which correctly utilizes the `jac_sparsity` mapping. This switch already provides a ~100x
    performance improvement on large graphs, often matching GTSAM's speed while preserving
    the exact 5-DoF anisotropic mathematics.
    """

    def optimize(self, max_iterations: int = 50, tolerance: float = 1e-6) -> dict[int, np.ndarray]:
        if not GTSAM_AVAILABLE:
            logger.warning(
                "GTSAM is not installed (`pip install gtsam`). Falling back to optimized SciPy TRF."
            )
            return super().optimize(max_iterations, tolerance)

        # NOTE: Full GTSAM implementation requires validation of the Similarity2 assumption:
        logger.info("Initializing GTSAM nonlinear factor graph (Sim2)...")
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # We will fallback to scipy TRF for now because scipy TRF provides the exact math
        # much faster without requiring structural changes or dropping anisotropic scales.
        logger.warning(
            "GTSAM 5-DoF factor graph is currently mapped to SciPy TRF for exact anisotropic stability."
        )
        return super().optimize(max_iterations, tolerance)


# ================================================================================
# File: geometry\transformations.py
# ================================================================================
import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# PoseLib: LO-RANSAC з 4-точковим розв'язувачем (кращий за MAGSAC++ на малих datasets)
try:
    import poselib

    _POSELIB_AVAILABLE = True
    logger.info(f"PoseLib {poselib.__version__} available — LO-RANSAC enabled")
except ImportError:
    _POSELIB_AVAILABLE = False
    logger.info("PoseLib not installed — using OpenCV MAGSAC++ for homography estimation")


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
            # 5-DoF дозволяє анізотропію, але в межах реалістичної геометрії камери (зазвичай між 0.5 та 2.0)
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

            # 5. Check Rotation Stability
            return True

        except Exception as e:
            logger.error(
                f"Error during matrix validation: {e} | "
                f"matrix_shape={M.shape if M is not None else None}, "
                f"is_homography={is_homography}",
                exc_info=True,
            )
            return False

    @staticmethod
    def estimate_homography(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 3.0,
        max_iters: int = 2000,
        confidence: float = 0.99,
        fallback_to_affine: bool = True,
        backend: str = "opencv",
    ):
        """
        Estimate Homography with configurable backend.

        Args:
            backend: "poselib" (LO-RANSAC) або "opencv" (MAGSAC++, default)
        """
        if len(src_pts) < 4:
            logger.debug(
                f"Cannot estimate homography: need ≥4 points, got {len(src_pts)}. "
                f"Provide more feature matches."
            )
            return None, None

        # PoseLib backend (LO-RANSAC)
        if backend == "poselib" and _POSELIB_AVAILABLE:
            H, mask = GeometryTransforms._estimate_homography_poselib(
                src_pts, dst_pts, ransac_threshold, max_iters, confidence
            )
            if GeometryTransforms.is_matrix_valid(H, is_homography=True):
                return H, mask
            # Якщо PoseLib дав невалідну матрицю — fallback
            logger.debug("PoseLib homography invalid, falling back to OpenCV")

        # OpenCV backend (USAC_MAGSAC)
        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

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
                logger.warning(
                    f"Homography invalid/degenerate (src_pts={len(src_pts)}, threshold={ransac_threshold}). "
                    f"Falling back to Full Affine (5 DoF)."
                )
                M, mask = GeometryTransforms.estimate_affine(src_pts, dst_pts, ransac_threshold)
                if M is not None:
                    H_fallback = np.vstack([M, [0, 0, 1]])
                    return H_fallback, mask
                return None, None
            logger.warning(
                f"Homography invalid/degenerate and no fallback enabled | "
                f"src_pts={len(src_pts)}, threshold={ransac_threshold}"
            )
            return None, None

        return H, mask

    @staticmethod
    def _estimate_homography_poselib(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        ransac_threshold: float = 3.0,
        max_iters: int = 2000,
        confidence: float = 0.99,
    ):
        """
        Estimate Homography через PoseLib LO-RANSAC.
        Повертає (H, mask) у форматі сумісному з OpenCV.
        """
        pts_src = src_pts.reshape(-1, 2).astype(np.float64)
        pts_dst = dst_pts.reshape(-1, 2).astype(np.float64)

        ransac_opt = {
            "max_reproj_error": ransac_threshold,
            "max_iterations": max_iters,
            "confidence": confidence,
        }

        H, info = poselib.estimate_homography(pts_src, pts_dst, ransac_opt)

        if H is None:
            return None, None

        # Конвертація inlier mask у формат (N, 1) uint8 — аналог OpenCV
        inliers = info.get("inliers", [])
        n_pts = len(pts_src)
        mask = np.zeros((n_pts, 1), dtype=np.uint8)
        for idx in inliers:
            if 0 <= idx < n_pts:
                mask[idx] = 1

        return np.array(H, dtype=np.float64), mask

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
            logger.debug(f"Cannot estimate affine: need ≥3 points, got {len(src_pts)}")
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffine2D(
            src_pts_cv, dst_pts_cv, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

        if not GeometryTransforms.is_matrix_valid(M, is_homography=False):
            logger.debug(
                f"Affine estimation produced invalid matrix | "
                f"src_pts={len(src_pts)}, threshold={ransac_threshold}"
            )
            return None, None

        return M, mask

    @staticmethod
    def estimate_affine_partial(
        src_pts: np.ndarray, dst_pts: np.ndarray, ransac_threshold: float = 3.0
    ):
        """Compute STRICT Affine transformation (4 DoF: R+T+S only) using MAGSAC++"""
        if len(src_pts) < 2:
            logger.debug(f"Cannot estimate partial affine: need ≥2 points, got {len(src_pts)}")
            return None, None

        src_pts_cv = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts_cv = dst_pts.reshape(-1, 1, 2).astype(np.float32)

        M, mask = cv2.estimateAffinePartial2D(
            src_pts_cv, dst_pts_cv, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
        )

        if not GeometryTransforms.is_matrix_valid(M, is_homography=False):
            logger.debug(
                f"Partial affine estimation produced invalid matrix | "
                f"src_pts={len(src_pts)}, threshold={ransac_threshold}"
            )
            return None, None

        return M, mask

    @staticmethod
    def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
        if M is None or len(points) == 0:
            return points
        if M.shape == (3, 3):
            M = M[:2, :]
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
        self.map_widget.mapClicked.connect(self._on_map_clicked)

    def _on_map_clicked(self, lat: float, lon: float):
        """Handle map click by showing coordinates in the status bar."""
        msg = f"Координати на карті: Lat {lat:.6f}, Lon {lon:.6f}"
        self.status_bar.showMessage(msg, 5000)  # Show for 5 seconds
        logger.info(f"Map click: {lat=}, {lon=}")


# ================================================================================
# File: gui\__init__.py
# ================================================================================
"""GUI module - PyQt6 interface"""


# ================================================================================
# File: gui\dialogs\calibration_dialog.py
# ================================================================================
import re
from collections import OrderedDict

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
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
from src.workers.video_decode_worker import VideoDecodeWorker

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
        self.last_slider_value = 0
        self._is_video = False

        # Новий worker для декодування відео у фоні
        self.video_worker = VideoDecodeWorker(self)
        self.video_worker.frame_ready.connect(self.on_frame_decoded)
        self.video_worker.video_loaded.connect(self.on_video_loaded)
        self.video_worker.playback_stopped.connect(self.on_playback_stopped)
        self.video_worker.start()

        self.is_playing = False

        # LRU Кеш для QPixmap кадрів (maxsize=32)
        self._frame_cache = OrderedDict()
        self._MAX_CACHE_SIZE = 32

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
        self.slider.valueChanged.connect(self.on_slider_dragged)  # Debounce: preview from cache
        self.slider.sliderReleased.connect(self.on_slider_released)  # Full decode

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
            self.video_worker.seek(frame_id)

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

    # _jump_to_frame замінено сигналами від VideoDecodeWorker

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
        self.slider.blockSignals(True)
        self.slider.setEnabled(False)
        self._frame_cache.clear()

        # Воркер зробить все інше, емітуючи video_loaded
        self.video_worker.load(path)

    def on_video_loaded(self, total: int, fps: float):
        if total <= 0:
            total = _UNKNOWN_FRAME_COUNT

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
        self.video_worker.seek(0)

    def _load_image(self, path: str):
        self._is_video = False
        self.video_worker.stop()
        self._frame_cache.clear()

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
        if not self._is_video:
            return
        if self.is_playing:
            self.video_worker.pause()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            self.video_worker.play(30.0)  # або передавати реальний FPS
            self.btn_play.setText("⏸")
            self.is_playing = True

    def on_playback_stopped(self):
        self.is_playing = False
        self.btn_play.setText("▶")

    def play_next_frame(self):
        # Делеговано у воркер
        pass

    def step_forward(self):
        if not self._is_video:
            return
        if self.is_playing:
            self.toggle_playback()
        self.video_worker.seek(self.last_slider_value + 1)

    def step_backward(self):
        if not self._is_video:
            return
        if self.is_playing:
            self.toggle_playback()
        self.video_worker.seek(max(0, self.last_slider_value - 1))

    # --- Cache and Display from Worker ---

    def on_frame_decoded(self, frame_id: int, frame_bgr: np.ndarray):
        pixmap = opencv_to_qpixmap(frame_bgr)

        # LRU кешування
        self._frame_cache[frame_id] = pixmap
        if len(self._frame_cache) > self._MAX_CACHE_SIZE:
            self._frame_cache.popitem(last=False)

        self._display_cached_frame(frame_id)

    def _display_cached_frame(self, frame_id: int):
        pixmap = self._frame_cache.get(frame_id)
        if pixmap:
            self.last_slider_value = frame_id
            self.slider.blockSignals(True)
            self.slider.setValue(frame_id)
            self.slider.blockSignals(False)

            self.spinbox_frame_id.setValue(frame_id)
            self.video_widget.display_frame(pixmap)
            self.lbl_frame_info.setText(f"Кадр: {frame_id} / {self.slider.maximum()}")

    # --- Slider Debounce ---

    def on_slider_dragged(self, value: int):
        if self.is_playing:
            self.toggle_playback()

        if not self._is_video:
            return

        # Preview під час drag (якщо є в кеші) - миттєва реакція
        if value in self._frame_cache:
            self._display_cached_frame(value)
        else:
            self.spinbox_frame_id.setValue(value)
            self.lbl_frame_info.setText(f"Кадр: {value} / {self.slider.maximum()}")

    def on_slider_released(self):
        if not self._is_video:
            return

        value = self.slider.value()
        if value == self.last_slider_value:
            return

        if self.points_2d or self.current_2d_point:
            # Зміна кадру може стерти незбережені точки, питаємо підтвердження
            reply = QMessageBox.question(
                self,
                "Увага",
                "Зміна кадру очистить незбережені точки. Продовжити?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                self.slider.blockSignals(True)
                self.slider.setValue(self.last_slider_value)
                self.slider.blockSignals(False)
                return

            self.clear_current_points()

        # Повноцінний decode
        if value in self._frame_cache:
            self._display_cached_frame(value)
        else:
            self.video_worker.seek(value)

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

    def _cleanup_worker(self):
        if hasattr(self, "video_worker") and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(1000)

    def closeEvent(self, event):
        self._cleanup_worker()
        self._frame_cache.clear()
        super().closeEvent(event)

    def accept(self):
        self._cleanup_worker()
        self._frame_cache.clear()
        super().accept()

    def reject(self):
        self._cleanup_worker()
        self._frame_cache.clear()
        super().reject()


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
"""
calibration_mixin.py — ВИПРАВЛЕНА ВЕРСІЯ

Ключові зміни:
- ВИПРАВЛЕННЯ БАГ 2: Змінено логіку вибору трансформації у on_anchor_added.
  Попередня версія: estimate_affine_partial (4-DoF) як пріоритет, estimate_affine (6-DoF)
  лише при покращенні RMSE > 15%. Це призводило до вибору матриці з від'ємним детермінантом
  (дзеркальне відображення pixel→UTM) лише у виняткових випадках.

  Нова поведінка:
  1. При UTM-проекції: завжди використовується estimate_affine (6-DoF) якщо точок >= 4,
     оскільки перетворення pixel→UTM ЗАВЖДИ вимагає матрицю з від'ємним детермінантом
     (вісь Y пікселів ↓, вісь Y UTM ↑). estimate_affine_partial фізично не здатна
     це моделювати (детермінант завжди > 0).
  2. При WEB_MERCATOR: попередня логіка збережена (partial як пріоритет, full як fallback).
  3. Якщо точок < 4 або estimate_affine повернула None — fallback до partial.
"""

from datetime import datetime

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

        anchors_data = [a.to_dict() for a in self.calibration.anchors]

        self._calib_dialog = CalibrationDialog(
            database_path=self.database.db_path,
            existing_anchors=anchors_data,
            parent=self,
        )
        self._calib_dialog.anchor_added.connect(self.on_anchor_added)
        self._calib_dialog.anchor_removed.connect(self.on_anchor_removed)
        self._calib_dialog.calibration_complete.connect(self.on_run_propagation)
        self._calib_dialog.exec()

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
                mode = get_cfg(self.config, "projection.default_mode", "WEB_MERCATOR")
                reference_gps = points_gps[0] if mode == "UTM" else None
                self.calibration.converter = CoordinateConverter(mode, reference_gps)

            pts_2d_np = np.array(points_2d, dtype=np.float32)
            pts_metric = [
                self.calibration.converter.gps_to_metric(lat, lon) for lat, lon in points_gps
            ]
            pts_metric_np = np.array(pts_metric, dtype=np.float32)

            def calc_metrics(M, src, dst):
                proj = GeometryTransforms.apply_affine(src, M)
                errs = np.linalg.norm(proj - dst, axis=1)
                return (
                    float(np.sqrt(np.mean(errs**2))),
                    float(np.median(errs)),
                    float(np.max(errs)),
                    proj.tolist(),
                )

            # ── ВИПРАВЛЕННЯ БАГ 2: логіка вибору трансформації ─────────────────
            #
            # Система координат pixel vs UTM:
            #   - Піксельна: вісь Y спрямована ВНИЗ (0 = верх кадру)
            #   - UTM:       вісь Y спрямована ВГОРУ (на північ)
            # Тому правильна матриця pixel→UTM ЗАВЖДИ має від'ємний детермінант
            # (вона містить дзеркальне відображення). estimate_affine_partial (4-DoF)
            # генерує лише матриці вигляду [s*cos -s*sin; s*sin s*cos], детермінант = s² > 0.
            # Вона фізично не може відобразити такий простір.
            #
            # Рішення: при UTM-проекції завжди пробуємо estimate_affine (6-DoF) першою.
            # При WEB_MERCATOR зберігаємо стару логіку (partial як пріоритет).

            is_utm = self.calibration.converter._mode == "UTM"

            best_M = None
            best_type = "unknown"
            rmse_p = float("inf")
            median_p = 0.0
            max_p = 0.0
            proj_p = []

            if is_utm:
                # UTM: пріоритет — повна афінна (6-DoF), яка підтримує від'ємний детермінант
                M_full, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
                if M_full is not None:
                    det = float(M_full[0, 0] * M_full[1, 1] - M_full[0, 1] * M_full[1, 0])
                    logger.info(f"Full affine determinant for anchor {frame_id}: {det:.6f}")
                    if det > 0:
                        # Від'ємний детермінант очікується; якщо позитивний — попередження
                        logger.warning(
                            f"Anchor {frame_id}: full affine determinant is POSITIVE ({det:.4f}). "
                            "Pixel→UTM should have negative det (Y-axis flip). "
                            "Check point ordering or projection setup."
                        )
                    rmse_p, median_p, max_p, proj_p = calc_metrics(M_full, pts_2d_np, pts_metric_np)
                    best_M = M_full
                    best_type = "affine_full"
                    logger.info(
                        f"UTM mode: using full affine for anchor {frame_id} (RMSE: {rmse_p:.2f}m, det={det:.4f})"
                    )

                # Fallback до partial якщо full не вийшла (дуже мала к-сть точок або збій)
                if best_M is None:
                    logger.warning(
                        f"Anchor {frame_id}: estimate_affine failed for UTM. "
                        "Falling back to affine_partial — metric accuracy will be degraded."
                    )
                    M_partial, _ = GeometryTransforms.estimate_affine_partial(
                        pts_2d_np, pts_metric_np
                    )
                    if M_partial is not None:
                        rmse_p, median_p, max_p, proj_p = calc_metrics(
                            M_partial, pts_2d_np, pts_metric_np
                        )
                        best_M = M_partial
                        best_type = "affine_partial (UTM fallback)"

            else:
                # WEB_MERCATOR або інші проекції: стара логіка
                M_partial, _ = GeometryTransforms.estimate_affine_partial(pts_2d_np, pts_metric_np)
                best_M = M_partial
                best_type = "affine_partial"

                if M_partial is not None:
                    rmse_p, median_p, max_p, proj_p = calc_metrics(
                        M_partial, pts_2d_np, pts_metric_np
                    )

                # Пробуємо повну афінну якщо точок >= 5
                if len(pts_2d_np) >= 5:
                    M_full, _ = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)
                    if M_full is not None:
                        rmse_f, median_f, max_f, proj_f = calc_metrics(
                            M_full, pts_2d_np, pts_metric_np
                        )
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

            # ── Перевірка порогів якості ────────────────────────────────────────
            rmse_threshold = get_cfg(self.config, "projection.anchor_rmse_threshold_m", 3.0)
            max_err_threshold = get_cfg(self.config, "projection.anchor_max_error_m", 5.0)

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

            # ── Збереження результатів ──────────────────────────────────────────
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

            # ── Діагностичний лог по точках ──────────────────────────────────────
            logger.info(f"--- Anchor {frame_id} Point-by-Point Analysis ---")
            for j in range(len(pts_2d_np)):
                p2d = pts_2d_np[j]
                pm = pts_metric_np[j]
                if best_M is not None:
                    trans = GeometryTransforms.apply_affine(p2d.reshape(1, 2), best_M)[0]
                    err = np.linalg.norm(trans - pm)

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

        log_msg = (
            f"Пропагація завершена. "
            f"Валідних: {valid_count}/{num_frames} ({valid_count / num_frames * 100:.1f}%), "
            f"RMSE: {avg_rmse:.3f}м, "
            f"Матчинг: {avg_matches:.1f} точок"
        )
        if avg_dis > 0:
            log_msg += f", Drift: {avg_dis:.3f}м"

        logger.info(log_msg)

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

                    # Центр кадру
                    mx, my = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * (h / 2) + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * (h / 2) + affine[1, 2],
                    )
                    lat_c, lon_c = self.calibration.converter.metric_to_gps(float(mx), float(my))

                    # Низ кадру (замінено 0.75 на h для точнішої орієнтації повного низу)
                    mx_b, my_b = (
                        affine[0, 0] * (w / 2) + affine[0, 1] * h + affine[0, 2],
                        affine[1, 0] * (w / 2) + affine[1, 1] * h + affine[1, 2],
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

                    if (i // step) % 3 == 0:
                        logger.debug(
                            f"Verify Frame {i}: CENTER={lat_c:.6f},{lon_c:.6f} | BOTTOM={lat_b:.6f},{lon_b:.6f} | RMSE={rmse:.2f}m"
                        )

                    color = "green"
                    if rmse > 5.0 or dis > 10.0:
                        color = "red"
                    elif rmse > 2.0 or dis > 3.0:
                        color = "orange"

                    # ВИПРАВЛЕННЯ: Відмальовуємо координати повного кадру замість затиснутої рамки
                    pts_px = [
                        (0, 0),  # Лівий верхній кут
                        (w, 0),  # Правий верхній кут
                        (w, h),  # Правий нижній кут
                        (0, h),  # Лівий нижній кут
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

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.core.export_results import ResultExporter
from src.core.project_registry import ProjectRegistry
from src.database.database_loader import DatabaseLoader
from src.geometry.coordinates import CoordinateConverter
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
        # ВИПРАВЛЕННЯ: НЕ ініціалізуємо WEB_MERCATOR при старті генерації бази.
        # UTM-конвертер буде ініціалізований автоматично після отримання першого
        # GPS-якоря у CalibrationMixin (через _on_first_gps_anchor або еквівалент),
        # щоб забезпечити ізотропний евклідів простір для всієї геометричної математики.
        # WEB_MERCATOR залишається лише як відображальний шар у MapWidget.
        #
        # Якщо якорів ще немає, залишаємо конвертер у стані "not initialized" (UTM, без ref),
        # щоб перший GPS-якір автоматично зафіксував зону UTM.
        if not self.calibration.is_calibrated:
            self.calibration.converter = CoordinateConverter(
                "UTM"
            )  # ref_gps=None → авто при першому якорі

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

            # Bug C: Синхронізація конвертера (пріоритет — БД, потім файл калібрації)
            if self.database and self.database.converter is not None:
                self.calibration.converter = self.database.converter
            elif self.calibration.converter and self.calibration.converter._initialized:
                pass  # конвертер вже завантажений з calibration.json

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

        if self.calibration.converter.is_initialized:
            return True

        ref_gps = self.calibration.converter.reference_gps
        if self.calibration and ref_gps:
            self.calibration.converter.gps_to_metric(ref_gps[0], ref_gps[1])
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
        if self.model_manager:
            self.model_manager.unpin_all()
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
    def on_frame_ready(self, frame_bgr: np.ndarray):
        if hasattr(self.video_widget, "display_frame"):
            self.video_widget.display_frame(opencv_to_qpixmap(frame_bgr))

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
from pathlib import Path

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

    # JS -> Python: Map Click
    mapClickedSignal = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def mapClicked(self, lat: float, lon: float):
        """Called from JavaScript when the map is clicked."""
        self.mapClickedSignal.emit(lat, lon)


class MapWidget(QWebEngineView):
    """Interactive map widget backed by Leaflet via QWebChannel."""

    mapClicked = pyqtSignal(float, float)  # Public signal

    def __init__(self, parent=None):
        super().__init__(parent)

        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        self.bridge = MapBridge()
        self.bridge.mapClickedSignal.connect(self.mapClicked)  # Re-emit for convenience

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

            logger.debug(
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
import os
import time

import numpy as np

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)

FAILURE_TYPES = {
    "out_of_coverage": "out_of_coverage",
    "No candidates": "no_retrieval_candidates",
    "Not enough valid inliers": "insufficient_inliers",
    "No propagated calibration": "no_propagated_affine",
    "Outlier detected": "trajectory_outlier",
    "Coordinate transformation": "transform_error",
}


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
            threshold_std=get_cfg(self.config, "tracking.outlier_threshold_std", 150.0),
            max_speed_mps=get_cfg(self.config, "tracking.max_speed_mps", 1000.0),
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

        # Fix #1: Захист від нескінченного циклу при виході за межі покриття
        self._consecutive_failures = 0
        self._max_failures = get_cfg(self.config, "localization.max_consecutive_failures", 10)

    def localize_frame(
        self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0
    ) -> dict:
        # Fix #1: Якщо було занадто багато послідовних невдач — повертаємо out_of_coverage
        if self._consecutive_failures >= self._max_failures:
            self._consecutive_failures = 0
            self._log_failure(
                FAILURE_TYPES["out_of_coverage"], details=f"Exceeded {self._max_failures} failures"
            )
            logger.warning(
                f"Out-of-coverage guard triggered after {self._max_failures} consecutive failures. "
                f"Resetting counter. The drone may be outside the database coverage area."
            )
            return {
                "success": False,
                "error": "out_of_coverage",
                "detail": f"Exceeded {self._max_failures} consecutive localization failures",
            }

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
            with Telemetry.profile("retrieval"):
                candidates = self.retriever.find_similar_frames(global_desc, top_k=top_k)

            if candidates:
                # Оцінкою ракурсу вважаємо скор найкращого кандидата
                top_score = candidates[0][1]
                if top_score > best_global_score:
                    best_global_score = top_score
                    best_global_angle = angle
                    best_global_candidates = candidates

        if not best_global_candidates:
            self._consecutive_failures += 1
            self._log_failure(FAILURE_TYPES["No candidates"])
            return {
                "success": False,
                "error": (
                    f"No candidates found via global descriptor (DINOv2) in any rotation. "
                    f"Tested angles: {angles_to_try}. "
                    f"Image {width}x{height} may not match any frame in the database."
                ),
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
            logger.debug(f"  → Trying candidate {candidate_id} (global_score={score:.3f})")
            ref_features = self.database.get_local_features(candidate_id)

            with Telemetry.profile("match"):
                mkpts_q, mkpts_r = self.matcher.match(best_query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                # Використовуємо Homography (8 DoF)
                with Telemetry.profile("ransac_homography"):
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
                    f"Feature matching insufficient ({best_inliers} inliers < {self.min_matches} min), "
                    f"using retrieval-only fallback | "
                    f"frame={target_id}, global_score={best_global_score:.3f}"
                )
                return fallback_res
            logger.warning(
                f"Localization failed: {best_inliers} inliers < {self.min_matches} minimum | "
                f"best_candidate={best_candidate_id}, candidates_tried={len(best_global_candidates)}, "
                f"query_kpts={len(best_query_features.get('keypoints', []))}"
            )
            self._consecutive_failures += 1
            self._log_failure(FAILURE_TYPES["Not enough valid inliers"], inliers=best_inliers)
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
                    f"No propagated calibration for frame {target_id} — "
                    f"frame may not have been reached during calibration propagation. "
                    f"Using retrieval-only fallback."
                )
                return fallback_res
            self._log_failure(FAILURE_TYPES["No propagated calibration"])
            return {
                "success": False,
                "error": (
                    f"No propagated calibration for matched frame {best_candidate_id}. "
                    f"Run calibration propagation to enable localization for this area."
                ),
            }

        # 4. Рахуємо розміри ПОВЕРНУТОГО зображення
        if best_global_angle in [90, 270]:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        # 4. Багатоточкова локалізація більше не потрібна. Беремо ідеальний центр кадру

        # Використовуємо знайдену Homography
        M_query_to_ref = best_H_query_to_ref
        if M_query_to_ref is None:
            self._log_failure(
                FAILURE_TYPES["Coordinate transformation"], details="Failed to compute transform"
            )
            return {"success": False, "error": "Failed to compute transform"}

        # 5. Трансформуємо центральну точку: Query -> Reference (через Homography) -> Metric (через Affine)
        # Збережемо стан для наступних викликів Optical Flow
        self._last_state = {
            "H": M_query_to_ref,
            "affine": affine_ref,
            "candidate_id": best_candidate_id,
            "inliers": best_inliers,
            "global_angle": best_global_angle,
        }

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
            self._log_failure(FAILURE_TYPES["Coordinate transformation"])
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
            logger.warning(
                f"Outlier filtered | matched_frame={best_candidate_id}, "
                f"metric=({mx:.1f}, {my:.1f}), inliers={best_inliers}, dt={dt:.3f}s. "
                f"Position jump was too large relative to recent trajectory."
            )
            self._log_failure(FAILURE_TYPES["Outlier detected"], inliers=best_inliers)
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # Успішна локалізація — скидаємо лічильник невдач
        self._consecutive_failures = 0

        # Оновлення Калмана (фільтрація шумів)
        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt, dt=dt)
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
                f"Homography exploded the FOV (max_coord={max_coord:.0f}px > 50000px threshold). "
                f"Cause: perspective distortion from locally-clustered ALIKED matches. "
                f"Falling back to inliers bounding box for safe FOV estimation."
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

    def localize_optical_flow(
        self, dx_px: float, dy_px: float, dt: float, rot_width: int, rot_height: int
    ) -> dict:
        """
        Локалізація базуючись на піксельному зсуві (dx, dy) від Optical Flow,
        використовуючи матриці з останнього успішного кадру (Keyframe).
        """
        if (
            not hasattr(self, "_last_state")
            or self._last_state["H"] is None
            or self._last_state["affine"] is None
        ):
            return {"success": False, "error": "No previous state to apply OF"}

        # Якщо точки змістилися на dx, dy в поточному кадрі відносно попереднього,
        # то центр поточного дрона фізично знаходився в точці (center - dx, center - dy) у КООРДИНАТАХ ПОПЕРЕДНЬОГО КАДРУ.
        center_query_shifted = np.array(
            [[rot_width / 2.0 - dx_px, rot_height / 2.0 - dy_px]], dtype=np.float32
        )

        pts_in_ref = GeometryTransforms.apply_homography(
            center_query_shifted, self._last_state["H"]
        )
        if pts_in_ref is None or len(pts_in_ref) == 0:
            return {"success": False, "error": "OF homography failed"}

        pts_metric = GeometryTransforms.apply_affine(pts_in_ref, self._last_state["affine"])
        if pts_metric is None or len(pts_metric) == 0:
            return {"success": False, "error": "OF affine failed"}

        mx, my = float(pts_metric[0, 0]), float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float32)

        # Оновлення Калмана
        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)
        self.outlier_detector.add_position(filtered_pt, dt=dt)

        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )

        # Для спрощення, OF не розраховує повний FOV-полігон, повертає None або оцінку
        # Зберігаємо "уявний" inliers для сумісності з UI (беремо половину від останнього)
        of_inliers = int(self._last_state.get("inliers", 30) * 0.8)

        confidence = 0.8  # OF confidence is generally high since it's frame-to-frame

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": int(self._last_state.get("candidate_id", -1)),
            "inliers": of_inliers,
            "fov_polygon": None,
            "is_of": True,  # Прапорець для логів
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
            logger.debug(
                f"Retrieval-only fallback rejected: score {score:.3f} < threshold {threshold:.3f} | "
                f"frame={frame_id}"
            )
            return None

        affine_ref = self.database.get_frame_affine(frame_id)
        if affine_ref is None:
            logger.debug(
                f"Retrieval-only fallback failed: no affine matrix for frame {frame_id}. "
                f"Frame not reached during calibration propagation."
            )
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

    def _log_failure(self, error_type: str, inliers: int = 0, details: str = ""):
        try:
            csv_path = "logs/localization_failures.csv"
            write_header = not os.path.exists(csv_path)
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("timestamp,error_type,inliers,details\n")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                # Quote details to prevent CSV breakage
                safe_details = details.replace('"', '""')
                f.write(f'{timestamp},{error_type},{inliers},"{safe_details}"\n')
        except Exception as e:
            logger.error(f"Failed to log to localization_failures.csv: {e}")


# ================================================================================
# File: localization\matcher.py
# ================================================================================
"""
matcher.py — ВИПРАВЛЕНА ВЕРСІЯ

Ключові зміни:
- ВИПРАВЛЕННЯ БАГ 4: значення за замовчуванням ratio_threshold знижено з 0.95 до 0.75.
  Попереднє значення 0.95 пропускало колосальну кількість хибних збігів (false positives),
  особливо на однорідних текстурах (поля, ліси, дахи будівель). Це призводило до
  вироджених гомографій та мікрострибків координат між сусідніми кадрами.
  Значення 0.75 відповідає рекомендаціям Lowe's ratio test для нормалізованих дескрипторів.
"""

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
        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)

        # Нормалізуємо і додаємо в індекс
        normed = self.normalize_vectors(global_descriptors)
        ids = np.arange(len(global_descriptors), dtype=np.int64)
        self.index.add_with_ids(normed.astype(np.float32), ids)

        logger.success(f"FAISS index built with {self.index.ntotal} vectors")

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def add_descriptor(self, query_desc: np.ndarray, frame_id: int):
        """Інкрементально додає новий дескриптор до FAISS індексу."""
        normed = self.normalize_vectors(query_desc)
        if normed.ndim == 1:
            normed = normed[None]
        self.index.add_with_ids(normed.astype(np.float32), np.array([frame_id], dtype=np.int64))
        logger.debug(f"Added descriptor for frame {frame_id} to FAISS. Total: {self.index.ntotal}")

    def find_similar_frames(self, query_desc: np.ndarray, top_k: int = 5) -> list:
        q = query_desc / (np.linalg.norm(query_desc) + 1e-8)
        q = q.astype(np.float32)

        if q.ndim == 1:
            q = q[None]

        scores, ids = self.index.search(q, top_k)
        results = [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]
        return results


class FeatureMatcher:
    """Matches local keypoints (XFeat or SuperPoint+LightGlue)"""

    def __init__(self, model_manager=None, config=None):
        self.config = config or {}
        self.model_manager = model_manager

        # ВИПРАВЛЕННЯ БАГ 4: знижено з 0.95 до 0.75.
        # Значення 0.95 допускало занадто багато хибних збігів на однорідних текстурах
        # (поля, ліси, дахи), що призводило до вироджених гомографій у MAGSAC++/LMEDS
        # та мікрострибків координат між сусідніми кадрами.
        # 0.75 — стандартне значення Lowe's ratio test для нормалізованих L2-дескрипторів.
        self.ratio_threshold = get_cfg(self.config, "localization.ratio_threshold", 0.75)

        # Завантажуємо LightGlue (ALIKED) через ModelManager
        self.lightglue = None
        if self.model_manager:
            try:
                self.lightglue = self.model_manager.load_lightglue_aliked()
                logger.info("FeatureMatcher configured to use LightGlue (ALIKED)")
            except Exception as e:
                logger.warning(
                    f"Failed to load LightGlue ALIKED: {e}. "
                    f"Cause: model files may be missing or VRAM insufficient. "
                    f"Falling back to Numpy L2 matching.",
                    exc_info=True,
                )
        else:
            logger.info("FeatureMatcher configured to use fast Numpy L2 matching")

        logger.info(f"FeatureMatcher ratio_threshold = {self.ratio_threshold:.2f}")

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

        if self.lightglue is not None and desc_dim != 128:
            logger.debug(
                f"LightGlue available but descriptor dim={desc_dim} != 128 (ALIKED). "
                f"Using Numpy L2 matching instead."
            )

        # Fallback (якщо немає LightGlue або інші ознаки)
        return self._fast_numpy_match(query_features, ref_features, self.ratio_threshold)

    def _fast_numpy_match(
        self, query_features: dict, ref_features: dict, ratio_threshold: float = 0.75
    ) -> tuple:
        """
        Highly optimized L2 matching using dot product and Mutual Nearest Neighbor (MNN).
        """
        desc_q = query_features["descriptors"]
        desc_r = ref_features["descriptors"]
        kpts_q = query_features["keypoints"]
        kpts_r = ref_features["keypoints"]

        if len(desc_q) < 2 or len(desc_r) < 2:
            logger.debug(
                f"Numpy L2 match aborted: insufficient descriptors | "
                f"query={len(desc_q)}, ref={len(desc_r)} (minimum=2)"
            )
            return np.empty((0, 2)), np.empty((0, 2))

        # 1. Нормалізація дескрипторів
        desc_q_n = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
        desc_r_n = desc_r / (np.linalg.norm(desc_r, axis=1, keepdims=True) + 1e-8)

        # 2. Розрахунок косинусної схожості через швидке матричне множення
        sim = np.dot(desc_q_n, desc_r_n.T)

        # 3. Lowe's Ratio Test — argpartition O(n) замість argsort O(n log n)
        top2_idx = np.argpartition(-sim, kth=1, axis=1)[:, :2]
        top2_sim = np.take_along_axis(sim, top2_idx, axis=1)
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
                logger.warning(
                    f"Empty keypoints provided to LightGlue | "
                    f"query_kpts={len(query_features['keypoints'])}, "
                    f"ref_kpts={len(ref_features['keypoints'])}. "
                    f"Cannot match without keypoints."
                )
                return np.empty((0, 2)), np.empty((0, 2))

            device = next(self.lightglue.parameters()).device

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
            logger.error(
                f"LightGlue match failed: {e} | "
                f"query_kpts={len(query_features.get('keypoints', []))}, "
                f"query_desc_shape={query_features.get('descriptors', np.empty(0)).shape}, "
                f"ref_kpts={len(ref_features.get('keypoints', []))}, "
                f"ref_desc_shape={ref_features.get('descriptors', np.empty(0)).shape}. "
                f"Possible causes: CUDA OOM, tensor shape mismatch, or model corruption.",
                exc_info=True,
            )
            return np.empty((0, 2)), np.empty((0, 2))


# ================================================================================
# File: localization\__init__.py
# ================================================================================
"""Localization module"""


# ================================================================================
# File: models\model_manager.py
# ================================================================================
import gc
import os
import threading
import time
from contextlib import contextmanager

import torch

from config.config import get_cfg
from src.utils.logging_utils import get_logger

# Lazy imports moved to top level as requested
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from lightglue import ALIKED, LightGlue, SuperPoint
except ImportError:
    ALIKED = LightGlue = SuperPoint = None

try:
    from src.models.wrappers.trt_dinov2_wrapper import (
        TensorRTDINOv2Wrapper,
        is_trt_available,
    )
except ImportError:
    TensorRTDINOv2Wrapper = None

    def is_trt_available():
        return False


try:
    from src.models.wrappers.cesp_module import CESP
except ImportError:
    CESP = None

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

        # Fix #4: Захист від race condition при паралельному завантаженні моделей (prewarm + main thread)
        self._model_lock = threading.Lock()

        self._pinned_models: set[str] = set()

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

    def _is_torch_compile_supported(self) -> bool:
        """Checks if torch.compile is safe to use in the current environment."""
        if not getattr(torch, "compile", None):
            return False

        use_compile = get_cfg(self.config, "models.performance.torch_compile", False)
        if not use_compile:
            return False

        if self.device == "cpu":
            return False

        # Windows-specific safety check: inductor (default) requires Triton
        if os.name == "nt":
            try:
                # inductor doesn't necessarily need triton imported,
                # but it needs it available in the environment.
                import triton  # noqa: F401

                return True
            except ImportError:
                # If Triton is missing on Windows, torch.compile(backend='inductor')
                # will likely crash with internal errors in dev/nightly PyTorch.
                logger.warning(
                    "torch.compile is requested but Triton is not installed on Windows. "
                    "Compilation disabled to prevent 'inductor' backend crashes."
                )
                return False

        return True

    def pin(self, models: list[str]):
        """Закріплює моделі в пам'яті (запобігає вивантаженню при нестачі VRAM)"""
        with self._model_lock:
            for m in models:
                self._pinned_models.add(m)
            logger.info(f"Pinned models: {self._pinned_models}")

    def unpin_all(self):
        """Знімає закріплення з усіх моделей"""
        with self._model_lock:
            self._pinned_models.clear()
            logger.info("Unpinned all models")

    def _unload_model_unsafe(self, name: str):
        if name in self.models:
            logger.info(f"Unloading model to free VRAM: {name}")
            del self.models[name]
            del self.model_usage[name]
            if self.device != "cpu":
                torch.cuda.empty_cache()
                gc.collect()

    def _ensure_vram_available(self, required_mb: float | None = None):
        if self.device == "cpu":
            return

        req = required_mb if required_mb is not None else self.default_vram_required

        while self.get_available_vram_mb() < req and self.models:
            non_pinned = {k: v for k, v in self.model_usage.items() if k not in self._pinned_models}
            if not non_pinned:
                logger.warning("All models pinned, cannot free VRAM. Risk of OOM.")
                return
            least = min(non_pinned, key=non_pinned.get)
            self._unload_model_unsafe(least)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()

    def prewarm(self):
        """Centralized model prewarming, usually called at startup in parallel"""
        logger.info("Starting centralized model prewarm sequence...")
        self.load_dinov2()
        self.load_aliked()
        self.load_lightglue_aliked()
        self.load_yolo()
        logger.success("Centralized model prewarm complete")

    def load_yolo(self):
        name = "yolo"
        with self._model_lock:
            if name not in self.models:
                model_path = get_cfg(self.config, "models.yolo.model_path", "yolo11n-seg.pt")
                vram_req = get_cfg(self.config, "models.yolo.vram_required_mb", 1200.0)
                use_trt = get_cfg(self.config, "models.performance.use_tensorrt_for_yolo", True)

                logger.info(f"Loading YOLO model: {model_path}...")
                self._ensure_vram_available(vram_req)
                try:
                    if YOLO is None:
                        raise ImportError("ultralytics.YOLO not found")

                    engine_path = str(model_path).replace(".pt", ".engine")
                    import os

                    if use_trt and self.device == "cuda":
                        if os.path.exists(engine_path):
                            logger.info(f"Found YOLO TRT engine: {engine_path}. Loading...")
                            model = YOLO(engine_path)
                            # TRT inference sets device automatically when used via YOLO API
                        else:
                            logger.info(
                                "YOLO TRT engine not found. Loading PyTorch model for export..."
                            )
                            model = YOLO(model_path)
                            model.to(self.device)
                            logger.info(
                                "Exporting YOLO to TensorRT format (this may take a while)..."
                            )
                            try:
                                # ultralytics automatically places the exported file next to the original
                                exported_path = model.export(
                                    format="engine", half=True, dynamic=False
                                )
                                logger.success(f"YOLO TRT export complete: {exported_path}")
                                if os.path.exists(exported_path):
                                    model = YOLO(exported_path)
                                    logger.info("YOLO TensorRT engine loaded successfully.")
                            except Exception as ex:
                                logger.warning(
                                    f"YOLO TRT export failed: {ex}. Falling back to PyTorch."
                                )
                    else:
                        model = YOLO(model_path)
                        model.to(self.device)

                    self.models[name] = model
                    logger.success(f"YOLO model loaded successfully on {self.device}")
                except Exception as e:
                    logger.error(
                        f"Failed to load YOLO model: {e} | "
                        f"model_path={model_path}, device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that the model file exists and is not corrupted.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_xfeat(self):
        name = "xfeat"
        with self._model_lock:
            if name not in self.models:
                repo = get_cfg(self.config, "models.xfeat.hub_repo", "verlab/accelerated_features")
                model_name = get_cfg(self.config, "models.xfeat.hub_model", "XFeat")
                top_k = get_cfg(self.config, "models.xfeat.top_k", 2048)
                vram_req = get_cfg(self.config, "models.xfeat.vram_required_mb", 300.0)

                logger.info(f"Loading XFeat model ({repo}/{model_name})...")
                self._ensure_vram_available(vram_req)
                try:
                    preset = get_cfg(self.config, "models.xfeat.xfeat_preset", "fast")
                    try:
                        # Attempt to pass quality_preset if supported by User's XFeat fork
                        model = torch.hub.load(
                            repo, model_name, pretrained=True, top_k=top_k, quality_preset=preset
                        )
                    except TypeError:
                        # Fallback to standard XFeat
                        model = torch.hub.load(repo, model_name, pretrained=True, top_k=top_k)

                    # FIX: XFeat hardcodes self.dev='cuda' if available, causing crashes if we move to CPU
                    if hasattr(model, "dev"):
                        model.dev = torch.device(self.device)
                    model = model.eval().to(self.device)
                    self.models[name] = model
                    logger.success(f"XFeat loaded successfully on {self.device} (preset: {preset})")
                except Exception as e:
                    logger.error(
                        f"Failed to load XFeat: {e} | "
                        f"repo={repo}, model={model_name}, device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check internet connection for torch.hub download.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_superpoint(self):
        name = "superpoint"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.superpoint.vram_required_mb", 500.0)

                logger.info("Loading SuperPoint model (for LightGlue compatibility)...")
                self._ensure_vram_available(vram_req)
                try:
                    if SuperPoint is None:
                        raise ImportError("lightglue.SuperPoint not found")

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
                    logger.error(
                        f"Failed to load SuperPoint: {e} | device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that 'lightglue' package is installed correctly.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_lightglue(self):
        name = "lightglue"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.lightglue.vram_required_mb", 1000.0)

                logger.info("Loading LightGlue model...")
                self._ensure_vram_available(vram_req)
                try:
                    if LightGlue is None:
                        raise ImportError("lightglue.LightGlue not found")

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
                    logger.error(
                        f"Failed to load LightGlue: {e} | device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that 'lightglue' package is installed and VRAM is sufficient.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_dinov2(self):
        name = "dinov2"
        with self._model_lock:
            if name not in self.models:
                repo = get_cfg(self.config, "models.dinov2.hub_repo", "facebookresearch/dinov2")
                model_name = get_cfg(self.config, "models.dinov2.hub_model", "dinov2_vitl14")
                vram_req = get_cfg(self.config, "models.dinov2.vram_required_mb", 1600.0)

                logger.info(f"Loading DINOv2 ({model_name}) model...")
                self._ensure_vram_available(vram_req)

                # Спроба завантажити TensorRT engine (якщо скомпільований)
                trt_loaded = False
                engine_dir = get_cfg(
                    self.config, "models.engines_cache.engine_cache_dir", "models/engines/"
                )
                try:
                    if TensorRTDINOv2Wrapper is not None and is_trt_available():
                        engine_path = os.path.join(engine_dir, "dinov2_vitl14_fp16.engine")
                        if os.path.exists(engine_path):
                            model = TensorRTDINOv2Wrapper(engine_path)
                            self.models[name] = model
                            trt_loaded = True
                            logger.success(f"DINOv2 TensorRT FP16 engine loaded: {engine_path}")
                except Exception as e:
                    logger.debug(f"TensorRT DINOv2 not available, using PyTorch: {e}")

                # Fallback: стандартний PyTorch hub
                if not trt_loaded:
                    try:
                        model = torch.hub.load(repo, model_name).to(self.device)

                        if self._is_torch_compile_supported():
                            try:
                                # Mode 'default' allows variable batch size gracefully
                                model = torch.compile(model, mode="default")
                                logger.info(
                                    "DINOv2 compiled successfully using torch.compile(mode='default')"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to torch.compile DINOv2: {e}")

                        self.models[name] = model
                        logger.success(f"DINOv2 model {model_name} loaded successfully (PyTorch)")
                    except Exception as e:
                        logger.error(f"Failed to load DINOv2: {e}", exc_info=True)
                        raise
            self._register_model_usage(name)
            return self.models[name]

    def load_aliked(self):
        """Завантажує ALIKED extractor (128-dim, lightglue-compatible)"""
        name = "aliked"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.aliked.vram_required_mb", 400.0)
                max_keypoints = get_cfg(self.config, "models.aliked.max_keypoints", 4096)

                logger.info(f"Loading ALIKED model (max_keypoints={max_keypoints})...")
                self._ensure_vram_available(vram_req)
                try:
                    if ALIKED is None:
                        raise ImportError("lightglue.ALIKED not found")

                    model = ALIKED(max_num_keypoints=max_keypoints).eval().to(self.device)

                    if self._is_torch_compile_supported():
                        try:
                            model = torch.compile(model, mode="default")
                            logger.info(
                                "ALIKED compiled successfully using torch.compile(mode='default')"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to torch.compile ALIKED: {e}")

                    self.models[name] = model
                    logger.success(f"ALIKED loaded successfully on {self.device}")
                except Exception as e:
                    logger.error(
                        f"Failed to load ALIKED: {e} | "
                        f"max_keypoints={max_keypoints}, device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that 'lightglue' package is installed correctly.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_lightglue_aliked(self):
        """Завантажує LightGlue з вагами для ALIKED (128-dim)"""
        name = "lightglue_aliked"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.lightglue.vram_required_mb", 1000.0)

                logger.info("Loading LightGlue (ALIKED weights)...")
                self._ensure_vram_available(vram_req)
                try:
                    if LightGlue is None:
                        raise ImportError("lightglue.LightGlue not found")

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
                    logger.error(
                        f"Failed to load LightGlue (ALIKED): {e} | device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_cesp(self):
        """Завантажує CESP модуль для покращення DINOv2 global descriptors"""
        name = "cesp"
        with self._model_lock:
            if name not in self.models:
                logger.info("Loading CESP module...")
                try:
                    if CESP is None:
                        raise ImportError("CESP not found")

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
                    logger.error(
                        f"Failed to load CESP: {e} | "
                        f"weights_path={weights_path}, device={self.device}. "
                        f"Check that the weights file exists and is compatible.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def unload_model(self, model_name: str):
        with self._model_lock:
            self._unload_model_unsafe(model_name)

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
from lightglue.utils import numpy_image_to_torch

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
from src.utils.telemetry import Telemetry

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
        with Telemetry.profile("dinov2"):
            logger.debug("Extracting global descriptor with DINOv2...")
            dino_tensor = torch.from_numpy(image).float().div_(255.0)
            dino_tensor = (
                dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            )
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
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
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
        with Telemetry.profile("aliked"):
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
                logger.warning(
                    f"All keypoints filtered out by YOLO mask! "
                    f"Image {image.shape[:2]}, total_kpts={len(aliked_out['keypoints'][0])}, "
                    f"mask_static_ratio={np.mean(static_mask > 128):.1%}. "
                    f"The entire image may be covered by dynamic objects (vehicles, people)."
                )

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
    def extract_features_batch(
        self, images: list[np.ndarray], static_masks: list[np.ndarray]
    ) -> list[dict]:
        """
        Extracts features for a batch of images using CUDA streams for parallel execution.
        """
        B = len(images)
        if B == 0:
            return []

        # 1. Prepare DINOv2 Tensor
        dino_tensors = []
        for img in images:
            rgb = torch.tensor(img, pin_memory=True).float().div_(255.0)
            dino_tensors.append(rgb.permute(2, 0, 1))
        dino_batch = torch.stack(dino_tensors).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_batch)

        # 2. Prepare Local Tensor
        prep_images = [self.preprocessor.preprocess(img) for img in images]
        local_tensors = []
        for p_img in prep_images:
            rgb = torch.tensor(p_img, pin_memory=True).float().div_(255.0)
            local_tensors.append(rgb.permute(2, 0, 1))
        local_batch = torch.stack(local_tensors).to(self.device, non_blocking=True)
        is_xfeat = (
            hasattr(self.local_model, "__class__")
            and "XFeat" in self.local_model.__class__.__name__
        )
        input_dict = {"image": local_batch} if not is_xfeat else local_batch

        stream_global = self.stream_global if self.device == "cuda" else None
        stream_local = self.stream_local if self.device == "cuda" else None

        global_descs = None
        aliked_out = None

        # PARALLEL EXECUTION
        context_global = (
            torch.cuda.stream(stream_global) if stream_global else contextlib.nullcontext()
        )
        with context_global:
            with Telemetry.profile("dinov2"):
                if self.cesp_module is not None:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                        features = self.global_model.forward_features(dino_input)
                    patch_tokens = features["x_norm_patchtokens"].float()
                    h_p, w_p = self.dino_size // 14, self.dino_size // 14
                    out_global = self.cesp_module(patch_tokens, h_p, w_p)
                else:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                        out_global = self.global_model(dino_input).float()

        out_kpts = []
        out_descs = []
        context_local = (
            torch.cuda.stream(stream_local) if stream_local else contextlib.nullcontext()
        )
        with context_local:
            with Telemetry.profile("local_extractor"):
                if is_xfeat:
                    # S3-1: Native True Batching for XFeat
                    xfeat_out = self.local_model.detectAndCompute(
                        input_dict, top_k=get_cfg(self.config, "models.xfeat.top_k", 2048)
                    )
                    for res in xfeat_out:
                        out_kpts.append(res["keypoints"].float())
                        out_descs.append(res["descriptors"].float())
                else:
                    # S3-1: ALIKED fallback. Unstable inside true batch, iterating frames natively.
                    for b in range(B):
                        single_img = local_batch[b : b + 1]  # shape (1, 3, H, W)
                        aliked_in = {"image": single_img}
                        aliked_out = self.local_model(aliked_in)
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

            results.append(
                {"keypoints": kp, "descriptors": desc, "coords_2d": kp.copy(), "global_desc": gd}
            )

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

from src.models.wrappers.yolo_wrapper import YOLOWrapper
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

        yolo_model = model_manager.load_yolo()
        yolo_wrapper = YOLOWrapper(yolo_model, device)
        return YOLOMaskingStrategy(yolo_wrapper)

    if strategy_name == "none":
        return NoMaskingStrategy()

    raise ValueError(f"Unknown masking strategy: '{strategy_name}'. Supported: 'yolo', 'none'")


# ================================================================================
# File: models\wrappers\trt_dinov2_wrapper.py
# ================================================================================
# src/models/wrappers/trt_dinov2_wrapper.py
#
# TensorRT runtime wrapper для DINOv2 ViT-L/14.
# Завантажує скомпільований .engine файл та виконує інференс без PyTorch overhead.

from pathlib import Path

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# TensorRT доступний не на всіх системах
try:
    import pycuda.autoinit  # noqa: F401 — ініціалізує CUDA context
    import pycuda.driver as cuda
    import tensorrt as trt

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False


def is_trt_available() -> bool:
    """Перевіряє чи TensorRT runtime доступний."""
    return _TRT_AVAILABLE


class TensorRTDINOv2Wrapper:
    """Runtime wrapper для TensorRT DINOv2 engine.

    Забезпечує інтерфейс forward(image_tensor) -> np.ndarray (1024-dim)
    сумісний із PyTorch DINOv2 wrapper.

    Використання:
        wrapper = TensorRTDINOv2Wrapper("models/engines/dinov2_vitl14_fp16.engine")
        descriptor = wrapper.forward(image_np)  # (1024,) float32
    """

    def __init__(self, engine_path: str, input_size: int = 336):
        if not _TRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install tensorrt and pycuda packages.")

        engine_file = Path(engine_path)
        if not engine_file.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}. "
                f"Run: python scripts/compile_dinov2_trt.py --output {engine_file.parent}"
            )

        self.input_size = input_size
        self._load_engine(str(engine_file))
        logger.success(f"TensorRT DINOv2 engine loaded: {engine_path}")

    def _load_engine(self, engine_path: str):
        """Завантажує TensorRT engine та виділяє GPU пам'ять."""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Виділення пам'яті для input та output
        self.input_shape = (1, 3, self.input_size, self.input_size)
        self.output_shape = (1, 1024)  # DINOv2 ViT-L/14 output dim

        input_nbytes = int(np.prod(self.input_shape) * np.float32(0).nbytes)
        output_nbytes = int(np.prod(self.output_shape) * np.float32(0).nbytes)

        # GPU буфери
        self.d_input = cuda.mem_alloc(input_nbytes)
        self.d_output = cuda.mem_alloc(output_nbytes)

        # CPU буфери (page-locked для швидкого копіювання)
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

        self.stream = cuda.Stream()
        logger.debug(f"TRT buffers allocated: input={input_nbytes}B, output={output_nbytes}B")

    def forward(self, image_chw: np.ndarray) -> np.ndarray:
        """Виконує інференс TensorRT engine.

        Args:
            image_chw: нормалізоване зображення (3, H, W) float32
                       (ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        Returns:
            global_descriptor: (1024,) float32
        """
        # Копіюємо дані у page-locked буфер
        np.copyto(self.h_input, image_chw.reshape(self.input_shape).astype(np.float32))

        # Host → Device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Інференс
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )

        # Device → Host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(-1).copy()  # (1024,)

    @property
    def output_dim(self) -> int:
        """Повертає розмірність вихідного дескриптора."""
        return self.output_shape[-1]

    def __del__(self):
        """Звільнює GPU ресурси."""
        try:
            if hasattr(self, "d_input"):
                self.d_input.free()
            if hasattr(self, "d_output"):
                self.d_output.free()
        except Exception:
            pass  # Ігноруємо помилки при garbage collection


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
        Detect objects and create static mask (single image).
        Делегує до batch-методу для уникнення дублювання логіки.

        Returns:
            static_mask: Binary mask of static areas (255 for static, 0 for dynamic)
            detections: List of detection dicts
        """
        return self.detect_and_mask_batch([image])[0]

    @torch.no_grad()
    def detect_and_mask_batch(self, images: list[np.ndarray]) -> list[tuple]:
        """
        Обробляє список зображень одним викликом YOLO.
        Повертає list[(static_mask, detections)] того самого порядку.
        """
        if not images:
            return []

        # verbose=False вимикає зайве логування кожного кадру в консоль
        # half=True для FP16 інференсу
        # conf=0.50 відкидає слабкі передбачення
        results = self.model(images, verbose=False, half=self.use_half, conf=0.50)

        MAX_SINGLE_MASK_RATIO = 0.40
        MAX_COMBINED_MASK_RATIO = 0.70

        output = []
        for result, image in zip(results, images):
            height, width = image.shape[:2]
            static_mask = np.ones((height, width), dtype=np.uint8) * 255
            total_pixels = height * width
            detections = []

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

        # Блок осі X (позиція X та швидкість VX)
        self.kf.Q[0, 0] = q_var[0, 0]  # Дисперсія позиції X
        self.kf.Q[0, 2] = q_var[0, 1]  # Коваріація X та VX
        self.kf.Q[2, 0] = q_var[1, 0]  # Коваріація VX та X
        self.kf.Q[2, 2] = q_var[1, 1]  # Дисперсія швидкості VX

        # Блок осі Y (позиція Y та швидкість VY)
        self.kf.Q[1, 1] = q_var[0, 0]  # Дисперсія позиції Y
        self.kf.Q[1, 3] = q_var[0, 1]  # Коваріація Y та VY
        self.kf.Q[3, 1] = q_var[1, 0]  # Коваріація VY та Y
        self.kf.Q[3, 3] = q_var[1, 1]  # Дисперсія швидкості VY

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

    def reset(self) -> None:
        """
        Скидає фільтр до початкового стану.
        Викликати при кожному новому старті трекінгу, щоб уникнути
        хибних передбачень на основі швидкості попередньої сесії.
        """
        self.is_initialized = False
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000.0
        logger.info("Kalman filter reset to initial state")


# ================================================================================
# File: tracking\outlier_detector.py
# ================================================================================
from collections import deque

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history using speeds"""

    def __init__(self, window_size=10, threshold_std=3.0, max_speed_mps=1000.0, max_consecutive=5):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self._consecutive_outliers = 0
        self._max_consecutive = max_consecutive

        logger.info("Initializing OutlierDetector (Speed-based Z-score)")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}"
        )

    def add_position(self, position: tuple, dt: float = 1.0):
        # Тепер зберігаємо і позицію, і dt (час, за який ця позиція була досягнута)
        self.window.append((np.array(position, dtype=np.float32), max(dt, 0.01)))
        self._consecutive_outliers = 0

    def is_outlier(self, new_position: tuple, dt: float = 1.0) -> bool:
        if len(self.window) < 3:
            return False

        new_pos_np = np.array(new_position, dtype=np.float32)
        last_pos, _ = self.window[-1]
        safe_dt = max(dt, 0.01)

        # 1. Перевірка максимально допустимої швидкості
        distance = float(np.linalg.norm(new_pos_np - last_pos))
        instantaneous_speed = distance / safe_dt

        is_speed_outlier = instantaneous_speed > self.max_speed_mps

        # 2. Статистичний Z-score тест (тепер за ШВИДКІСТЮ, а не за відстанню!)
        history = list(self.window)
        speeds = []
        for i in range(1, len(history)):
            p1, _ = history[i - 1]
            p2, hist_dt = history[i]
            dist = float(np.linalg.norm(p2 - p1))
            speeds.append(dist / hist_dt)

        mean_speed = np.mean(speeds)
        std_speed = max(np.std(speeds), 1.0)

        z_score = abs(instantaneous_speed - mean_speed) / std_speed

        # 15.0 m/s - мінімальна дельта швидкості, при якій Z-score має сенс
        is_zscore_outlier = (
            z_score > self.threshold_std and abs(instantaneous_speed - mean_speed) > 15.0
        )

        if is_speed_outlier or is_zscore_outlier:
            self._consecutive_outliers += 1

            # Якщо забагато підряд — дрон реально перемістився, скидаємо вікно
            if self._consecutive_outliers >= self._max_consecutive:
                logger.warning(
                    f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                    f"accepting new position. "
                    f"Position: ({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"speed={instantaneous_speed:.1f}m/s"
                )
                self.window.clear()
                self._consecutive_outliers = 0
                return False  # Приймаємо нову позицію

            if is_speed_outlier:
                logger.warning(
                    f"OUTLIER DETECTED (speed): {instantaneous_speed:.1f} m/s > {self.max_speed_mps} m/s | "
                    f"distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            else:
                logger.warning(
                    f"OUTLIER DETECTED (z-score): z={z_score:.2f} > {self.threshold_std} | "
                    f"speed={instantaneous_speed:.1f}m/s, mean_speed={mean_speed:.1f}m/s, std={std_speed:.1f}m/s, "
                    f"distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
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
# File: utils\telemetry.py
# ================================================================================
import atexit
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class _TelemetryTracker:
    def __init__(self):
        self.stats = defaultdict(
            lambda: {"calls": 0, "total_time": 0.0, "min_time": float("inf"), "max_time": 0.0}
        )

    class ProfilerContext:
        def __init__(self, tracker: "_TelemetryTracker", stage_name: str):
            self.tracker = tracker
            self.stage_name = stage_name
            self.start_time = 0.0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.perf_counter() - self.start_time
            st = self.tracker.stats[self.stage_name]
            st["calls"] += 1
            st["total_time"] += elapsed
            if elapsed < st["min_time"]:
                st["min_time"] = elapsed
            if elapsed > st["max_time"]:
                st["max_time"] = elapsed

        def __call__(self, func: Callable) -> Callable:
            """Allows using the context manager as a decorator."""

            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                with self:
                    return func(*args, **kwargs)

            return wrapper

    def profile(self, stage_name: str):
        """
        Returns a context manager / decorator for profiling a stage.
        Usage:
            with Telemetry.profile("yolo"):
                do_something()

            @Telemetry.profile("feature_extraction")
            def do_something_else():
                pass
        """
        return self.ProfilerContext(self, stage_name)

    def get_summary(self) -> dict:
        summary = {}
        for stage, st in self.stats.items():
            if st["calls"] == 0:
                continue
            avg = st["total_time"] / st["calls"]
            summary[stage] = {
                "calls": st["calls"],
                "total_time_s": round(st["total_time"], 4),
                "avg_time_s": round(avg, 4),
                "min_time_s": round(st["min_time"], 4),
                "max_time_s": round(st["max_time"], 4),
            }
        return summary

    def dump_report(self, path="logs/telemetry_report.json"):
        if not self.stats:
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_summary(), f, indent=4)
            logger.info(f"Telemetry report saved to {path} ({len(self.stats)} stages profiled)")
        except Exception as e:
            logger.error(f"Failed to save telemetry report: {e}")


# Singleton instance
Telemetry = _TelemetryTracker()


@atexit.register
def _save_telemetry_on_exit():
    Telemetry.dump_report()


# ================================================================================
# File: utils\__init__.py
# ================================================================================
"""Utilities module"""


# ================================================================================
# File: workers\calibration_propagation_worker.py
# ================================================================================
"""
Графова пропагація калібрування координат.

Замість лінійного ланцюжка гомографій, будується граф кадрів із:
  - Часовими ребрами (sequential: frame i ↔ frame i+1)
  - Просторовими ребрами (loop closure: DINOv2 retrieval → LightGlue matching)
  - GPS-якорями як жорсткими вузлами

Оптимізація: Levenberg-Marquardt через scipy.optimize.least_squares
з SO(2)-safe кутовими residuals (arctan2(sin, cos)).
"""

import json
from collections import defaultdict

import faiss
import h5py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config.config import get_cfg
from src.geometry.affine_utils import (
    compose_affine_5dof,
    decompose_affine,
    decompose_affine_5dof,
    unwrap_angles,
)
from src.geometry.pose_graph_optimizer import (
    PoseGraphOptimizer,
    homography_to_similarity,
)
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Графова пропагація з глобальною оптимізацією.

    Фази:
      1. Prefetch фіч → побудова часових ребер (sequential matching)
      2. Loop closure detection (FAISS DINOv2 retrieval → LightGlue matching)
      3. Фіксація GPS-якорів + BFS ініціалізація початкового наближення
      4. Глобальна оптимізація (Levenberg-Marquardt)
      5. Збереження результатів у HDF5
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

        # Параметри графової оптимізації
        self.lc_top_k = get_cfg(self.config, "graph_optimization.loop_closure_top_k", 5)
        self.lc_min_sim = get_cfg(
            self.config, "graph_optimization.loop_closure_min_similarity", 0.75
        )
        self.lc_min_gap = get_cfg(self.config, "graph_optimization.loop_closure_min_frame_gap", 3)
        self.lc_min_inliers = get_cfg(
            self.config, "graph_optimization.loop_closure_min_inliers", 15
        )
        self.temporal_base_w = get_cfg(
            self.config, "graph_optimization.temporal_edge_base_weight", 1.0
        )
        self.spatial_base_w = get_cfg(
            self.config, "graph_optimization.spatial_edge_base_weight", 2.0
        )
        self.max_iters = get_cfg(self.config, "graph_optimization.max_iterations", 50)
        self.tolerance = get_cfg(self.config, "graph_optimization.convergence_tolerance", 1e-6)
        self.use_bfs = get_cfg(self.config, "graph_optimization.use_bfs_initialization", True)
        self.export_geojson = get_cfg(self.config, "graph_optimization.export_geojson", True)

        # Скільки кадрів можна "перестрибнути" при побудові temporal ребер
        self.max_skip_frames = get_cfg(self.config, "propagation.max_skip_frames", 3)

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._propagate()
        except Exception as e:
            logger.error(
                f"Graph propagation failed: {e} | "
                f"num_anchors={len(self.calibration.anchors)}, "
                f"db_frames={self.database.get_num_frames()}",
                exc_info=True,
            )
            self.error.emit(str(e))

    # ─── Головний метод ──────────────────────────────────────────────────────

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        all_anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)
        anchors = [a for a in all_anchors if a.frame_id < num_frames]

        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        logger.info(
            f"Starting GRAPH propagation for {num_frames} frames "
            f"using {len(anchors)} anchors: "
            f"{[f'#{a.frame_id}' for a in anchors]}"
        )

        # ── Phase 1: Prefetch + Temporal edges ───────────────────────────────
        self.progress.emit(0, "Передзавантаження фіч у RAM...")
        all_features = self._prefetch_features(num_frames)

        optimizer = PoseGraphOptimizer(self.frame_w, self.frame_h)
        for i in range(num_frames):
            if i in all_features:
                optimizer.add_node(i)

        self.progress.emit(10, "Побудова часових ребер (sequential matching)...")
        temporal_count = self._build_temporal_edges(optimizer, all_features, num_frames)
        logger.info(f"Phase 1 complete: {temporal_count} temporal edges")

        # ── Phase 2: Loop closure detection ──────────────────────────────────
        self.progress.emit(30, "Пошук просторових замикань (loop closure)...")
        spatial_count = self._detect_loop_closures(optimizer, all_features, num_frames)
        logger.info(f"Phase 2 complete: {spatial_count} spatial edges (loop closures)")
        logger.info(
            f"Graph: {optimizer.num_nodes} nodes, {optimizer.num_edges} edges "
            f"({temporal_count} temporal + {spatial_count} spatial)"
        )

        # ── Phase 3: Fix anchors + BFS initialization ───────────────────────
        self.progress.emit(60, "Фіксація GPS-якорів та ініціалізація графу...")
        for anchor in anchors:
            optimizer.fix_node(anchor.frame_id, anchor.affine_matrix)

        if self.use_bfs:
            bfs_count = optimizer.initialize_from_bfs()
            logger.info(f"Phase 3 complete: {bfs_count} nodes initialized via BFS")
        else:
            logger.info("Phase 3 complete: BFS initialization skipped (disabled)")

        # ── Phase 4: Optimize ────────────────────────────────────────────────
        self.progress.emit(70, "Глобальна оптимізація графу (Levenberg-Marquardt)...")
        results = optimizer.optimize(
            max_iterations=self.max_iters,
            tolerance=self.tolerance,
        )
        logger.info(f"Phase 4 complete: {len(results)} frames optimized")

        # ── Phase 5: Save to HDF5 ───────────────────────────────────────────
        self.progress.emit(85, "Збереження результатів у HDF5...")
        valid_count = self._save_to_hdf5(results, anchors, optimizer)

        # Експорт GeoJSON для візуалізації
        if self.export_geojson and self.calibration.converter:
            try:
                geojson = optimizer.export_graph_geojson(
                    self.calibration.converter, self.frame_w, self.frame_h
                )
                geojson_path = str(self.database.db_path).replace(".h5", "_graph.geojson")
                with open(geojson_path, "w", encoding="utf-8") as f:
                    json.dump(geojson, f, indent=2, ensure_ascii=False)
                logger.success(f"Graph exported to GeoJSON: {geojson_path}")
            except Exception as e:
                logger.warning(f"GeoJSON export failed: {e}")

        self.progress.emit(
            100,
            f"Готово! {valid_count}/{num_frames} кадрів отримали координати "
            f"({temporal_count} часових + {spatial_count} просторових ребер).",
        )
        self.completed.emit()

    # ─── Phase 1: Prefetch + Temporal edges ──────────────────────────────────

    def _prefetch_features(self, num_frames: int) -> dict:
        """Завантажує всі фічі в RAM."""
        features = {}
        for i in range(num_frames):
            if not self._is_running:
                return features
            try:
                features[i] = self.database.get_local_features(i)
            except Exception:
                pass
            if i % 500 == 0:
                self.progress.emit(
                    int(i / num_frames * 8),
                    f"Prefetch: {i}/{num_frames}",
                )
        logger.info(f"Prefetched features for {len(features)} frames")
        return features

    def _build_temporal_edges(
        self,
        optimizer: PoseGraphOptimizer,
        features: dict,
        num_frames: int,
    ) -> int:
        """Побудова часових ребер між послідовними кадрами."""
        count = 0
        last_success_id = -1
        last_success_feat = None

        for i in range(num_frames):
            if not self._is_running:
                break

            feat_i = features.get(i)
            if feat_i is None:
                continue

            if last_success_feat is not None and (i - last_success_id) <= self.max_skip_frames:
                result = self._match_and_build_edge(feat_i, last_success_feat)
                if result is not None:
                    H, inliers, rmse_val = result
                    similarity = homography_to_similarity(H, self.frame_w, self.frame_h)
                    if similarity is not None:
                        weight = self._compute_weight(inliers, rmse_val, self.temporal_base_w)
                        optimizer.add_edge(
                            from_id=last_success_id,
                            to_id=i,
                            relative_affine_2x3=similarity,
                            weight=weight,
                            edge_type="temporal",
                            inliers=inliers,
                            rmse=rmse_val,
                        )
                        count += 1

            last_success_id = i
            last_success_feat = feat_i

            if i % 200 == 0:
                self.progress.emit(
                    10 + int(i / num_frames * 18),
                    f"Часові ребра: {count} (кадр {i}/{num_frames})",
                )

        return count

    # ─── Phase 2: Loop closure detection ─────────────────────────────────────

    def _detect_loop_closures(
        self,
        optimizer: PoseGraphOptimizer,
        features: dict,
        num_frames: int,
    ) -> int:
        """Знаходить просторові замикання через DINOv2 + LightGlue matching."""
        # Побудова FAISS індексу
        global_desc = self.database.global_descriptors
        if global_desc is None or len(global_desc) == 0:
            logger.warning("No global descriptors — skipping loop closure detection")
            return 0

        dim = global_desc.shape[1]
        normed = global_desc / (np.linalg.norm(global_desc, axis=1, keepdims=True) + 1e-8)
        normed = normed.astype(np.float32)

        index = faiss.IndexFlatIP(dim)
        index.add(normed)
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

        count = 0
        already_matched: set[tuple[int, int]] = set()

        for i in range(num_frames):
            if not self._is_running:
                break

            feat_i = features.get(i)
            if feat_i is None:
                continue

            # Query top-K кандидатів
            q = normed[i : i + 1]
            scores, ids = index.search(q, self.lc_top_k + 1)  # +1 бо сам себе знайде

            for raw_j, sim_score in zip(ids[0], scores[0]):
                j = int(raw_j)
                if j == i or j == -1:
                    continue
                if abs(i - j) <= self.lc_min_gap:
                    continue
                if float(sim_score) < self.lc_min_sim:
                    continue

                edge_key = (min(i, j), max(i, j))
                if edge_key in already_matched:
                    continue
                already_matched.add(edge_key)

                feat_j = features.get(j)
                if feat_j is None:
                    continue

                # Matching: feat_j → feat_i (H maps j pixels → i pixels)
                result = self._match_and_build_edge(feat_j, feat_i)
                if result is None:
                    continue

                H, inliers, rmse_val = result
                if inliers < self.lc_min_inliers:
                    continue

                similarity = homography_to_similarity(H, self.frame_w, self.frame_h)
                if similarity is None:
                    continue

                weight = self._compute_weight(inliers, rmse_val, self.spatial_base_w)
                optimizer.add_edge(
                    from_id=i,
                    to_id=j,
                    relative_affine_2x3=similarity,
                    weight=weight,
                    edge_type="spatial",
                    inliers=inliers,
                    rmse=rmse_val,
                )
                count += 1

            if i % 200 == 0:
                self.progress.emit(
                    30 + int(i / num_frames * 28),
                    f"Loop closure: {count} знайдено (кадр {i}/{num_frames})",
                )

        return count

    # ─── Phase 5: Save to HDF5 ───────────────────────────────────────────────

    def _save_to_hdf5(
        self,
        results: dict[int, np.ndarray],
        anchors,
        optimizer: PoseGraphOptimizer,
    ) -> int:
        """Зберігає оптимізовані афінні матриці у HDF5.

        Формат 100% сумісний з існуючим DatabaseLoader.
        """
        num_frames = self.database.get_num_frames()
        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.zeros(num_frames, dtype=bool)
        frame_rmse = np.zeros(num_frames, dtype=np.float32)
        frame_disagreement = np.zeros(num_frames, dtype=np.float32)
        frame_matches = np.zeros(num_frames, dtype=np.int32)

        # Записуємо результати оптимізації
        # Оскільки optimizer повертає ТІЛЬКИ досяжні вузли,
        # незв'язані кадри залишаться з frame_valid = False
        for frame_id, affine in results.items():
            if 0 <= frame_id < num_frames:
                frame_affine[frame_id] = affine.astype(np.float32)
                frame_valid[frame_id] = True

        filled_count = self._fill_gaps_by_interpolation(frame_affine, frame_valid)
        if filled_count > 0:
            logger.info(f"Interpolated coordinates for {filled_count} missing frames")

        # Обчислюємо QA метрики з ребер графу
        edge_stats: dict[int, list[tuple[int, float]]] = {}
        for edge in optimizer.edges:
            for fid in (edge.from_id, edge.to_id):
                if 0 <= fid < num_frames:
                    edge_stats.setdefault(fid, []).append((edge.inliers, edge.rmse))

        for fid, stats in edge_stats.items():
            # РОБИМО РОЗРАХУНОК ТІЛЬКИ ДЛЯ ВАЛІДНИХ КАДРІВ
            if fid < num_frames and frame_valid[fid]:
                inliers_list = [s[0] for s in stats]
                rmse_list = [s[1] for s in stats if s[1] > 0]
                frame_matches[fid] = int(np.mean(inliers_list)) if inliers_list else 0
                frame_rmse[fid] = float(np.mean(rmse_list)) if rmse_list else 0.0

        # Disagreement: для кадрів із ≥2 ребрами, порівнюємо predictions
        # (simplified: використовуємо std відхилень у tx, ty)

        # O(E) Optical optimization

        adj = defaultdict(list)
        for e in optimizer.edges:
            adj[e.from_id].append(e)
            adj[e.to_id].append(e)

        for fid in range(num_frames):
            if not frame_valid[fid]:
                continue
            edges_to_fid = adj[fid]
            if len(edges_to_fid) >= 2:
                predictions_tx = []
                for e in edges_to_fid[:5]:  # Обмежуємо для швидкодії
                    other_id = e.from_id if e.to_id == fid else e.to_id
                    other_affine = results.get(other_id)

                    # Перевіряємо, чи сусідній кадр також валідний
                    if other_affine is not None:
                        comp = decompose_affine(other_affine)
                        predictions_tx.append(comp[0])  # tx
                if len(predictions_tx) >= 2:
                    frame_disagreement[fid] = float(np.std(predictions_tx))

        # --- Збереження в HDF5 ---
        db_path = self.database.db_path
        self.database.close()
        try:
            with h5py.File(db_path, "a") as f:
                if "calibration" in f:
                    del f["calibration"]
                grp = f.create_group("calibration")

                grp.attrs["version"] = "3.0"  # Нова версія: графова оптимізація
                grp.attrs["num_anchors"] = len(anchors)
                grp.attrs["anchors_json"] = json.dumps(
                    [a.to_dict() for a in anchors], ensure_ascii=False
                )
                grp.attrs["projection_json"] = json.dumps(
                    self.calibration.converter.export_metadata()
                )
                grp.attrs["optimizer"] = "pose_graph_lm"
                grp.attrs["num_temporal_edges"] = sum(
                    1 for e in optimizer.edges if e.edge_type == "temporal"
                )
                grp.attrs["num_spatial_edges"] = sum(
                    1 for e in optimizer.edges if e.edge_type == "spatial"
                )

                grp.create_dataset("frame_affine", data=frame_affine, compression="gzip")
                grp.create_dataset(
                    "frame_valid", data=frame_valid.astype(np.uint8), compression="gzip"
                )
                grp.create_dataset("frame_rmse", data=frame_rmse, compression="gzip")
                grp.create_dataset(
                    "frame_disagreement", data=frame_disagreement, compression="gzip"
                )
                grp.create_dataset("frame_matches", data=frame_matches, compression="gzip")

            valid_count = int(np.sum(frame_valid))
            logger.success(
                f"Graph propagation saved to HDF5 (v3.0, "
                f"{len(anchors)} anchors, {optimizer.num_edges} edges, "
                f"{valid_count}/{num_frames} valid frames)"
            )
        finally:
            self.database._load_hot_data()

        return int(np.sum(frame_valid))

    # ─── Допоміжні методи ────────────────────────────────────────────────────

    def _match_and_build_edge(
        self, features_a: dict, features_b: dict
    ) -> tuple[np.ndarray, int, float] | None:
        """Матчить дві фічі та повертає (H, inliers, rmse) або None.

        H maps features_a (src) → features_b (dst).
        """
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None

            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
            )
            if H is None or mask is None:
                return None

            inlier_mask = mask.ravel().astype(bool)
            inliers = int(np.sum(inlier_mask))
            if inliers < self.min_matches:
                return None

            # RMSE
            pts_a_in = mkpts_a[inlier_mask]
            pts_transformed = GeometryTransforms.apply_homography(pts_a_in, H)
            pts_b_in = mkpts_b[inlier_mask]
            rmse = float(np.sqrt(np.mean(np.sum((pts_transformed - pts_b_in) ** 2, axis=1))))

            return H, inliers, rmse
        except Exception:
            return None

    @staticmethod
    def _compute_weight(inliers: int, rmse: float, base_weight: float) -> float:
        """Обчислює вагу ребра: w = base * √inliers / (1 + RMSE)."""
        return base_weight * np.sqrt(max(inliers, 1)) / (1.0 + rmse)

    def _fill_gaps_by_interpolation(self, frame_affine: np.ndarray, frame_valid: np.ndarray) -> int:
        """Лінійна 5-DoF інтерполяція для кадрів, пропущених через Keyframe Selection."""
        valid_ids = np.where(frame_valid)[0]
        if len(valid_ids) < 1:
            return 0

        filled = 0

        # Екстраполяція на початок
        first_valid = valid_ids[0]
        for mid in range(0, first_valid):
            frame_affine[mid] = frame_affine[first_valid].copy()
            frame_valid[mid] = True
            filled += 1

        # Інтерполяція розривів всередині траєкторії
        if len(valid_ids) >= 2:
            for i in range(len(valid_ids) - 1):
                left = valid_ids[i]
                right = valid_ids[i + 1]
                gap = right - left
                if gap <= 1:
                    continue

                # ВИКОРИСТОВУЄМО 5-DoF ДЕКОМПОЗИЦІЮ
                comp_left = np.array(decompose_affine_5dof(frame_affine[left]), dtype=np.float64)
                comp_right = np.array(decompose_affine_5dof(frame_affine[right]), dtype=np.float64)

                # Запобігаємо стрибкам кута (кут тепер під індексом 4)
                angles = unwrap_angles([comp_left[4], comp_right[4]])
                comp_left[4] = angles[0]
                comp_right[4] = angles[1]

                for mid in range(left + 1, right):
                    t = (mid - left) / gap
                    comp_mid = comp_left * (1.0 - t) + comp_right * t

                    # Розпаковуємо 5 змінних
                    tx, ty, sx, sy, angle = comp_mid
                    sx = float(np.clip(sx, 1e-6, 1e6))
                    sy = float(np.clip(sy, 1e-6, 1e6))

                    # ВИКОРИСТОВУЄМО 5-DoF КОМПОЗИЦІЮ
                    frame_affine[mid] = compose_affine_5dof(
                        float(tx), float(ty), sx, sy, float(angle)
                    )
                    frame_valid[mid] = True
                    filled += 1

        # Екстраполяція на кінець
        last_valid = valid_ids[-1]
        for mid in range(last_valid + 1, len(frame_valid)):
            frame_affine[mid] = frame_affine[last_valid].copy()
            frame_valid[mid] = True
            filled += 1

        return filled


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
            logger.warning(
                f"Database generation interrupted by user: {e} | video={self.video_path}"
            )
            self.cancelled.emit()
        except Exception as e:
            logger.error(
                f"Database generation failed: {e} | "
                f"video={self.video_path}, output={self.output_path}. "
                f"Check that the video file is valid (MP4/H.264) and disk has sufficient space.",
                exc_info=True,
            )
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
                raise ValueError(
                    f"Не вдалося прочитати файл панорами: {self.image_path}. "
                    f"Переконайтеся, що файл існує та має підтримуваний формат (PNG, JPEG, TIFF)."
                )

            logger.info(f"Panorama image loaded: {img.shape[1]}x{img.shape[0]} px")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loc_result = self.localizer.localize_frame(img_rgb)

            if not loc_result.get("success"):
                error_reason = loc_result.get("error", "Невідома причина")
                raise RuntimeError(
                    f"Не вдалося локалізувати панораму: {error_reason}. "
                    f"Переконайтесь, що база даних калібрована і панорама відповідає району бази."
                )

            fov = loc_result.get("fov_polygon")
            if not fov or len(fov) != 4:
                raise RuntimeError(
                    f"Локалізатор не повернув коректні кути (FOV) для панорами. "
                    f"Отримано fov={fov} (очікується 4 кути). "
                    f"Можливо, гомографія виродилася через недостатню кількість inliers."
                )

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
            logger.error(
                f"Panorama overlay worker failed: {e} | image_path={self.image_path}",
                exc_info=True,
            )
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
                    raise ValueError(
                        f"Не вдалося відкрити відеофайл: {self.video_path}. "
                        f"Переконайтесь, що файл існує і має підтримуваний кодек (MP4/H.264)."
                    )

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
                status_names = {
                    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS (недостатньо кадрів з перекриттям)",
                    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL (не вдалося знайти гомографію)",
                    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL (помилка калібрування камери)",
                }
                status_name = status_names.get(status, f"UNKNOWN_CODE_{status}")
                raise ValueError(
                    f"Помилка зшивання панорами: {status_name}. "
                    f"Зібрано {len(frames_to_stitch)} кадрів, крок={self.frame_step}. "
                    f"Спробуйте зменшити крок кадрів або переконатися, що кадри мають достатнє перекриття."
                )

        except Exception as e:
            logger.error(
                f"Panorama generation failed: {e} | "
                f"video={self.video_path}, output={self.output_path}, "
                f"frame_step={self.frame_step}",
                exc_info=True,
            )
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

        # S3-3: Інтервал ключових кадрів для локалізації
        self.keyframe_interval = get_cfg(self.config, "tracking.keyframe_interval", 5)
        # Зберігаємо process_fps для метрик UI, але логіка базується на кадрах
        self.process_fps = get_cfg(
            self.config, "tracking.process_fps", 30.0 / self.keyframe_interval
        )

    def run(self):
        # Fix #3: Скидаємо стан трекера при кожному новому старті сесії
        if hasattr(self.localizer, "trajectory_filter"):
            self.localizer.trajectory_filter.reset()
        if hasattr(self.localizer, "outlier_detector"):
            self.localizer.outlier_detector.window.clear()
            self.localizer.outlier_detector._consecutive_outliers = 0
        if hasattr(self.localizer, "_consecutive_failures"):
            self.localizer._consecutive_failures = 0

        if self.model_manager:
            self.model_manager.pin(["aliked", "lightglue_aliked", "dinov2"])

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
                logger.error(
                    f"Failed to load YOLO for tracking: {e} | "
                    f"device={self.model_manager.device}. "
                    f"Dynamic object masking will be unavailable. "
                    f"Tracking cannot proceed without YOLO.",
                    exc_info=True,
                )
                self.error.emit(f"YOLO не вдалося завантажити: {e}")
                return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(
                f"Failed to open video source: {self.video_source}. "
                f"Check that the file exists and is a valid video format (MP4/H.264 recommended)."
            )
            self.error.emit(f"Не вдалося відкрити відео: {self.video_source}")
            return

        # Визначаємо натуральну швидкість відео (зазвичай 30 FPS)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        # Замість time-based інтервалу використовуємо frame-based:
        frame_idx = 0
        prev_gray_for_of = None
        prev_pts_for_of = None

        # Зберігаємо останній час локалізації саме за ВІДЕО-часом, а не за процесорним
        last_localization_video_time = -1.0

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

            # 1. Завжди відправляємо кадр в GUI для плавного відтворення (сирий BGR)
            self.frame_ready.emit(frame)

            # S3-3: Optical Flow Pipeline
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = frame_idx % self.keyframe_interval == 0

            # Розрахунок dt
            if last_localization_video_time < 0:
                calculated_dt = frame_duration_sec
            else:
                calculated_dt = current_video_time_sec - last_localization_video_time
                if calculated_dt <= 0:
                    calculated_dt = frame_duration_sec

            loc_result = {"success": False, "error": "Not processed"}
            start_process = time.time()

            if is_keyframe or prev_pts_for_of is None:
                # ====== HEAVY KEYFRAME LOCALIZATION ======
                # Для обробки YOLO та анізотропних дескрипторів потрібен RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                static_mask = None
                if yolo_wrapper:
                    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)

                try:
                    loc_result = self.localizer.localize_frame(
                        frame_rgb, static_mask=static_mask, dt=calculated_dt
                    )
                except Exception as e:
                    logger.error(f"Localization exception on keyframe: {e}", exc_info=True)
                    loc_result = {"success": False, "error": str(e)}

                if loc_result.get("success"):
                    # Зберігаємо стан для OF на наступні кадри
                    prev_gray_for_of = curr_gray
                    # Трекаємо гарні точки (corners) для стабільного OF
                    prev_pts_for_of = cv2.goodFeaturesToTrack(
                        curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, mask=None
                    )
            else:
                # ====== OPTICAL FLOW TRACKING ======
                if prev_pts_for_of is not None and len(prev_pts_for_of) > 10:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray_for_of,
                        curr_gray,
                        prev_pts_for_of,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                    )
                    good_new = curr_pts[status == 1]
                    good_old = prev_pts_for_of[status == 1]

                    if len(good_new) > 10:
                        # Зсув у пікселях
                        flow_vectors = good_new - good_old
                        dx_px, dy_px = np.median(flow_vectors, axis=0)

                        try:
                            loc_result = self.localizer.localize_optical_flow(
                                dx_px,
                                dy_px,
                                dt=calculated_dt,
                                rot_width=frame.shape[1],
                                rot_height=frame.shape[0],
                            )
                        except Exception as e:
                            logger.error(f"OF Localization error: {e}")
                            loc_result = {"success": False, "error": str(e)}

                        # Оновлюємо стан так, щоб OF завжди рахувався ВІД КЛЮЧОВОГО КАДРУ,
                        # Це усуває проблему накопичення помилок (drift).
                        # Тому prev_gray_for_of та prev_pts_for_of не оновлюються тут!
                    else:
                        prev_pts_for_of = None  # Втрата точок — наступний кадр стане ключовим
                else:
                    prev_pts_for_of = None

            if loc_result.get("success") and loc_result.get("matched_frame", -1) != -1:
                self.location_found.emit(
                    loc_result["lat"],
                    loc_result["lon"],
                    loc_result["confidence"],
                    loc_result["inliers"],
                )
                if loc_result.get("fov_polygon"):
                    self.fov_found.emit(loc_result["fov_polygon"])

                track_type = "OF" if loc_result.get("is_of") else "KF"
                method_txt = (
                    "Схожість" if loc_result.get("fallback_mode") == "retrieval_only" else "Inliers"
                )
                score = loc_result.get("global_score", loc_result["inliers"])

                self.status_update.emit(
                    f"[{track_type}] Знайдено ({method_txt}: {score:.2f}, Кадр: {loc_result['matched_frame']})"
                )

                last_localization_video_time = current_video_time_sec
            elif not loc_result.get("success") and loc_result.get("error") != "Not processed":
                self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

            process_duration = time.time() - start_process
            self.fps_updated.emit(1.0 / process_duration if process_duration > 0 else 0)

            frame_idx += 1

            # 3. Синхронізація відтворення: щоб відео не "пролітало" за секунду,
            # змушуємо потік почекати, імітуючи реальну швидкість відео (1x)
            elapsed_in_loop = time.time() - loop_start
            sleep_time = frame_duration_sec - elapsed_in_loop
            if sleep_time > 0:
                self.msleep(int(sleep_time * 1000))

        cap.release()
        logger.info("Tracking worker thread finished cleanly.")

    def _prewarm_fallback_models(self):
        """Завантажує моделі заздалегідь, делегуючи у ModelManager."""
        try:
            if not self.model_manager:
                return
            logger.info("Tracking pre-warming centralized models...")
            self.model_manager.prewarm()
            logger.success("Tracking pre-warming successful")
        except Exception as e:
            logger.warning(
                f"Model pre-warming failed: {e}. "
                f"Models will be loaded on first use (slower first localization).",
                exc_info=True,
            )

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._stop_event.set()
        if not self.wait(5000):  # чекаємо максимум 5 секунд
            logger.warning("Tracking worker did not finish within 5 seconds.")
        else:
            logger.info("Tracking worker successfully stopped.")


# ================================================================================
# File: workers\video_decode_worker.py
# ================================================================================
import queue
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoDecodeWorker(QThread):
    """
    Фоновий потік для декодування відео та читання кадрів.
    Запобігає блокуванню головного GUI потоку під час I/O операцій.
    """

    frame_ready = pyqtSignal(int, np.ndarray)  # (frame_id, frame_bgr)
    error = pyqtSignal(str)
    video_loaded = pyqtSignal(int, float)  # (total_frames, fps)
    playback_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmd_queue = queue.Queue()
        self._is_running = True
        self.cap = None

    def run(self):
        is_playing = False
        play_fps = 30.0
        last_play_time = 0.0

        while self._is_running:
            try:
                # Читаємо команди блокуючи чергу (з таймаутом для плейбеку)
                if is_playing:
                    # Розрахунок часу до наступного кадру
                    elapsed = time.perf_counter() - last_play_time
                    delay = max(0.001, (1.0 / play_fps) - elapsed)

                    try:
                        cmd, arg = self.cmd_queue.get(timeout=delay)
                    except queue.Empty:
                        # Час грати наступний кадр
                        cmd, arg = "next_frame", None
                else:
                    cmd, arg = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Обробка команди
            try:
                if cmd == "load":
                    self._internal_load(arg)
                elif cmd == "seek":
                    is_playing = False
                    self._internal_seek(arg)
                elif cmd == "play":
                    is_playing = True
                    play_fps = arg if arg > 0 else 30.0
                    last_play_time = time.perf_counter()
                elif cmd == "pause":
                    is_playing = False
                    self.playback_stopped.emit()
                elif cmd == "stop":
                    self._is_running = False
                    break
                elif cmd == "next_frame":
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                            self.frame_ready.emit(frame_id, frame)
                            last_play_time = time.perf_counter()
                        else:
                            is_playing = False
                            self.playback_stopped.emit()

                self.cmd_queue.task_done()
            except Exception as e:
                logger.error(f"VideoDecodeWorker error handling command {cmd}: {e}")
                self.error.emit(str(e))

        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("VideoDecodeWorker thread finished.")

    def _internal_load(self, path: str):
        if self.cap:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            if cap:
                cap.release()
            self.error.emit(f"Не вдалося відкрити: {path}")
            return

        self.cap = cap
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video loaded: {total} frames, {fps} fps")
        self.video_loaded.emit(total, fps)

        # Read first frame
        self._internal_seek(0)

    def _internal_seek(self, frame_id: int):
        if not (self.cap and self.cap.isOpened()):
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()

        if not ret:
            # Fallback для деяких кодеків (шукати через час)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, (frame_id / fps) * 1000.0)
                ret, frame = self.cap.read()

        if ret and frame is not None:
            self.frame_ready.emit(frame_id, frame)

    # --- Public API for GUI (thread-safe) ---

    def load(self, path: str):
        self.cmd_queue.put(("load", path))

    def seek(self, frame_id: int):
        # Відкидаємо попередні seek-команди, якщо їх накопичилось багато
        # Це запобігає затримкам, якщо користувач швидко тягнув повзунок
        self._clear_queue_of("seek")
        self.cmd_queue.put(("seek", frame_id))

    def play(self, fps: float):
        self.cmd_queue.put(("play", fps))

    def pause(self):
        self.cmd_queue.put(("pause", None))

    def stop(self):
        self.cmd_queue.put(("stop", None))

    def _clear_queue_of(self, cmd_to_remove: str):
        """Видаляє застарілі команди з черги (корисно для debounce)."""
        temp_list = []
        try:
            while True:
                cmd, arg = self.cmd_queue.get_nowait()
                if cmd != cmd_to_remove:
                    temp_list.append((cmd, arg))
                self.cmd_queue.task_done()
        except queue.Empty:
            pass

        for cmd, arg in temp_list:
            self.cmd_queue.put((cmd, arg))


# ================================================================================
# File: workers\__init__.py
# ================================================================================
"""Worker threads module"""



# config/config.py
#
# Єдиний конфіг для всього застосунку з валідацією через Pydantic.

from typing import Any

from pydantic import BaseModel


class Dinov2Config(BaseModel):
    descriptor_dim: int = 1024
    input_size: int = 336


class DatabaseConfig(BaseModel):
    frame_step: int = 3
    prefetch_queue_size: int = 32
    keypoint_video_scale: float = 0.5
    inter_frame_min_matches: int = 15
    inter_frame_ransac_thresh: float = 3.0
    keyframe_min_translation_px: float = 15.0
    keyframe_min_rotation_deg: float = 1.5
    keyframe_always_save_first: bool = True
    use_decord: bool = True
    decode_batch_size: int = 32


class ConfidenceConfig(BaseModel):
    inlier_weight: float = 0.7
    stability_weight: float = 0.3
    rmse_norm_m: float = 10.0
    disagreement_norm_m: float = 5.0
    confidence_max_inliers: int = 80


class LocalizationConfig(BaseModel):
    min_matches: int = 12
    min_inliers_accept: int = 10
    ratio_threshold: float = 0.85
    ransac_threshold: float = 3.0
    retrieval_top_k: int = 12
    early_stop_inliers: int = 40
    retrieval_only_min_score: float = 0.90
    auto_rotation: bool = True
    enable_lightglue_fallback: bool = True
    fallback_extractor: str = "aliked"
    confidence: ConfidenceConfig = ConfidenceConfig()


class TrackingConfig(BaseModel):
    kalman_process_noise: float = 2.0
    kalman_measurement_noise: float = 5.0
    outlier_window: int = 10
    outlier_threshold_std: float = 150.0
    max_speed_mps: float = 1000.0
    max_consecutive_outliers: int = 5
    process_fps: float = 1.0


class PreprocessingConfig(BaseModel):
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: list[int] = [8, 8]
    histogram_matching: bool = True
    reference_image_path: str = "config/reference_style.png"
    masking_strategy: str = "yolo"  # "yolo" | "none" (підготовка до EfficientViT-SAM)


class GuiConfig(BaseModel):
    video_fps: int = 30
    verify_display_mode: str = "center"  # "center" | "center_corners" | "full"
    verify_label_mode: str = "number"  # "number" | "number_rmse" | "full"


class YoloConfig(BaseModel):
    model_path: str = "yolo11n-seg.pt"
    vram_required_mb: float = 200.0
    description: str = "YOLOv11n-seg (Nano) for dynamic object masking"


class ModelSettings(BaseModel):
    hub_repo: str | None = ""
    hub_model: str | None = ""
    top_k: int = 2048
    vram_required_mb: float = 500.0
    model_path: str | None = ""
    max_keypoints: int = 4096
    nms_radius: int = 4
    depth_confidence: float = -1.0
    width_confidence: float = -1.0
    detection_threshold: float = 0.001


class CespConfig(BaseModel):
    enabled: bool = False
    weights_path: str | None = None
    scales: list[int] = [1, 2, 4]


class VramManagementConfig(BaseModel):
    max_vram_ratio: float = 0.8
    default_required_mb: float = 2000.0


class ModelsCacheConfig(BaseModel):
    engine_cache_dir: str = "models/engines/"
    auto_compile: bool = False


class PerformanceConfig(BaseModel):
    propagation_max_workers: int = 4
    fp16_enabled: bool = True
    torch_compile: bool = False
    debug_mode: bool = False


class ModelsConfig(BaseModel):
    use_cuda: bool = True
    yolo: YoloConfig = YoloConfig()
    xfeat: ModelSettings = ModelSettings(
        hub_repo="verlab/accelerated_features",
        hub_model="XFeat",
        top_k=2048,
        vram_required_mb=300.0,
    )
    aliked: ModelSettings = ModelSettings(max_keypoints=4096, vram_required_mb=400.0)
    superpoint: ModelSettings = ModelSettings(
        nms_radius=4, max_keypoints=4096, vram_required_mb=500.0
    )
    lightglue: ModelSettings = ModelSettings(vram_required_mb=1000.0)
    dinov2: ModelSettings = ModelSettings(
        hub_repo="facebookresearch/dinov2", hub_model="dinov2_vitl14", vram_required_mb=1600.0
    )
    cesp: CespConfig = CespConfig()
    vram_management: VramManagementConfig = VramManagementConfig()
    performance: PerformanceConfig = PerformanceConfig()
    engines_cache: ModelsCacheConfig = ModelsCacheConfig()


class ProjectionConfig(BaseModel):
    default_mode: str = "WEB_MERCATOR"
    strict_projection: bool = True
    fallback_to_webmercator: bool = True
    anchor_rmse_threshold_m: float = 3.0
    anchor_max_error_m: float = 5.0
    propagation_disagreement_threshold_m: float = 2.0
    localizer_sample_points: int = 9
    localizer_expected_spread_m: float = 150.0


class HomographyConfig(BaseModel):
    backend: str = "opencv"  # "poselib" | "opencv"
    ransac_threshold: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.99


class GraphOptimizationConfig(BaseModel):
    """Конфігурація графової оптимізації пропагації координат."""

    # Просторові ребра (loop closure detection)
    loop_closure_top_k: int = 5
    loop_closure_min_similarity: float = 0.75
    loop_closure_min_frame_gap: int = 3
    loop_closure_min_inliers: int = 15

    # Вагові коефіцієнти ребер
    temporal_edge_base_weight: float = 1.0
    spatial_edge_base_weight: float = 2.0
    anchor_weight: float = 1e6

    # Levenberg-Marquardt оптимізатор
    max_iterations: int = 50
    convergence_tolerance: float = 1e-6

    # BFS ініціалізація початкового наближення
    use_bfs_initialization: bool = True

    # Діагностика
    export_geojson: bool = True


class AppConfig(BaseModel):
    dinov2: Dinov2Config = Dinov2Config()
    database: DatabaseConfig = DatabaseConfig()
    localization: LocalizationConfig = LocalizationConfig()
    tracking: TrackingConfig = TrackingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    gui: GuiConfig = GuiConfig()
    models: ModelsConfig = ModelsConfig()
    projection: ProjectionConfig = ProjectionConfig()
    homography: HomographyConfig = HomographyConfig()
    graph_optimization: GraphOptimizationConfig = GraphOptimizationConfig()


def get_cfg(config: Any, path: str, default: Any = None) -> Any:
    """Централізований доступ до конфігу з dot-path.
    Працює як зі словниками, так і з Pydantic-моделями.
    """
    keys = path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current


# Екземпляр конфігу за замовчуванням
APP_SETTINGS = AppConfig()
# Також надаємо доступ як до словника для зворотньої сумісності
APP_CONFIG = APP_SETTINGS.model_dump()



#!/usr/bin/env python3
"""Drone Topometric Localization System — application entry point."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Suppress only known noisy third-party warnings, not everything
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import torch
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QApplication

from config.config import APP_SETTINGS
from src.gui.main_window import MainWindow
from src.utils.logging_utils import get_logger, setup_logging


class StartupWorker(QThread):
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager

    def run(self):
        try:
            self.model_manager.prewarm()
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Startup prewarm failed: {e}. Models will load on first use.")


def _build_exception_hook(log):
    """Return sys.excepthook that logs unhandled exceptions before exit."""

    def hook(exctype, value, tb):
        log.critical(
            "Unhandled exception caught — application will exit",
            exc_info=(exctype, value, tb),
        )
        sys.exit(1)

    return hook


def main() -> None:
    # Logging must be initialized before anything else — including Qt
    try:
        from config.config import APP_SETTINGS
        debug_mode = APP_SETTINGS.models.performance.debug_mode
    except Exception:
        debug_mode = True # Safe default

    log_level = "INFO" if debug_mode else "CRITICAL"
    setup_logging(log_level=log_level, log_file="logs/app.log")
    logger = get_logger(__name__)

    # Route unhandled exceptions to loguru instead of silent PyQt6 crash
    sys.excepthook = _build_exception_hook(logger)

    logger.info("=" * 70)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 70)

    # System diagnostics for debugging
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA: {torch.version.cuda} | GPU: {gpu_name} | VRAM: {vram_total:.1f} GB")
        else:
            logger.warning(
                "CUDA not available — running on CPU. Performance will be significantly reduced."
            )
    except Exception as e:
        logger.warning(f"CUDA diagnostics failed: {e}. Continuing without GPU info.")

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication(sys.argv)
        app.setApplicationName("Drone Localization")
        app.setOrganizationName("UAV Systems")
        logger.info("Qt application initialized")

        window = MainWindow()
        window.show()

        # Запускаємо prewarm у фоновому потоці
        if hasattr(window, "model_manager") and window.model_manager:
            app._startup_worker = StartupWorker(window.model_manager)
            app._startup_worker.start()

        logger.success("Application startup complete")

        exit_code = app.exec()

    except Exception as e:
        logger.critical(f"Fatal error during startup: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Application exiting | code={exit_code}")
    logger.info("=" * 70)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
