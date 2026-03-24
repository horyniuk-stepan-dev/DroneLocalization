import json

import numpy as np

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AnchorCalibration:
    """
    Одна точка прив'язки GPS — конкретний кадр з афінною матрицею та повними QA-метриками.
    """

    def __init__(self, frame_id: int, affine_matrix: np.ndarray, qa_data: dict = None):
        self.frame_id = frame_id
        self.affine_matrix = affine_matrix

        # QA fields
        self.qa_data = qa_data or {}

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
        self.created_at = self.qa_data.get("created_at", "")
        self.updated_at = self.qa_data.get("updated_at", self.created_at)
        self.notes = self.qa_data.get("notes", "")
        self.quality_flag = self.qa_data.get("quality_flag", "normal")  # 'normal', 'warning', 'bad'

    def pixel_to_metric(self, x: float, y: float) -> tuple:
        pt = np.array([[x, y]], dtype=np.float32)
        result = GeometryTransforms.apply_affine(pt, self.affine_matrix)[0]
        return float(result[0]), float(result[1])

    def to_dict(self) -> dict:
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
    def from_dict(data: dict) -> "AnchorCalibration":
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

    VERSION = "2.2"

    def __init__(self):
        self.anchors = []

    @property
    def is_calibrated(self) -> bool:
        return len(self.anchors) > 0

    def add_anchor(self, frame_id: int, affine_matrix: np.ndarray, qa_data: dict = None):
        existing = next((a for a in self.anchors if a.frame_id == frame_id), None)
        if existing:
            existing.affine_matrix = affine_matrix
            if qa_data:
                # Оновлюємо QA дані
                new_anchor = AnchorCalibration(frame_id, affine_matrix, qa_data)
                existing.qa_data = new_anchor.qa_data
                existing.rmse_m = new_anchor.rmse_m
                existing.max_err_m = new_anchor.max_err_m
                existing.inliers_count = new_anchor.inliers_count
                existing.points_2d = new_anchor.points_2d
                existing.points_gps = new_anchor.points_gps
                existing.updated_at = new_anchor.updated_at
            logger.info(f"Updated anchor for frame {frame_id}")
        else:
            self.anchors.append(AnchorCalibration(frame_id, affine_matrix, qa_data))
            self.anchors.sort(key=lambda a: a.frame_id)
            logger.info(
                f"Added new anchor for frame {frame_id}. Total anchors: {len(self.anchors)}"
            )

    def get_anchor(self, frame_id: int) -> AnchorCalibration | None:
        return next((a for a in self.anchors if a.frame_id == frame_id), None)

    def remove_anchor(self, frame_id: int) -> bool:
        initial_len = len(self.anchors)
        self.anchors = [a for a in self.anchors if a.frame_id != frame_id]
        success = len(self.anchors) < initial_len
        if success:
            logger.info(f"Removed anchor for frame {frame_id}")
        return success

    def get_metric_position(self, frame_id: int, x: float, y: float) -> tuple | None:
        if not self.is_calibrated:
            return None
        if len(self.anchors) == 1 or frame_id <= self.anchors[0].frame_id:
            return self.anchors[0].pixel_to_metric(x, y)
        if frame_id >= self.anchors[-1].frame_id:
            return self.anchors[-1].pixel_to_metric(x, y)

        # Знаходимо сегмент — тепер логіка одна і читається зверху вниз
        for i in range(len(self.anchors) - 1):
            a1, a2 = self.anchors[i], self.anchors[i + 1]
            if a1.frame_id <= frame_id <= a2.frame_id:
                dist_1 = abs(frame_id - a1.frame_id)
                dist_2 = abs(frame_id - a2.frame_id)
                total = dist_1 + dist_2
                if total == 0:
                    return a1.pixel_to_metric(x, y)
                w2 = dist_1 / total  # ближчий до a2 → більше вагу a2
                m1 = a1.pixel_to_metric(x, y)
                m2 = a2.pixel_to_metric(x, y)
                return m1[0] * (1 - w2) + m2[0] * w2, m1[1] * (1 - w2) + m2[1] * w2
        return None

    def save(self, path: str):
        if not self.is_calibrated:
            raise RuntimeError("Немає даних для збереження")

        # Зберігаємо також метадані проєкції
        data = {
            "version": self.VERSION,
            "projection": CoordinateConverter.export_projection_metadata(),
            "anchors": [a.to_dict() for a in self.anchors],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.success(
            f"MultiAnchorCalibration saved: {path} (v{self.VERSION}, {len(self.anchors)} anchors)"
        )

    def load(self, path: str):
        logger.info(f"Loading MultiAnchorCalibration from: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.anchors.clear()
        version = data.get("version", "1.0")

        # 1. Відновлення проєкції
        if "projection" in data:
            CoordinateConverter.load_projection_metadata(data["projection"])
        elif "reference_gps" in data and data["reference_gps"] is not None:
            # Fallback для v2.0
            CoordinateConverter.configure_projection("UTM", data["reference_gps"])
        else:
            # Fallback для v1.0 або відсутніх даних
            logger.warning(
                "No projection metadata found in calibration file. Defaulting to WEB_MERCATOR fallback."
            )
            CoordinateConverter.configure_projection("WEB_MERCATOR")

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
        logger.success(f"Loaded {len(self.anchors)} anchors (file version: {version})")
