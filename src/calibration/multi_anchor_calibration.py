try:
    import orjson as _json_lib

    _USE_ORJSON = True
except ImportError:
    import json as _json_lib  # type: ignore[assignment]  # fallback якщо orjson не встановлено

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

        # Fallback — лінійна інтерполяція (якщо PCHIP недоступний або NaN)
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

        # Зберігаємо також метадані проєкції
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
