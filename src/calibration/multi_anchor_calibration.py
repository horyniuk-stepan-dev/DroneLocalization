import numpy as np
import json
from src.geometry.transformations import GeometryTransforms
from src.geometry.coordinates import CoordinateConverter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AnchorCalibration:
    """Одна точка прив'язки GPS — конкретний кадр з affine матрицею"""

    def __init__(self, frame_id: int, affine_matrix: np.ndarray):
        self.frame_id = frame_id
        self.affine_matrix = affine_matrix

    def pixel_to_metric(self, x: float, y: float) -> tuple:
        pt = np.array([[x, y]], dtype=np.float32)
        result = GeometryTransforms.apply_affine(pt, self.affine_matrix)[0]
        return float(result[0]), float(result[1])

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "affine_matrix": self.affine_matrix.tolist()
        }

    @staticmethod
    def from_dict(data: dict) -> "AnchorCalibration":
        return AnchorCalibration(
            frame_id=int(data["frame_id"]),
            affine_matrix=np.array(data["affine_matrix"], dtype=np.float32)
        )


class MultiAnchorCalibration:
    """Менеджер декількох якорів калібрування"""

    def __init__(self):
        self.anchors = []
        # Додано референсну точку для ініціалізації UTM
        self.reference_gps = None

    @property
    def is_calibrated(self) -> bool:
        return len(self.anchors) > 0

    def add_anchor(self, frame_id: int, affine_matrix: np.ndarray):
        existing = next((a for a in self.anchors if a.frame_id == frame_id), None)
        if existing:
            existing.affine_matrix = affine_matrix
            logger.info(f"Updated anchor for frame {frame_id}")
        else:
            self.anchors.append(AnchorCalibration(frame_id, affine_matrix))
            self.anchors.sort(key=lambda a: a.frame_id)
            logger.info(f"Added new anchor for frame {frame_id}. Total anchors: {len(self.anchors)}")

    def get_metric_position(self, frame_id: int, x: float, y: float) -> tuple | None:
        if not self.is_calibrated:
            return None

        if len(self.anchors) == 1:
            return self.anchors[0].pixel_to_metric(x, y)

        anchor_1 = self.anchors[0]
        anchor_2 = self.anchors[-1]

        for i in range(len(self.anchors) - 1):
            if self.anchors[i].frame_id <= frame_id <= self.anchors[i + 1].frame_id:
                anchor_1 = self.anchors[i]
                anchor_2 = self.anchors[i + 1]
                break

        if frame_id <= self.anchors[0].frame_id:
            anchor_1 = anchor_2 = self.anchors[0]
        elif frame_id >= self.anchors[-1].frame_id:
            anchor_1 = anchor_2 = self.anchors[-1]

        metric_1 = anchor_1.pixel_to_metric(x, y)
        metric_2 = anchor_2.pixel_to_metric(x, y)

        dist_1 = abs(frame_id - anchor_1.frame_id)
        dist_2 = abs(frame_id - anchor_2.frame_id)
        total_dist = dist_1 + dist_2

        if total_dist == 0:
            return metric_1

        weight_1 = dist_2 / total_dist
        weight_2 = dist_1 / total_dist

        blend_x = metric_1[0] * weight_1 + metric_2[0] * weight_2
        blend_y = metric_1[1] * weight_1 + metric_2[1] * weight_2

        return blend_x, blend_y

    def save(self, path: str):
        if not self.is_calibrated:
            raise RuntimeError("Немає даних для збереження")

        data = {
            "reference_gps": self.reference_gps,  # Зберігаємо базову координату
            "anchors": [a.to_dict() for a in self.anchors]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.success(f"MultiAnchorCalibration saved: {path} ({len(self.anchors)} anchors)")

    def load(self, path: str):
        logger.info(f"Loading MultiAnchorCalibration from: {path}")
        with open(path, 'r') as f:
            data = json.load(f)

        self.anchors.clear()

        # Ініціалізація UTM конвертера, якщо у файлі збережено референсну GPS точку
        if "reference_gps" in data and data["reference_gps"] is not None:
            self.reference_gps = data["reference_gps"]
            CoordinateConverter.gps_to_metric(self.reference_gps[0], self.reference_gps[1])
            logger.info(f"UTM Projection initialized from loaded reference GPS: {self.reference_gps}")
        else:
            logger.warning("No reference GPS found in calibration file. UTM converter is not initialized.")

        # Підтримка старого формату файлів
        if "affine_matrix" in data and "calib_frame_id" in data:
            anchor = AnchorCalibration(
                frame_id=int(data.get("calib_frame_id", 0)),
                affine_matrix=np.array(data["affine_matrix"], dtype=np.float32)
            )
            self.anchors.append(anchor)
        # Новий формат файлів
        elif "anchors" in data:
            for item in data["anchors"]:
                self.anchors.append(AnchorCalibration.from_dict(item))

        self.anchors.sort(key=lambda a: a.frame_id)
        logger.success(f"Loaded {len(self.anchors)} anchors")