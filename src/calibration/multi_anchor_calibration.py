import numpy as np
import json
from src.geometry.transformations import GeometryTransforms
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
        self.is_calibrated = False

    def add_anchor(self, frame_id: int, affine_matrix: np.ndarray):
        new_anchor = AnchorCalibration(frame_id, affine_matrix)
        self.anchors = [a for a in self.anchors if a.frame_id != frame_id]
        self.anchors.append(new_anchor)
        self.anchors.sort(key=lambda a: a.frame_id)
        self.is_calibrated = True
        logger.info(f"Added anchor for frame {frame_id}. Total anchors: {len(self.anchors)}")

    def blend_metric(self, frame_id: int, metric_1: tuple, metric_2: tuple, anchor_1, anchor_2) -> tuple:
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
        data = {"anchors": [a.to_dict() for a in self.anchors]}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.success(f"MultiAnchorCalibration saved: {path} ({len(self.anchors)} anchors)")

    def load(self, path: str):
        logger.info(f"Loading MultiAnchorCalibration from: {path}")
        with open(path, 'r') as f:
            data = json.load(f)

        if "affine_matrix" in data and "calib_frame_id" in data:
            anchor = AnchorCalibration(
                frame_id=int(data.get("calib_frame_id", 0)),
                affine_matrix=np.array(data["affine_matrix"], dtype=np.float32)
            )
            self.anchors = [anchor]
        else:
            self.anchors = [AnchorCalibration.from_dict(a) for a in data["anchors"]]
            self.anchors.sort(key=lambda a: a.frame_id)

        self.is_calibrated = True
        logger.success(f"Loaded {len(self.anchors)} anchors")