import json
import os
import shutil
import numpy as np
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_VALID_AFFINE_SHAPES = {(2, 3), (3, 3)}


class AnchorCalibration:
    """Single GPS anchor — a reference frame with its affine matrix (pixel → metric)."""

    def __init__(self, frame_id: int, affine_matrix: np.ndarray):
        if affine_matrix is None or tuple(affine_matrix.shape) not in _VALID_AFFINE_SHAPES:
            raise ValueError(
                f"affine_matrix must be (2,3) or (3,3), got {getattr(affine_matrix, 'shape', None)}"
            )
        self.frame_id = frame_id
        self.affine_matrix = affine_matrix.astype(np.float32)

    def pixel_to_metric(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[x, y]], dtype=np.float32)
        result = GeometryTransforms.apply_affine(pt, self.affine_matrix)[0]
        return float(result[0]), float(result[1])

    def to_dict(self) -> dict:
        return {
            "frame_id":     self.frame_id,
            "affine_matrix": self.affine_matrix.tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> "AnchorCalibration":
        matrix = np.array(data["affine_matrix"], dtype=np.float32)
        return AnchorCalibration(
            frame_id=int(data["frame_id"]),
            affine_matrix=matrix,
        )


class MultiAnchorCalibration:
    """Manages multiple GPS anchor points for localization calibration."""

    def __init__(self):
        self.anchors: list[AnchorCalibration] = []
        self.is_calibrated: bool = False

    # ── Anchor management ────────────────────────────────────────────────────

    def add_anchor(self, frame_id: int, affine_matrix: np.ndarray) -> None:
        """Add or replace anchor for frame_id. Keeps list sorted by frame_id."""
        anchor = AnchorCalibration(frame_id, affine_matrix)  # validates shape
        self.anchors = [a for a in self.anchors if a.frame_id != frame_id]
        self.anchors.append(anchor)
        self.anchors.sort(key=lambda a: a.frame_id)
        self.is_calibrated = True
        logger.info(f"Anchor added: frame={frame_id}, total={len(self.anchors)}")

    def remove_anchor(self, frame_id: int) -> bool:
        """Remove anchor by frame_id. Returns True if found and removed."""
        before = len(self.anchors)
        self.anchors = [a for a in self.anchors if a.frame_id != frame_id]
        removed = len(self.anchors) < before
        if removed:
            self.is_calibrated = len(self.anchors) > 0
            logger.info(f"Anchor removed: frame={frame_id}, remaining={len(self.anchors)}")
        return removed

    def reset(self) -> None:
        """Clear all anchors."""
        self.anchors.clear()
        self.is_calibrated = False
        logger.info("MultiAnchorCalibration reset")

    # ── Interpolation ────────────────────────────────────────────────────────

    def blend_metric(
        self,
        frame_id: int,
        metric_1: tuple,
        metric_2: tuple,
        anchor_1: AnchorCalibration,
        anchor_2: AnchorCalibration,
    ) -> tuple[float, float]:
        """Linear interpolation between two anchor metric coordinates by frame distance."""
        dist_1 = abs(frame_id - anchor_1.frame_id)
        dist_2 = abs(frame_id - anchor_2.frame_id)
        total  = dist_1 + dist_2
        if total == 0:
            return metric_1
        w1 = dist_2 / total
        w2 = dist_1 / total
        return metric_1[0] * w1 + metric_2[0] * w2, metric_1[1] * w1 + metric_2[1] * w2

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if not self.is_calibrated:
            raise RuntimeError("No calibration data to save")

        data = {"anchors": [a.to_dict() for a in self.anchors]}

        # Atomic write — prevent corrupt file on crash
        tmp_path = path + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            shutil.move(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.success(f"Calibration saved: {path} ({len(self.anchors)} anchors)")

    def load(self, path: str) -> None:
        logger.info(f"Loading calibration: {path}")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt calibration file: {e}")

        # Backward-compat: legacy single-anchor format
        if "affine_matrix" in data and "calib_frame_id" in data:
            logger.warning("Legacy single-anchor format detected — consider re-calibrating")
            anchors = [AnchorCalibration(
                frame_id=int(data["calib_frame_id"]),
                affine_matrix=np.array(data["affine_matrix"], dtype=np.float32),
            )]
        elif "anchors" in data:
            anchors = [AnchorCalibration.from_dict(a) for a in data["anchors"]]
        else:
            raise ValueError(f"Unknown calibration format: keys={list(data.keys())}")

        anchors.sort(key=lambda a: a.frame_id)
        self.anchors = anchors
        self.is_calibrated = True
        logger.success(f"Loaded {len(self.anchors)} anchors")
