try:
    import orjson as _json_lib

    _USE_ORJSON = True
except ImportError:
    import json as _json_lib

    _USE_ORJSON = False
from datetime import datetime
from typing import Any

import numpy as np

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms
from src.security.at_rest import decrypt_bytes, get_passphrase, is_encrypted
from src.security.project_scan import assert_project_writable
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Single source of truth for 5-DoF center-parameterized PCHIP interpolation
from src.geometry.affine_utils import build_5dof_pchip as _build_5dof_pchip
from src.geometry.affine_utils import sample_5dof_pchip as _sample_5dof_pchip


class AnchorCalibration:
    """A single GPS anchor point bound to a specific frame with affine transformation and full QA metrics."""

    def __init__(
        self, frame_id: int, affine_matrix: np.ndarray, qa_data: dict[str, Any] | None = None
    ):
        self.frame_id = frame_id
        self.affine_matrix = affine_matrix
        self.update_qa(qa_data or {})

    def update_qa(self, qa_data: dict[str, Any]) -> None:
        """Updates anchor QA metrics in-place without re-creating the object."""
        self.qa_data = qa_data

        # Primary quality metrics
        self.rmse_m = float(self.qa_data.get("rmse_m", 0.0))
        self.median_err_m = float(self.qa_data.get("median_err_m", 0.0))
        self.max_err_m = float(self.qa_data.get("max_err_m", 0.0))
        self.inliers_count = int(self.qa_data.get("inliers_count", 0))

        # Point collections
        self.points_2d = self.qa_data.get("points_2d", [])  # [[x,y], ...]
        self.points_gps = self.qa_data.get("points_gps", [])  # [[lat,lon], ...]
        self.points_metric = self.qa_data.get("points_metric", [])  # [[mx,my], ...]

        # Metadata and UI flags
        self.transform_type = self.qa_data.get("transform_type", "unknown")
        self.projection_mode = self.qa_data.get("projection_mode", "WEB_MERCATOR")
        self.created_at = self.qa_data.get("created_at", datetime.now().isoformat())
        self.updated_at = self.qa_data.get("updated_at", self.created_at)
        self.notes = self.qa_data.get("notes", "")
        self.quality_flag = self.qa_data.get("quality_flag", "normal")  # 'normal', 'warning', 'bad'

    def pixel_to_metric(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[x, y]], dtype=np.float64)
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
        # Support legacy formats without qa_data
        qa = data.get("qa_data", {})

        # Handle legacy v1.0/v2.0 flat dictionary layouts
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
            affine_matrix=np.array(data["affine_matrix"], dtype=np.float64),
            qa_data=qa,
        )


class MultiAnchorCalibration:
    """Manages multiple calibration anchors with versioning, projection support, and PCHIP interpolation."""

    VERSION: str = "2.3"

    def __init__(
        self,
        converter: CoordinateConverter | None = None,
        log_scale_interp: bool | None = None,
    ) -> None:
        self.anchors: list[AnchorCalibration] = []
        self.converter = converter or CoordinateConverter("WEB_MERCATOR")
        # Log-scale scale interpolation between anchors. None -> read from APP_CONFIG default False.
        if log_scale_interp is None:
            try:
                from config import APP_CONFIG, get_cfg

                log_scale_interp = bool(
                    get_cfg(APP_CONFIG, "graph_optimization.log_scale_interp", False)
                )
            except Exception:
                log_scale_interp = False
        self._log_scale_interp = bool(log_scale_interp)
        self._interp: Any = None  # Cached PCHIP interpolator (build_5dof_pchip)
        self._interp_sign: float = -1.0  # Determinant sign of anchor matrices (Y-flip)
        self._interp_range: tuple[float, float] | None = None  # [first_frame, last_frame]
        self._ref_px: tuple[float, float] = (0.0, 0.0)  # Decomposition reference pixel
        self._frame_size: tuple[int, int] | None = None  # (width, height)

    def set_frame_size(self, width: int, height: int) -> None:
        """Sets frame dimensions: interpolation is parameterized around the frame center."""
        new_size = (int(width), int(height))
        if new_size != self._frame_size and new_size[0] > 0 and new_size[1] > 0:
            self._frame_size = new_size
            self._rebuild_interpolators()

    def _reference_pixel(self) -> tuple[float, float]:
        """Calculates the reference pixel for matrix decomposition (frame center or point centroid)."""
        if self._frame_size:
            return self._frame_size[0] / 2.0, self._frame_size[1] / 2.0
        # Fallback: centroid of 2D points across all anchors
        pts = [p for a in self.anchors for p in (a.points_2d or [])]
        if pts:
            arr = np.asarray(pts, dtype=np.float64)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
        return 0.0, 0.0

    def _rebuild_interpolators(self) -> None:
        """Rebuilds the 5-DoF PCHIP interpolator from anchor matrices.

        Interpolates (rx, ry, sx, sy, angle) where (rx, ry) is the metric position of the
        reference pixel (frame center). The determinant sign is preserved and reapplied
        during matrix composition to ensure proper orientation and coordinate system Y-flip.
        """
        self._interp = None
        self._interp_range = None
        if len(self.anchors) < 2:
            logger.debug(f"Interpolator not built: need ≥2 anchors, have {len(self.anchors)}")
            return

        dets = np.array(
            [np.linalg.det(a.affine_matrix[:2, :2]) for a in self.anchors], dtype=np.float64
        )
        n_neg = int(np.sum(dets < 0))
        if 0 < n_neg < len(dets):
            logger.warning(
                f"Anchors have MIXED determinant signs ({n_neg}/{len(dets)} negative). "
                f"One of the anchors is likely mirrored (swapped lat/lon?). "
                f"Interpolation may be unreliable — recheck anchor points."
            )

        cx, cy = self._reference_pixel()
        self._ref_px = (cx, cy)

        # Shared builder: 5-DoF center-parameterized PCHIP interpolation
        ids = [a.frame_id for a in self.anchors]
        affines = [a.affine_matrix for a in self.anchors]
        self._interp, self._interp_sign, self._interp_range = _build_5dof_pchip(
            ids, affines, (cx, cy), log_scale=self._log_scale_interp
        )

    def _get_interpolated_matrix(self, frame_id: float) -> np.ndarray | None:
        """Returns the interpolated 2x3 affine matrix for a given frame_id."""
        return _sample_5dof_pchip(
            self._interp,
            self._interp_sign,
            self._interp_range,
            self._ref_px,
            frame_id,
            log_scale=self._log_scale_interp,
        )

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
                qa_data["updated_at"] = datetime.now().isoformat()
                existing.update_qa(qa_data)
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

    def clear(self) -> None:
        """Clears all anchors and resets calibration state."""
        self.anchors.clear()
        self._interp = None
        self._interp_range = None
        from src.geometry.coordinates import CoordinateConverter

        self.converter = CoordinateConverter("WEB_MERCATOR")
        logger.info("Cleared all anchors and reset calibration state.")

    def get_metric_position(self, frame_id: int, x: float, y: float) -> tuple[float, float] | None:
        if not self.is_calibrated:
            return None

        # Single anchor: extrapolation not supported, return direct conversion
        if len(self.anchors) == 1:
            return self.anchors[0].pixel_to_metric(x, y)

        # Exact anchor hit: reset accumulated drift and return direct transformation
        exact_anchor = self.get_anchor(frame_id)
        if exact_anchor is not None:
            logger.debug(
                f"Exact anchor hit at frame {frame_id} — using direct affine (drift reset)"
            )
            return exact_anchor.pixel_to_metric(x, y)

        # PCHIP interpolation
        if self._interp is not None:
            M = self._get_interpolated_matrix(float(frame_id))
            if M is not None:
                pt = np.array([[x, y]], dtype=np.float64)
                result = GeometryTransforms.apply_affine(pt, M)[0]
                return float(result[0]), float(result[1])

        # Fallback linear interpolation
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

    def set_gsd_calculator(self, gsd_calculator) -> None:
        """Links GSD calculator for metadata and inspection purposes."""
        self._gsd = gsd_calculator
        if self._gsd:
            logger.info(f"GSD Calculator linked: {self._gsd.gsd_m_per_px * 100:.2f} cm/px")

    def save(self, path: str) -> None:
        """Saves anchors and projection metadata to JSON."""
        assert_project_writable(path)
        data = {
            "version": self.VERSION,
            "projection": self.converter.export_metadata(),
            "frame_size": list(self._frame_size) if self._frame_size else None,
            "anchors": [a.to_dict() for a in self.anchors],
        }

        # Atomic write to prevent file corruption
        from src.utils.atomic_io import atomic_write_bytes

        if _USE_ORJSON:
            raw = _json_lib.dumps(
                data,
                option=_json_lib.OPT_INDENT_2 | getattr(_json_lib, "OPT_NON_STR_KEYS", 0),
            )
            atomic_write_bytes(path, raw)
        else:
            atomic_write_bytes(
                path, _json_lib.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            )
        logger.success(
            f"MultiAnchorCalibration saved: {path} (v{self.VERSION}, {len(self.anchors)} anchors)"
        )

    def load(self, path: str) -> None:
        logger.info(f"Loading MultiAnchorCalibration from: {path}")
        with open(path, "rb") as f:
            content = f.read()

        # Transparently decrypt at-rest encrypted calibration
        if is_encrypted(content):
            content = decrypt_bytes(content, get_passphrase())

        if _USE_ORJSON:
            data = _json_lib.loads(content)
        else:
            data = _json_lib.loads(content.decode("utf-8"))

        self.anchors.clear()
        version = data.get("version", "1.0")

        # Restore coordinate projection
        if "projection" in data:
            self.converter = CoordinateConverter.from_metadata(data["projection"])
        elif "reference_gps" in data and data["reference_gps"] is not None:
            self.converter = CoordinateConverter("UTM", tuple(data["reference_gps"]))
        else:
            logger.warning(
                "No projection metadata found in calibration file. Defaulting to WEB_MERCATOR fallback."
            )
            self.converter = CoordinateConverter("WEB_MERCATOR")

        # Load anchors
        if version == "1.0" and "affine_matrix" in data and "calib_frame_id" in data:
            anchor = AnchorCalibration(
                frame_id=int(data.get("calib_frame_id", 0)),
                affine_matrix=np.array(data["affine_matrix"], dtype=np.float64),
            )
            self.anchors.append(anchor)
        elif "anchors" in data:
            for item in data["anchors"]:
                self.anchors.append(AnchorCalibration.from_dict(item))

        # Restore frame dimensions
        fs = data.get("frame_size")
        if fs and len(fs) == 2 and int(fs[0]) > 0 and int(fs[1]) > 0:
            self._frame_size = (int(fs[0]), int(fs[1]))

        self.anchors.sort(key=lambda a: a.frame_id)
        self._rebuild_interpolators()
        logger.success(f"Loaded {len(self.anchors)} anchors (file version: {version})")
