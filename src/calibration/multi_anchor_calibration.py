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
    compose_affine_5dof as _compose_affine_5dof,
)
from src.geometry.affine_utils import (
    decompose_affine_5dof as _decompose_affine_5dof,
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
            affine_matrix=np.array(data["affine_matrix"], dtype=np.float64),
            qa_data=qa,
        )


class MultiAnchorCalibration:
    """Менеджер декількох якорів калібрування з підтримкою версіонування та проєкцій"""

    VERSION: str = "2.3"

    def __init__(self, converter: CoordinateConverter | None = None) -> None:
        self.anchors: list[AnchorCalibration] = []
        self.converter = converter or CoordinateConverter("WEB_MERCATOR")
        self._interp: PchipInterpolator | None = None  # кешований інтерполятор
        self._interp_sign: float = -1.0  # знак det якірних матриць (Y-flip)
        self._interp_range: tuple[float, float] | None = None  # [перший, останній] якір
        self._ref_px: tuple[float, float] = (0.0, 0.0)  # опорний піксель декомпозиції
        self._frame_size: tuple[int, int] | None = None  # (width, height) кадру

    def set_frame_size(self, width: int, height: int) -> None:
        """Розмір кадру: інтерполяція параметризується навколо центру кадру."""
        new_size = (int(width), int(height))
        if new_size != self._frame_size and new_size[0] > 0 and new_size[1] > 0:
            self._frame_size = new_size
            self._rebuild_interpolators()

    def _reference_pixel(self) -> tuple[float, float]:
        """Опорний піксель для декомпозиції (центр кадру або центроїд точок)."""
        if self._frame_size:
            return self._frame_size[0] / 2.0, self._frame_size[1] / 2.0
        # Fallback: центроїд точок усіх якорів — стабільна точка всередині кадру
        pts = [p for a in self.anchors for p in (a.points_2d or [])]
        if pts:
            arr = np.asarray(pts, dtype=np.float64)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
        return 0.0, 0.0

    def _rebuild_interpolators(self) -> None:
        """
        Перебудовує PCHIP-інтерполятор на основі 5-DoF декомпозиції якірних матриць.

        ВИПРАВЛЕНО (критичний баг): попередня 4-DoF декомпозиція (tx, ty, scale,
        angle) не кодувала віддзеркалення осі Y, а якірні матриці pixel→metric
        ЗАВЖДИ мають det < 0 (піксельна вісь Y ↓, метрична ↑). Реконструйована
        матриця виходила з det > 0 — Y-складова дзеркалилась, і позиції між
        якорями їхали на десятки метрів, тоді як точно на якорі результат був
        правильним → різкі стрибки біля якорів.

        Тепер: інтерполюються (rx, ry, sx, sy, angle), де (rx, ry) — метрична
        позиція ОПОРНОГО ПІКСЕЛЯ (центр кадру), а глобальний знак det
        зберігається окремо і відновлюється при композиції. Параметризація
        навколо центру кадру (а не пікселя (0,0)) прибирає "гойдання" центру
        при зміні кута між якорями.
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
        self._interp_sign = -1.0 if n_neg * 2 >= len(dets) else 1.0

        cx, cy = self._reference_pixel()
        self._ref_px = (cx, cy)

        ids = np.array([a.frame_id for a in self.anchors], dtype=np.float64)
        comps = np.zeros((len(self.anchors), 5), dtype=np.float64)
        for i, a in enumerate(self.anchors):
            M = a.affine_matrix
            _, _, sx, sy, angle = _decompose_affine_5dof(M)
            # Метрична позиція опорного пікселя — інтерполюємо саме її
            rx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
            ry = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
            comps[i] = (rx, ry, sx, sy, angle)

        # Розгортаємо кути для коректної інтерполяції через межу ±π
        comps[:, 4] = _unwrap_angles(comps[:, 4])

        # extrapolate=False + clamp у _get_interpolated_matrix:
        # кубічна екстраполяція за межами діапазону якорів розліталась.
        self._interp = PchipInterpolator(ids, comps, extrapolate=False)
        self._interp_range = (float(ids[0]), float(ids[-1]))

    def _get_interpolated_matrix(self, frame_id: float) -> np.ndarray | None:
        """Повертає інтерпольовану афінну матрицю 2x3 для заданого frame_id."""
        if self._interp is None or self._interp_range is None:
            return None
        # Clamp: за межами діапазону якорів використовуємо крайній якір
        fid = float(np.clip(frame_id, self._interp_range[0], self._interp_range[1]))
        components = self._interp(fid)  # (5,): rx, ry, sx, sy, angle
        if components is None or np.any(np.isnan(components)):
            return None
        rx, ry, sx, sy, angle = components
        # Захист від вироджених значень масштабу
        sx = float(np.clip(sx, 1e-6, 1e6))
        sy = float(np.clip(sy, 1e-6, 1e6))
        M = _compose_affine_5dof(0.0, 0.0, sx, sy, float(angle), sign=self._interp_sign)
        # Трансляція: опорний піксель має відобразитись точно у (rx, ry)
        cx, cy = self._ref_px
        M[0, 2] = float(rx) - (M[0, 0] * cx + M[0, 1] * cy)
        M[1, 2] = float(ry) - (M[1, 0] * cx + M[1, 1] * cy)
        return M

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
        """Очищає всі якорі та скидає стан калібрування."""
        self.anchors.clear()
        self._interp = None
        self._interp_range = None
        from src.geometry.coordinates import CoordinateConverter
        self.converter = CoordinateConverter("WEB_MERCATOR")
        logger.info("Cleared all anchors and reset calibration state.")

    def get_metric_position(self, frame_id: int, x: float, y: float) -> tuple[float, float] | None:
        if not self.is_calibrated:
            return None

        # Якщо якір один — екстраполяція неможлива, повертаємо його координати
        if len(self.anchors) == 1:
            return self.anchors[0].pixel_to_metric(x, y)

        # Phase 1.2: Перевірка чи frame_id = один із якорів → reset drift
        exact_anchor = self.get_anchor(frame_id)
        if exact_anchor is not None:
            # Точне потрапляння на якір = скидаємо накопичений drift
            logger.debug(f"Exact anchor hit at frame {frame_id} — using direct affine (drift reset)")
            return exact_anchor.pixel_to_metric(x, y)

        # Decomposition-based PCHIP: інтерполяція через tx/ty/scale/angle
        if self._interp is not None:
            M = self._get_interpolated_matrix(float(frame_id))
            if M is not None:
                pt = np.array([[x, y]], dtype=np.float64)
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

    def get_metric_position_with_depth(
        self,
        frame_id: int,
        x: float,
        y: float,
        depth_scale: float = 1.0,
    ) -> tuple[float, float] | None:
        """Версія get_metric_position з корекцією масштабу через depth.

        depth_scale — відносний масштаб з DepthEstimator.get_relative_scale().
        При depth_scale > 1: об'єкт ближче (нижча висота) → більший pixel scale.
        При depth_scale < 1: об'єкт далі (вища висота) → менший pixel scale.
        """
        result = self.get_metric_position(frame_id, x, y)
        if result is None:
            return None

        mx, my = result

        # Нормалізуємо depth_scale відносно reference (медіана всіх якорів).
        ref_depth = getattr(self, '_reference_depth_scale', 1.0)
        if ref_depth > 1e-6:
            correction = depth_scale / ref_depth
            # Обмежуємо корекцію: максимум 2x в будь-який бік
            correction = float(np.clip(correction, 0.5, 2.0))

            # TODO: У майбутньому тут можна додати зміщення відносно оптичного центру
            # для більш точної компенсації паралаксу. Поки що — логуємо.
            if abs(correction - 1.0) > 0.05:
                logger.debug(
                    f"Depth scale correction: ref={ref_depth:.3f}, "
                    f"current={depth_scale:.3f}, ratio={correction:.3f}"
                )

        return mx, my

    def set_reference_depth_scale(self, depth_scale: float) -> None:
        """Встановлює референсний depth_scale (зі збудови БД)."""
        self._reference_depth_scale = float(depth_scale)
        logger.info(f"Reference depth scale set: {depth_scale:.4f}")

    def set_gsd_calculator(self, gsd_calculator) -> None:
        """Встановлює калькулятор GSD для прив'язки до фізичного масштабу."""
        self._gsd = gsd_calculator
        if self._gsd:
            logger.info(f"GSD Calculator linked: {self._gsd.gsd_m_per_px*100:.2f} cm/px")

    def save(self, path: str) -> None:
        """Збереження якорів та метаданих проєкції у JSON."""
        data = {
            "version": self.VERSION,
            "projection": self.converter.export_metadata(),
            "frame_size": list(self._frame_size) if self._frame_size else None,
            "anchors": [a.to_dict() for a in self.anchors],
        }

        # Атомарний запис: калібрування — критичні дані, обрізаний JSON
        # при краші/конкурентному збереженні означає втрату всіх якорів.
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
                affine_matrix=np.array(data["affine_matrix"], dtype=np.float64),
            )
            self.anchors.append(anchor)
        elif "anchors" in data:
            # Новій формат (список якорів)
            for item in data["anchors"]:
                self.anchors.append(AnchorCalibration.from_dict(item))

        # Відновлення розміру кадру (для параметризації навколо центру)
        fs = data.get("frame_size")
        if fs and len(fs) == 2 and int(fs[0]) > 0 and int(fs[1]) > 0:
            self._frame_size = (int(fs[0]), int(fs[1]))

        self.anchors.sort(key=lambda a: a.frame_id)
        self._rebuild_interpolators()
        logger.success(f"Loaded {len(self.anchors)} anchors (file version: {version})")
