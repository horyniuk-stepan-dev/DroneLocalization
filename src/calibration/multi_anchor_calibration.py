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
        self.affine_matrix = affine_matrix  # (2, 3) піксель→метрика для цього кадру

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
    """
    GPS калібрування з декількома якорями.

    Кожен якір — це один кадр з набором точок пікселів→GPS,
    з якого будується своя affine матриця.

    При трансформації координат довільного кадру використовується
    зважений блендінг між двома найближчими якорями:

        GPS(frame_i) = w_left  * GPS_via_anchor_left(frame_i)
                     + w_right * GPS_via_anchor_right(frame_i)

    де w = 1 / distance (обернена відстань до якоря по номеру кадру).
    """

    def __init__(self):
        self.anchors: list[AnchorCalibration] = []
        self.is_calibrated = False
        logger.info("MultiAnchorCalibration initialized")

    @property
    def anchor_frame_ids(self) -> list[int]:
        return [a.frame_id for a in self.anchors]

    def add_anchor(self, frame_id: int, points_2d: list, points_gps: list) -> dict:
        """
        Додати або оновити якір для конкретного кадру.
        Обчислює affine матрицю з пікселів→GPS для цього кадру.
        """
        logger.info(f"Adding anchor at frame {frame_id} with {len(points_2d)} points")

        if len(points_2d) < 3:
            raise ValueError("Потрібно мінімум 3 точки для калібрування")

        pts_2d_np = np.array(points_2d, dtype=np.float32)
        pts_metric = []

        for lat, lon in points_gps:
            x, y = CoordinateConverter.gps_to_metric(lat, lon)
            pts_metric.append((x, y))

        pts_metric_np = np.array(pts_metric, dtype=np.float32)
        M, inliers = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)

        if M is None:
            raise ValueError(f"Не вдалося обчислити affine для кадру {frame_id}")

        # RMSE
        transformed = GeometryTransforms.apply_affine(pts_2d_np, M)
        errors = np.linalg.norm(pts_metric_np - transformed, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        inliers_count = int(np.sum(inliers)) if inliers is not None else len(points_2d)

        # Оновлюємо або додаємо якір
        existing = next((a for a in self.anchors if a.frame_id == frame_id), None)
        if existing:
            existing.affine_matrix = M
            logger.info(f"Updated anchor at frame {frame_id}: RMSE={rmse:.4f}m")
        else:
            self.anchors.append(AnchorCalibration(frame_id, M))
            # Тримаємо відсортованими за frame_id
            self.anchors.sort(key=lambda a: a.frame_id)
            logger.info(f"Added anchor at frame {frame_id}: RMSE={rmse:.4f}m")

        self.is_calibrated = True

        return {
            "status": "success",
            "frame_id": frame_id,
            "rmse_meters": rmse,
            "inliers_count": inliers_count,
            "total_anchors": len(self.anchors)
        }

    def get_surrounding_anchors(self, frame_id: int) -> tuple:
        """
        Знайти два якорі що оточують frame_id (лівий і правий).
        Повертає (left_anchor, right_anchor).
        Якщо frame_id поза діапазоном — повертає найближчий з обох боків.
        """
        if not self.anchors:
            return None, None

        # Якщо лише один якір — він і лівий, і правий
        if len(self.anchors) == 1:
            return self.anchors[0], self.anchors[0]

        sorted_anchors = self.anchors  # вже відсортовані

        # frame_id ліворуч від всіх якорів
        if frame_id <= sorted_anchors[0].frame_id:
            return sorted_anchors[0], sorted_anchors[0]

        # frame_id праворуч від всіх якорів
        if frame_id >= sorted_anchors[-1].frame_id:
            return sorted_anchors[-1], sorted_anchors[-1]

        # Знаходимо між якими якорями знаходиться frame_id
        for i in range(len(sorted_anchors) - 1):
            if sorted_anchors[i].frame_id <= frame_id <= sorted_anchors[i + 1].frame_id:
                return sorted_anchors[i], sorted_anchors[i + 1]

        return sorted_anchors[-1], sorted_anchors[-1]

    def blend_metric(
        self,
        frame_id: int,
        metric_via_left: tuple,
        metric_via_right: tuple,
        left_anchor: "AnchorCalibration",
        right_anchor: "AnchorCalibration"
    ) -> tuple:
        """
        Зважений блендінг метричних координат між двома якорями.
        Вага пропорційна до оберненої відстані по номеру кадру.
        """
        if left_anchor.frame_id == right_anchor.frame_id:
            return metric_via_left

        d_left = abs(frame_id - left_anchor.frame_id)
        d_right = abs(frame_id - right_anchor.frame_id)
        total = d_left + d_right

        if total == 0:
            return metric_via_left

        # Чим ближче до якоря — тим більша вага
        w_left = d_right / total   # якщо frame близько до left: d_left малий → w_left великий
        w_right = d_left / total

        blended_x = w_left * metric_via_left[0] + w_right * metric_via_right[0]
        blended_y = w_left * metric_via_left[1] + w_right * metric_via_right[1]

        return blended_x, blended_y

    # ------------------------------------------------------------------
    # Сумісність зі старим GPSCalibration API (для localizer/main_window)
    # ------------------------------------------------------------------

    @property
    def calib_frame_id(self) -> int | None:
        """Повертає frame_id першого якоря (для сумісності)"""
        return self.anchors[0].frame_id if self.anchors else None

    def pixel_to_metric(self, x_2d: float, y_2d: float,
                        frame_id: int = None,
                        anchor: "AnchorCalibration" = None) -> tuple:
        """
        Трансформація пікселя calib-кадру → метрика.
        anchor або frame_id вказує який якір використовувати.
        """
        if not self.is_calibrated:
            raise RuntimeError("Калібрування не виконано")

        if anchor is not None:
            return anchor.pixel_to_metric(x_2d, y_2d)

        if frame_id is not None:
            a = next((a for a in self.anchors if a.frame_id == frame_id), None)
            if a:
                return a.pixel_to_metric(x_2d, y_2d)

        # Fallback: перший якір
        return self.anchors[0].pixel_to_metric(x_2d, y_2d)

    def transform_to_gps(self, x_2d: float, y_2d: float,
                         frame_id: int = None,
                         anchor: "AnchorCalibration" = None) -> tuple:
        """Піксель → GPS через вказаний або перший якір"""
        mx, my = self.pixel_to_metric(x_2d, y_2d, frame_id=frame_id, anchor=anchor)
        return CoordinateConverter.metric_to_gps(mx, my)

    def save(self, path: str):
        """Зберегти всі якорі у JSON"""
        if not self.is_calibrated:
            raise RuntimeError("Немає даних для збереження")

        data = {
            "anchors": [a.to_dict() for a in self.anchors]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.success(f"MultiAnchorCalibration saved: {path} ({len(self.anchors)} anchors)")

    def load(self, path: str):
        """Завантажити якорі з JSON"""
        logger.info(f"Loading MultiAnchorCalibration from: {path}")
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Підтримка старого формату (один якір)
            if "affine_matrix" in data and "calib_frame_id" in data:
                logger.warning("Old single-anchor format detected, converting...")
                anchor = AnchorCalibration(
                    frame_id=int(data.get("calib_frame_id", 0)),
                    affine_matrix=np.array(data["affine_matrix"], dtype=np.float32)
                )
                self.anchors = [anchor]
            else:
                self.anchors = [AnchorCalibration.from_dict(a) for a in data["anchors"]]
                self.anchors.sort(key=lambda a: a.frame_id)

            self.is_calibrated = True
            logger.success(
                f"Loaded {len(self.anchors)} anchors: "
                f"frames {self.anchor_frame_ids}"
            )
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            raise