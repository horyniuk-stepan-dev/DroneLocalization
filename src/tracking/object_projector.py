import numpy as np
from dataclasses import dataclass
from src.geometry.transformations import GeometryTransforms
from src.tracking.object_tracker import TrackedObject

@dataclass
class ObjectGPS:
    track_id: int
    class_name: str
    lat: float
    lon: float
    confidence: float


class ObjectProjector:
    """Проєктує піксельні координати об'єктів у GPS через наявні H та affine матриці."""
    
    def __init__(self, calibration_manager):
        self.calibration_manager = calibration_manager
        
    def _apply_rotation(self, px_x: float, px_y: float, angle: int, frame_w: int, frame_h: int) -> tuple[float, float]:
        """Обертає координати відповідно до повороту кадру (0, 90, 180, 270)."""
        if angle == 0:
            return px_x, px_y
        elif angle == 90:
            return px_y, frame_w - 1 - px_x
        elif angle == 180:
            return frame_w - 1 - px_x, frame_h - 1 - px_y
        elif angle == 270:
            return frame_h - 1 - px_y, px_x
        return px_x, px_y

    def project_objects(
        self,
        objects: list[TrackedObject],
        H: np.ndarray,          # Homography query->ref
        affine: np.ndarray,     # Affine ref->metric
        rotation_angle: int,    # Кут обертання кадру
        frame_w: int,
        frame_h: int
    ) -> list[ObjectGPS]:
        """Трансформує центри bbox: Query px -> Ref px (H) -> Metric (Affine) -> GPS."""
        
        if not objects or H is None or affine is None:
            return []
        
        if not self.calibration_manager.converter.is_initialized:
            return []
            
        objects_gps = []
        
        for obj in objects:
            px_x, px_y = obj.center_px
            
            # 1. Враховуємо обертання кадру
            rx, ry = self._apply_rotation(px_x, px_y, rotation_angle, frame_w, frame_h)
            
            # 2. Query pixels -> Reference pixels (Homography)
            pt_query = np.array([[rx, ry]], dtype=np.float64)
            try:
                pt_ref = GeometryTransforms.apply_homography(pt_query, H)
                
                # 3. Reference pixels -> Metric (Affine calibration)
                pt_metric = GeometryTransforms.apply_affine(pt_ref, affine)
                
                # 4. Metric -> GPS (WGS84)
                lat, lon = self.calibration_manager.converter.metric_to_gps(
                    float(pt_metric[0, 0]), float(pt_metric[0, 1])
                )
                
                objects_gps.append(ObjectGPS(
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    lat=lat,
                    lon=lon,
                    confidence=obj.confidence
                ))
            except Exception as e:
                # В разі виродженої матриці або інших помилок математики
                continue
                
        return objects_gps
