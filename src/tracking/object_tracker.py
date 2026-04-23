import numpy as np
from dataclasses import dataclass

try:
    import supervision as sv
except ImportError:
    sv = None


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    center_px: tuple[float, float]  # (x, y)


class ObjectTracker:
    """Обгортка над ByteTrack для трекінгу об'єктів між кадрами."""

    def __init__(self, config: dict):
        self.config = config
        
        if sv is None:
            raise ImportError("Package 'supervision' is required for ObjectTracker. Run 'pip install supervision'")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.config.get("track_activation_threshold", 0.25),
            lost_track_buffer=self.config.get("lost_track_buffer", 30),
            minimum_matching_threshold=self.config.get("minimum_matching_threshold", 0.8),
            frame_rate=30,  # Default
        )
        
        # COCO class names matching YOLO
        self._class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light", 10: "fire hydrant",
            11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse",
            18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
            22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
            26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
            34: "baseball bat", 35: "baseball glove", 36: "skateboard",
            37: "surfboard", 38: "tennis racket", 39: "bottle",
            40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
            48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
            52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
            60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
            68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
            72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
            76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }

    def update(self, detections: list[dict], frame_shape: tuple) -> list[TrackedObject]:
        """Оновити трекер новими детекціями. Повертає список відстежених об'єктів.
        detections: [{"class_id": int, "confidence": float, "bbox": [x1, y1, x2, y2]}, ...]
        """
        tracked_objects = []
        
        if not detections:
            # supervision requires sv.Detections object even if empty to update track states
            sv_detections = sv.Detections.empty()
        else:
            # Convert list of dicts to sv.Detections
            bboxes = []
            confidences = []
            class_ids = []
            
            for d in detections:
                bboxes.append(d["bbox"])
                confidences.append(d["confidence"])
                class_ids.append(d["class_id"])
            
            sv_detections = sv.Detections(
                xyxy=np.array(bboxes, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )

        # Update tracker
        tracked_sv_detections = self.tracker.update_with_detections(sv_detections)
        
        # Convert back to TrackedObject
        if tracked_sv_detections is not None and len(tracked_sv_detections) > 0:
            for i in range(len(tracked_sv_detections)):
                xyxy = tracked_sv_detections.xyxy[i].tolist()
                class_id = tracked_sv_detections.class_id[i]
                confidence = tracked_sv_detections.confidence[i]
                track_id = tracked_sv_detections.tracker_id[i]
                
                class_name = self._class_names.get(class_id, f"Class {class_id}")
                
                center_x = (xyxy[0] + xyxy[2]) / 2.0
                center_y = (xyxy[1] + xyxy[3]) / 2.0
                
                tracked_objects.append(TrackedObject(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=class_name,
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=float(confidence),
                    center_px=(float(center_x), float(center_y))
                ))
                
        return tracked_objects

    def reset(self):
        """Скинути стан трекера (при новій сесії)."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.config.get("track_activation_threshold", 0.25),
            lost_track_buffer=self.config.get("lost_track_buffer", 30),
            minimum_matching_threshold=self.config.get("minimum_matching_threshold", 0.8),
            frame_rate=30,
        )
