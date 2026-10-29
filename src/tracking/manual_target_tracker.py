"""Single-object trackers for objects the operator outlines on the video.

Unlike ObjectTracker (YOLO + ByteTrack, fixed COCO classes) this follows any
user-drawn box with an OpenCV visual tracker (CSRT -> KCF -> MIL fallback).
When the tracker loses the target it is re-acquired by template matching in a
growing window around the last known box.

Frames are processed at most at ``max_side`` px on the long side; boxes in and
out are always in ORIGINAL frame pixels (x1, y1, x2, y2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_FACTORIES = {
    "csrt": ("TrackerCSRT_create", "TrackerCSRT"),
    "kcf": ("TrackerKCF_create", "TrackerKCF"),
    "mil": ("TrackerMIL_create", "TrackerMIL"),
}


def create_cv_tracker(kind: str):
    """Create an OpenCV tracker, falling back csrt -> kcf -> mil."""
    order = [kind] + [k for k in ("csrt", "kcf", "mil") if k != kind]
    for name in order:
        fn_name, cls_name = _FACTORIES.get(name, (None, None))
        if fn_name is None:
            continue
        for mod in (cv2, getattr(cv2, "legacy", None)):
            if mod is None:
                continue
            fn = getattr(mod, fn_name, None)
            if fn is None:
                cls = getattr(mod, cls_name, None)
                fn = getattr(cls, "create", None) if cls is not None else None
            if fn is None:
                continue
            try:
                return fn(), name
            except Exception:  # noqa: BLE001 — try the next backend
                continue
    raise RuntimeError("No OpenCV visual tracker available (csrt/kcf/mil)")


@dataclass
class ManualTarget:
    target_id: int
    bbox: tuple[float, float, float, float]  # original px, x1 y1 x2 y2
    lost: bool = False
    lost_frames: int = 0
    reason: str = ""  # why lost: out_of_frame | drift | tracker
    lat: float | None = None
    lon: float | None = None
    # internal
    _tracker: object = field(default=None, repr=False)
    _template: np.ndarray | None = field(default=None, repr=False)
    _frames_since_template: int = 0
    _low_ncc_frames: int = 0

    @property
    def center_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        return {
            "id": self.target_id,
            "bbox": [float(v) for v in self.bbox],
            "lost": self.lost,
            "reason": self.reason,
            "lat": self.lat,
            "lon": self.lon,
        }


class ManualTargetTracker:
    def __init__(
        self,
        tracker_type: str = "csrt",
        max_side: int = 960,
        reacquire_ncc: float = 0.6,
        template_refresh_frames: int = 30,
        min_box_px: int = 6,
        drift_ncc: float = 0.3,
        drift_frames: int = 3,
        edge_margin_px: int = 3,
    ):
        self.drift_ncc = float(drift_ncc)
        self.drift_frames = max(int(drift_frames), 1)
        self.edge_margin_px = int(edge_margin_px)
        self.tracker_type = str(tracker_type).lower()
        self.max_side = max(int(max_side), 64)
        self.reacquire_ncc = float(reacquire_ncc)
        self.template_refresh_frames = max(int(template_refresh_frames), 1)
        self.min_box_px = int(min_box_px)
        self._targets: dict[int, ManualTarget] = {}
        self._next_id = 1
        self._backend = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _scale(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        return min(1.0, self.max_side / float(max(h, w)))

    def _small(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        s = self._scale(frame)
        if s < 1.0:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return frame, s

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def _init_tracker(self, t: ManualTarget, small: np.ndarray, s: float) -> bool:
        x1, y1, x2, y2 = (v * s for v in t.bbox)
        box = (
            int(round(x1)),
            int(round(y1)),
            max(int(round(x2 - x1)), 2),
            max(int(round(y2 - y1)), 2),
        )
        tracker, backend = create_cv_tracker(self.tracker_type)
        if self._backend != backend:
            self._backend = backend
            logger.info(f"Manual target tracker backend: {backend}")
        try:
            res = tracker.init(small, box)
        except cv2.error as e:
            logger.warning(f"Manual tracker init failed: {e}")
            return False
        if res is False:
            return False
        t._tracker = tracker
        gray = self._gray(small)
        bx, by, bw, bh = box
        tpl = gray[max(by, 0) : by + bh, max(bx, 0) : bx + bw]
        t._template = tpl.copy() if tpl.size else None
        t._frames_since_template = 0
        return True

    # ── public API ───────────────────────────────────────────────────────────
    @property
    def targets(self) -> list[ManualTarget]:
        return list(self._targets.values())

    def add(self, frame_bgr: np.ndarray, bbox_xyxy) -> int | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
        x1, x2 = sorted((max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))))
        y1, y2 = sorted((max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))))
        if x2 - x1 < self.min_box_px or y2 - y1 < self.min_box_px:
            return None
        t = ManualTarget(target_id=self._next_id, bbox=(x1, y1, x2, y2))
        small, s = self._small(frame_bgr)
        if not self._init_tracker(t, small, s):
            return None
        self._targets[t.target_id] = t
        self._next_id += 1
        return t.target_id

    def remove(self, target_id: int) -> None:
        self._targets.pop(int(target_id), None)

    def clear(self) -> None:
        self._targets.clear()

    def update(self, frame_bgr: np.ndarray) -> list[ManualTarget]:
        if not self._targets:
            return []
        small, s = self._small(frame_bgr)
        sh, sw = small.shape[:2]
        gray = None
        for t in self._targets.values():
            ok, box = False, None
            if t._tracker is not None:
                try:
                    ok, box = t._tracker.update(small)
                except cv2.error:
                    ok = False
            reason = "tracker"
            if ok:
                bx, by, bw, bh = box
                ok = bw >= 2 and bh >= 2
            if ok and self._touches_edge(bx, by, bw, bh, sw, sh):
                # CSRT clamps a leaving target to the border and then rides
                # along with the camera, so border contact means "left frame".
                ok, reason = False, "out_of_frame"
            if ok:
                if gray is None:
                    gray = self._gray(small)
                if self._ncc_to_template(t, gray, box) < self.drift_ncc:
                    t._low_ncc_frames += 1
                    if t._low_ncc_frames >= self.drift_frames:
                        ok, reason = False, "drift"
                else:
                    t._low_ncc_frames = 0
            if ok:
                t.bbox = (bx / s, by / s, (bx + bw) / s, (by + bh) / s)
                t.lost, t.lost_frames, t.reason = False, 0, ""
                t._frames_since_template += 1
                if t._frames_since_template >= self.template_refresh_frames:
                    self._refresh_template(t, gray, box)
                continue
            # Lost: stop the visual tracker (no edge riding), keep the last good
            # box, re-acquire only by a template match fully inside the frame.
            if not t.lost:
                t.reason = reason
            t.lost = True
            t._tracker = None
            t._low_ncc_frames = 0
            t.lost_frames += 1
            if gray is None:
                gray = self._gray(small)
            if self._reacquire(t, gray, s):
                self._init_tracker(t, small, s)
                t.lost, t.lost_frames, t.reason = False, 0, ""
        return self.targets

    def _touches_edge(self, bx, by, bw, bh, sw, sh) -> bool:
        m = self.edge_margin_px
        return bx <= m or by <= m or bx + bw >= sw - m or by + bh >= sh - m

    @staticmethod
    def _ncc_to_template(t: ManualTarget, gray: np.ndarray, box) -> float:
        if t._template is None or t._template.size == 0:
            return 1.0
        bx, by, bw, bh = (int(round(v)) for v in box)
        patch = gray[max(by, 0) : by + bh, max(bx, 0) : bx + bw]
        if patch.size == 0:
            return 0.0
        th, tw = t._template.shape[:2]
        if patch.shape != (th, tw):
            patch = cv2.resize(patch, (tw, th))
        return float(cv2.matchTemplate(patch, t._template, cv2.TM_CCOEFF_NORMED).max())

    def _refresh_template(self, t: ManualTarget, gray: np.ndarray, box) -> None:
        bx, by, bw, bh = (int(round(v)) for v in box)
        patch = gray[max(by, 0) : by + bh, max(bx, 0) : bx + bw]
        if patch.size == 0 or t._template is None:
            return
        # refresh only while the box still looks like the target (no drift lock-in)
        if patch.shape != t._template.shape:
            patch_r = cv2.resize(patch, (t._template.shape[1], t._template.shape[0]))
        else:
            patch_r = patch
        ncc = float(cv2.matchTemplate(patch_r, t._template, cv2.TM_CCOEFF_NORMED).max())
        if ncc >= self.reacquire_ncc:
            t._template = patch.copy()
        t._frames_since_template = 0

    def _reacquire(self, t: ManualTarget, gray: np.ndarray, s: float) -> bool:
        tpl = t._template
        if tpl is None or tpl.size == 0:
            return False
        th, tw = tpl.shape[:2]
        cx, cy = (v * s for v in t.center_px)
        grow = min(2.5 + 0.1 * t.lost_frames, 8.0)
        half_w, half_h = tw * grow / 2.0, th * grow / 2.0
        gh, gw = gray.shape[:2]
        x0, y0 = int(max(cx - half_w, 0)), int(max(cy - half_h, 0))
        x1, y1 = int(min(cx + half_w, gw)), int(min(cy + half_h, gh))
        if x1 - x0 <= tw or y1 - y0 <= th:
            return False
        res = cv2.matchTemplate(gray[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < self.reacquire_ncc:
            return False
        nx, ny = x0 + max_loc[0], y0 + max_loc[1]
        if self._touches_edge(nx, ny, tw, th, gw, gh):
            return False  # a partial view at the border is not a reliable match
        t.bbox = (nx / s, ny / s, (nx + tw) / s, (ny + th) / s)
        return True
