"""Localize probe frames of a local (no-GPS) layer against geographic layers.

Runs in a QThread: reads the probe frames from the layer's source video
(DB slot ``i`` = video frame ``i * frame_step``) and localizes each one
independently (session reset per frame, so no temporal prior links probes
that are far apart in time) against every reference layer and keeps the
strongest verified fix. Emits one record per probe; the consensus and
the operator review happen in the GUI thread.
"""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AutoGeorefWorker(QThread):
    progress = pyqtSignal(int, str)  # percent, message
    completed = pyqtSignal(list)  # list[dict] — one record per probe
    failed = pyqtSignal(str)

    def __init__(self, localizers, video_path: str, frame_ids, frame_step: int, parent=None):
        """``localizers``: list of (source_id, Localizer), one per reference layer."""
        super().__init__(parent)
        self.localizers = list(localizers)
        self.video_path = str(video_path)
        self.frame_ids = [int(f) for f in frame_ids]
        self.frame_step = max(int(frame_step), 1)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.failed.emit(f"Не вдалося відкрити відео: {self.video_path}")
            return
        records = []
        n = len(self.frame_ids)
        try:
            for k, fid in enumerate(self.frame_ids):
                if self._stop:
                    break
                self.progress.emit(int(100 * k / max(n, 1)), f"Кадр {fid} ({k + 1}/{n})")
                rec = {"frame_id": fid, "success": False}
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid * self.frame_step)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    rec["error"] = "кадр не прочитано з відео"
                    records.append(rec)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                best, errors, hits = None, [], []
                for sid, localizer in self.localizers:
                    if self._stop:
                        break
                    try:
                        localizer.reset_session()
                        res = localizer.localize_frame(rgb)
                    except Exception as e:  # one bad frame/layer must not stop the run
                        logger.warning(f"Auto-georef: frame {fid} vs {sid} raised: {e}")
                        errors.append(f"{sid}: {e}")
                        continue
                    ok = (
                        bool(res.get("success"))
                        and not res.get("fallback_mode")
                        and "lat" in res
                        and "lon" in res
                    )
                    if not ok:
                        errors.append(f"{sid}: {res.get('error', 'не знайдено')}")
                        continue
                    hits.append(sid)
                    key = (float(res.get("confidence") or 0.0), int(res.get("inliers") or 0))
                    if best is None or key > best[0]:
                        best = (key, sid, res)
                if best is None:
                    rec["error"] = "; ".join(errors)
                else:
                    _, sid, res = best
                    rec.update(
                        success=True,
                        lat=res["lat"],
                        lon=res["lon"],
                        confidence=float(res.get("confidence") or 0.0),
                        inliers=int(res.get("inliers") or 0),
                        matched_frame=res.get("matched_frame"),
                        source_id=sid,
                        layers_hit=hits,
                    )
                records.append(rec)
        finally:
            cap.release()
        self.progress.emit(100, "Готово")
        self.completed.emit(records)
