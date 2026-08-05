import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import get_cfg
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RealtimeTrackingWorker(QThread):
    """Real-time localization worker thread (Optimized for XFeat + YOLO11)"""

    frame_ready = pyqtSignal(np.ndarray)
    location_found = pyqtSignal(float, float, float, int)
    # Precise keyframe localization signal (anchor_fix)
    anchor_fix = pyqtSignal()
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    fov_found = pyqtSignal(list)
    objects_detected = pyqtSignal(object)  # list[TrackedObject]
    objects_gps_updated = pyqtSignal(object)  # list[ObjectGPS]
    debug_view_ready = pyqtSignal(str, np.ndarray)  # (channel_name, BGR image)

    def __init__(self, video_source: str, localizer, model_manager=None, config=None):
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.model_manager = model_manager
        self.config = config or {}
        self._stop_event = threading.Event()

        # Keyframe interval for localization
        self.keyframe_interval = get_cfg(self.config, "tracking.keyframe_interval", 5)
        self.process_fps = get_cfg(
            self.config, "tracking.process_fps", 30.0 / self.keyframe_interval
        )
        self.tracking_config = get_cfg(self.config, "object_tracking", {})
        # Forward-backward optical flow check
        self.of_fb_check = get_cfg(self.config, "tracking.of_fb_check", False)
        self.of_fb_max_px = get_cfg(self.config, "tracking.of_fb_max_px", 2.0)
        self.of_stride = max(1, int(get_cfg(self.config, "tracking.of_stride", 1)))
        self.of_half_res = bool(get_cfg(self.config, "tracking.of_half_res", False))
        self.of_local_speed = bool(get_cfg(self.config, "tracking.of_local_speed", False))

        # ── Debug views ───────────────────────────────────────────────────────
        self._debug_lock = threading.Lock()
        self._debug_channels = set()
        self._debug_max_width = get_cfg(self.config, "debug_views.max_width", 640)
        self._debug_dino_pca = get_cfg(self.config, "debug_views.dino_pca_enabled", True)
        self._debug_inflight = {}  # {channel: monotonic emit time} — backpressure
        self._debug_inflight_stale_sec = 1.0

        # Latency Tracker when monitoring is enabled
        self._latency_tracker = None
        if get_cfg(self.config, "models.performance.log_latency_stats", False):
            from src.utils.latency_tracker import LatencyTracker

            self._latency_tracker = LatencyTracker(
                log_interval=get_cfg(self.config, "models.performance.latency_log_interval", 100),
                logger=logger,
            )

    def _models_to_pin(self) -> list[str]:
        """Model names to pin in VRAM during tracking."""
        local = str(get_cfg(self.config, "models.local_extractor", "aliked")).lower()
        if local not in ("rdd", "xfeat"):
            if local != "aliked":
                logger.warning(
                    f"models.local_extractor={local!r} is not supported — defaulting to ALIKED"
                )
            local = "aliked"
        if local == "xfeat":
            return ["xfeat", "dinov2"]
        return [local, "dinov2", f"lightglue_{local}"]

    def run(self):
        # Reset session state via public API
        if hasattr(self.localizer, "reset_session"):
            self.localizer.reset_session()

        # Debug views: fresh backpressure state for new session
        with self._debug_lock:
            self._debug_inflight.clear()

        if self.model_manager:
            # Pin active neural models in VRAM for selected local extractor
            self.model_manager.pin(self._models_to_pin())

        from src.tracking.object_projector import ObjectProjector
        from src.tracking.object_tracker import ObjectTracker

        object_tracker = None
        object_projector = None

        is_tracking_enabled = False
        if isinstance(self.tracking_config, dict):
            is_tracking_enabled = self.tracking_config.get("enabled", False)
        else:
            is_tracking_enabled = getattr(self.tracking_config, "enabled", False)

        if is_tracking_enabled:
            tracker_cfg = (
                self.tracking_config
                if isinstance(self.tracking_config, dict)
                else self.tracking_config.model_dump()
            )
            try:
                object_tracker = ObjectTracker(tracker_cfg)
                object_projector = ObjectProjector(self.localizer.calibration)
                logger.info("Object tracking enabled")
            except Exception as e:
                logger.error(f"Failed to initialize object tracking: {e}")

        # Fix 6: Pre-warm fallback models when starting tracking
        threading.Thread(target=self._prewarm_fallback_models, daemon=True).start()

        logger.info(f"Starting tracking from source: {self.video_source}")

        yolo_wrapper = None
        if self.model_manager:
            try:
                yolo_model = self.model_manager.load_yolo()
                yolo_wrapper = YOLOWrapper(yolo_model, self.model_manager.device)
                logger.success("YOLO loaded for dynamic object masking in tracking loop")
            except Exception as e:
                logger.error(
                    f"Failed to load YOLO for tracking: {e} | "
                    f"device={self.model_manager.device}. "
                    f"Dynamic object masking will be unavailable. "
                    f"Tracking cannot proceed without YOLO.",
                    exc_info=True,
                )
                self.error.emit(f"YOLO failed to load: {e}")
                return

        from src.video.video_source import VideoSource, VideoSourceConfig

        if isinstance(self.video_source, VideoSource):
            video_src = self.video_source
        else:
            v_config = VideoSourceConfig(source=str(self.video_source))
            video_src = VideoSource(v_config)

        if not video_src.is_opened:
            logger.error(
                f"Failed to open video source: {self.video_source}. "
                f"Check that the source is available."
            )
            self.error.emit(f"Failed to open video source: {self.video_source}")
            return

        video_fps = video_src.fps
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        frame_idx = 0
        prev_gray_for_of = None
        prev_gray_half_for_of = None
        prev_pts_for_of = None
        last_tracked_objects = []

        last_localization_video_time = -1.0
        last_keyframe_video_time = -1.0
        last_of_video_time = -1.0

        stream_start_time = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()

            ret, frame = video_src.read()
            if not ret:
                logger.info("End of video stream or connection lost.")
                self.status_update.emit("Video stream ended or connection lost.")
                break

            if video_src.is_live:
                current_video_time_sec = time.time() - stream_start_time
            else:
                # Video timestamp in seconds
                current_video_time_sec = video_src.pos_msec / 1000.0
                # Fallback: estimate from frame index if position is 0
                if current_video_time_sec <= 0:
                    current_video_time_sec = video_src.pos_frames * frame_duration_sec

            # 1. Always emit raw frame to GUI for smooth playback
            self.frame_ready.emit(frame)

            # S3-3: Optical Flow Pipeline
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = frame_idx % self.keyframe_interval == 0

            # dt calculation logic
            _is_of_frame = not (is_keyframe or prev_pts_for_of is None)
            if is_keyframe or prev_pts_for_of is None:
                if last_keyframe_video_time < 0:
                    calculated_dt = self.keyframe_interval * frame_duration_sec
                else:
                    calculated_dt = current_video_time_sec - last_keyframe_video_time
                    if calculated_dt <= 0:
                        calculated_dt = self.keyframe_interval * frame_duration_sec
            else:
                _dt_base = (
                    last_of_video_time if self.of_local_speed else last_localization_video_time
                )
                if _dt_base < 0:
                    calculated_dt = frame_duration_sec
                else:
                    calculated_dt = current_video_time_sec - _dt_base
                    if calculated_dt <= 0:
                        calculated_dt = frame_duration_sec

            _of_computed = _is_of_frame and (
                self.of_stride <= 1 or (frame_idx % self.of_stride) == 0
            )
            if _of_computed or is_keyframe:
                last_of_video_time = current_video_time_sec

            loc_result = {"success": False, "error": "Not processed"}
            start_process = time.time()

            if is_keyframe or prev_pts_for_of is None:
                # ====== HEAVY KEYFRAME LOCALIZATION ======
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                static_mask = None
                detections = []
                if yolo_wrapper:
                    static_mask, detections = yolo_wrapper.detect_and_mask(frame_rgb)

                # Debug views snapshot
                with self._debug_lock:
                    active_debug = set(self._debug_channels)
                debug_collector = None
                if active_debug & {"matches", "dino", "depth"}:
                    from src.localization.debug_collector import DebugCollector

                    debug_collector = DebugCollector(
                        want_matches="matches" in active_debug,
                        want_dino_pca=("dino" in active_debug) and self._debug_dino_pca,
                        want_depth="depth" in active_debug,
                    )

                try:
                    loc_result = self.localizer.localize_frame(
                        frame_rgb,
                        static_mask=static_mask,
                        dt=calculated_dt,
                        collector=debug_collector,
                    )
                except Exception as e:
                    import torch

                    torch.cuda.empty_cache()
                    logger.error(f"Localization exception on keyframe: {e}", exc_info=True)
                    loc_result = {"success": False, "error": str(e)}

                if active_debug:
                    self._render_debug(
                        active_debug, frame_rgb, detections, static_mask, debug_collector
                    )

                last_keyframe_video_time = current_video_time_sec

                if (
                    loc_result.get("success")
                    and loc_result.get("fallback_mode") != "retrieval_only"
                ):
                    prev_gray_for_of = curr_gray
                    prev_gray_half_for_of = (
                        cv2.resize(curr_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                        if self.of_half_res
                        else None
                    )
                    prev_pts_for_of = cv2.goodFeaturesToTrack(
                        curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, mask=None
                    )

                if object_tracker and detections is not None:
                    tracked_objects = object_tracker.update(detections, frame.shape)
                    last_tracked_objects = tracked_objects
                    self.objects_detected.emit(tracked_objects)
                    loc_state = getattr(self.localizer, "last_state", None)
                    if object_projector and loc_state:
                        H = loc_state.get("H")
                        affine = loc_state.get("affine")
                        angle = loc_state.get("global_angle", 0)

                        if H is not None and affine is not None:
                            scale = loc_state.get(
                                "scale", getattr(self.localizer, "_last_scale", 1.0)
                            )

                            from copy import copy as _shallow_copy

                            scaled_tracked_objects = []
                            for obj in tracked_objects:
                                s_obj = _shallow_copy(obj)
                                s_obj.center_px = (
                                    obj.center_px[0] * scale,
                                    obj.center_px[1] * scale,
                                )
                                s_obj.bbox = [c * scale for c in obj.bbox]
                                scaled_tracked_objects.append(s_obj)

                            objects_gps = object_projector.project_objects(
                                scaled_tracked_objects,
                                H,
                                affine,
                                angle,
                                int(frame.shape[1] * scale),
                                int(frame.shape[0] * scale),
                            )
                            if objects_gps:
                                obj_summary = ", ".join(
                                    [f"{obj.class_name} #{obj.track_id}" for obj in objects_gps]
                                )
                                logger.debug(
                                    f"Tracked {len(objects_gps)} objects (KF): {obj_summary}"
                                )
            else:
                # ====== OPTICAL FLOW TRACKING ======
                if self.of_stride > 1 and (frame_idx % self.of_stride) != 0:
                    if object_tracker:
                        self.objects_detected.emit(last_tracked_objects)
                elif prev_pts_for_of is not None and len(prev_pts_for_of) > 10:
                    of_scale = (
                        0.5 if (self.of_half_res and prev_gray_half_for_of is not None) else 1.0
                    )
                    if of_scale != 1.0:
                        g_prev = prev_gray_half_for_of
                        g_curr = cv2.resize(
                            curr_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                        )
                        pts_prev = np.ascontiguousarray(
                            prev_pts_for_of * of_scale, dtype=np.float32
                        )
                    else:
                        g_prev, g_curr, pts_prev = prev_gray_for_of, curr_gray, prev_pts_for_of

                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        g_prev,
                        g_curr,
                        pts_prev,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                    )
                    keep = status.reshape(-1) == 1

                    # Forward-backward check
                    if self.of_fb_check and keep.any():
                        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                            g_curr,
                            g_prev,
                            curr_pts,
                            None,
                            winSize=(15, 15),
                            maxLevel=2,
                        )
                        rt_err = np.linalg.norm(
                            back_pts.reshape(-1, 2) - pts_prev.reshape(-1, 2), axis=1
                        )
                        fb_ok = (back_status.reshape(-1) == 1) & (rt_err <= self.of_fb_max_px)
                        if int((keep & fb_ok).sum()) >= 10:
                            keep = keep & fb_ok

                    good_new = curr_pts.reshape(-1, 2)[keep]
                    good_old = pts_prev.reshape(-1, 2)[keep]
                    if of_scale != 1.0:
                        good_new = good_new / of_scale
                        good_old = good_old / of_scale

                    if len(good_new) > 10:
                        flow_vectors = good_new - good_old
                        dx_px, dy_px = np.median(flow_vectors, axis=0)

                        flow_affine = None
                        flow_quality = None
                        try:
                            S_of, of_mask = cv2.estimateAffinePartial2D(
                                good_old,
                                good_new,
                                method=cv2.RANSAC,
                                ransacReprojThreshold=3.0,
                            )
                            if S_of is not None and np.all(np.isfinite(S_of)):
                                flow_affine = S_of
                                if of_mask is not None and len(of_mask) > 0:
                                    inlier_ratio = float(of_mask.sum()) / len(of_mask)
                                    n_norm = min(1.0, len(good_new) / 120.0)
                                    flow_quality = inlier_ratio * n_norm
                        except cv2.error as e:
                            logger.debug(f"OF affine estimation failed: {e}")

                        try:
                            loc_result = self.localizer.localize_optical_flow(
                                dx_px,
                                dy_px,
                                dt=calculated_dt,
                                rot_width=frame.shape[1],
                                rot_height=frame.shape[0],
                                flow_affine=flow_affine,
                                flow_quality=flow_quality,
                            )
                        except Exception as e:
                            logger.error(f"OF Localization error: {e}")
                            loc_result = {"success": False, "error": str(e)}

                        if object_tracker:
                            self.objects_detected.emit(last_tracked_objects)
                    else:
                        prev_pts_for_of = None
                else:
                    prev_pts_for_of = None

            if loc_result.get("success") and loc_result.get("matched_frame", -1) != -1:
                self.location_found.emit(
                    loc_result["lat"],
                    loc_result["lon"],
                    loc_result["confidence"],
                    loc_result["inliers"],
                )
                if not loc_result.get("is_of"):
                    self.anchor_fix.emit()
                if loc_result.get("fov_polygon"):
                    self.fov_found.emit(loc_result["fov_polygon"])

                track_type = "OF" if loc_result.get("is_of") else "KF"
                method_txt = (
                    "Similarity"
                    if loc_result.get("fallback_mode") == "retrieval_only"
                    else "Inliers"
                )
                score = loc_result.get("global_score", loc_result["inliers"])

                self.status_update.emit(
                    f"[{track_type}] Found ({method_txt}: {score:.2f}, Frame: {loc_result['matched_frame']})"
                )

                last_localization_video_time = current_video_time_sec

                if hasattr(self.localizer, "db_manager") and self.localizer.db_manager is not None:
                    try:
                        self.localizer.db_manager.set_active_by_gps(
                            loc_result["lat"], loc_result["lon"]
                        )
                        self.localizer.db_manager.update_retriever_positions(
                            loc_result["lat"], loc_result["lon"]
                        )
                    except Exception as e:
                        logger.debug(f"set_active_by_gps failed: {e}")
            elif not loc_result.get("success") and loc_result.get("error") != "Not processed":
                self.status_update.emit(f"Lost: {loc_result.get('error', 'Unknown error')}")

            process_duration = time.time() - start_process
            if self._latency_tracker is not None:
                self._latency_tracker.record(process_duration)
            self.fps_updated.emit(1.0 / process_duration if process_duration > 0 else 0)

            frame_idx += 1

            if not video_src.is_live:
                elapsed_in_loop = time.time() - loop_start
                sleep_time = frame_duration_sec - elapsed_in_loop
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))

        video_src.release()
        logger.info("Tracking worker thread finished cleanly.")

    def _prewarm_fallback_models(self):
        """Pre-warms models via ModelManager."""
        try:
            if not self.model_manager:
                return
            logger.info("Tracking pre-warming centralized models...")
            self.model_manager.prewarm()
            logger.success("Tracking pre-warming successful")
        except Exception as e:
            logger.warning(
                f"Model pre-warming failed: {e}. "
                f"Models will be loaded on first use (slower first localization).",
                exc_info=True,
            )

    def set_debug_channels(self, channels) -> None:
        """GUI -> worker: active debug channels (thread-safe)."""
        with self._debug_lock:
            self._debug_channels = set(channels or [])

    def _render_debug(self, active, frame_rgb, detections, static_mask, collector) -> None:
        """Renders active debug channels and emits BGR frames to GUI."""
        from src.workers import debug_renderers as dr

        mw = self._debug_max_width

        def emit_if_free(channel, render_fn):
            now = time.monotonic()
            with self._debug_lock:
                ts = self._debug_inflight.get(channel)
                if ts is not None and (now - ts) < self._debug_inflight_stale_sec:
                    return
            try:
                img = render_fn()
            except Exception as e:
                logger.debug(f"{channel} debug render failed: {e}")
                return
            with self._debug_lock:
                self._debug_inflight[channel] = time.monotonic()
            self.debug_view_ready.emit(channel, img)

        if "yolo" in active:
            emit_if_free("yolo", lambda: dr.render_yolo(frame_rgb, detections, static_mask, mw))
        if collector is None:
            return
        if "matches" in active and collector.rotated_frame is not None:
            emit_if_free("matches", lambda: dr.render_matches(collector, mw))
        if "dino" in active and collector.rotated_frame is not None:
            emit_if_free("dino", lambda: dr.render_dino(collector, mw, self._debug_dino_pca))
        if "depth" in active and collector.depth_map is not None:
            emit_if_free("depth", lambda: dr.render_depth(collector, mw))

    def mark_debug_channel_free(self, channel) -> None:
        """GUI -> worker: confirmation that channel frame was consumed (thread-safe)."""
        with self._debug_lock:
            self._debug_inflight.pop(channel, None)

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._stop_event.set()
        if not self.wait(5000):
            logger.warning("Tracking worker did not finish within 5 seconds.")
        else:
            logger.info("Tracking worker successfully stopped.")
