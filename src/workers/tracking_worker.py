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
    fps_updated = pyqtSignal(float)
    error = pyqtSignal(str)
    status_update = pyqtSignal(str)
    fov_found = pyqtSignal(list)
    objects_detected = pyqtSignal(object)  # list[TrackedObject]
    objects_gps_updated = pyqtSignal(object)  # list[ObjectGPS]
    debug_view_ready = pyqtSignal(str, np.ndarray)  # (channel_name, готове BGR-зображення)

    def __init__(self, video_source: str, localizer, model_manager=None, config=None):
        super().__init__()
        self.video_source = video_source
        self.localizer = localizer
        self.model_manager = model_manager
        self.config = config or {}
        self._stop_event = threading.Event()

        # S3-3: Інтервал ключових кадрів для локалізації
        self.keyframe_interval = get_cfg(self.config, "tracking.keyframe_interval", 5)
        # Зберігаємо process_fps для метрик UI, але логіка базується на кадрах
        self.process_fps = get_cfg(
            self.config, "tracking.process_fps", 30.0 / self.keyframe_interval
        )
        self.tracking_config = get_cfg(self.config, "object_tracking", {})

        # ── Debug views (вікна «очима моделей») ─────────────────────────────
        # Порожній набір каналів ⇒ нуль overhead: колектор не створюється.
        self._debug_lock = threading.Lock()
        self._debug_channels = set()
        self._debug_max_width = get_cfg(self.config, "debug_views.max_width", 640)
        self._debug_dino_pca = get_cfg(self.config, "debug_views.dino_pca_enabled", True)
        self._debug_inflight = {}  # {канал: monotonic-час emit} — backpressure (self-healing)
        self._debug_inflight_stale_sec = 1.0  # авто-скидання, якщо ack від GUI не прийшов

    def run(self):
        # Fix #3: скидаємо стан сесії через публічний API (без приватних полів)
        if hasattr(self.localizer, "reset_session"):
            self.localizer.reset_session()

        # Debug views: свіжий старт backpressure-стану для нової сесії
        with self._debug_lock:
            self._debug_inflight.clear()

        if self.model_manager:
            self.model_manager.pin(["aliked", "lightglue_aliked", "dinov2"])

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

        # Fix 6: Pre-warm fallback моделей при старті трекінгу
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
                self.error.emit(f"YOLO не вдалося завантажити: {e}")
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
            self.error.emit(f"Не вдалося відкрити відеоджерело: {self.video_source}")
            return

        video_fps = video_src.fps
        if video_fps <= 0:
            video_fps = 30.0
        frame_duration_sec = 1.0 / video_fps

        # Замість time-based інтервалу використовуємо frame-based:
        frame_idx = 0
        prev_gray_for_of = None
        prev_pts_for_of = None
        last_tracked_objects = []  # Кеш об'єктів з останнього ключового кадру для OF-кадрів

        # Зберігаємо останній час локалізації саме за ВІДЕО-часом, а не за процесорним
        last_localization_video_time = -1.0
        # Час останнього ОБРОБЛЕНОГО keyframe-а (навіть якщо він був відхилений як outlier)
        # Це потрібно для коректного dt в outlier_detector: якщо всі keyframe-и
        # відхиляються, last_localization_video_time залишається -1, і dt = 0.033s,
        # що штучно завищує швидкість у 5× (keyframe_interval=5).
        last_keyframe_video_time = -1.0

        stream_start_time = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()

            ret, frame = video_src.read()
            if not ret:
                logger.info("End of video stream or connection lost.")
                self.status_update.emit("Відеопотік завершено або втрачено.")
                break

            if video_src.is_live:
                current_video_time_sec = time.time() - stream_start_time
            else:
                # Отримуємо поточний час САМОГО ВІДЕО у секундах (публічний API)
                current_video_time_sec = video_src.pos_msec / 1000.0
                # Fallback: деякі кодеки повертають 0 — рахуємо за номером кадру
                if current_video_time_sec <= 0:
                    current_video_time_sec = video_src.pos_frames * frame_duration_sec

            # 1. Завжди відправляємо кадр в GUI для плавного відтворення (сирий BGR)
            self.frame_ready.emit(frame)

            # S3-3: Optical Flow Pipeline
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_keyframe = frame_idx % self.keyframe_interval == 0

            # Розрахунок dt — різний для KF та OF
            if is_keyframe or prev_pts_for_of is None:
                # Для ключових кадрів: dt = час від ПОПЕРЕДНЬОГО ключового кадру
                # (навіть якщо він був відхилений як outlier)
                if last_keyframe_video_time < 0:
                    calculated_dt = self.keyframe_interval * frame_duration_sec
                else:
                    calculated_dt = current_video_time_sec - last_keyframe_video_time
                    if calculated_dt <= 0:
                        calculated_dt = self.keyframe_interval * frame_duration_sec
            else:
                # Для OF-кадрів: dt = час від останньої УСПІШНОЇ локалізації
                if last_localization_video_time < 0:
                    calculated_dt = frame_duration_sec
                else:
                    calculated_dt = current_video_time_sec - last_localization_video_time
                    if calculated_dt <= 0:
                        calculated_dt = frame_duration_sec

            loc_result = {"success": False, "error": "Not processed"}
            start_process = time.time()

            if is_keyframe or prev_pts_for_of is None:
                # ====== HEAVY KEYFRAME LOCALIZATION ======
                # Для обробки YOLO та анізотропних дескрипторів потрібен RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                static_mask = None
                detections = []
                if yolo_wrapper:
                    static_mask, detections = yolo_wrapper.detect_and_mask(frame_rgb)

                # Debug views: знімок активних каналів + opt-in колектор.
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

                # Завжди оновлюємо час останнього keyframe, навіть якщо він rejected
                last_keyframe_video_time = current_video_time_sec

                # БАГФІКС (OF-шов): retrieval-only fallback не має H і не
                # оновлює _last_state — ребейз OF-точок на ньому дав би OF
                # з новими точками на старій гомографії.
                if loc_result.get("success") and loc_result.get("fallback_mode") != "retrieval_only":
                    # Зберігаємо стан для OF на наступні кадри
                    prev_gray_for_of = curr_gray
                    # Трекаємо гарні точки (corners) для стабільного OF
                    prev_pts_for_of = cv2.goodFeaturesToTrack(
                        curr_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, mask=None
                    )

                if object_tracker and detections is not None:
                    tracked_objects = object_tracker.update(detections, frame.shape)
                    # ОНОВЛЕНО: Завжди оновлюємо кеш, навіть якщо порожній, щоб об'єкти могли зникати
                    last_tracked_objects = tracked_objects
                    self.objects_detected.emit(tracked_objects)
                    loc_state = getattr(self.localizer, "last_state", None)
                    if object_projector and loc_state:
                        H = loc_state.get("H")
                        affine = loc_state.get("affine")
                        angle = loc_state.get("global_angle", 0)

                        if H is not None and affine is not None:
                            # Фікс: масштабуємо об'єкти до нормалізованого простору гомографії
                            scale = loc_state.get(
                                "scale", getattr(self.localizer, "_last_scale", 1.0)
                            )

                            # Shallow copy достатньо: перезаписуємо лише center_px і bbox
                            # (deepcopy на кожен об'єкт кожного keyframe — зайвий CPU)
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
                            self.objects_gps_updated.emit(objects_gps)

                # ВИПРАВЛЕНО (A1): раніше тут був torch.cuda.empty_cache() після
                # КОЖНОГО keyframe — це синхронізує GPU і повертає блоки драйверу,
                # через що наступні алокації йдуть повільним cudaMalloc (10–100 мс
                # "податку" на keyframe). При OOM кеш чиститься у except-гілці вище.
            else:
                # ====== OPTICAL FLOW TRACKING ======
                if prev_pts_for_of is not None and len(prev_pts_for_of) > 10:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray_for_of,
                        curr_gray,
                        prev_pts_for_of,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                    )
                    good_new = curr_pts[status == 1]
                    good_old = prev_pts_for_of[status == 1]

                    if len(good_new) > 10:
                        # Зсув у пікселях (fallback, якщо симілярність не зійдеться)
                        flow_vectors = good_new - good_old
                        dx_px, dy_px = np.median(flow_vectors, axis=0)

                        # B4: повна симілярність (R+T+S) замість чистої трансляції —
                        # враховує обертання дрона та зміну висоти між keyframe-ами.
                        # flow_quality (0..1) — чесна оцінка якості OF для Kalman R.
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
                        except cv2.error:
                            pass

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

                        # Оновлюємо стан так, щоб OF завжди рахувався ВІД КЛЮЧОВОГО КАДРУ,
                        # Це усуває проблему накопичення помилок (drift).
                        # Тому prev_gray_for_of та prev_pts_for_of не оновлюються тут!

                        # На OF-кадрах: повторно emit останні відомі об'єкти для візуальної
                        # безперервності (YOLO не запускається, тому нових детекцій немає)
                        if object_tracker:
                            self.objects_detected.emit(last_tracked_objects)
                    else:
                        prev_pts_for_of = None  # Втрата точок — наступний кадр стане ключовим
                else:
                    prev_pts_for_of = None

            if loc_result.get("success") and loc_result.get("matched_frame", -1) != -1:
                self.location_found.emit(
                    loc_result["lat"],
                    loc_result["lon"],
                    loc_result["confidence"],
                    loc_result["inliers"],
                )
                if loc_result.get("fov_polygon"):
                    self.fov_found.emit(loc_result["fov_polygon"])

                track_type = "OF" if loc_result.get("is_of") else "KF"
                method_txt = (
                    "Схожість" if loc_result.get("fallback_mode") == "retrieval_only" else "Inliers"
                )
                score = loc_result.get("global_score", loc_result["inliers"])

                self.status_update.emit(
                    f"[{track_type}] Знайдено ({method_txt}: {score:.2f}, Кадр: {loc_result['matched_frame']})"
                )

                last_localization_video_time = current_video_time_sec

                # Мульти-режим: оновлюємо активні бази за поточною GPS-позицією
                if hasattr(self.localizer, "db_manager") and self.localizer.db_manager is not None:
                    try:
                        self.localizer.db_manager.set_active_by_gps(
                            loc_result["lat"], loc_result["lon"]
                        )
                        # Фонова перебудова FAISS-підмножини у GeoAwareRetriever-ах
                        self.localizer.db_manager.update_retriever_positions(
                            loc_result["lat"], loc_result["lon"]
                        )
                    except Exception as e:
                        logger.debug(f"set_active_by_gps failed: {e}")
            elif not loc_result.get("success") and loc_result.get("error") != "Not processed":
                self.status_update.emit(f"Втрата: {loc_result.get('error', 'Невідома помилка')}")

            process_duration = time.time() - start_process
            self.fps_updated.emit(1.0 / process_duration if process_duration > 0 else 0)

            frame_idx += 1

            # 3. Синхронізація відтворення (тільки для файлів)
            if not video_src.is_live:
                elapsed_in_loop = time.time() - loop_start
                sleep_time = frame_duration_sec - elapsed_in_loop
                if sleep_time > 0:
                    self.msleep(int(sleep_time * 1000))

        video_src.release()
        logger.info("Tracking worker thread finished cleanly.")

    def _prewarm_fallback_models(self):
        """Завантажує моделі заздалегідь, делегуючи у ModelManager."""
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
        """GUI → worker: набір активних debug-каналів (thread-safe).

        Порожній набір ⇒ нуль overhead. Викликається з GUI-потоку при зміні
        видимості вікон і при старті трекінгу.
        """
        with self._debug_lock:
            self._debug_channels = set(channels or [])

    def _render_debug(self, active, frame_rgb, detections, static_mask, collector) -> None:
        """Рендерить активні debug-канали й emit-ить готові BGR-кадри у GUI.

        Лише на keyframe-ах, у worker-потоці. Емітяться свіжі масиви (не аліаси
        кадру/колектора), тож безпечно між потоками.

        Backpressure «drop замість черги»: на канал одночасно ≤1 кадр «у льоті».
        Поки GUI не підтвердив попередній (mark_debug_channel_free), нові кадри
        цього каналу не рендеряться і не emit-яться — GUI-черга не росте, ми
        показуємо найсвіжіший кадр, а не відстаємо. Кожен рендер у своєму try:
        помилка одного вікна не валить локалізацію чи інші вікна.
        """
        from src.workers import debug_renderers as dr

        mw = self._debug_max_width

        def emit_if_free(channel, render_fn):
            now = time.monotonic()
            with self._debug_lock:
                ts = self._debug_inflight.get(channel)
                # Свіжий in-flight → drop. Застарілий (ack втрачено?) → self-heal,
                # рендеримо знову, щоб канал не «замерзав» назавжди.
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
            emit_if_free(
                "yolo", lambda: dr.render_yolo(frame_rgb, detections, static_mask, mw)
            )
        if collector is None:
            return
        if "matches" in active and collector.rotated_frame is not None:
            emit_if_free("matches", lambda: dr.render_matches(collector, mw))
        if "dino" in active and collector.rotated_frame is not None:
            emit_if_free("dino", lambda: dr.render_dino(collector, mw, self._debug_dino_pca))
        if "depth" in active and collector.depth_map is not None:
            emit_if_free("depth", lambda: dr.render_depth(collector, mw))

    def mark_debug_channel_free(self, channel) -> None:
        """GUI → worker: підтвердження, що кадр каналу спожито (thread-safe).

        Знімає in-flight позначку, дозволяючи emit наступного кадру цього
        каналу. Викликається зі слота _on_debug_view_ready у GUI-потоці.
        """
        with self._debug_lock:
            self._debug_inflight.pop(channel, None)

    def stop(self):
        logger.info("Stopping tracking worker...")
        self._stop_event.set()
        if not self.wait(5000):  # чекаємо максимум 5 секунд
            logger.warning("Tracking worker did not finish within 5 seconds.")
        else:
            logger.info("Tracking worker successfully stopped.")
