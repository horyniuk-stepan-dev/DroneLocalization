from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import h5py
import numpy as np
import torch

from config.config import get_cfg
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.models.wrappers.masking_strategy import create_masking_strategy
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseBuilder:
    """Builds HDF5 topometric database from reference video using XFeat & DINOv2"""

    def __init__(self, output_path, matcher=None, config=None):
        self.output_path = output_path
        self.config = config or {}
        self.matcher = matcher
        db_cfg = self.config.get("database", {})
        self.descriptor_dim = get_cfg(self.config, "dinov2.descriptor_dim", 384)
        self.prefetch_size = get_cfg(self.config, "database.prefetch_queue_size", 32)
        self.kp_scale_cfg = get_cfg(self.config, "database.keypoint_video_scale", 0.5)
        self.db_file = None

        logger.info(f"DatabaseBuilder initialized with output: {output_path}")
        if self.matcher:
            logger.info("Using provided FeatureMatcher for inter-frame poses")
        logger.info(f"DINOv2 descriptor dimension: {self.descriptor_dim}")

    def build_from_video(
        self,
        video_path: str,
        model_manager,
        progress_callback=None,
        save_keypoint_video: bool = True,
    ):
        """
        Process video and build database.
        """
        logger.info(f"Starting database build from video: {video_path}")

        # Читаємо налаштування з конфігу (з дефолтом)
        frame_step = get_cfg(self.config, "database.frame_step", 3)
        if frame_step < 1:
            frame_step = 1

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError(f"Не вдалося відкрити відео: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Обчислюємо скільки кадрів РЕАЛЬНО буде оброблено
        num_frames = (total_frames + frame_step - 1) // frame_step
        effective_fps = original_fps / frame_step

        if num_frames <= 0:
            cap.release()
            logger.error(
                f"Invalid frame count ({num_frames}). Video might be corrupted or uses unsupported codec."
            )
            raise ValueError(
                "OpenCV не зміг розпізнати відео. Файл пошкоджений або використовує непідтримуваний кодек. "
                "Спробуйте переконвертувати відео у стандартний MP4 (H.264)."
            )

        logger.info(
            f"Video properties: {width}x{height}, {total_frames} total frames, {original_fps:.2f} FPS"
        )
        logger.info(
            f"Processing with step={frame_step} -> {num_frames} frames to process ({effective_fps:.2f} effective FPS)"
        )

        # Ініціалізуємо запис відео з keypoints
        kp_video_path = None
        kp_writer = None
        kp_scale = 1.0  # ЗАВЖДИ 1.0, щоб координати відео і бази HDF5 збігалися (виправлення багу масштабу)
        if save_keypoint_video:
            try:
                kp_width = int(width * kp_scale)
                kp_height = int(height * kp_scale)

                kp_video_path = str(Path(self.output_path).with_suffix("")) + "_keypoints.mp4"

                # Attempt H.264 (avc1)
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                kp_writer = cv2.VideoWriter(
                    kp_video_path, fourcc, effective_fps, (kp_width, kp_height)
                )

                if not kp_writer or not kp_writer.isOpened():
                    logger.warning("H.264 (avc1) codec not available, falling back to mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    kp_writer = cv2.VideoWriter(
                        kp_video_path, fourcc, effective_fps, (kp_width, kp_height)
                    )

                if kp_writer and kp_writer.isOpened():
                    logger.info(
                        f"Keypoint video will be saved to: {kp_video_path} at {kp_width}x{kp_height} (Scale: {kp_scale})"
                    )
                else:
                    logger.warning(
                        "Failed to initialize any compatible video codec, keypoint video disabled"
                    )
                    kp_writer = None
            except Exception as e:
                logger.warning(f"VideoWriter initialization crashed: {e}")
                kp_writer = None

        # Ініціалізуємо стратегію маскування (YOLO / none / ...)
        masking_strategy_name = get_cfg(self.config, "preprocessing.masking_strategy", "yolo")
        logger.info(f"Loading masking strategy: {masking_strategy_name}")
        masking_strategy = create_masking_strategy(
            masking_strategy_name, model_manager, model_manager.device
        )

        local_model = model_manager.load_aliked()
        nv_model = model_manager.load_dinov2()

        cesp = None
        if get_cfg(self.config, "models.cesp.enabled", False):
            try:
                cesp = model_manager.load_cesp()
            except Exception:
                logger.warning("CESP loading failed during DB build, continuing without it")

        feature_extractor = FeatureExtractor(
            local_model, nv_model, model_manager.device, config=self.config, cesp_module=cesp
        )
        logger.success("All models loaded successfully")

        # Fix 10: Dynamic descriptor dimension detection to avoid broadcast errors
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                free_mb, total_mb = torch.cuda.mem_get_info()
                logger.info(f"VRAM before dimension detection: {free_mb / (1024**2):.1f}MB free")

            logger.info("Detecting descriptor dimension...")
            if hasattr(nv_model, "embed_dim"):
                self.descriptor_dim = int(nv_model.embed_dim)
            else:
                # Use a small dummy tensor directly to save VRAM
                with torch.no_grad():
                    dino_size = get_cfg(self.config, "dinov2.input_size", 336)
                    dummy_input = torch.zeros((1, 3, dino_size, dino_size)).to(model_manager.device)
                    # Use the same logic as FeatureExtractor
                    if cesp is not None:
                        features = nv_model.forward_features(dummy_input)
                        patch_tokens = features["x_norm_patchtokens"]
                        h_patches, w_patches = dino_size // 14, dino_size // 14
                        dummy_out = cesp(patch_tokens, h_patches, w_patches)[0]
                        self.descriptor_dim = int(dummy_out.shape[0])
                    else:
                        dummy_out = nv_model(dummy_input)[0]
                        self.descriptor_dim = int(dummy_out.shape[0])

            logger.info(f"Detected global descriptor dimension: {self.descriptor_dim}")
        except Exception as e:
            import traceback

            logger.warning(f"Failed to detect descriptor dimension: {e}\n{traceback.format_exc()}")
            logger.warning(f"Using default dimension: {self.descriptor_dim}")

        # Create empty database structure
        logger.info("Creating HDF5 database structure...")
        self.create_hdf5_structure(num_frames, width, height)

        current_pose = np.eye(3, dtype=np.float32)
        prev_features = None

        # Adaptive Keyframe Selection (П4)
        saved_count = 0  # лічильник РЕАЛЬНО записаних кадрів
        frame_index_map: list[int] = []  # список збережених frame_id
        use_keyframe_selection = (
            get_cfg(self.config, "database.keyframe_min_translation_px", 0.0) > 0
        )
        if use_keyframe_selection:
            logger.info(
                f"Adaptive keyframe selection ENABLED "
                f"(min_translation={get_cfg(self.config, 'database.keyframe_min_translation_px', 15.0)}px, "
                f"min_rotation={get_cfg(self.config, 'database.keyframe_min_rotation_deg', 1.5)}°)"
            )

        # cuDNN benchmark conditionally (Fix 5)

        if torch.cuda.is_available():
            model_type = get_cfg(self.config, "localization.fallback_extractor", "aliked")
            if model_type in ("xfeat", "aliked"):  # CNN-based types
                torch.backends.cudnn.benchmark = True
                logger.info(f"cuDNN benchmark ENABLED for {model_type}")

        # Increased prefetch queue (Fix 5)
        frame_queue = Queue(maxsize=self.prefetch_size)

        def prefetch_frames():
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                if i % frame_step != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig_frame_idx = i // frame_step
                frame_queue.put((orig_frame_idx, (frame, frame_rgb)))

            frame_queue.put((-1, None))

        prefetch_thread = Thread(target=prefetch_frames, daemon=True)
        prefetch_thread.start()

        try:
            self.db_file = h5py.File(self.output_path, "a")
            logger.info(f"Opened HDF5 file for writing: {self.output_path}")

            # YOLO micro-batching (П8)
            yolo_batch_size = get_cfg(self.config, "database.yolo_batch_size", 2)
            if yolo_batch_size > 1:
                logger.info(f"YOLO micro-batching ENABLED (batch_size={yolo_batch_size})")
            pending_frames: list[tuple] = []  # буфер (idx, frame, frame_rgb)

            def _flush_mask_batch(batch: list) -> list:
                """Обробляє батч через MaskingStrategy, повертає (idx, frame, frame_rgb, static_mask)."""
                images_rgb = [b[2] for b in batch]
                masks_list = masking_strategy.get_mask_batch(images_rgb)
                return [(b[0], b[1], b[2], m) for b, m in zip(batch, masks_list)]

            def _process_single_frame(
                p_idx,
                p_frame,
                p_frame_rgb,
                p_static_mask,
                current_pose,
                prev_features,
                saved_count,
                frame_index_map,
            ):
                """Обробляє один кадр після YOLO: feature extraction, pose, keyframe selection."""
                features = feature_extractor.extract_features(p_frame_rgb, p_static_mask)
                features["coords_2d"] = features["keypoints"]

                if kp_writer is not None:
                    kp_frame = self._draw_keypoints_frame(
                        p_frame, features["keypoints"], p_static_mask, p_idx, num_frames
                    )
                    if kp_scale != 1.0:
                        kp_w = int(width * kp_scale)
                        kp_h = int(height * kp_scale)
                        kp_frame = cv2.resize(kp_frame, (kp_w, kp_h), interpolation=cv2.INTER_AREA)
                    kp_writer.write(kp_frame)

                if p_idx == 0 or prev_features is None:
                    current_pose = np.eye(3, dtype=np.float64)
                    save_this_frame = True
                    save_this_frame = True
                else:
                    H_step = self._compute_inter_frame_H(prev_features, features)
                    if H_step is not None:
                        current_pose = current_pose @ H_step.astype(np.float64)
                        if use_keyframe_selection:
                            save_this_frame = self._is_significant_motion(H_step)
                        else:
                            save_this_frame = True
                    else:
                        logger.warning(
                            f"Frame {p_idx}: inter-frame match failed, reusing previous pose"
                        )
                    if use_keyframe_selection and H_step is not None:
                        save_this_frame = self._is_significant_motion(H_step, width, height)
                    else:
                        save_this_frame = True

                prev_features = features

                # ЗАВЖДИ зберігаємо pose для повного ланцюга пропагації,
                # навіть якщо кадр не є keyframe (пропущений через малий рух).
                # Без цього frame_poses[frame_id] = zeros → пропагація ламається.
                if self.db_file:
                    self.db_file["global_descriptors"]["frame_poses"][p_idx] = current_pose

                if save_this_frame:
                    frame_index_map.append(p_idx)
                    # Зберігаємо за ОРИГІНАЛЬНИМ індексом p_idx, а не послідовним
                    # Це зберігає frame_id ↔ slot identity для калібрування/пропагації
                    self.save_frame_data(p_idx, features, current_pose)
                    saved_count += 1

                    if saved_count % 100 == 0:
                        progress_pct = int((p_idx + 1) / num_frames * 100)
                        logger.info(
                            f"Saved {saved_count} keyframes from {p_idx + 1}/{num_frames} processed "
                            f"({progress_pct}%)"
                        )

                progress_percent = int((p_idx + 1) / num_frames * 100)
                if progress_callback:
                    progress_callback(progress_percent)

                return current_pose, prev_features, saved_count

            while True:
                idx, data = frame_queue.get()

                if idx != -1 and data is not None:
                    frame, frame_rgb = data
                    pending_frames.append((idx, frame, frame_rgb))
                    if len(pending_frames) < yolo_batch_size:
                        continue  # накопичуємо батч

                # Якщо EOF або батч повний — обробляємо все накопичене
                if not pending_frames:
                    break

                processed = _flush_mask_batch(pending_frames)
                pending_frames = []

                for p_idx, p_frame, p_frame_rgb, p_static_mask in processed:
                    current_pose, prev_features, saved_count = _process_single_frame(
                        p_idx,
                        p_frame,
                        p_frame_rgb,
                        p_static_mask,
                        current_pose,
                        prev_features,
                        saved_count,
                        frame_index_map,
                    )

                if idx == -1:
                    break

            # Save the index mapping
            if "metadata" in self.db_file:
                self.db_file["metadata"].attrs["actual_num_frames"] = db_index
                self.db_file["metadata"].create_dataset(
                    "frame_index_map", data=np.array(frame_index_map, dtype=np.int32)
                )

        except Exception as e:
            logger.error(f"Error during database building: {e}")
            raise
        finally:
            # Зберігаємо frame_index_map і actual_num_frames у metadata
            if self.db_file and saved_count > 0:
                try:
                    meta = self.db_file["metadata"]
                    meta.attrs["actual_num_frames"] = saved_count
                    if "frame_index_map" not in meta:
                        meta.create_dataset(
                            "frame_index_map",
                            data=np.array(frame_index_map, dtype=np.int32),
                        )
                    if use_keyframe_selection:
                        logger.info(
                            f"Keyframe selection: {saved_count}/{num_frames} frames saved "
                            f"({100 - saved_count / num_frames * 100:.1f}% reduction)"
                        )
                except Exception as e:
                    logger.warning(f"Could not save frame_index_map: {e}")

            prefetch_thread.join(timeout=5)
            if kp_writer is not None:
                kp_writer.release()
            if self.db_file:
                self.db_file.close()
            cap.release()

        logger.success(f"Database build completed successfully: {self.output_path}")

    def _is_significant_motion(self, H: np.ndarray) -> bool:
        """Повертає True якщо гомографія H відповідає значному руху."""
        min_t = get_cfg(self.config, "database.keyframe_min_translation_px", 15.0)
        min_r = get_cfg(self.config, "database.keyframe_min_rotation_deg", 1.5)

        cx, cy = 1920.0 / 2.0, 1080.0 / 2.0
        p_src = np.array([cx, cy, 1.0], dtype=np.float64)
        p_dst = H.astype(np.float64) @ p_src
        if p_dst[2] != 0:
            p_dst /= p_dst[2]
        translation = float(np.linalg.norm(p_dst[:2] - np.array([cx, cy])))

        if translation >= min_t:
            return True

        A = H[:2, :2].astype(np.float64)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return True
        angle_rad = np.arctan2(A[1, 0], A[0, 0])
        angle_deg = abs(np.degrees(angle_rad))
        return angle_deg >= min_r

    def _draw_keypoints_frame(
        self,
        frame_bgr: np.ndarray,
        keypoints: np.ndarray,
        static_mask: np.ndarray,
        frame_id: int,
        total_frames: int,
    ) -> np.ndarray:
        vis = frame_bgr.copy()

        if static_mask is not None:
            dynamic_zone = static_mask == 0
            if dynamic_zone.any():
                overlay = vis.copy()
                overlay[dynamic_zone] = (0, 0, 200)
                cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        for x, y in keypoints:
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(vis, (cx, cy), radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(vis, (cx, cy), radius=4, color=(0, 180, 0), thickness=1)

        info_lines = [
            f"Frame: {frame_id:05d} / {total_frames:05d}",
            f"Keypoints: {len(keypoints)}",
            f"Dynamic mask: {'YES' if static_mask is not None else 'NO'}",
        ]
        panel_h = len(info_lines) * 28 + 14
        cv2.rectangle(vis, (0, 0), (340, panel_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (0, 0), (340, panel_h), (80, 80, 80), 1)

        for idx, line in enumerate(info_lines):
            cv2.putText(
                vis,
                line,
                (8, 22 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        legend_y = vis.shape[0] - 10
        cv2.circle(vis, (12, legend_y - 4), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            "XFeat keypoint",
            (22, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(vis, (200, legend_y - 10), (218, legend_y + 2), (0, 0, 200), -1)
        cv2.putText(
            vis,
            "YOLO dynamic zone",
            (224, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1,
            cv2.LINE_AA,
        )

        return vis

    def _compute_inter_frame_H(self, fa: dict, fb: dict) -> np.ndarray | None:
        """
        Обчислює H(fb → fa): гомографію з поточного кадру в попередній.
        """
        min_matches = get_cfg(self.config, "database.inter_frame_min_matches", 15)
        ransac_thresh = get_cfg(self.config, "database.inter_frame_ransac_thresh", 3.0)

        if self.matcher is None:
            from src.localization.matcher import FeatureMatcher

            self.matcher = FeatureMatcher(config=self.config)

        mkpts_a, mkpts_b = self.matcher.match(fa, fb)

        if len(mkpts_a) < min_matches:
            return None

        from src.geometry.transformations import GeometryTransforms

        H, mask = GeometryTransforms.estimate_homography(
            mkpts_a, mkpts_b, ransac_threshold=ransac_thresh
        )

        if H is None or int(np.sum(mask)) < min_matches:
            return None

        return H.astype(np.float32)

    def _is_significant_motion(self, H: np.ndarray, frame_w: int, frame_h: int) -> bool:
        """
        Повертає True якщо гомографія H відповідає значному руху.
        H: (3,3) float32 — матриця з frame_b до frame_a.
        """
        min_t = get_cfg(self.config, "database.keyframe_min_translation_px", 15.0)
        min_r = get_cfg(self.config, "database.keyframe_min_rotation_deg", 1.5)

        # Трансляція: зсув центру кадру через H
        cx, cy = frame_w / 2.0, frame_h / 2.0
        p_src = np.array([cx, cy, 1.0], dtype=np.float64)
        p_dst = H.astype(np.float64) @ p_src
        p_dst /= p_dst[2]
        translation = np.linalg.norm(p_dst[:2] - np.array([cx, cy]))

        if translation >= min_t:
            return True

        # Кут: з лінійної частини H (2×2 зліва вгорі)
        A = H[:2, :2].astype(np.float64)
        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return True  # вироджена матриця → вважаємо рухом
        angle_rad = np.arctan2(A[1, 0], A[0, 0])
        angle_deg = abs(np.degrees(angle_rad))
        return angle_deg >= min_r

    def create_hdf5_structure(self, num_frames: int, width: int, height: int):
        """Create optimal HDF5 hierarchy with pre-allocated chunked arrays (schema v2)"""
        compression = get_cfg(self.config, "database.hdf5_compression", "lzf")
        chunk_f = get_cfg(self.config, "database.hdf5_chunk_frames", 64)
        max_kps = get_cfg(self.config, "database.max_keypoints_stored", 2048)
        local_desc_dim = 128  # ALIKED descriptor dim

        logger.info(
            f"Creating HDF5 v2 structure for {num_frames} frames "
            f"(compression={compression}, chunks={chunk_f}, max_kps={max_kps})"
        )

        with h5py.File(self.output_path, "w") as f:
            # --- global_descriptors: chunked ---
            g1 = f.create_group("global_descriptors")
            g1.create_dataset(
                "descriptors",
                shape=(num_frames, self.descriptor_dim),
                dtype="float32",
                compression=compression,
                chunks=(min(256, num_frames), self.descriptor_dim),
            )
            g1.create_dataset(
                "frame_poses",
                shape=(num_frames, 3, 3),
                dtype="float64",
                compression=compression,
                chunks=(min(256, num_frames), 3, 3),
            )

            # --- local_features: PRE-ALLOCATED chunked arrays (НОВА СХЕМА v2) ---
            lf = f.create_group("local_features")
            lf.create_dataset(
                "keypoints",
                shape=(num_frames, max_kps, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, 2),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "descriptors",
                shape=(num_frames, max_kps, local_desc_dim),
                dtype="float16",  # float16: -50% розміру (П2)
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, local_desc_dim),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "coords_2d",
                shape=(num_frames, max_kps, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, 2),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "kp_counts",  # скільки keypoints у кожному кадрі
                shape=(num_frames,),
                dtype="int16",
                compression=compression,
                chunks=(min(num_frames, 4096),),
                fillvalue=0,
            )
            # Розміри кадру — зберігаємо ОДИН РАЗ у групі
            lf.attrs["frame_width"] = width
            lf.attrs["frame_height"] = height

            g3 = f.create_group("metadata")
            g3.attrs["num_frames"] = num_frames
            g3.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            g3.attrs["frame_width"] = width
            g3.attrs["frame_height"] = height
            g3.attrs["descriptor_dim"] = self.descriptor_dim
            g3.attrs["hdf5_schema"] = "v2"  # версія схеми для зворотної сумісності
            g3.attrs["max_keypoints"] = max_kps

        logger.success("HDF5 v2 structure created successfully")

    def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
        """Save extracted data for a single frame via slice assignment (schema v2)"""
        # global — без змін
        self.db_file["global_descriptors"]["descriptors"][frame_id] = features["global_desc"]
        self.db_file["global_descriptors"]["frame_poses"][frame_id] = pose_2d

        # local — slice assignment замість create_group + create_dataset
        kps = features["keypoints"]
        descs = features["descriptors"]
        c2d = features["coords_2d"]

        max_kps = self.db_file["local_features"]["keypoints"].shape[1]
        n = min(len(kps), max_kps)

        lf = self.db_file["local_features"]
        lf["keypoints"][frame_id, :n] = kps[:n]
        lf["descriptors"][frame_id, :n] = descs[:n].astype("float16")
        lf["coords_2d"][frame_id, :n] = c2d[:n]
        lf["kp_counts"][frame_id] = n
