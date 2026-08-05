import gc
import math
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from config import get_active_descriptor_cfg, get_cfg
from src.database import keyframe_selector, keypoint_video_writer
from src.database.db_writer import DbWriter
from src.database.frame_processor import FrameProcessor
from src.database.video_frame_source import EOF_INDEX, VideoFrameSource
from src.localization.matcher import FeatureMatcher
from src.models.wrappers.feature_extractor import FeatureExtractor
from src.models.wrappers.masking_strategy import create_masking_strategy
from src.security.project_scan import assert_project_writable
from src.utils.logging_utils import get_logger, silent_output
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


class DatabaseBuilder:
    """Builds HDF5 topometric database from reference video using XFeat & DINOv2.

    Orchestration only. The three phases live in their own modules
    (IMPROVEMENT_PLAN п.1.3):

    * :class:`src.database.video_frame_source.VideoFrameSource` — decode + prefetch
    * :class:`src.database.frame_processor.FrameProcessor` — per-frame work
    * :class:`src.database.db_writer.DbWriter` — HDF5/LanceDB storage, end-to-end

    Model loading, dimension detection and the YOLO micro-batching loop stay
    here, because they are what ties the three together.
    """

    def __init__(self, output_path, matcher=None, config=None):
        # Protection against writing into an encrypted project.
        assert_project_writable(output_path)
        self.output_path = output_path
        self.config = config or {}
        self.matcher = matcher
        self.descriptor_dim = get_active_descriptor_cfg(self.config).descriptor_dim
        # Global descriptor dimension from VLAD dictionary when enabled.
        if get_cfg(self.config, "models.vlad.enabled", False):
            _vocab = get_cfg(self.config, "models.vlad.vocab_path", None)
            if _vocab and Path(_vocab).exists():
                from src.models.wrappers.vlad_aggregator import VladAggregator

                self.descriptor_dim = VladAggregator.load(_vocab).out_dim
                logger.info(f"VLAD enabled: global descriptor dim = {self.descriptor_dim}")
            else:
                logger.warning(
                    f"models.vlad.enabled=True but vocab not found ({_vocab!r}) — "
                    f"building with CLS descriptors (dim={self.descriptor_dim})"
                )
        # Store extra SIFT features for fallback matching
        self.store_sift = get_cfg(self.config, "database.store_sift_features", False)
        self.sift_max_kps = get_cfg(self.config, "database.sift_max_keypoints", 2048)
        self.prefetch_size = get_cfg(self.config, "database.prefetch_queue_size", 32)
        self.kp_scale_cfg = get_cfg(self.config, "database.keypoint_video_scale", 0.5)
        self.use_lancedb = get_cfg(self.config, "database.use_lancedb", True)

        # DbWriter owns creation and writing of HDF5 + LanceDB structures.
        self.writer = DbWriter(output_path, config=self.config, descriptor_dim=self.descriptor_dim)

        logger.info(f"DatabaseBuilder initialized with output: {output_path}")
        if self.matcher:
            logger.info("Using provided FeatureMatcher for inter-frame poses")
        logger.info(f"DINOv2 descriptor dimension: {self.descriptor_dim}")

    # ------------------------------------------------------------------
    # Backwards-compatible storage handles (delegate to DbWriter)
    # ------------------------------------------------------------------

    @property
    def db_file(self):
        return self.writer.db_file

    @property
    def lance_table(self):
        return self.writer.lance_table

    @property
    def lance_batch(self):
        return self.writer.lance_batch

    def create_hdf5_structure(self, *args, **kwargs):
        """Deprecated shim — see :meth:`DbWriter.create_structure`."""
        self.writer.descriptor_dim = self.descriptor_dim
        self.writer.local_descriptor_dim = getattr(self, "local_descriptor_dim", 128)
        return self.writer.create_structure(*args, **kwargs)

    def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
        """Deprecated shim — see :meth:`DbWriter.save_frame_data`."""
        return self.writer.save_frame_data(frame_id, features, pose_2d)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_video(
        self,
        video_path: str,
        model_manager,
        progress_callback=None,
        save_keypoint_video: bool = True,
        project_manager=None,
    ):
        """
        Process video and build database.
        """
        self._temp_model_manager = model_manager
        self._project_manager = project_manager
        logger.info(f"Starting database build from video: {video_path}")

        # Initialise the video-frame source (VideoFrameSource)
        source = VideoFrameSource(
            video_path,
            frame_step=get_cfg(self.config, "database.frame_step", 3),
            use_decord=get_cfg(self.config, "database.use_decord", True),
            decode_batch_size=get_cfg(self.config, "database.decode_batch_size", 32),
            prefetch_size=self.prefetch_size,
        )
        width, height = source.width, source.height
        num_frames = source.num_frames

        # Save the reference resolution to the project
        if (
            self._project_manager
            and hasattr(self._project_manager, "settings")
            and self._project_manager.settings
        ):
            self._project_manager.settings.ref_frame_width = width
            self._project_manager.settings.ref_frame_height = height
            self._project_manager.save_project()
            logger.info(f"Reference resolution saved to project: {width}x{height}")

        # Initialise the keypoint-overlay video writer
        kp_scale = 1.0  # ALWAYS 1.0 so video coordinates match HDF5 DB coordinates (scale bug fix)
        kp_writer = self._open_keypoint_writer(
            save_keypoint_video, width, height, kp_scale, source.effective_fps
        )

        # Initialise the masking strategy (YOLO / none / ...)
        masking_strategy_name = get_cfg(self.config, "preprocessing.masking_strategy", "yolo")
        logger.info(f"Loading masking strategy: {masking_strategy_name}")
        masking_strategy = create_masking_strategy(
            masking_strategy_name, model_manager, model_manager.device
        )

        local_ext_type = get_cfg(self.config, "localization.fallback_extractor", "aliked")
        if local_ext_type == "xfeat":
            local_model = model_manager.load_xfeat()
        else:
            local_model = model_manager.load_local_extractor()

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

        # Patchify: multi-scale patch descriptors
        use_patchify = get_cfg(self.config, "localization.use_patchify", False)
        patchify = None
        if use_patchify:
            from src.localization.patchify import PatchifyRetrieval

            patchify_grids = get_cfg(
                self.config, "localization.patchify_grids", [[1, 1], [2, 2], [3, 3]]
            )
            patchify_batch = get_cfg(self.config, "localization.patchify_batch_size", 1)
            patchify = PatchifyRetrieval(
                feature_extractor,
                descriptor_dim=self.descriptor_dim,
                grids=patchify_grids,
                batch_size=patchify_batch,
            )
            logger.info(f"Patchify ENABLED for DB build: {patchify.num_patches} patches per frame")

        # Phase 2.1: Depth estimator integration
        self._depth_estimator = None
        depth_backend = get_cfg(self.config, "models.depth_estimator.backend", "depth_anything_v2")
        if depth_backend != "none":
            try:
                from src.depth.depth_estimator import DepthEstimator

                self._depth_estimator = DepthEstimator.build(
                    backend=depth_backend, device=model_manager.device
                )
                logger.info(f"Depth estimator ({depth_backend}) initialized for DB build")
            except Exception as e:
                logger.warning(f"Failed to initialize depth estimator: {e}")

        self._detect_descriptor_dims(
            feature_extractor, nv_model, cesp, model_manager, local_ext_type
        )

        # Create empty database structure
        logger.info("Creating HDF5 database structure...")
        self.writer.descriptor_dim = self.descriptor_dim
        self.writer.local_descriptor_dim = self.local_descriptor_dim
        self.writer.create_structure(
            num_frames,
            width,
            height,
            use_patchify=use_patchify,
            num_patches=patchify.num_patches if patchify else 0,
            frame_step=source.frame_step,
            source_total_frames=source.total_frames,
        )

        # Adaptive Keyframe Selection
        keyframe_criterion = get_cfg(self.config, "database.keyframe_criterion", "step")
        max_overlap = get_cfg(self.config, "database.keyframe_max_overlap", 0.5)
        max_gap_frames = get_cfg(self.config, "database.keyframe_max_gap_frames", 0)
        use_keyframe_selection = (
            get_cfg(self.config, "database.keyframe_min_translation_px", 0.0) > 0
            or keyframe_criterion == "overlap"
        )
        if use_keyframe_selection and keyframe_criterion == "overlap":
            logger.info(
                f"Adaptive keyframe selection ENABLED (criterion=overlap: keep a frame once "
                f"overlap with the last keyframe drops to {max_overlap:.0%}, "
                f"forced keyframe every {max_gap_frames} frames)"
            )
        elif use_keyframe_selection:
            logger.info(
                f"Adaptive keyframe selection ENABLED "
                f"(min_translation={get_cfg(self.config, 'database.keyframe_min_translation_px', 15.0)}px, "
                f"min_rotation={get_cfg(self.config, 'database.keyframe_min_rotation_deg', 1.5)}°)"
            )

        processor = FrameProcessor(
            feature_extractor=feature_extractor,
            db_writer=self.writer,
            compute_inter_frame_h=self._compute_inter_frame_H,
            is_significant_motion=self._is_significant_motion,
            draw_keypoints=self._draw_keypoints_frame,
            config=self.config,
            width=width,
            height=height,
            num_frames=num_frames,
            patchify=patchify,
            depth_estimator=self._depth_estimator,
            kp_writer=kp_writer,
            kp_scale=kp_scale,
            use_keyframe_selection=use_keyframe_selection,
            always_save_first=get_cfg(self.config, "database.keyframe_always_save_first", True),
            keyframe_criterion=keyframe_criterion,
            overlap_gate=lambda H, w, h: keyframe_selector.is_overlap_below(
                H, w, h, max_overlap=max_overlap
            ),
            keyframe_max_gap_frames=max_gap_frames,
            progress_callback=progress_callback,
        )

        # cuDNN benchmark is now set globally at startup by HardwareProfile.apply_torch_backends()
        # (previously was conditional on CNN model type; now all architectures benefit)

        # Increased prefetch queue (Fix 5)
        frame_queue = source.start_prefetch()

        try:
            self.writer.open()

            # YOLO micro-batching
            yolo_batch_size = get_cfg(self.config, "database.yolo_batch_size", 1)
            if yolo_batch_size > 1:
                logger.info(f"YOLO micro-batching ENABLED (batch_size={yolo_batch_size})")
            pending_frames: list[tuple] = []  # buffer (idx, frame, frame_rgb)

            def _flush_mask_batch(batch: list) -> list:
                """Processes batch through MaskingStrategy, returns (idx, frame, frame_rgb, static_mask)."""
                images_rgb = [b[2] for b in batch]
                with Telemetry.profile("yolo"):
                    masks_list = masking_strategy.get_mask_batch(images_rgb)
                return [(b[0], b[1], b[2], m) for b, m in zip(batch, masks_list)]

            with torch.no_grad():
                while True:
                    idx, data = frame_queue.get()

                    if idx != EOF_INDEX and data is not None:
                        frame, frame_rgb = data
                        pending_frames.append((idx, frame, frame_rgb))
                        if len(pending_frames) < yolo_batch_size:
                            continue  # accumulate batch

                    # If EOF or batch full — process all accumulated
                    if not pending_frames:
                        break

                    processed = _flush_mask_batch(pending_frames)
                    pending_frames = []

                    for p_idx, p_frame, p_frame_rgb, p_static_mask in processed:
                        processor.process(p_idx, p_frame, p_frame_rgb, p_static_mask)
                        # empty_cache() after every frame synchronised the GPU and
                        # forced subsequent allocations through slow cudaMalloc —
                        # the main build bottleneck. Flush infrequently for hygiene.
                        if torch.cuda.is_available() and p_idx % 500 == 0 and p_idx > 0:
                            torch.cuda.empty_cache()
                    if idx == EOF_INDEX:
                        break

        except Exception as e:
            logger.error(
                f"Error during database building: {e} | "
                f"video={video_path}, output={self.output_path}, "
                f"processed_frames={processor.saved_count}",
                exc_info=True,
            )
            raise
        finally:
            self.writer.finalize_vectors(processor.saved_count)

            # Save frame_index_map and actual_num_frames to metadata
            self.writer.write_frame_index_map(
                processor.saved_count,
                processor.frame_index_map,
                num_frames,
                use_keyframe_selection,
            )

            source.join(timeout=5)
            if kp_writer is not None:
                kp_writer.release()
            self.writer.close()
            source.release()

        logger.success(f"Database build completed successfully: {self.output_path}")

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _open_keypoint_writer(
        self,
        save_keypoint_video: bool,
        width: int,
        height: int,
        kp_scale: float,
        effective_fps: float,
    ):
        """Opens the keypoint-overlay VideoWriter, or returns None."""
        if not save_keypoint_video:
            return None

        kp_writer = None
        try:
            import os
            import sys

            kp_width = int(width * kp_scale)
            kp_height = int(height * kp_scale)
            kp_video_path = str(Path(self.output_path).with_suffix("")) + "_keypoints.mp4"

            # Codec order:
            # - Windows: avc1 requires openh264.dll — skip to avoid FFmpeg noise
            # - XVID: reliable cross-platform without third-party DLLs
            # - mp4v: absolute fallback
            codecs = []
            if sys.platform != "win32":
                codecs.append("avc1")  # H.264 — only on Linux/macOS with native support
            codecs += ["XVID", "mp4v"]

            for codec_name in codecs:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                # Suppress C-level stderr from FFmpeg (fd=2) to avoid OpenH264 noise
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                old_stderr_fd = os.dup(2)
                os.dup2(devnull_fd, 2)
                try:
                    with silent_output():
                        kp_writer = cv2.VideoWriter(
                            kp_video_path, fourcc, effective_fps, (kp_width, kp_height)
                        )
                finally:
                    os.dup2(old_stderr_fd, 2)
                    os.close(old_stderr_fd)
                    os.close(devnull_fd)

                if kp_writer and kp_writer.isOpened():
                    logger.info(
                        f"Keypoint video: {kp_video_path} | {kp_width}x{kp_height} | codec={codec_name}"
                    )
                    break
                kp_writer = None
            else:
                logger.warning("No compatible video codec found, keypoint video disabled")
                kp_writer = None

        except Exception as e:
            logger.warning(f"VideoWriter initialization crashed: {e}")
            kp_writer = None

        return kp_writer

    def _detect_descriptor_dims(
        self, feature_extractor, nv_model, cesp, model_manager, local_ext_type: str
    ) -> None:
        """Fix 10: Dynamic descriptor dimension detection to avoid broadcast errors."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                free_mb, total_mb = torch.cuda.mem_get_info()
                logger.info(f"VRAM before dimension detection: {free_mb / (1024**2):.1f}MB free")

            logger.info("Detecting descriptor dimension...")
            if feature_extractor.vlad_aggregator is not None:
                # With VLAD the dimension is set by the codebook (out_dim),
                # not the backbone. Probing nv_model would return the CLS size (1024),
                # and the LanceDB/HDF5 schema would diverge from what FeatureExtractor
                # actually writes (256) → Arrow cast error on the first flush.
                self.descriptor_dim = int(feature_extractor.global_descriptor_dim)
            elif hasattr(nv_model, "embed_dim"):
                self.descriptor_dim = int(nv_model.embed_dim)
            else:
                # Use a small dummy tensor directly to save VRAM
                with torch.no_grad():
                    dino_size = get_active_descriptor_cfg(self.config).input_size
                    dummy_input = torch.zeros((1, 3, dino_size, dino_size)).to(model_manager.device)
                    # Use the same logic as FeatureExtractor
                    if cesp is not None:
                        features = nv_model.forward_features(dummy_input)
                        patch_tokens = features["x_norm_patchtokens"]
                        # Grid from the actual token count (DINOv3 patch=16, not 14)
                        side = int(math.isqrt(int(patch_tokens.shape[1])))
                        h_patches, w_patches = side, side
                        dummy_out = cesp(patch_tokens, h_patches, w_patches)[0]
                        self.descriptor_dim = int(dummy_out.shape[0])
                    else:
                        dummy_out = nv_model(dummy_input)[0]
                        self.descriptor_dim = int(dummy_out.shape[0])

            logger.info(f"Detected global descriptor dimension: {self.descriptor_dim}")
        except Exception as e:
            logger.warning(
                f"Failed to detect descriptor dimension: {e}\n{traceback.format_exc()}"
                f"Falling back to configured default: {self.descriptor_dim}"
            )
            logger.warning(f"Using default dimension: {self.descriptor_dim}")

        # Detect local descriptor dimension
        try:
            with torch.no_grad():
                dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                dummy_feats = feature_extractor.extract_features(dummy_img)
                if len(dummy_feats["descriptors"]) > 0:
                    self.local_descriptor_dim = dummy_feats["descriptors"].shape[-1]
                else:
                    # Fallback default dims
                    self.local_descriptor_dim = 64 if local_ext_type == "xfeat" else 128
            logger.info(f"Detected local descriptor dimension: {self.local_descriptor_dim}")
        except Exception as e:
            logger.warning(f"Failed to detect local feature dimension: {e}. Using 128 as fallback.")
            self.local_descriptor_dim = 128

    # ------------------------------------------------------------------
    # Thin wrappers over the extracted pure helpers
    # ------------------------------------------------------------------

    def _draw_keypoints_frame(
        self,
        frame_bgr: np.ndarray,
        keypoints: np.ndarray,
        static_mask: np.ndarray,
        frame_id: int,
        total_frames: int,
    ) -> np.ndarray:
        """Delegates to ``keypoint_video_writer.draw_keypoints_frame`` (pure,
        headless-tested); kept as a thin method to avoid call-site churn."""
        return keypoint_video_writer.draw_keypoints_frame(
            frame_bgr, keypoints, static_mask, frame_id, total_frames
        )

    def _compute_inter_frame_H(self, fa: dict, fb: dict) -> np.ndarray | None:
        """H(fb -> fa): homography from current frame to previous.
        Calculation moved to ``keyframe_selector.compute_inter_frame_homography``
        (headless-tested); only lazy matcher initialization remains here.
        """
        if self.matcher is None:
            # Try to get model_manager from context if available
            mm = getattr(self, "_temp_model_manager", None)
            self.matcher = FeatureMatcher(model_manager=mm, config=self.config)

        return keyframe_selector.compute_inter_frame_homography(
            self.matcher,
            fa,
            fb,
            min_matches=get_cfg(self.config, "database.inter_frame_min_matches", 15),
            ransac_thresh=get_cfg(self.config, "database.inter_frame_ransac_thresh", 3.0),
            homography_backend=get_cfg(self.config, "homography.backend", "opencv"),
            use_mad_ransac=get_cfg(self.config, "homography.use_mad_ransac", True),
            mad_k_factor=get_cfg(self.config, "homography.mad_k_factor", 2.5),
        )

    def _is_significant_motion(self, H: np.ndarray, frame_w: int, frame_h: int) -> bool:
        """True if H corresponds to significant motion (keyframe selection).
        Logic moved to ``keyframe_selector.is_significant_motion``
        (headless-tested); only reading thresholds from config remains here.
        """
        return keyframe_selector.is_significant_motion(
            H,
            frame_w,
            frame_h,
            min_translation_px=get_cfg(self.config, "database.keyframe_min_translation_px", 15.0),
            min_rotation_deg=get_cfg(self.config, "database.keyframe_min_rotation_deg", 1.5),
        )
