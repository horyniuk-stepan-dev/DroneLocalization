"""Per-frame processing for the database build.

Extracted verbatim from the ``_process_single_frame`` closure inside
``DatabaseBuilder.build_from_video`` (IMPROVEMENT_PLAN item 1.3, splitting
``db_builder``). Owns everything that happens to ONE already-decoded, already-
masked frame: feature extraction, patch descriptors, depth scale, keypoint
video, the pose chain and the keyframe decision — then hands the result to the
``DbWriter``.

The loop state that used to live in local variables of ``build_from_video``
(``current_pose``, ``prev_features``, ``saved_count``, ``frame_index_map``) is
now instance state here, which is why the module is deliberately stateful.

This module imports no torch and no h5py: it talks to injected collaborators
only, so it is exercisable headlessly.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from config import get_cfg
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Processes decoded+masked frames into database rows.

    Args:
        feature_extractor: provides ``extract_features(rgb, mask)``.
        db_writer: :class:`src.database.db_writer.DbWriter`.
        compute_inter_frame_h: ``(prev_features, features) -> H | None``.
        is_significant_motion: ``(H, width, height) -> bool`` keyframe gate.
        draw_keypoints: ``(bgr, kps, mask, idx, total) -> bgr`` overlay.
    """

    def __init__(
        self,
        *,
        feature_extractor,
        db_writer,
        compute_inter_frame_h: Callable,
        is_significant_motion: Callable,
        draw_keypoints: Callable,
        config: dict | None = None,
        width: int = 0,
        height: int = 0,
        num_frames: int = 0,
        patchify=None,
        depth_estimator=None,
        kp_writer=None,
        kp_scale: float = 1.0,
        use_keyframe_selection: bool = False,
        always_save_first: bool = True,
        keyframe_criterion: str = "step",
        overlap_gate: Callable | None = None,
        keyframe_max_gap_frames: int = 0,
        progress_callback: Callable | None = None,
    ):
        self.feature_extractor = feature_extractor
        self.db_writer = db_writer
        self.compute_inter_frame_h = compute_inter_frame_h
        self.is_significant_motion = is_significant_motion
        self.draw_keypoints = draw_keypoints
        self.config = config or {}
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.patchify = patchify
        self.depth_estimator = depth_estimator
        self.kp_writer = kp_writer
        self.kp_scale = kp_scale
        self.use_keyframe_selection = use_keyframe_selection
        # database.keyframe_always_save_first: the config key existed before
        # but was never read — the first frame was always saved unconditionally.
        self.always_save_first = always_save_first
        # "step" = legacy adjacent-frame motion; "overlap" = displacement
        # accumulated since the last KEPT keyframe (config keyframe_criterion).
        self.keyframe_criterion = keyframe_criterion
        self.overlap_gate = overlap_gate
        self.keyframe_max_gap_frames = keyframe_max_gap_frames
        self.progress_callback = progress_callback

        self.store_sift = get_cfg(self.config, "database.store_sift_features", False)
        self.sift_max_kps = get_cfg(self.config, "database.sift_max_keypoints", 2048)

        # Loop state (was local to build_from_video)
        self.current_pose = np.eye(3, dtype=np.float32)
        self.prev_features: dict | None = None
        self.saved_count = 0
        self.frame_index_map: list[int] = []
        self._last_depth_scale: float | None = None
        # Overlap criterion state: pose of the last KEPT keyframe and how many
        # frames have been processed since it.
        self._pose_at_last_keyframe: np.ndarray | None = None
        self._frames_since_keyframe = 0

    def process(self, p_idx: int, p_frame, p_frame_rgb, p_static_mask) -> None:
        """Processes one frame: feature extraction, pose calculation, keyframe selection."""
        features = self.feature_extractor.extract_features(p_frame_rgb, p_static_mask)
        features["coords_2d"] = features["keypoints"]

        # Patchify descriptors
        if self.patchify is not None:
            features["patch_descriptors"] = self.patchify.compute_patch_descriptors(p_frame_rgb)

        # Depth scale estimation (reuse scale across interval to save compute)
        features["depth_scale"] = np.float32(
            self._last_depth_scale if self._last_depth_scale is not None else 1.0
        )
        if self.depth_estimator is not None:
            depth_every = max(1, int(get_cfg(self.config, "database.depth_every_n", 10)))
            if p_idx % depth_every == 0 or self._last_depth_scale is None:
                try:
                    self._last_depth_scale = float(
                        self.depth_estimator.get_relative_scale(p_frame_rgb)
                    )
                    features["depth_scale"] = np.float32(self._last_depth_scale)
                except Exception as e:
                    logger.warning(f"Depth estimation failed for frame {p_idx}: {e}")

        if self.kp_writer is not None:
            kp_frame = self.draw_keypoints(
                p_frame, features["keypoints"], p_static_mask, p_idx, self.num_frames
            )
            if self.kp_scale != 1.0:
                kp_w = int(self.width * self.kp_scale)
                kp_h = int(self.height * self.kp_scale)
                kp_frame = cv2.resize(kp_frame, (kp_w, kp_h), interpolation=cv2.INTER_AREA)
            self.kp_writer.write(kp_frame)

        if p_idx == 0 or self.prev_features is None:
            self.current_pose = np.eye(3, dtype=np.float64)
            save_this_frame = self.always_save_first
        else:
            H_step = self.compute_inter_frame_h(self.prev_features, features)
            if H_step is not None:
                self.current_pose = self.current_pose @ H_step.astype(np.float64)
                if not self.use_keyframe_selection:
                    save_this_frame = True
                elif self.keyframe_criterion == "overlap":
                    save_this_frame = self._overlap_says_keyframe(p_idx)
                else:
                    save_this_frame = self.is_significant_motion(H_step, self.width, self.height)
            else:
                logger.warning(f"Frame {p_idx}: inter-frame match failed, reusing previous pose")
                save_this_frame = True

        self.prev_features = features

        # Always write pose for full propagation chain
        self.db_writer.write_pose(p_idx, self.current_pose)

        if save_this_frame:
            # SIFT features extracted only for keyframes being saved
            if self.store_sift:
                try:
                    from src.localization.matcher import extract_sift_features

                    sift_feats = extract_sift_features(
                        p_frame_rgb, p_static_mask, self.sift_max_kps
                    )
                    features["sift_keypoints"] = sift_feats["keypoints"]
                    features["sift_descriptors"] = sift_feats["descriptors"]
                except Exception as e:
                    logger.warning(f"SIFT extraction failed for frame {p_idx}: {e}")
            self.frame_index_map.append(p_idx)
            # Save using original frame index p_idx for calibration mapping
            self.db_writer.save_frame_data(p_idx, features, self.current_pose)
            self.saved_count += 1
            self._pose_at_last_keyframe = self.current_pose.copy()
            self._frames_since_keyframe = 0

            if self.saved_count % 100 == 0:
                progress_pct = int((p_idx + 1) / self.num_frames * 100)
                logger.info(
                    f"Saved {self.saved_count} keyframes from {p_idx + 1}/{self.num_frames} "
                    f"processed ({progress_pct}%)"
                )
        else:
            self._frames_since_keyframe += 1

        progress_percent = int((p_idx + 1) / self.num_frames * 100)
        if self.progress_callback:
            self.progress_callback(progress_percent)

    def _overlap_says_keyframe(self, p_idx: int) -> bool:
        """Determines if frame is a keyframe based on overlap with last saved keyframe."""
        if self._pose_at_last_keyframe is None or self.overlap_gate is None:
            return True

        if (
            self.keyframe_max_gap_frames > 0
            and self._frames_since_keyframe >= self.keyframe_max_gap_frames
        ):
            logger.warning(
                f"Frame {p_idx}: forced keyframe — {self._frames_since_keyframe} frames "
                f"without one (keyframe_max_gap_frames)"
            )
            return True

        try:
            H_rel = np.linalg.inv(self._pose_at_last_keyframe) @ self.current_pose
        except np.linalg.LinAlgError:
            logger.warning(f"Frame {p_idx}: singular keyframe pose — forcing keyframe")
            return True

        return bool(self.overlap_gate(H_rel, self.width, self.height))
