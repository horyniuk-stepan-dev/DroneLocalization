"""Database-builder configuration."""

from typing import Literal

from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    frame_step: int = 30
    prefetch_queue_size: int = 32
    keypoint_video_scale: float = 0.5
    inter_frame_min_matches: int = 15
    inter_frame_ransac_thresh: float = 3.0
    keyframe_min_translation_px: float = 15.0
    keyframe_min_rotation_deg: float = 1.5
    keyframe_always_save_first: bool = True
    # --- Adaptive sampling by real image displacement (flag-gated, default OFF) ---
    # "step"    — legacy: motion measured between ADJACENT processed frames
    #             (keyframe_min_translation_px / _min_rotation_deg above).
    # "overlap" — motion accumulated since the LAST KEPT keyframe: a frame is
    #             kept once its overlap with that keyframe drops to
    #             keyframe_max_overlap. This is what "the image has moved by
    #             50%" actually means; the "step" criterion cannot express it,
    #             because adjacent-frame displacement never reaches half a frame.
    keyframe_criterion: Literal["step", "overlap"] = "step"
    # Fraction of the keyframe's area still visible in the current frame.
    # 0.5 = keep a frame once half the picture is new. Lower = fewer keyframes.
    keyframe_max_overlap: float = 0.5
    # Safety net for "overlap": force a keyframe after this many CONSECUTIVE
    # skipped frames, i.e. kept keyframes are at most N+1 frames apart.
    # Degrading matching (farmland, water, fog) makes the accumulated drift, and an over-optimistic overlap estimate
    # would otherwise open exactly the anchor gap that broke the map in the
    # VO-guards stage-8 incident. 0 disables the limit.
    keyframe_max_gap_frames: int = 60
    use_decord: bool = True
    decode_batch_size: int = 32
    # A6: Depth-Anything на кожному K-му кадрі збудови (масштаб змінюється
    # повільно, повний інференс на кожен кадр — марна трата 20-35% часу)
    depth_every_n: int = 10
    use_lancedb: bool = True
    # RESEARCH 2.2: зберігати SIFT-ознаки keyframe-ів (група sift_features у
    # HDF5) для аварійного фолбека localization.sift_fallback. ~1 МБ/кадр f16.
    store_sift_features: bool = False
    sift_max_keypoints: int = 2048
    lancedb_batch_size: int = 64
    lancedb_index_min_frames: int = 256
    yolo_batch_size: int = 1
