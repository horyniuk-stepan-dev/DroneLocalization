"""Database-builder configuration."""

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
    use_decord: bool = True
    decode_batch_size: int = 32
    # A6: Depth-Anything на кожному K-му кадрі збудови (масштаб змінюється
    # повільно, повний інференс на кожен кадр — марна трата 20-35% часу)
    depth_every_n: int = 10
    use_lancedb: bool = True
    lancedb_batch_size: int = 64
    lancedb_index_min_frames: int = 256
    yolo_batch_size: int = 1
