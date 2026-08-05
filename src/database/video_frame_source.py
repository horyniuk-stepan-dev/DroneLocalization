"""Video decoding for the database build (decord with cv2 fallback).

Extracted verbatim from ``DatabaseBuilder.build_from_video`` (IMPROVEMENT_PLAN
item 1.3, splitting ``db_builder``). This module owns ONLY decoding and frame
prefetching: opening the video, reporting geometry/FPS, and feeding a bounded
queue of ``(slot_index, (frame_bgr, frame_rgb))`` pairs from a daemon thread.

It knows nothing about models, HDF5 or keyframes. The slot index it emits is
``video_frame_index // frame_step`` — the DB slot identity pinned by
``tests/integration/test_db_builder_characterization.py`` (invariant 3).
"""

from __future__ import annotations

from queue import Queue
from threading import Thread

import cv2

from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)

#: Sentinel put on the queue after the last frame: ``(EOF_INDEX, None)``.
EOF_INDEX = -1


class VideoFrameSource:
    """Opens a video and prefetches decoded frames into a bounded queue.

    Args:
        video_path: path to the reference video.
        frame_step: keep every ``frame_step``-th frame (values < 1 are clamped
            to 1, as in the original builder).
        use_decord: try the decord reader first; falls back to cv2 on any
            import/initialisation failure.
        decode_batch_size: decord batch read size.
        prefetch_size: max queued frames (backpressure for the consumer).
    """

    def __init__(
        self,
        video_path: str,
        frame_step: int = 3,
        use_decord: bool = True,
        decode_batch_size: int = 32,
        prefetch_size: int = 32,
    ):
        self.video_path = video_path
        self.frame_step = frame_step if frame_step >= 1 else 1
        self.decode_batch_size = decode_batch_size
        self.prefetch_size = prefetch_size

        self._vr = None
        self._cap = None
        self._thread: Thread | None = None
        self.use_decord = use_decord

        if self.use_decord:
            try:
                import decord

                decord.bridge.set_bridge("numpy")
                # FFMPEG multi-threaded CPU decode is usually the most stable fallback
                # GPU decode requires custom decord builds on Windows
                self._vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                logger.info("Decord VideoReader initialized successfully.")
            except ImportError:
                logger.warning("decord not installed, falling back to cv2.VideoCapture")
                self.use_decord = False
            except Exception as e:
                logger.warning(
                    f"Failed to initialize decord VideoReader: {e}. Falling back to cv2.VideoCapture"
                )
                self.use_decord = False

        if not self.use_decord:
            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                logger.error(
                    f"Failed to open video: {video_path}. "
                    f"Check that the file exists and uses a supported codec (H.264/H.265 recommended)."
                )
                raise ValueError(f"Failed to open video: {video_path}")

        if self.use_decord:
            self.total_frames = len(self._vr)
            # Sample first frame to get dims
            h, w, _c = self._vr.get_batch([0]).shape[1:]
            self.width, self.height = int(w), int(h)
            self.original_fps = self._vr.get_avg_fps()
        else:
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.original_fps = self._cap.get(cv2.CAP_PROP_FPS)

        # Actual frame count to process
        self.num_frames = (self.total_frames + self.frame_step - 1) // self.frame_step
        self.effective_fps = self.original_fps / self.frame_step

        if self.num_frames <= 0:
            self.release()
            logger.error(
                f"Invalid frame count ({self.num_frames}). "
                f"Video might be corrupted or uses unsupported codec."
            )
            raise ValueError(
                f"Could not parse video '{video_path}'. File may be corrupt or use unsupported codec."
            )

        logger.info(
            f"Video properties: {self.width}x{self.height}, "
            f"{self.total_frames} total frames, {self.original_fps:.2f} FPS"
        )
        logger.info(
            f"Processing with step={self.frame_step} -> {self.num_frames} frames to process "
            f"({self.effective_fps:.2f} effective FPS)"
        )

    # ------------------------------------------------------------------
    # Prefetching
    # ------------------------------------------------------------------

    def start_prefetch(self) -> Queue:
        """Starts the decode thread and returns the queue it feeds.

        Queue items are ``(slot_index, (frame_bgr, frame_rgb))``; the stream
        always terminates with ``(EOF_INDEX, None)``.
        """
        queue: Queue = Queue(maxsize=self.prefetch_size)
        self._thread = Thread(
            target=self._prefetch_frames, args=(queue,), name="DbBuildPrefetch", daemon=True
        )
        self._thread.start()
        return queue

    def _prefetch_frames(self, frame_queue: Queue) -> None:
        if self.use_decord:
            # Decord provides batched read
            indices = list(range(0, self.total_frames, self.frame_step))
            for chunk_start in range(0, len(indices), self.decode_batch_size):
                chunk_indices = indices[chunk_start : chunk_start + self.decode_batch_size]

                with Telemetry.profile("video_read"):
                    # Decord returns RGB (B, H, W, C)
                    frames_rgb = self._vr.get_batch(chunk_indices).asnumpy()

                for i, frame_rgb in enumerate(frames_rgb):
                    orig_frame_idx = chunk_indices[i] // self.frame_step
                    with Telemetry.profile("rgb_to_bgr"):
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    frame_queue.put((orig_frame_idx, (frame_bgr, frame_rgb)))
        else:
            for i in range(self.total_frames):
                with Telemetry.profile("video_read"):
                    ret, frame = self._cap.read()
                if not ret:
                    break

                if i % self.frame_step != 0:
                    continue

                with Telemetry.profile("bgr_to_rgb"):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig_frame_idx = i // self.frame_step
                frame_queue.put((orig_frame_idx, (frame, frame_rgb)))

        frame_queue.put((EOF_INDEX, None))

    def join(self, timeout: float = 5) -> None:
        """Waits for the prefetch thread (no-op if it was never started)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def release(self) -> None:
        """Releases the underlying decoder."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._vr = None
