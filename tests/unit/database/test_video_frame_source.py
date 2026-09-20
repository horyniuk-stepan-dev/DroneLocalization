from queue import Queue

import pytest

from src.database.video_frame_source import EOF_INDEX, VideoFrameSource


class _FailingDecordReader:
    def get_batch(self, _indices):
        raise RuntimeError("synthetic decoder failure")


def test_prefetch_failure_always_terminates_queue_and_propagates_error():
    source = VideoFrameSource.__new__(VideoFrameSource)
    source.use_decord = True
    source.total_frames = 1
    source.frame_step = 1
    source.decode_batch_size = 1
    source._vr = _FailingDecordReader()
    source._prefetch_error = None

    frame_queue = Queue()
    source._prefetch_frames(frame_queue)

    assert frame_queue.get_nowait() == (EOF_INDEX, None)
    with pytest.raises(RuntimeError, match="Video decoding failed") as exc_info:
        source.raise_if_failed()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
