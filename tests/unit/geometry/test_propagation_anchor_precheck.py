"""Missing anchor slots stop propagation before any matching (fail fast)."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.workers.propagation_pipeline import PropagationPipeline


def pipeline(anchor_ids):
    database = SimpleNamespace(metadata={}, get_num_frames=lambda: 4)
    anchors = [SimpleNamespace(frame_id=i, affine_matrix=np.eye(2, 3)) for i in anchor_ids]
    return PropagationPipeline(
        database,
        SimpleNamespace(anchors=anchors),
        Mock(),
        completed_callback=Mock(),
        cancelled_callback=Mock(),
        error_callback=Mock(),
    )


def test_missing_anchor_slot_fails_before_matching(monkeypatch):
    p = pipeline([0, 2])
    monkeypatch.setattr(p, "_prefetch_features", lambda _: {0: {}, 1: {}, 3: {}})
    temporal, loops, save = Mock(), Mock(), Mock()
    monkeypatch.setattr(p, "_build_temporal_edges", temporal)
    monkeypatch.setattr(p, "_detect_loop_closures", loops)
    monkeypatch.setattr(p, "_save_to_hdf5", save)

    p._propagate()

    temporal.assert_not_called()
    loops.assert_not_called()
    save.assert_not_called()
    p._completed_cb.assert_not_called()
    p._error_cb.assert_called_once()
    message = p._error_cb.call_args[0][0]
    assert "no features at exact DB slots" in message
    assert "[2]" in message


def test_present_anchor_slots_reach_matching(monkeypatch):
    p = pipeline([0, 3])
    monkeypatch.setattr(p, "_prefetch_features", lambda _: {0: {}, 1: {}, 3: {}})

    class ReachedMatchingError(Exception):
        pass

    def temporal(*_args, **_kwargs):
        raise ReachedMatchingError

    monkeypatch.setattr(p, "_build_temporal_edges", temporal)
    with pytest.raises(ReachedMatchingError):
        p._propagate()
    p._error_cb.assert_not_called()
