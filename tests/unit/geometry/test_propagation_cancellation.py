from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.workers.propagation_pipeline import PropagationPipeline


def pipeline():
    database = SimpleNamespace(metadata={}, get_num_frames=lambda: 2)
    anchor = SimpleNamespace(frame_id=0, affine_matrix=np.eye(2, 3))
    return PropagationPipeline(
        database,
        SimpleNamespace(anchors=[anchor]),
        Mock(),
        completed_callback=Mock(),
        cancelled_callback=Mock(),
        error_callback=Mock(),
    )


def test_cancel_before_start_does_not_access_features(monkeypatch):
    p = pipeline()
    prefetch = Mock()
    monkeypatch.setattr(p, "_prefetch_features", prefetch)
    p.stop()
    p._propagate()
    prefetch.assert_not_called()
    p._cancelled_cb.assert_called_once()
    p._completed_cb.assert_not_called()


def test_cancel_during_prefetch_does_not_build_or_save(monkeypatch):
    p = pipeline()

    def prefetch(_):
        p.stop()
        return {0: {}, 1: {}}

    monkeypatch.setattr(p, "_prefetch_features", prefetch)
    build, save = Mock(), Mock()
    monkeypatch.setattr(p, "_build_temporal_edges", build)
    monkeypatch.setattr(p, "_save_to_hdf5", save)
    p._propagate()
    build.assert_not_called()
    save.assert_not_called()
    p._completed_cb.assert_not_called()
    p._cancelled_cb.assert_called_once()


def test_cancel_from_progress_callback_stops_before_next_stage(monkeypatch):
    p = pipeline()
    p._progress_cb = lambda *_: p.stop()
    prefetch = Mock()
    monkeypatch.setattr(p, "_prefetch_features", prefetch)
    p._propagate()
    prefetch.assert_not_called()
    p._cancelled_cb.assert_called_once()
