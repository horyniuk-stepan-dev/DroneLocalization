"""Conformance tests for the structural Protocols in src.interfaces.

The fake-based checks run everywhere (no heavy deps). The real-implementation
checks import torch/cv2-backed modules and are skipped where those are absent
(e.g. the pure-Python CI sandbox); they run on full installs and in gpu CI.
"""

import numpy as np
import pytest

from src.interfaces import (
    FrameDatabase,
    GlobalDescriptorExtractor,
    LocalFeatureExtractor,
    Retriever,
)


# ── fake implementers (always available) ──────────────────────────────────────
class _FakeRetriever:
    def find_similar_frames(self, query_desc, top_k=5):
        return []

    def add_descriptor(self, query_desc, frame_id):
        pass


class _FakeFeatureExtractor:
    def extract_global_descriptor(self, image):
        return np.zeros(4)

    def extract_global_descriptors_multi(self, images):
        return np.zeros((len(images), 4))

    def extract_local_features(self, image, static_mask=None):
        return {}


class _FakeDatabase:
    def get_local_features(self, frame_id):
        return {}

    def get_frame_affine(self, frame_id):
        return None

    def get_frame_size(self, frame_id):
        return (0, 0)

    def get_num_frames(self):
        return 0


def test_fakes_conform():
    assert isinstance(_FakeRetriever(), Retriever)
    assert isinstance(_FakeFeatureExtractor(), GlobalDescriptorExtractor)
    assert isinstance(_FakeFeatureExtractor(), LocalFeatureExtractor)
    assert isinstance(_FakeDatabase(), FrameDatabase)


def test_missing_method_does_not_conform():
    class _NoAddDescriptor:
        def find_similar_frames(self, query_desc, top_k=5):
            return []

    assert not isinstance(_NoAddDescriptor(), Retriever)


# ── real implementers (skipped when heavy deps are unavailable) ───────────────
def _real_classes():
    try:
        from src.database.database_loader import DatabaseLoader
        from src.localization.matcher import FastRetrieval, LanceDBRetrieval
        from src.models.wrappers.feature_extractor import FeatureExtractor
    except Exception as exc:  # torch / lancedb / cv2 backends absent
        pytest.skip(f"heavy deps unavailable: {exc}")
    return FastRetrieval, LanceDBRetrieval, FeatureExtractor, DatabaseLoader


def test_real_retrievers_conform():
    FastRetrieval, LanceDBRetrieval, _, _ = _real_classes()
    assert issubclass(FastRetrieval, Retriever)
    assert issubclass(LanceDBRetrieval, Retriever)


def test_real_feature_extractor_conforms():
    _, _, FeatureExtractor, _ = _real_classes()
    assert issubclass(FeatureExtractor, GlobalDescriptorExtractor)
    assert issubclass(FeatureExtractor, LocalFeatureExtractor)


def test_real_database_conforms():
    *_, DatabaseLoader = _real_classes()
    assert issubclass(DatabaseLoader, FrameDatabase)
