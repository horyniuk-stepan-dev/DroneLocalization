"""The in-frame retry of localize_frame reuses the first attempt's caches.

localize_frame retries once with the scale/angle priors cleared when the first
attempt ends in "Not enough valid inliers" or a retrieval-only fallback. The
retry used to start with empty caches, so ALIKED and the 14 patchify crop
passes ran again for a view the first attempt had already processed.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("faiss")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fixtures"))
from localizer_fakes import build_localizer, synthetic_frame  # noqa: E402

from src.localization.candidate_retriever import CandidateRetriever  # noqa: E402


def test_retry_receives_the_first_attempts_caches(monkeypatch):
    loc = build_localizer()
    loc._scale_manager._prior = 1.0  # a prior exists -> a failed attempt is retried
    seen = []

    def attempt(*args, _caches=None, **kwargs):
        seen.append(_caches)
        _caches.setdefault("features", {})[("view", 0.0, 1.0)] = "from first attempt"
        return {"success": False, "error": "Not enough valid inliers (3 < 12)"}

    monkeypatch.setattr(loc, "_localize_frame_impl", attempt)
    loc.localize_frame(synthetic_frame(1))

    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert seen[1]["features"][("view", 0.0, 1.0)] == "from first attempt"


def test_without_a_prior_there_is_one_attempt(monkeypatch):
    loc = build_localizer()
    calls = []
    monkeypatch.setattr(
        loc,
        "_localize_frame_impl",
        lambda *a, _caches=None, **k: calls.append(_caches) or {"success": False, "error": "x"},
    )
    loc.localize_frame(synthetic_frame(1))
    assert len(calls) == 1


class _CountingPatchify:
    grids = ((1, 1), (2, 2), (3, 3))
    descriptor_dim = 4

    def __init__(self):
        self.computed = 0

    def compute_patch_descriptors(self, frame):
        self.computed += 1
        return "descriptors"

    def search(self, descriptors, top_k=10):
        return [(7, 0.9)]


def test_expand_computes_crop_descriptors_once_per_view_and_cache():
    patchify = _CountingPatchify()
    retriever = CandidateRetriever(None, None, patchify, {})
    view, other_view = object(), object()
    cache: dict = {}

    retriever.expand(view, [(1, 0.5)], top_k=5, desc_cache=cache)
    retriever.expand(view, [(1, 0.5)], top_k=5, desc_cache=cache)
    assert patchify.computed == 1
    retriever.expand(other_view, [(1, 0.5)], top_k=5, desc_cache=cache)
    assert patchify.computed == 2
    retriever.expand(view, [(1, 0.5)], top_k=5)  # no cache: old behaviour
    assert patchify.computed == 3


def test_patch_cache_entry_holds_its_view_and_ignores_a_reused_id():
    patchify = _CountingPatchify()
    view, other = object(), object()
    cache: dict = {}
    CandidateRetriever._patch_descriptors(patchify, view, cache)
    ((key, entry),) = cache.items()
    assert entry[0] is view  # kept alive: its id cannot be handed to another array

    # An entry under ``other``'s id that belongs to a different view is not served.
    cache[(id(other),) + key[1:]] = (view, "stale")
    assert CandidateRetriever._patch_descriptors(patchify, other, cache) == "descriptors"
    assert patchify.computed == 2
