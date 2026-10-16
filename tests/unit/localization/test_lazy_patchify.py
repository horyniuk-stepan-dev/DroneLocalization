"""localization.patchify_lazy: the patch search runs only after the standard candidates fail.

The eager path computes the 14 crop descriptors (patchify) before every retrieval
verification; on the 2026-10-16 GTX 1650 replay the patchify-only candidates were
reached on 29 of 133 such frames. The lazy path verifies the standard candidates
first and runs the patch search on the same view only when they fail.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("faiss")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fixtures"))
from localizer_fakes import build_localizer, synthetic_frame  # noqa: E402

NUM_FRAMES = 8  # FakeDatabase


def _instrumented(monkeypatch, lazy, standard_ok, add_patch_frame=True):
    config = {"localization": {"retrieval_top_k": 3}}
    if lazy:
        config["localization"]["patchify_lazy"] = True
    loc = build_localizer(config)
    calls = {"expand": 0, "verify": [], "extra": []}
    retriever = loc._candidate_retriever
    monkeypatch.setattr(retriever, "patchify_for", lambda source_id=None: object())

    def expand(rotated_frame, base, top_k, source_id=None, desc_cache=None):
        calls["expand"] += 1
        known = {int(frame) for frame, _ in base}
        extra = [f for f in range(NUM_FRAMES) if f not in known][:1] if add_patch_frame else []
        calls["extra"] = extra
        return list(base) + [(frame, 0.1) for frame in extra]

    monkeypatch.setattr(retriever, "expand", expand)
    real_verify = loc._geometric_verifier.verify

    def verify(query_features, candidates, database, ref_cache=None, tag=None):
        calls["verify"].append((tag, [int(c[0]) for c in candidates]))
        if tag == "retrieval" and not standard_ok:
            return None
        return real_verify(query_features, candidates, database, ref_cache=ref_cache, tag=tag)

    monkeypatch.setattr(loc._geometric_verifier, "verify", verify)
    return loc, calls


def test_lazy_skips_the_patch_search_when_the_standard_candidates_verify(monkeypatch):
    loc, calls = _instrumented(monkeypatch, lazy=True, standard_ok=True)
    result = loc.localize_frame(synthetic_frame(1))

    assert result.get("success")
    assert calls["expand"] == 0
    assert [tag for tag, _ in calls["verify"]] == ["retrieval"]


def test_lazy_runs_the_patch_search_after_the_standard_candidates_fail(monkeypatch):
    loc, calls = _instrumented(monkeypatch, lazy=True, standard_ok=False)
    result = loc.localize_frame(synthetic_frame(1))

    assert calls["expand"] == 1
    assert [tag for tag, _ in calls["verify"]][:2] == ["retrieval", "patchify"]
    extra = calls["extra"]
    assert len(extra) == 1 and extra[0] not in calls["verify"][0][1]
    assert calls["verify"][1][1] == extra  # only the patchify-only candidates
    assert result.get("success")
    assert result.get("matched_frame") == extra[0]


def test_lazy_without_new_candidates_verifies_nothing_more(monkeypatch):
    loc, calls = _instrumented(monkeypatch, lazy=True, standard_ok=False, add_patch_frame=False)
    loc.localize_frame(synthetic_frame(1))

    assert calls["expand"] >= 1
    assert "patchify" not in [tag for tag, _ in calls["verify"]]


def test_eager_default_searches_patches_before_verifying(monkeypatch):
    loc, calls = _instrumented(monkeypatch, lazy=False, standard_ok=True)
    result = loc.localize_frame(synthetic_frame(1))

    assert result.get("success")
    assert calls["expand"] == 1
    assert calls["verify"][0][0] == "retrieval"
    assert calls["extra"][0] in calls["verify"][0][1]  # merged list, as before
