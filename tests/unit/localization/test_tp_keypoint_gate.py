"""localization.temporal_prior_kp_min_inliers: the prior-check keypoint cap only after a strong fix.

A plain localization.temporal_prior_max_keypoints=2048 cut the wb2019 recovery on
2026-10-17 (161 -> 91 fixes): its borderline spatial-prior matches need every
keypoint. With the gate the cap applies only when the previous frame's fix had at
least N inliers; after a weak or failed fix the prior check uses all keypoints.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("faiss")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fixtures"))
from localizer_fakes import build_localizer, synthetic_frame  # noqa: E402


def _loc(cap, min_inliers):
    return build_localizer(
        {
            "localization": {
                "temporal_prior_max_keypoints": cap,
                "temporal_prior_kp_min_inliers": min_inliers,
            }
        }
    )


@pytest.mark.parametrize(
    "cap,min_inliers,prev,expected",
    [
        (0, 0, 500, 0),  # no cap configured
        (2048, 0, 0, 2048),  # ungated: always (behaviour before the gate)
        (2048, 150, 0, 0),  # after a failure: every keypoint
        (2048, 150, 149, 0),  # after a weak fix: every keypoint
        (2048, 150, 150, 2048),
        (2048, 150, 800, 2048),
    ],
)
def test_budget_follows_the_previous_fix(cap, min_inliers, prev, expected):
    loc = _loc(cap, min_inliers)
    loc._prev_fix_inliers = prev
    assert loc._tp_keypoint_budget() == expected


def test_localize_frame_records_the_fix_inliers_and_resets_on_failure(monkeypatch):
    loc = _loc(2048, 10)
    result = loc.localize_frame(synthetic_frame(1))
    assert result.get("success")
    assert loc._prev_fix_inliers == int(result["inliers"]) > 0

    monkeypatch.setattr(loc._geometric_verifier, "verify", lambda *a, **k: None)
    monkeypatch.setattr(loc, "_localize_by_reference_frame", lambda *a, **k: None)
    failed = loc.localize_frame(synthetic_frame(2))
    assert not failed.get("success")
    assert loc._prev_fix_inliers == 0

    loc._prev_fix_inliers = 500
    loc.reset_session()
    assert loc._prev_fix_inliers == 0
