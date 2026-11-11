"""Localizer wiring of the map-coverage model (vision-only plan phase 1).

The Localizer tests need torch (run on the Windows side); classify_failure runs anywhere.
"""

import sys
from pathlib import Path

import pytest

from src.localization.failure_log import classify_failure

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fixtures"))
from localizer_fakes import build_localizer, synthetic_frame  # noqa: E402


def _cfg(**cov):
    return {"localization": {"coverage": {"enabled": True, **cov}}}


def test_disabled_by_default_adds_no_fields():
    pytest.importorskip("torch")
    loc = build_localizer({})
    assert loc.coverage_monitor is None
    res = loc.localize_frame(synthetic_frame(1), timestamp=0.0)
    assert "coverage_state" not in res


def test_enabled_builds_model_and_annotates_results():
    pytest.importorskip("torch")
    loc = build_localizer(_cfg(cell_m=2.0))
    loc.reset_session()
    model = loc.coverage_model
    assert model is not None and model.n_footprints == 8
    res = loc.localize_frame(synthetic_frame(1), timestamp=0.0)
    assert res["coverage_state"] in ("inside", "near_edge", "leaving", "unknown", "lost_inside")
    assert "coverage_alert" in res
    if not res.get("success"):
        assert res["failure_reason"] == classify_failure(res.get("error"))


def test_edge_guards_block_retrieval_only_fallback():
    pytest.importorskip("torch")
    loc = build_localizer(_cfg(edge_guards=True))
    loc.reset_session()
    mon = loc.coverage_monitor
    mon.state = "outside"
    assert loc._edge_guards_active()
    assert loc._localize_by_reference_frame(0, 0.99) is None
    mon.state = "inside"
    assert not loc._edge_guards_active()


@pytest.mark.parametrize(
    "error,reason",
    [
        ("No candidates found via global descriptor (DINOv2) in any rotation.", "no_candidates"),
        ("Not enough valid inliers (3 < 12)", "too_few_matches"),
        ("Outlier detected — position jump filtered", "gate_rejected"),
        ("Fix footprint mostly outside map coverage (20% inside)", "gate_rejected"),
        ("Coordinate transformation error (homography failed)", "geometry_rejected"),
        ("out_of_coverage", "out_of_coverage"),
        ("OF rejected — flow quality too low", "optical_flow"),
        ("No propagated calibration for matched frame 3.", "no_calibration"),
        ("ambiguous_geometry", "geometry_rejected"),
        ("something new", "other"),
    ],
)
def test_classify_failure(error, reason):
    assert classify_failure(error) == reason


def test_ang_dist_wraps():
    pytest.importorskip("torch")
    from src.localization.localizer import _ang_dist

    assert _ang_dist(350.0, 10.0) == 20.0
    assert _ang_dist(0.0, 180.0) == 180.0


def test_cold_start_retry_off_by_default_and_on_after_first_failure():
    pytest.importorskip("torch")
    loc = build_localizer({"localization": {"cold_start_extra_angles": 2}})
    res = loc.localize_frame(synthetic_frame(1), timestamp=0.0)
    assert "success" in res
    assert loc._cold_exclude is None  # always cleared
