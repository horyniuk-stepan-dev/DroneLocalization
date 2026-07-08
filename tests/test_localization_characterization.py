"""Characterization snapshot for Localizer — guards behaviour during the 1.1 refactor.

Runs on a full install (build_localizer imports Localizer -> torch/faiss). Workflow:
    CAPTURE=1 pytest tests/test_localization_characterization.py   # BEFORE refactor: save baseline
    pytest tests/test_localization_characterization.py             # after each step: must match

Any behavioural drift in localize_frame / localize_optical_flow fails the assert.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Import the fakes directly from tests/fixtures to avoid a name clash with the
# stray site-packages ``tests`` package that ultralytics installs.
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from localizer_fakes import build_localizer, synthetic_frame  # noqa: E402

_BASELINE = Path(__file__).parent / "fixtures" / "localization_baseline.json"


def _clean(obj: Any) -> Any:
    """JSON-comparable + tolerant of last-ULP float noise on the same machine."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 9)
    return obj


def run_scenarios() -> dict[str, Any]:
    snap: dict[str, Any] = {}

    # A) success/deterministic result on a fresh localizer
    loc = build_localizer()
    snap["frame1"] = loc.localize_frame(synthetic_frame(1))
    ls = loc.last_state
    snap["last_state"] = None if ls is None else {
        k: ls.get(k) for k in ("candidate_id", "inliers", "global_angle", "source_id")
    }

    # B) A3 temporal prior: a second identical keyframe should reuse the angle
    #    (1 forward pass, not a full 4-angle scan). Track via the fake's call counter.
    calls_before = loc.feature_extractor.calls
    snap["frame1_repeat"] = loc.localize_frame(synthetic_frame(1))
    snap["extractor_calls_second_keyframe"] = loc.feature_extractor.calls - calls_before

    # C) optical flow after a successful keyframe
    snap["optical_flow"] = loc.localize_optical_flow(
        dx_px=5.0, dy_px=3.0, dt=1.0, rot_width=640, rot_height=480
    )

    # D) fresh localizer, reset_session then localize another frame
    loc2 = build_localizer()
    loc2.reset_session()
    snap["after_reset"] = loc2.localize_frame(synthetic_frame(2))

    return _clean(snap)


def test_characterization_matches_baseline() -> None:
    snap = run_scenarios()
    if os.environ.get("CAPTURE") == "1" or not _BASELINE.exists():
        _BASELINE.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
        pytest.skip(f"baseline captured at {_BASELINE} ({len(snap)} scenarios)")
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert snap == baseline, "localize_frame behaviour changed vs baseline (refactor introduced a diff)"
