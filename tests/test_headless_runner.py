"""Regression tests for HeadlessRunner config access.

The P0 bug: APP_CONFIG is a plain dict (APP_SETTINGS.model_dump()), but
_setup_project / _build_localizer used it as a Pydantic model
(.model_dump(), .models), raising AttributeError before any localization ran.

Two layers:
  * test_app_config_is_a_plain_dict_contract - pure config, runs anywhere.
  * test_headless_setup_and_build_use_dict_config - drives the real methods
    with every heavy collaborator mocked; needs PyQt6/torch/etc., so it
    self-skips where those are unavailable (e.g. the Linux CI sandbox).
"""
from unittest.mock import MagicMock, patch

import pytest

from config import APP_CONFIG, APP_SETTINGS, get_cfg

try:
    import src.core.headless_runner as _hr
except Exception as _e:  # PyQt6 / torch / lancedb absent in this environment
    _hr = None
    _HR_SKIP = f"HeadlessRunner deps unavailable: {type(_e).__name__}"
else:
    _HR_SKIP = None


def test_app_config_is_a_plain_dict_contract():
    """APP_CONFIG must stay a plain dict; the Pydantic model is APP_SETTINGS.

    This is the invariant headless_runner violated. No heavy deps needed.
    """
    assert isinstance(APP_CONFIG, dict)
    assert not hasattr(APP_CONFIG, "model_dump")
    assert hasattr(APP_SETTINGS, "model_dump")
    # The exact accesses _build_localizer performs must work on the dict:
    assert isinstance(get_cfg(APP_CONFIG, "models.cesp.enabled", False), bool)
    merged = {**APP_CONFIG, "_model_manager": object()}
    assert "_model_manager" in merged and "models" in merged


_MOCK_TARGETS = (
    "QCoreApplication", "ModelManager", "MultiAnchorCalibration",
    "CoordinatesBroker", "ProjectManager", "MultiDatabaseManager",
    "MultiCalibrationManager", "DatabaseLoader", "FeatureExtractor",
    "FeatureMatcher", "Localizer",
)


@pytest.mark.skipif(_hr is None, reason=_HR_SKIP or "")
def test_headless_setup_and_build_use_dict_config():
    """Drive real _setup_project (multi-source) + _build_localizer with every
    heavy collaborator mocked. If APP_CONFIG.model_dump()/.models is
    reintroduced, evaluating those args raises AttributeError and fails here.
    """
    mocks = {name: patch.object(_hr, name) for name in _MOCK_TARGETS}
    started = {name: p.start() for name, p in mocks.items()}
    try:
        runner = _hr.HeadlessRunner(project_dir="/tmp/proj", video_source="v.mp4")

        # Force the multi-source branch (exercises the line-60 config arg).
        pm = started["ProjectManager"].return_value
        pm.load_project.return_value = True
        pm.settings.get_enabled_sources.return_value = [
            MagicMock(source_id="main"), MagicMock(source_id="cam2"),
        ]

        dbm = started["MultiDatabaseManager"].return_value
        dbm.all_source_ids = ["main"]
        db = MagicMock(is_propagated=True, metadata={})
        dbm.get_database.return_value = db

        calib = MagicMock()
        calib.converter.is_initialized = True
        started["MultiCalibrationManager"].return_value.get.return_value = calib

        runner._setup_project()
        assert runner.database is db

        localizer = runner._build_localizer()
        assert localizer is started["Localizer"].return_value

        # Config handed to Localizer must be a plain dict carrying the manager.
        _, kwargs = started["Localizer"].call_args
        assert isinstance(kwargs["config"], dict)
        assert kwargs["config"]["_model_manager"] is runner.model_manager
        assert "models" in kwargs["config"]
    finally:
        for p in mocks.values():
            p.stop()
