"""Telemetry reports: latest + per-run copy, explicit path, no writes from tests."""

import json
import os

from src.utils.telemetry import TELEMETRY_DIR_ENV, _TelemetryTracker


def _tracker_with_stage():
    tracker = _TelemetryTracker()
    with tracker.profile("stage"):
        pass
    return tracker


def test_default_dump_writes_latest_and_per_run_copy(tmp_path, monkeypatch):
    monkeypatch.setenv(TELEMETRY_DIR_ENV, str(tmp_path))
    _tracker_with_stage().dump_report()

    latest = tmp_path / "telemetry_report.json"
    runs = list((tmp_path / "telemetry").glob("telemetry_*.json"))
    assert latest.exists()
    assert len(runs) == 1
    assert json.loads(latest.read_text(encoding="utf-8")) == json.loads(
        runs[0].read_text(encoding="utf-8")
    )
    assert json.loads(latest.read_text(encoding="utf-8"))["stage"]["calls"] == 1


def test_explicit_path_writes_only_that_file(tmp_path, monkeypatch):
    monkeypatch.setenv(TELEMETRY_DIR_ENV, str(tmp_path / "default"))
    target = tmp_path / "custom" / "report.json"
    _tracker_with_stage().dump_report(str(target))

    assert target.exists()
    assert not (tmp_path / "default").exists()


def test_per_run_copies_are_pruned(tmp_path):
    folder = tmp_path / "telemetry"
    folder.mkdir()
    for i in range(5):
        (folder / f"telemetry_20260101-00000{i}_1.json").write_text("{}", encoding="utf-8")

    _TelemetryTracker._prune_run_reports(folder, keep=3)

    kept = sorted(p.name for p in folder.glob("telemetry_*.json"))
    assert kept == [f"telemetry_20260101-00000{i}_1.json" for i in (2, 3, 4)]


def test_empty_tracker_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(TELEMETRY_DIR_ENV, str(tmp_path))
    _TelemetryTracker().dump_report()
    assert not os.listdir(tmp_path)


def test_test_session_does_not_point_at_application_logs():
    # Root conftest.py redirects the atexit report of a pytest run.
    assert os.environ.get(TELEMETRY_DIR_ENV)
