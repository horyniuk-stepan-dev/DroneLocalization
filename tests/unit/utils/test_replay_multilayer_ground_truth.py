"""Focused contract checks for the chronological multi-layer replay evaluator."""

import csv
import json
from types import SimpleNamespace

import pytest

from scripts.replay_multilayer_ground_truth import (
    parse_args,
    replay_slots,
    summarize,
    write_report,
)


class FakeCapture:
    def __init__(self):
        self.position = 0

    def set(self, _property, position):
        self.position = position

    def read(self):
        if self.position == 90:
            return False, None
        return True, f"image-{self.position}"


class FakeSearch:
    def __init__(self):
        self.handoff = SimpleNamespace(pending=None, reason=None)
        self.calls = 0

    def search(self):
        self.calls += 1
        sid = "a" if self.calls < 3 else "b"
        verification = SimpleNamespace(candidate_id=12, inliers=40, rmse=0.5)
        return [
            SimpleNamespace(
                source_id=sid,
                quality=25.0,
                gps=(10.0, 20.0),
                prepared=(verification,),
            )
        ]


class FakeManager:
    def __init__(self):
        self.calls = 0

    def get_matches_by_source(self, *_args, **_kwargs):
        self.calls += 1
        sid = "a" if self.calls < 3 else "b"
        return {sid: [(12, 0.9)]}


class FakeLocalizer:
    def __init__(self):
        self._layer_search = FakeSearch()
        self.db_manager = FakeManager()
        self.calls = []

    def localize_frame(self, frame, *, dt, timestamp):
        self.calls.append((frame, dt, timestamp))
        self.db_manager.get_matches_by_source(None)
        self._layer_search.search()
        if timestamp == 0:
            self._layer_search.handoff.pending = "a"
            self._layer_search.handoff.reason = "awaiting_confirmation"
            return {"success": False, "status": "lost", "error": "awaiting_confirmation"}
        sid = "a" if timestamp == 1 else "b"
        self._layer_search.handoff.pending = None
        self._layer_search.handoff.reason = None
        return {
            "success": True,
            "status": "confirmed",
            "source_id": sid,
            "source_changed": True,
            "lat": 10.2,
            "lon": 20.0,
            "raw_lat": 10.1,
            "raw_lon": 20.0,
            "matched_frame": 12,
            "search": {"verifications": 2, "hypotheses": 3},
        }


def test_replay_keeps_query_gt_out_of_localizer_and_records_handoff(tmp_path):
    slots = [
        {
            "slot": i,
            "video_frame": i * 30,
            "timestamp": float(i),
            "camera_agl": 500 + i * 100,
            "ground_center_gps": [10.0, 20.0],
        }
        for i in range(4)
    ]
    localizer = FakeLocalizer()
    ticks = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03])
    rows = replay_slots(
        slots,
        FakeCapture(),
        localizer,
        {"a", "b"},
        clock=lambda: next(ticks),
        distance=lambda a, b: abs(a[0] - b[0]) * 100,
        false_confirmation_m=5.0,
    )
    assert localizer.calls == [
        ("image-0", 1.0, 0.0),
        ("image-30", 1.0, 1.0),
        ("image-60", 1.0, 2.0),
    ]
    assert [row["status"] for row in rows] == ["lost", "confirmed", "confirmed", "decode_failed"]
    assert rows[0]["pending_source_id"] == "a"
    assert rows[1]["bootstrap"] and not rows[1]["handoff"]
    assert rows[2]["handoff"] and rows[2]["previous_confirmed_source_id"] == "a"
    assert rows[2]["observed_sources"] == ["b"]
    assert rows[2]["retrieved_candidates_by_source"] == {
        "b": {"count": 1, "unique_frames": [12], "top_score": 0.9}
    }
    assert rows[2]["raw_error_m"] == pytest.approx(10.0)
    assert rows[2]["final_error_m"] == pytest.approx(20.0)
    assert rows[3]["observed_candidates"] == []
    summary = summarize(rows, [600.0, 800.0], 5.0)
    assert summary["confirmed"] == 2
    assert summary["failed"] == 2
    assert summary["handoff_count"] == 1
    assert summary["verified_observations_by_source"] == {"a": 2, "b": 1}
    assert summary["retrieved_hypotheses_by_source"] == {"a": 2, "b": 1}
    assert summary["false_confirmations"] == 2
    assert summary["altitude_bins"]["600-800m"]["attempted"] == 2
    assert summary["warm_processing_ms"]["n"] == 2

    report = {"rows": rows, **summary}
    json_path, csv_path = tmp_path / "out.json", tmp_path / "out.csv"
    write_report(report, json_path, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["handoff_count"] == 1
    with csv_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert json.loads(csv_rows[2]["observed_candidates"])[0]["source_id"] == "b"


def test_cli_accepts_repeated_sources_and_rejects_duplicate_ids(tmp_path):
    paths = [tmp_path / name for name in ("a.h5", "a.json", "b.h5", "b.json", "v.mp4", "gt.json")]
    for path in paths:
        path.touch()
    common = [
        "--video",
        str(paths[4]),
        "--gt",
        str(paths[5]),
        "--json",
        str(tmp_path / "out.json"),
        "--csv",
        str(tmp_path / "out.csv"),
    ]
    args = parse_args(
        [
            "--source",
            "a",
            str(paths[0]),
            str(paths[1]),
            "--source",
            "b",
            str(paths[2]),
            str(paths[3]),
            *common,
        ]
    )
    assert [source.source_id for source in args.sources] == ["a", "b"]
    assert args.altitude_edges == [600.0, 800.0, 1000.0]

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--source",
                "a",
                str(paths[0]),
                str(paths[1]),
                "--source",
                "a",
                str(paths[2]),
                str(paths[3]),
                *common,
            ]
        )

    legacy = parse_args(["--source", "a", str(paths[0]), str(paths[1]), "--legacy-search", *common])
    assert legacy.legacy_search and legacy.config_overrides == {}
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--source",
                "a",
                str(paths[0]),
                str(paths[1]),
                "--legacy-search",
                "--layer-search",
                "early_stop=true",
                *common,
            ]
        )


def test_layer_search_overrides_are_typed_and_validated():
    from scripts.replay_multilayer_ground_truth import layer_search_overrides

    assert layer_search_overrides(None) == {}
    assert layer_search_overrides(["motion_gate=true", "motion_gate_growth_mps=12"]) == {
        "motion_gate": True,
        "motion_gate_growth_mps": 12.0,
    }
    for bad in (
        ["no_such_key=1"],
        ["enabled=false"],
        ["motion_gate"],
        ["early_stop_max_scale_ratio=0.5"],
        ["early_stop=true", "early_stop=false"],
    ):
        with pytest.raises(ValueError):
            layer_search_overrides(bad)


def test_set_overrides_are_nested_typed_and_validated():
    from scripts.replay_multilayer_ground_truth import config_overrides

    assert config_overrides(None) == {}
    assert config_overrides(["localization.use_patchify=true", "models.vlad.pca_dim=128"]) == {
        "localization.use_patchify": True,
        "models.vlad.pca_dim": 128,
    }
    for bad in (
        ["localization.no_such_key=1"],
        ["localization.layer_search.motion_gate=true"],
        ["localization.use_patchify"],
        ["localization=1"],
        ["localization.use_patchify=1"],
        ['localization.retrieval_top_k="many"'],
        ["localization.geometric_min_inlier_ratio=1.5"],
        ["localization.use_patchify=true", "localization.use_patchify=false"],
    ):
        with pytest.raises(ValueError):
            config_overrides(bad)


class FakeLegacyLocalizer:
    """Pre-layer-search path: no confirmation state, no layer search."""

    def localize_frame(self, frame, *, dt, timestamp):
        if timestamp == 4:
            return {"success": False, "error": "Not enough valid inliers"}
        offset = 0.0001 if timestamp == 2 else 0.0
        lat = 10.0 + 0.001 * timestamp + offset
        return {
            "success": True,
            "lat": lat,
            "lon": 20.0,
            "raw_lat": lat,
            "raw_lon": 20.0,
            "source_id": "a",
            "fallback_mode": "retrieval_only" if timestamp == 3 else None,
        }


def test_legacy_results_count_as_confirmed_and_steps_are_scored():
    slots = [
        {
            "slot": i,
            "video_frame": i * 10,  # FakeCapture fails only at frame 90
            "timestamp": float(i),
            "camera_agl": 500,
            "ground_center_gps": [10.0 + 0.001 * i, 20.0],
        }
        for i in range(5)
    ]
    ticks = iter(float(i) for i in range(10))
    rows = replay_slots(
        slots,
        FakeCapture(),
        FakeLegacyLocalizer(),
        {"a"},
        clock=lambda: next(ticks),
        false_confirmation_m=10.0,
        legacy=True,
    )
    assert [row["status"] for row in rows] == ["confirmed"] * 4 + ["failed"]
    assert rows[2]["lat"] == pytest.approx(10.0021)
    assert rows[2]["gt_lat"] == pytest.approx(10.002)
    assert rows[3]["fallback_mode"] == "retrieval_only"
    summary = summarize(rows, [600.0], 10.0)
    # one 11 m excursion at slot 2 shows up as two 11 m step errors
    assert summary["raw_step_error_m"]["n"] == 3
    assert summary["raw_step_error_m"]["max"] == pytest.approx(11.06, abs=0.01)
    assert summary["raw_step_errors_over_threshold"] == 2
    assert summary["final_step_error_m"]["median"] == pytest.approx(11.06, abs=0.01)
    assert summary["confirmed_fallback_modes"] == {"retrieval_only": 1}

    ticks = iter(float(i) for i in range(10))
    strict = replay_slots(
        slots, FakeCapture(), FakeLegacyLocalizer(), {"a"}, clock=lambda: next(ticks)
    )
    assert strict[0]["status"] == "invalid_result" and not strict[0]["confirmed"]
