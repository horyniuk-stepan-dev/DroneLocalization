"""Pure helpers of scripts/replay_tracking_ground_truth.py (no torch needed)."""

import time

import numpy as np
import pytest

import scripts.replay_tracking_ground_truth as R

LAT0, LON0 = 48.42, 26.19
M_PER_DEG_LAT = 111_195.0


def _ll(east_m, north_m):
    lat = LAT0 + north_m / M_PER_DEG_LAT
    lon = LON0 + east_m / (M_PER_DEG_LAT * np.cos(np.radians(LAT0)))
    return lat, lon


def _row(frame, est_en, gt_en, success=True, gt_valid=True, path="kf"):
    lat, lon = _ll(*est_en)
    glat, glon = _ll(*gt_en)
    return {
        "frame": frame,
        "path": path,
        "success": success,
        "gt_valid": gt_valid,
        "lat": lat if success else None,
        "lon": lon if success else None,
        "gt_lat": glat,
        "gt_lon": glon,
    }


def test_cli_rejects_missing_inputs_before_loading_models(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        R.main(
            [
                "--source", "main", str(tmp_path / "no.h5"), str(tmp_path / "no.json"),
                "--video", str(tmp_path / "v.mp4"),
                "--frames-gt", str(tmp_path / "f.jsonl"),
                "--json", str(tmp_path / "r.json"),
            ]
        )  # fmt: skip
    assert exc.value.code == 2
    assert "database does not exist" in capsys.readouterr().err


def test_cli_validates_new_tracking_flags(tmp_path, capsys):
    from scripts.replay_multilayer_ground_truth import config_overrides

    got = config_overrides(
        [
            "tracking.smoother_odom_huber_k=1.2",
            "tracking.of_noise_growth_per_step=1.0",
            "tracking.outlier_reacquire_consistent=true",
        ]
    )
    assert got == {
        "tracking.smoother_odom_huber_k": 1.2,
        "tracking.of_noise_growth_per_step": 1.0,
        "tracking.outlier_reacquire_consistent": True,
    }


def test_local_vector_metres():
    v = R.local_vector_m(_ll(0, 0), _ll(30.0, -40.0))
    assert np.allclose(v, [30.0, -40.0], atol=0.05)


def test_step_errors_flags_jump_and_ignores_constant_bias():
    rows = [_row(i, (10.0 * i + 4.0, 3.0), (10.0 * i, 0.0)) for i in range(5)]
    # a constant 5 m bias is NOT a jump: steps match the truth
    assert max(s["step_error_m"] for s in R.step_errors(rows)) < 0.05
    rows[3] = _row(3, (30.0 + 4.0 + 40.0, 3.0), (30.0, 0.0))  # +40 m east spike
    errs = [s["step_error_m"] for s in R.step_errors(rows)]
    assert errs[2] > 39.0 and errs[3] > 39.0


def test_step_chain_spans_failures_and_skips_invalid_gt():
    rows = [
        _row(0, (0, 0), (0, 0)),
        _row(1, (0, 0), (10, 0), success=False),
        _row(2, (500, 0), (20, 0), gt_valid=False),
        _row(3, (55, 0), (30, 0)),
    ]
    steps = R.step_errors(rows)
    assert len(steps) == 1 and steps[0]["frame"] == 3
    assert abs(steps[0]["step_error_m"] - 25.0) < 0.1  # re-acquisition jump counted


def test_worker_clock_rules():
    c = R.WorkerClock(keyframe_interval=30, frame_duration=1 / 30, of_local_speed=False)
    assert c.dt(0.0, True) == pytest.approx(1.0)  # first keyframe: interval * frame
    c.last_kf = 0.0
    c.last_loc = 0.0
    assert c.dt(1.0, True) == pytest.approx(1.0)
    assert c.dt(0.5, False) == pytest.approx(0.5)  # OF: from last localization
    c2 = R.WorkerClock(30, 1 / 30, of_local_speed=True)
    c2.last_loc, c2.last_of = 0.0, 0.4
    assert c2.dt(0.5, False) == pytest.approx(0.1)  # OF local: from last OF frame
    assert c2.dt(0.4, False) == pytest.approx(1 / 30)  # non-positive -> one frame


def test_summarize_counts_jumps_per_path():
    rows = [
        _row(i, (10.0 * i, 0), (10.0 * i, 0), path="kf" if i % 3 == 0 else "of") for i in range(9)
    ]
    rows[4] = _row(4, (40.0 + 25.0, 0), (40.0, 0), path="of")
    for r in rows:
        r["final_error_m"] = 0.0
        r["processing_ms"] = 5.0
        r["stages_ms"] = {"match": 2.0} if r["path"] == "kf" else {}
    s = R.summarize(rows, jump_m=10.0)
    assert s["jumps"] == 2 and s["jump_frames"] == [4, 5]
    assert s["keyframe"]["attempted"] == 3 and s["optical_flow"]["attempted"] == 6
    assert s["stages_ms"]["kf:match"]["n"] == 3


def test_stage_timer_accumulates_and_resets():
    class Obj:
        def work(self, x):
            time.sleep(0.002)
            return x * 2

    o = Obj()
    t = R.StageTimer()
    t.wrap(o, "work", "w")
    t.wrap(o, "missing", "m")  # absent methods are ignored
    assert o.work(3) == 6 and o.work(1) == 2
    got = t.take()
    assert set(got) == {"w"} and got["w"] >= 3.0
    assert t.take() == {}
