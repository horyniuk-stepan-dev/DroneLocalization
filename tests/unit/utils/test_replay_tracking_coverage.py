"""summarize_coverage of scripts/replay_tracking_ground_truth.py (plan 1.4 gates)."""

from scripts.replay_tracking_ground_truth import summarize_coverage


def _row(t, state, gin, gd, alert=False, path="kf", success=True, confirmed=True, err=1.0):
    return {
        "final_error_m": err,
        "t": t,
        "path": path,
        "success": success,
        "coverage_state": state,
        "coverage_alert": alert,
        "gt_inside": gin,
        "gt_dist_to_edge_m": gd,
        "fix_confirmed": confirmed,
    }


def test_exit_lead_time_reentry_and_false_alerts():
    rows = [
        _row(0, "inside", True, 900),
        _row(1, "inside", True, 700),
        _row(2, "near_edge", True, 650, alert=True),  # time-to-edge warning far out
        _row(10, "near_edge", True, 20),
        _row(12, "outside", False, -50, alert=True, success=False, confirmed=None),
        _row(13, "outside", False, -80, success=True, confirmed=False),
        _row(30, "outside", True, 30, success=True, confirmed=False),
        _row(31, "near_edge", True, 60, success=True, confirmed=True),
    ]
    s = summarize_coverage(rows, 300.0)
    assert s["exits"] == [{"t": 12, "lead_s": 10}]
    assert s["exits_warned"] == 1
    assert s["false_alerts"] == 0
    assert s["confirmed_fixes_truly_outside"] == 0
    assert s["confirmed_fixes_truly_outside_false"] == 0
    assert s["reentry_to_first_confirmed_fix_s"] == [1]
    assert s["outside_vs_lost_inside_correct"] == "2/3"


def test_alert_without_approach_is_false_and_bad_outside_fix_counted():
    rows = [_row(0, "inside", True, 900)]
    rows += [_row(1, "near_edge", True, 800, alert=True)]
    rows += [_row(1 + k, "near_edge", True, 800) for k in range(1, 70)]
    rows += [_row(80, "outside", False, -100, err=60.0)]
    s = summarize_coverage(rows, 300.0)
    assert s["false_alert_t"] == [1]
    assert s["confirmed_fixes_truly_outside_false"] == 1


def test_no_coverage_rows():
    assert summarize_coverage([{"t": 0}], 300.0) is None
