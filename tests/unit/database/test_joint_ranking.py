"""Joint candidate list of all layers (localization.joint_candidate_ranking) + priority checks."""

from types import SimpleNamespace

import numpy as np

from src.database.source_priority import (
    JointCandidate,
    coverage_cells,
    effective_patchify_priority,
    joint_order,
    next_priority,
    priority_conflicts,
    priority_steps,
    verification_runs,
)

PRIO = {"main": 0, "2": 1, "3": 2}


def std(sid, frame, score):
    return JointCandidate(sid, frame, score, False)


def patch(sid, frame, score):
    return JointCandidate(sid, frame, score, True)


def ids(order):
    return [(c.source_id, c.frame_id, "p" if c.patch else "s") for c in order]


def test_main_layer_within_the_step_goes_first_like_choose_source():
    # rank_step 2: main 3rd behind layer 2's 1st -> main first; 4th -> layer 2 first
    near = [std("2", 1, 0.9), std("2", 2, 0.89), std("main", 7, 0.88)]
    far = near[:2] + [std("2", 3, 0.885), std("main", 7, 0.88)]

    assert ids(joint_order(near, PRIO, PRIO, 2, 2))[0] == ("main", 7, "s")
    assert ids(joint_order(far, PRIO, PRIO, 2, 2))[0] == ("2", 1, "s")


def test_step_zero_is_plain_score_order():
    pool = [std("main", 7, 0.80), std("2", 1, 0.90), std("3", 4, 0.85)]

    assert ids(joint_order(pool, PRIO, PRIO, 0, 0)) == [
        ("2", 1, "s"),
        ("3", 4, "s"),
        ("main", 7, "s"),
    ]


def test_patch_candidates_are_ranked_among_themselves_not_by_raw_score():
    # patchify scores can exceed 1; the best patch competes with the standard
    # candidate `patch_penalty` places below the best standard one
    pool = [
        patch("main", 50, 2.3),
        std("main", 7, 0.9),
        std("main", 8, 0.8),
        std("main", 9, 0.7),
    ]

    # patch place 1 + 2 = 3 ties with frame 9 (place 3): standard first
    assert ids(joint_order(pool, PRIO, PRIO, 2, 2)) == [
        ("main", 7, "s"),
        ("main", 8, "s"),
        ("main", 9, "s"),
        ("main", 50, "p"),
    ]


def test_patch_penalty_zero_puts_the_best_patch_right_after_the_best_standard():
    pool = [patch("main", 50, 0.4), std("main", 7, 0.9), std("main", 8, 0.8)]

    assert ids(joint_order(pool, PRIO, PRIO, 2, 0)) == [
        ("main", 7, "s"),
        ("main", 50, "p"),
        ("main", 8, "s"),
    ]


def test_patchify_priority_orders_the_patches_of_the_layers():
    pool = [patch("3", 60, 0.9), patch("2", 50, 0.8), std("main", 7, 0.9)]
    by_altitude = {"main": 0, "2": 1, "3": 2}

    order = ids(joint_order(pool, PRIO, by_altitude, 2, 2))
    assert order.index(("2", 50, "p")) < order.index(("3", 60, "p"))

    reversed_patch = {"main": 0, "2": 2, "3": 1}
    order = ids(joint_order(pool, PRIO, reversed_patch, 2, 2))
    assert order.index(("3", 60, "p")) < order.index(("2", 50, "p"))


def test_top_k_cuts_the_list():
    pool = [std("main", f, 0.9 - 0.01 * f) for f in range(10)]

    assert len(joint_order(pool, PRIO, PRIO, 2, 2, top_k=5)) == 5
    assert joint_order([], PRIO, PRIO, 2, 2, top_k=5) == []


def test_priority_steps_ignore_gaps_and_share_equal_numbers():
    assert priority_steps({"main": 0, "2": 2, "3": 3}) == {"main": 0, "2": 1, "3": 2}
    assert priority_steps({"a": 0, "b": 0, "c": 5}) == {"a": 0, "b": 0, "c": 1}


def test_verification_runs_group_consecutive_candidates_of_one_layer():
    order = [std("main", 1, 0.9), patch("main", 5, 0.5), std("2", 3, 0.8), std("main", 2, 0.7)]

    runs = verification_runs(order)

    assert [(sid, [c.frame_id for c in run]) for sid, run in runs] == [
        ("main", [1, 5]),
        ("2", [3]),
        ("main", [2]),
    ]


def test_patchify_priority_defaults_to_the_layer_priority():
    assert effective_patchify_priority({"priority": 2}) == 2
    assert effective_patchify_priority({"priority": 2, "patchify_priority": None}) == 2
    assert effective_patchify_priority({"priority": 2, "patchify_priority": 0}) == 0
    assert effective_patchify_priority(SimpleNamespace(priority=1, patchify_priority=3)) == 3


def test_next_priority_is_one_past_the_largest():
    assert next_priority([]) == 0
    assert next_priority([0, 1]) == 2
    assert next_priority([0, 5]) == 6


def test_coverage_cells_skip_frames_without_position():
    gps = np.array([[49.005, 26.005], [np.nan, np.nan], [49.006, 26.006]])

    assert len(coverage_cells(gps)) == 1
    assert coverage_cells(None) == set()


def test_overlapping_layers_with_one_number_are_reported():
    a = {"source_id": "main", "priority": 0}
    b = {"source_id": "2", "priority": 0, "patchify_priority": 1}
    c = {"source_id": "far", "priority": 0}
    overlaps = {frozenset({"main", "2"})}

    def overlap(x, y):
        return frozenset({x["source_id"], y["source_id"]}) in overlaps

    messages = priority_conflicts([a, b, c], overlap)

    assert messages == ["layers 'main' and '2' overlap but share priority 0"]


def test_shared_patchify_priority_is_reported_too():
    a = SimpleNamespace(source_id="main", priority=0, patchify_priority=None)
    b = SimpleNamespace(source_id="2", priority=1, patchify_priority=0)

    messages = priority_conflicts([a, b], lambda x, y: True)

    assert messages == ["layers 'main' and '2' overlap but share patchify priority 0"]


def test_a_patchify_priority_that_follows_the_layer_priority_is_not_reported_twice():
    a = {"source_id": "main", "priority": 0}
    b = {"source_id": "2", "priority": 0}

    assert priority_conflicts([a, b], lambda x, y: True) == [
        "layers 'main' and '2' overlap but share priority 0"
    ]
