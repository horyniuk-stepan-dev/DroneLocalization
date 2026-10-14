"""Layer choice of the legacy multi-layer retrieval with a priority rank margin."""

from src.database.source_priority import choose_source, main_before_patchify

PRIORITY = {"main": 0, "2": 1, "3": 2}


def test_without_margin_the_best_top_score_wins():
    per_source = {"main": [(1, 0.80)], "2": [(5, 0.85)]}

    assert choose_source(per_source, PRIORITY, 0) == "2"


def test_main_at_rank_two_beats_another_layer_at_rank_one():
    per_source = {"main": [(1, 0.80), (2, 0.60)], "2": [(5, 0.85), (6, 0.70)]}

    assert choose_source(per_source, PRIORITY, 1) == "main"


def test_main_far_down_the_ranking_loses():
    other = [(frame, 0.95 - 0.01 * frame) for frame in range(9)]  # ranks 1..9
    per_source = {"main": [(1, 0.50)], "2": other}  # main's best is rank 10

    assert choose_source(per_source, PRIORITY, 3) == "2"
    assert choose_source(per_source, PRIORITY, 9) == "main"


def test_priority_order_decides_among_layers_within_the_margin():
    per_source = {
        "3": [(1, 0.90), (2, 0.89)],  # ranks 1, 2
        "2": [(4, 0.88)],  # rank 3
        "main": [(7, 0.70)],  # rank 4
    }

    assert choose_source(per_source, PRIORITY, 1) == "3"  # only ranks 1-2 count
    assert choose_source(per_source, PRIORITY, 2) == "2"
    assert choose_source(per_source, PRIORITY, 3) == "main"


def test_equal_scores_go_to_the_higher_priority_layer():
    per_source = {"2": [(5, 0.8)], "main": [(1, 0.8)]}

    assert choose_source(per_source, PRIORITY, 0) == "main"


def test_no_candidates():
    assert choose_source({}, PRIORITY, 3) is None
    assert choose_source({"main": []}, PRIORITY, 3) is None


def test_patchify_first_and_main_second_puts_main_first():
    expanded = [(50, 0.90), (7, 0.70)]  # 50 found only by patchify, 7 by standard retrieval
    main = [(3, 0.85), (4, 0.60)]

    assert main_before_patchify(expanded, {7}, main, 3) is True


def test_main_far_below_the_patchify_candidate_keeps_patchify_first():
    expanded = [(50, 0.95)] + [(frame, 0.94 - 0.01 * frame) for frame in range(8)]
    main = [(3, 0.50)]  # rank 10 behind patchify's rank 1

    assert main_before_patchify(expanded, set(range(8)), main, 3) is False
    assert main_before_patchify(expanded, set(range(8)), main, 9) is True


def test_no_patchify_only_candidates_or_no_main_or_margin_off():
    assert main_before_patchify([(7, 0.9)], {7}, [(3, 0.8)], 3) is False
    assert main_before_patchify([(50, 0.9)], set(), [], 3) is False
    assert main_before_patchify([(50, 0.9)], set(), [(3, 0.8)], 0) is False


def test_main_above_patchify_is_always_first():
    assert main_before_patchify([(50, 0.5)], set(), [(3, 0.9)], 1) is True
