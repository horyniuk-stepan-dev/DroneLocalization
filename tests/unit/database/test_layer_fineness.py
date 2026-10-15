"""Layer accuracy in the layer priority and adaptive patchify (source_priority helpers)."""

import math
import random
from types import SimpleNamespace

import numpy as np

from src.database.source_priority import (
    choose_source,
    choose_source_weighted,
    effective_steps,
    grid_factors,
    layer_gsd,
    patchify_layers,
    priority_conflicts,
)

# relief project: main 1000 m, 2 = 2000 m, 3 = 500 m (pixel sizes measured from the calibration)
PRIO = {"main": 0, "2": 1, "3": 2}
GSD = {"main": 0.72, "2": 1.51, "3": 0.33}


def affines(scales, valid=None):
    a = np.zeros((len(scales), 2, 3))
    for i, s in enumerate(scales):
        a[i, 0, 0] = a[i, 1, 1] = s
    return SimpleNamespace(frame_affine=a, frame_valid=valid)


def test_layer_pixel_size_is_the_median_affine_scale_of_valid_frames():
    assert math.isclose(layer_gsd(affines([0.5, 0.5, 0.7])), 0.5)
    assert math.isclose(layer_gsd(affines([0.5, 9.0, 9.0], valid=[True, False, False])), 0.5)
    flat = SimpleNamespace(frame_affine=np.array([[0.4, 0, 1, 0, 0.4, 2]]), frame_valid=None)
    assert math.isclose(layer_gsd(flat), 0.4)
    assert layer_gsd(SimpleNamespace(frame_affine=None)) is None


def test_effective_steps_add_the_fineness_term():
    steps = effective_steps(PRIO, GSD, 2.0)

    assert math.isclose(steps["3"], 2.0)
    assert round(steps["main"], 1) == 2.3
    assert round(steps["2"], 1) == 5.4
    assert effective_steps(PRIO, GSD, 0.0) == {"main": 0.0, "2": 1.0, "3": 2.0}
    assert effective_steps(PRIO, {"main": None}, 2.0) == {"main": 0.0, "2": 1.0, "3": 2.0}


def test_two_layers_one_step_apart_choose_like_choose_source():
    rng = random.Random(7)
    for _ in range(200):
        per_source = {
            sid: sorted(((f, rng.random()) for f in range(5)), key=lambda c: -c[1])
            for sid in ("main", "2")
        }
        assert choose_source_weighted(per_source, {"main": 0.0, "2": 1.0}, 2) == choose_source(
            per_source, {"main": 0, "2": 1}, 2
        )


def test_fine_layer_wins_with_enough_weight():
    # 500 m layer's best is 3rd, main's 1st; place = rank + 2 x effective step:
    # weight 3 -> main 1 + 6.75, 500 m 3 + 4 -> 500 m; weight 2 -> main 1 + 4.5 -> main
    per_source = {"main": [(1, 0.9), (2, 0.85)], "3": [(7, 0.84)], "2": [(9, 0.5)]}

    assert choose_source_weighted(per_source, effective_steps(PRIO, GSD, 3.0), 2) == "3"
    assert choose_source_weighted(per_source, effective_steps(PRIO, GSD, 2.0), 2) == "main"
    assert choose_source_weighted(per_source, effective_steps(PRIO, GSD, 0.0), 2) == "main"


def test_patchify_is_skipped_when_a_real_layer_is_close_in_scale():
    factors = grid_factors([[1, 1], [2, 2], [3, 3]])

    assert patchify_layers(0.72, GSD, factors, 1.5) == set()
    assert patchify_layers(0.45, GSD, factors, 1.5) == set()


def test_patchify_runs_on_the_layer_whose_crops_fill_the_gap():
    two_layers = {"3": 0.33, "2": 1.51}
    factors = grid_factors([[1, 1], [2, 2], [3, 3]])

    assert patchify_layers(0.72, two_layers, factors, 1.5) == {"2"}
    assert patchify_layers(1.4, two_layers, factors, 1.5) == set()


def test_unknown_query_scale_keeps_patchify_on_every_layer():
    assert patchify_layers(None, GSD, [2.0, 3.0], 1.5) is None
    assert patchify_layers(0.7, {"main": None}, [2.0], 1.5) is None


def test_grid_factors():
    assert grid_factors([[1, 1], [2, 2], (3, 3)]) == [1.0, 2.0, 3.0]
    assert math.isclose(grid_factors([[2, 3]])[0], math.sqrt(6))
    assert grid_factors([["x"], [0, 2]]) == []


def test_layers_told_apart_by_accuracy_are_not_reported():
    a = {"source_id": "main", "priority": 0}
    b = {"source_id": "3", "priority": 0}

    assert priority_conflicts([a, b], lambda x, y: True, lambda x, y: True) == []
    assert len(priority_conflicts([a, b], lambda x, y: True, lambda x, y: False)) == 1
