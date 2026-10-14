"""MultiDatabaseManager.get_best_match with localization.source_priority_rank_margin."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.database.multi_database_manager import MultiDatabaseManager


def manager(margin):
    result = MultiDatabaseManager.__new__(MultiDatabaseManager)
    result._config = {"localization": {"source_priority_rank_margin": margin}}
    result._active_source_ids = {"main", "2"}
    result._sources = {"main": SimpleNamespace(priority=0), "2": SimpleNamespace(priority=1)}
    result._databases = {sid: SimpleNamespace() for sid in ("main", "2")}
    result._retrievers = {
        "main": Mock(find_similar_frames=Mock(return_value=[(10, 0.80), (11, 0.60)])),
        "2": Mock(find_similar_frames=Mock(return_value=[(3, 0.85), (4, 0.70)])),
    }
    return result


def test_old_behaviour_without_margin():
    assert manager(0).get_best_match(np.ones(2), top_k=2) == ("2", [(3, 0.85), (4, 0.70)])


def test_main_layer_wins_within_the_margin():
    assert manager(1).get_best_match(np.ones(2), top_k=2) == ("main", [(10, 0.80), (11, 0.60)])


def test_every_layers_candidates_of_the_last_call_are_kept():
    db = manager(0)
    db.get_best_match(np.ones(2), top_k=2)

    assert db.last_matches_by_source == {
        "main": [(10, 0.80), (11, 0.60)],
        "2": [(3, 0.85), (4, 0.70)],
    }
