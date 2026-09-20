import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.database.multi_database_manager import MultiDatabaseManager
from src.database.schema_fingerprint import build_components


def manager():
    result = MultiDatabaseManager.__new__(MultiDatabaseManager)
    result._config = {}
    result._active_source_ids = {"a", "b"}
    result._sources = {sid: SimpleNamespace(priority=0) for sid in ("a", "b")}
    schema = build_components({}, descriptor_dim=2, local_descriptor_dim=128)
    result._databases = {
        sid: SimpleNamespace(metadata={"schema_components": json.dumps(schema)})
        for sid in ("a", "b")
    }
    result._retrievers = {
        "a": Mock(find_similar_frames=Mock(return_value=[(0, 0.99)])),
        "b": Mock(find_similar_frames=Mock(return_value=[(0, 0.7)])),
    }
    return result


def test_keeps_identical_frame_ids_in_separate_sources():
    db = manager()
    assert db.get_matches_by_source(np.ones(2), top_k=3) == {"a": [(0, 0.99)], "b": [(0, 0.7)]}
    for retriever in db._retrievers.values():
        assert retriever.find_similar_frames.call_args.kwargs == {"top_k": 3}


def test_known_schema_mismatch_never_reaches_matcher():
    db = manager()
    db._databases["a"].metadata["schema_components"] = {"descriptor_dim": 5}
    assert set(db.get_matches_by_source(np.ones(2))) == {"b"}
    db._retrievers["a"].find_similar_frames.assert_not_called()


def test_legacy_schema_requires_explicit_strictness_policy():
    db = manager()
    db._databases["a"].metadata = {}
    assert set(db.get_matches_by_source(np.ones(2))) == {"a", "b"}
    assert set(db.get_matches_by_source(np.ones(2), require_schema=True)) == {"b"}
