"""LanceDB retrieval must not treat IVF-PQ distances as exact cosine scores."""

import lancedb
import numpy as np
import pytest

from src.localization.matcher import LanceDBRetrieval


@pytest.fixture
def indexed_table(tmp_path):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(320, 16)).astype(np.float32)
    table = lancedb.connect(str(tmp_path)).create_table(
        "vectors",
        data=[{"frame_id": i, "vector": vector} for i, vector in enumerate(vectors)],
    )
    table.create_index(metric="cosine", num_partitions=8, num_sub_vectors=4)
    return table, vectors


def test_small_indexed_table_uses_exact_cosine(indexed_table):
    table, vectors = indexed_table
    retriever = LanceDBRetrieval(table)

    matches = retriever.find_similar_frames(vectors[17], top_k=3)

    assert matches[0][0] == 17
    assert matches[0][1] == pytest.approx(1.0, abs=1e-5)


def test_large_table_path_refines_indexed_distances(indexed_table):
    table, vectors = indexed_table
    retriever = LanceDBRetrieval(table)
    retriever.EXACT_SEARCH_MAX_ROWS = 0  # Exercise the large-map path on a small fixture.

    matches = retriever.find_similar_frames(vectors[17], top_k=3)

    assert matches[0][0] == 17
    assert matches[0][1] == pytest.approx(1.0, abs=1e-5)


def test_legacy_query_reranks_shortlist_from_original_vectors():
    class Query:
        def __init__(self):
            self.limit_value = None

        def metric(self, metric):
            assert metric == "cosine"
            return self

        def limit(self, count):
            self.limit_value = count
            return self

        def select(self, columns):
            assert columns == ["frame_id", "vector"]
            return self

        def to_list(self):
            return [
                {"frame_id": 0, "vector": [0.0, 1.0]},
                {"frame_id": 1, "vector": [-1.0, 0.0]},
                {"frame_id": 2, "vector": [1.0, 0.0]},
            ][: self.limit_value]

    class Table:
        def count_rows(self):
            return 3

        def search(self, vector):
            return Query()

    matches = LanceDBRetrieval(Table()).find_similar_frames(np.array([1.0, 0.0]), 3)

    assert matches == [(2, 1.0), (0, 0.0), (1, -1.0)]
