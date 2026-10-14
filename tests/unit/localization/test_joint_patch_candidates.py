"""CandidateRetriever.patch_candidates: one layer's patch hits for the joint candidate list."""

from src.localization.candidate_retriever import CandidateRetriever


class FakePatchify:
    grids = [(1, 1), (2, 2)]
    descriptor_dim = 256

    def __init__(self, hits):
        self.hits = hits
        self.queries = 0

    def compute_patch_descriptors(self, frame):
        self.queries += 1
        return f"descriptors of {frame}"

    def search(self, descriptors, top_k=10):
        return self.hits[:top_k]


class FakeManager:
    def __init__(self, databases):
        self.databases = databases

    def get_database(self, source_id):
        return self.databases.get(source_id)


def retriever(indexes):
    databases = {sid: object() for sid in indexes}
    by_db = {id(db): indexes[sid] for sid, db in databases.items()}
    return CandidateRetriever(
        FakeManager(databases), None, None, {}, patchify_factory=lambda db: by_db[id(db)]
    )


def test_hits_of_the_asked_layer_only():
    main, other = FakePatchify([(5, 0.6), (9, 0.4)]), FakePatchify([(77, 0.9)])
    r = retriever({"main": main, "2": other})

    assert r.patch_candidates("view", 10, source_id="main") == [(5, 0.6), (9, 0.4)]
    assert r.patch_candidates("view", 1, source_id="2") == [(77, 0.9)]


def test_crop_descriptors_of_one_view_are_computed_once_per_frame():
    shared = FakePatchify([(5, 0.6)])
    r = retriever({"main": shared, "2": shared})
    cache = {}
    view = ["the prepared view"]

    r.patch_candidates(view, 10, source_id="main", desc_cache=cache)
    r.patch_candidates(view, 10, source_id="2", desc_cache=cache)
    r.patch_candidates(["another view"], 10, source_id="2", desc_cache=cache)

    assert shared.queries == 2


def test_layer_without_patch_index_or_view_gives_nothing():
    r = retriever({"main": None})

    assert r.patch_candidates("view", 10, source_id="main") == []
    assert r.patch_candidates(None, 10, source_id="main") == []
    assert r.patch_candidates("view", 10, source_id="unknown") == []


def test_failing_patch_search_gives_nothing():
    class Broken(FakePatchify):
        def search(self, descriptors, top_k=10):
            raise RuntimeError("faiss")

    assert retriever({"main": Broken([])}).patch_candidates("view", 10, source_id="main") == []
