"""Patchify expansion with several layers: patch hits only join their own layer's candidates."""

from src.localization.candidate_retriever import CandidateRetriever


class FakePatchify:
    def __init__(self, hits):
        self.hits = hits
        self.queries = 0

    def compute_patch_descriptors(self, frame):
        self.queries += 1
        return "descriptors"

    def search(self, descriptors, top_k=10):
        return self.hits[:top_k]


class FakeDatabase:
    def __init__(self, name):
        self.name = name


class FakeManager:
    def __init__(self, databases):
        self.databases = databases

    def get_database(self, source_id):
        return self.databases.get(source_id)


def _multi(databases, indexes):
    built = []

    def factory(database):
        built.append(database.name)
        return indexes.get(database.name)

    manager = FakeManager(databases)
    retriever = CandidateRetriever(manager, None, None, {}, patchify_factory=factory)
    return retriever, built, manager


def test_each_layer_uses_its_own_patch_index():
    databases = {"main": FakeDatabase("main"), "2": FakeDatabase("2")}
    indexes = {"main": FakePatchify([(7, 0.9)]), "2": FakePatchify([(40, 0.8)])}
    retriever, built, _ = _multi(databases, indexes)

    from_main = retriever.expand("frame", [(1, 0.5)], top_k=5, source_id="main")
    from_two = retriever.expand("frame", [(3, 0.6)], top_k=5, source_id="2")

    assert dict(from_main) == {1: 0.5, 7: 0.9}
    assert dict(from_two) == {3: 0.6, 40: 0.8}
    assert sorted(built) == ["2", "main"]


def test_layer_index_is_built_once_and_again_after_a_reload():
    databases = {"main": FakeDatabase("main")}
    retriever, built, manager = _multi(databases, {"main": FakePatchify([(7, 0.9)])})

    retriever.expand("frame", [(1, 0.5)], top_k=5, source_id="main")
    retriever.expand("frame", [(1, 0.5)], top_k=5, source_id="main")
    manager.databases["main"] = FakeDatabase("main")  # layer database reloaded
    retriever.expand("frame", [(1, 0.5)], top_k=5, source_id="main")

    assert built == ["main", "main"]


def test_layer_without_patch_descriptors_or_unknown_layer_keeps_candidates():
    databases = {"main": FakeDatabase("main")}
    retriever, _, _ = _multi(databases, {})  # factory returns None: no patch descriptors
    base = [(1, 0.5), (2, 0.4)]

    assert retriever.expand("frame", base, top_k=5, source_id="main") == base
    assert retriever.expand("frame", base, top_k=5, source_id="missing") == base
    assert retriever.expand("frame", base, top_k=5, source_id=None) == base


def test_single_database_mode_keeps_its_index():
    patchify = FakePatchify([(9, 0.7)])
    retriever = CandidateRetriever(None, None, patchify, {})

    merged = retriever.expand("frame", [(1, 0.5)], top_k=5)

    assert dict(merged) == {1: 0.5, 9: 0.7}
    assert patchify.queries == 1


def test_failing_patch_search_falls_back_to_the_candidates():
    class Broken(FakePatchify):
        def search(self, descriptors, top_k=10):
            raise RuntimeError("faiss gone")

    retriever, _, _ = _multi({"main": FakeDatabase("main")}, {"main": Broken([])})

    assert retriever.expand("frame", [(1, 0.5)], top_k=5, source_id="main") == [(1, 0.5)]


def test_retrieve_exposes_every_layers_candidates():
    class Manager(FakeManager):
        def get_best_match(self, descriptor, top_k=8):
            self.last_matches_by_source = {"main": [(1, 0.8)], "2": [(5, 0.9)]}
            return "2", [(5, 0.9)]

    retriever = CandidateRetriever(Manager({}), None, None, {})

    assert retriever.retrieve("descriptor", 5) == ("2", [(5, 0.9)])
    assert retriever.last_by_source == {"main": [(1, 0.8)], "2": [(5, 0.9)]}
