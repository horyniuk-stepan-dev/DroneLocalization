"""Юніт-тести GeometricVerifier (аудит §3.3).

Клас свідомо винесли з Localizer як stateless-колаборатора «щоб було
тестовано» — і жодного тесту не написали. Саме тут роздутий лічильник інлаєрів
(§1.1) перетворювався на вибір ХИБНОГО кадру БД: перемагає той кандидат, у
якого `inliers` більший.

Тести не потребують ні GPU, ні бази: матчер і БД — фейки.
"""

import numpy as np
import pytest

from src.localization.geometric_verifier import GeometricVerifier


class FakeDB:
    """Мінімальна БД: frame_id → dict фіч."""

    def __init__(self, feats: dict):
        self._feats = feats
        self.reads = []

    def get_local_features(self, frame_id):
        self.reads.append(frame_id)
        if frame_id not in self._feats:
            raise ValueError(f"немає кадру {frame_id}")
        return self._feats[frame_id]


class FakeMatcher:
    """Повертає заздалегідь задані пари точок для (кількості) кандидата."""

    def __init__(self, plan: dict):
        # plan: frame_id → (mkpts_q, mkpts_r)
        self.plan = plan
        self.calls = []
        self._order = []

    def match(self, q, r):
        fid = int(r["_id"])
        self.calls.append(fid)
        return self.plan.get(fid, (np.empty((0, 2)), np.empty((0, 2))))


def _feats(n: int, dim: int = 8, fid: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, dim)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return {
        "keypoints": rng.uniform(0, 500, (n, 2)).astype(np.float32),
        "descriptors": d,
        "image_size": np.array([500, 500], dtype=np.int32),
        "_id": fid,
    }


def _identity_matches(n: int, noise: float = 0.5, seed: int = 1):
    """n відповідностей, що узгоджуються з H ≈ I."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, 500.0, (n, 2))
    r = q + rng.normal(0.0, noise, (n, 2))
    return q.astype(np.float32), r.astype(np.float32)


def _garbage_matches(n: int, seed: int = 2):
    """n відповідностей без спільної геометрії."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.0, 500.0, (n, 2))
    r = rng.uniform(0.0, 500.0, (n, 2))
    return q.astype(np.float32), r.astype(np.float32)


def _verifier(**kw):
    defaults = dict(
        min_matches=12,
        ransac_thresh=3.0,
        homography_backend="opencv",
        use_mad_ransac=True,
        mad_k_factor=2.5,
        early_stop_inliers=1000,  # вимикаємо early-stop, якщо тест не просить
    )
    defaults.update(kw)
    matcher = defaults.pop("matcher")
    return GeometricVerifier(matcher=matcher, **defaults)


class TestCandidateSelection:
    def test_picks_the_geometrically_consistent_candidate(self):
        good_q, good_r = _identity_matches(40)
        bad_q, bad_r = _garbage_matches(40)
        db = FakeDB({1: _feats(40, fid=1), 2: _feats(40, fid=2)})
        matcher = FakeMatcher({1: (bad_q, bad_r), 2: (good_q, good_r)})
        ver = _verifier(matcher=matcher)

        res = ver.verify(_feats(40), [(1, 0.9), (2, 0.5)], db)

        assert res is not None
        assert res.candidate_id == 2, "переміг кандидат без спільної геометрії"

    def test_mad_does_not_let_garbage_outrank_a_good_candidate(self):
        """Регресія §1.1.

        Кандидат 1 — сміття (40 випадкових пар), кандидат 2 — 25 чесних.
        Зі старим MAD (поріг по ВСІХ парах) сміттєвий міг відрапортувати всі 40
        «інлаєрів» і обійти чесні 25. Після виправлення маска лише звужується.
        """
        bad_q, bad_r = _garbage_matches(40, seed=5)
        good_q, good_r = _identity_matches(25, seed=6)
        db = FakeDB({1: _feats(40, fid=1), 2: _feats(25, fid=2)})
        matcher = FakeMatcher({1: (bad_q, bad_r), 2: (good_q, good_r)})
        ver = _verifier(matcher=matcher)

        res = ver.verify(_feats(40), [(1, 0.99), (2, 0.10)], db)

        assert res is not None
        assert res.candidate_id == 2
        assert res.inliers <= 25

    def test_returns_none_when_nothing_reaches_min_matches(self):
        q, r = _identity_matches(5)
        db = FakeDB({1: _feats(5, fid=1)})
        ver = _verifier(matcher=FakeMatcher({1: (q, r)}))
        assert ver.verify(_feats(5), [(1, 0.9)], db) is None

    def test_empty_candidate_list_returns_none(self):
        ver = _verifier(matcher=FakeMatcher({}))
        assert ver.verify(_feats(10), [], FakeDB({})) is None


class TestEarlyStop:
    def test_early_stop_skips_remaining_candidates(self):
        good_q, good_r = _identity_matches(40)
        db = FakeDB({1: _feats(40, fid=1), 2: _feats(40, fid=2), 3: _feats(40, fid=3)})
        matcher = FakeMatcher({1: (good_q, good_r), 2: (good_q, good_r), 3: (good_q, good_r)})
        ver = _verifier(matcher=matcher, early_stop_inliers=10)

        res = ver.verify(_feats(40), [(1, 0.9), (2, 0.8), (3, 0.7)], db)

        assert res is not None
        assert matcher.calls == [1], f"early-stop не спрацював, матчили {matcher.calls}"

    def test_without_early_stop_all_candidates_are_tried(self):
        good_q, good_r = _identity_matches(40)
        db = FakeDB({1: _feats(40, fid=1), 2: _feats(40, fid=2)})
        matcher = FakeMatcher({1: (good_q, good_r), 2: (good_q, good_r)})
        ver = _verifier(matcher=matcher, early_stop_inliers=10_000)
        ver.verify(_feats(40), [(1, 0.9), (2, 0.8)], db)
        assert matcher.calls == [1, 2]


class TestRefCache:
    def test_ref_cache_prevents_repeated_db_reads(self):
        good_q, good_r = _identity_matches(40)
        db = FakeDB({1: _feats(40, fid=1)})
        ver = _verifier(matcher=FakeMatcher({1: (good_q, good_r)}))

        cache = {}
        ver.verify(_feats(40), [(1, 0.9)], db, ref_cache=cache)
        first = list(db.reads)
        # Кеш заповнюється лише префільтром; без нього verify читає щоразу —
        # фіксуємо фактичний контракт, а не бажаний.
        ver.verify(_feats(40), [(1, 0.9)], db, ref_cache={1: db._feats[1]})
        assert db.reads == first, "передане у ref_cache має уникати читання з БД"


class TestMnnCounts:
    def test_returns_none_on_degenerate_query(self):
        ver = _verifier(matcher=FakeMatcher({}))
        assert ver.mnn_counts({"descriptors": None}, [(1, 0.5)], FakeDB({})) is None
        assert ver.mnn_counts({"descriptors": np.empty((0, 8))}, [(1, 0.5)], FakeDB({})) is None

    def test_identical_descriptors_score_high(self):
        f = _feats(30, fid=1, seed=3)
        db = FakeDB({1: f})
        ver = _verifier(matcher=FakeMatcher({}))
        scored = ver.mnn_counts(f, [(1, 0.9)], db)
        assert scored is not None
        n_pairs, fid, _score = scored[0]
        assert fid == 1
        assert n_pairs > 0, "однакові набори дескрипторів мають дати MNN-пари"

    def test_unreadable_candidate_scores_zero_not_raises(self):
        db = FakeDB({})  # будь-який frame_id → ValueError
        ver = _verifier(matcher=FakeMatcher({}))
        scored = ver.mnn_counts(_feats(20), [(99, 0.5)], db, ref_cache={})
        assert scored == [(0, 99, 0.5)]

    def test_mismatched_descriptor_dim_scores_zero(self):
        q = _feats(20, dim=8, fid=0)
        db = FakeDB({1: _feats(20, dim=16, fid=1)})
        ver = _verifier(matcher=FakeMatcher({}))
        scored = ver.mnn_counts(q, [(1, 0.5)], db, ref_cache={})
        assert scored[0][0] == 0


class TestPrefilter:
    def test_prefilter_keeps_requested_number(self):
        good_q, good_r = _identity_matches(40)
        target = _feats(30, fid=2, seed=11)
        db = FakeDB(
            {
                1: _feats(30, fid=1, seed=21),
                2: target,
                3: _feats(30, fid=3, seed=31),
            }
        )
        matcher = FakeMatcher({1: (good_q, good_r), 2: (good_q, good_r), 3: (good_q, good_r)})
        ver = _verifier(
            matcher=matcher,
            prefilter_enabled=True,
            prefilter_keep=1,
            early_stop_inliers=10_000,
        )
        # query == кандидат 2 → MNN має вивести саме його
        ver.verify(target, [(1, 0.9), (2, 0.9), (3, 0.9)], db)
        assert len(matcher.calls) == 1
        assert matcher.calls[0] == 2

    def test_prefilter_keeps_full_list_when_no_mnn_pairs(self):
        """Консервативність: якщо жоден кандидат не дав пар — список не ріжеться."""
        q = _feats(20, dim=8, fid=0)
        db = FakeDB({1: _feats(20, dim=16, fid=1), 2: _feats(20, dim=16, fid=2)})
        good_q, good_r = _identity_matches(40)
        matcher = FakeMatcher({1: (good_q, good_r), 2: (good_q, good_r)})
        ver = _verifier(
            matcher=matcher,
            prefilter_enabled=True,
            prefilter_keep=1,
            early_stop_inliers=10_000,
        )
        ver.verify(q, [(1, 0.9), (2, 0.8)], db)
        assert matcher.calls == [1, 2]


class TestResultShape:
    def test_result_fields_are_consistent(self):
        good_q, good_r = _identity_matches(40)
        db = FakeDB({1: _feats(40, fid=1)})
        ver = _verifier(matcher=FakeMatcher({1: (good_q, good_r)}))
        res = ver.verify(_feats(40), [(1, 0.9)], db)

        assert res is not None
        assert res.H_query_to_ref.shape == (3, 3)
        assert len(res.mkpts_q_in) == len(res.mkpts_r_in) == res.inliers
        assert res.inliers <= res.total_matches == 40
        assert res.rmse == pytest.approx(res.rmse)  # не NaN
        assert np.isfinite(res.rmse)
