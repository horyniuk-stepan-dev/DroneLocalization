"""Тести retrieval-метрик (Recall@K / SDM@K / PDM@K), Трек 6 / §3.2.

Ключова перевірка — реконструйована f(R_i) відтворює три приклади зі статті
AnyVisLoc §5.2 (arXiv:2503.10692): R_i≈0.27 → успіх (похибка 1.4 м), 0.35 →
успіх (2.1 м), 1.37 → провал (649 м). Решта — ранг-зважування, пороги,
крайові випадки.
"""

import math

import numpy as np
import pytest

import scripts.retrieval_metrics as M


class TestPdmScoreVsPaperAnchors:
    """f(R_i) з дефолтами статті (alpha=0.85, lam=6, l=1.67)."""

    def test_paper_success_regime_high(self):
        # R=0.27 і 0.35 — «успішні» retrieval у статті → скор близький до 1
        assert float(M.pdm_score(0.27)) == pytest.approx(0.97013, abs=1e-4)
        assert float(M.pdm_score(0.35)) == pytest.approx(0.95257, abs=1e-4)
        assert float(M.pdm_score(0.27)) > 0.9
        assert float(M.pdm_score(0.35)) > 0.9

    def test_paper_failure_regime_low(self):
        # R=1.37 — «провал» локалізації у статті → скор майже 0
        assert float(M.pdm_score(1.37)) == pytest.approx(0.04229, abs=1e-4)
        assert float(M.pdm_score(1.37)) < 0.05

    def test_success_scores_above_failure(self):
        assert float(M.pdm_score(0.27)) > float(M.pdm_score(1.37))

    def test_half_at_alpha(self):
        assert float(M.pdm_score(M.PDM_ALPHA)) == pytest.approx(0.5, abs=1e-12)

    def test_near_one_at_zero_not_exactly_one(self):
        f0 = float(M.pdm_score(0.0))
        assert f0 == pytest.approx(0.99394, abs=1e-4)
        assert f0 < 1.0  # логістика не досягає 1

    def test_zero_beyond_diagonal_clamp(self):
        assert float(M.pdm_score(1.67)) == 0.0  # R == l → без перекриття
        assert float(M.pdm_score(2.0)) == 0.0
        # без l-кліпу той самий R дав би малий, але додатний скор
        assert float(M.pdm_score(2.0, l_diag=None)) > 0.0

    def test_monotonic_decreasing(self):
        rs = np.linspace(0.0, 1.6, 20)
        f = M.pdm_score(rs)
        assert np.all(np.diff(f) < 0)

    def test_lambda_controls_sharpness(self):
        # більша lambda → різкіше падіння навколо alpha (далі від 0.5 при R>alpha)
        soft = float(M.pdm_score(1.1, lam=4.0, l_diag=None))
        sharp = float(M.pdm_score(1.1, lam=8.0, l_diag=None))
        assert sharp < soft


class TestRi:
    def test_normalization(self):
        assert float(M.r_i(50.0, 100.0)) == pytest.approx(0.5)

    def test_array_broadcast(self):
        r = M.r_i(np.array([10.0, 50.0, 200.0]), 100.0)
        np.testing.assert_allclose(r, [0.1, 0.5, 2.0])

    def test_nonpositive_width_raises(self):
        with pytest.raises(ValueError):
            M.r_i(10.0, 0.0)
        with pytest.raises(ValueError):
            M.r_i(10.0, -5.0)


class TestRecallAtK:
    def test_hit_within_threshold(self):
        assert M.recall_at_k([10.0, 3.0, 20.0], thresh_m=5.0) == 1.0

    def test_no_hit(self):
        assert M.recall_at_k([10.0, 8.0, 20.0], thresh_m=5.0) == 0.0

    def test_k_truncation_excludes_later_hit(self):
        # хіт лише на ранзі 2; k=1 його не бачить
        assert M.recall_at_k([10.0, 3.0, 20.0], k=1, thresh_m=5.0) == 0.0
        assert M.recall_at_k([10.0, 3.0, 20.0], k=2, thresh_m=5.0) == 1.0

    def test_empty(self):
        assert M.recall_at_k([]) == 0.0


class TestSdmAtK:
    def test_perfect_is_one(self):
        assert M.sdm_at_k([0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_rank_weighting_prefers_top_rank(self):
        # той самий набір відстаней, близький хіт на ранзі 1 vs ранзі 3
        top = M.sdm_at_k([0.0, 100.0, 100.0], s=0.2)
        bottom = M.sdm_at_k([100.0, 100.0, 0.0], s=0.2)
        assert top > bottom

    def test_large_distance_no_overflow(self):
        # exp(-s·d) з великим d не має падати/попереджати
        val = M.sdm_at_k([1e6, 1e6], s=1.0)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_empty(self):
        assert M.sdm_at_k([]) == 0.0


class TestPdmAtK:
    def test_perfect_overlap_high(self):
        # усі кандидати в центрі → R=0 → f≈0.994
        val = M.pdm_at_k([0.0, 0.0, 0.0], footprint_width_m=100.0)
        assert val == pytest.approx(0.99394, abs=1e-4)

    def test_rank_weighting_prefers_top_rank(self):
        # близький кандидат (R=0.1) на ранзі 1 vs ранзі 3, решта без перекриття
        top = M.pdm_at_k([10.0, 200.0, 200.0], 100.0)
        bottom = M.pdm_at_k([200.0, 200.0, 10.0], 100.0)
        assert top > bottom
        assert top == pytest.approx(0.98902 * 3 / 6, abs=1e-4)
        assert bottom == pytest.approx(0.98902 * 1 / 6, abs=1e-4)

    def test_footprint_scalar_equals_array(self):
        scalar = M.pdm_at_k([10.0, 50.0], 100.0)
        array = M.pdm_at_k([10.0, 50.0], [100.0, 100.0])
        assert scalar == pytest.approx(array)

    def test_per_candidate_footprint(self):
        # той самий d, але ширший футпринт → менший R → вищий скор
        narrow = M.pdm_at_k([50.0], [100.0])
        wide = M.pdm_at_k([50.0], [400.0])
        assert wide > narrow

    def test_all_beyond_diagonal_is_zero(self):
        assert M.pdm_at_k([500.0, 500.0], 100.0) == 0.0

    def test_nonpositive_width_raises(self):
        with pytest.raises(ValueError):
            M.pdm_at_k([10.0], 0.0)

    def test_empty(self):
        assert M.pdm_at_k([], 100.0) == 0.0


class TestQueryFromPositions:
    def test_euclidean_distances(self):
        q = M.query_from_positions(
            gt_xy=(0.0, 0.0),
            candidate_xy=[(3.0, 4.0), (0.0, 0.0)],
            footprint_width_m=100.0,
        )
        np.testing.assert_allclose(q["distances_m"], [5.0, 0.0])
        assert q["footprint_width_m"] == 100.0

    def test_feeds_into_pdm(self):
        q = M.query_from_positions((0.0, 0.0), [(0.0, 0.0)], 100.0)
        val = M.pdm_at_k(q["distances_m"], q["footprint_width_m"])
        assert val == pytest.approx(0.99394, abs=1e-4)


class TestComputeRetrievalMetrics:
    def test_mean_over_queries(self):
        q_good = M.query_from_positions((0.0, 0.0), [(0, 0), (0, 0), (0, 0)], 100.0)
        q_bad = M.query_from_positions((0.0, 0.0), [(1000, 0), (1000, 0), (1000, 0)], 100.0)
        out = M.compute_retrieval_metrics([q_good, q_bad], k=3)
        assert out["n_queries"] == 2
        assert out["k"] == 3
        assert out["recall@k"] == pytest.approx(0.5)  # 1 і 0
        assert out["pdm@k"] == pytest.approx((0.99394 + 0.0) / 2, abs=1e-4)
        assert 0.0 < out["sdm@k"] < 1.0

    def test_empty_queries(self):
        out = M.compute_retrieval_metrics([], k=5)
        assert out["n_queries"] == 0
        assert out["recall@k"] == 0.0
        assert out["pdm@k"] == 0.0
        assert out["sdm@k"] == 0.0

    def test_larger_pdm_for_better_retrieval(self):
        # гейт-семантика: краще (ближче) retrieval → вищий PDM@K
        better = [M.query_from_positions((0.0, 0.0), [(20, 0)], 100.0)]
        worse = [M.query_from_positions((0.0, 0.0), [(90, 0)], 100.0)]
        pdm_better = M.compute_retrieval_metrics(better, k=1)["pdm@k"]
        pdm_worse = M.compute_retrieval_metrics(worse, k=1)["pdm@k"]
        assert pdm_better > pdm_worse


def test_module_constants_match_paper():
    # дефолти = рекомендації статті для 4:3
    assert M.PDM_L == 1.67
    assert M.PDM_LAMBDA == 6.0
    assert 0.5 * M.PDM_L < M.PDM_ALPHA < M.PDM_L  # alpha «трохи вище l/2»
    assert math.isclose(M.SDM_S, 1.0 / M.RECALL_THRESH_M)
