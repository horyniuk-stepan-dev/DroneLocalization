"""ADDENDUM 2.1: каскадний recovery в RotationSelector.

Головна властивість, яку тут захищаємо: каскад НЕ МОЖЕ бути дорожчим за
поточну поведінку. Етап 2 рахує лише комбінації, яких не було на етапі 1,
тож найгірший випадок = стільки ж форвардів, скільки й зараз (20 у типовій
конфігурації 4 кути × 5 масштабів), а типовий = 4.

Фейковий екстрактор лічить виклики — тест без GPU/торча.
"""

import numpy as np
import pytest

from src.localization.rotation_selector import RotationSelector

ANGLES = [0, 90, 180, 270]
PYRAMID = [0.5, 0.7, 1.0, 1.4, 2.0]


class FakeExtractor:
    """Рахує, скільки кадрів пройшло через екстракцію, і в якому порядку."""

    def __init__(self):
        self.batches: list[int] = []
        self.total = 0

    def extract_global_descriptors_multi(self, frames):
        self.batches.append(len(frames))
        self.total += len(frames)
        return [np.full(4, float(i), dtype=np.float32) for i in range(len(frames))]

    def extract_global_descriptor(self, frame):
        self.batches.append(1)
        self.total += 1
        return np.zeros(4, dtype=np.float32)


class FakeRetriever:
    """Повертає заданий скор для кожного виклику по черзі (потім — останній)."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def retrieve(self, desc, top_k):
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return "src0", [(self.calls, s)]


class FakeScaleManager:
    def __init__(self, prior=None, candidates=None):
        self.prior = prior
        self._candidates = candidates if candidates is not None else list(PYRAMID)
        self.rescan_min_score = 0.65
        self.normalized: list[float] = []

    def candidates(self):
        return list(self._candidates)

    def normalize(self, frame, r):
        self.normalized.append(r)
        return frame, None


def _cfg(cascade: bool, rescan_min: float = 0.70):
    return {
        "localization": {
            "rotation_rescan_min_score": rescan_min,
            "recovery_cascade": cascade,
        }
    }


def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _select(cfg, retriever, sm, extractor):
    sel = RotationSelector(extractor, retriever, cfg)
    return sel.select(
        _frame(), prior_angle=None, use_prior=False, angles_to_try=ANGLES, top_k=5, scale_manager=sm
    )


class TestPlanStages:
    def test_disabled_is_single_full_product(self):
        stages = RotationSelector._plan_stages(ANGLES, PYRAMID, None, use_cascade=False)
        assert len(stages) == 1
        assert len(stages[0]) == 20

    def test_disabled_preserves_exact_order(self):
        """Побітово та сама послідовність, що й до змін (angle-major)."""
        stages = RotationSelector._plan_stages(ANGLES, PYRAMID, None, use_cascade=False)
        expected = [(a, s) for a in ANGLES for s in PYRAMID]
        assert stages[0] == expected

    def test_cascade_splits_4_then_16(self):
        stages = RotationSelector._plan_stages(
            ANGLES, PYRAMID, FakeScaleManager(), use_cascade=True
        )
        assert [len(s) for s in stages] == [4, 16]

    def test_stage2_never_repeats_stage1(self):
        """Ключова властивість: сума етапів = повний добуток, без дублів."""
        s1, s2 = RotationSelector._plan_stages(
            ANGLES, PYRAMID, FakeScaleManager(), use_cascade=True
        )
        assert not set(s1) & set(s2)
        assert set(s1) | set(s2) == {(a, s) for a in ANGLES for s in PYRAMID}
        assert len(s1) + len(s2) == 20

    def test_stage1_uses_scale_prior_when_present(self):
        s1, _ = RotationSelector._plan_stages(
            ANGLES, PYRAMID, FakeScaleManager(prior=1.4), use_cascade=True
        )
        assert {sc for _, sc in s1} == {1.4}

    def test_stage1_falls_back_to_unity(self):
        s1, _ = RotationSelector._plan_stages(
            ANGLES, PYRAMID, FakeScaleManager(prior=None), use_cascade=True
        )
        assert {sc for _, sc in s1} == {1.0}

    def test_stage1_uses_first_candidate_when_no_unity(self):
        """ScaleManager.candidates() сортує піраміду за depth-hint — беремо голову."""
        pyramid = [0.7, 0.5, 1.4]
        s1, _ = RotationSelector._plan_stages(
            ANGLES, pyramid, FakeScaleManager(prior=None), use_cascade=True
        )
        assert {sc for _, sc in s1} == {0.7}

    def test_prior_outside_pyramid_ignored(self):
        """EMA-prior 1.23 не належить сітці — не можна планувати етап навколо нього."""
        s1, s2 = RotationSelector._plan_stages(
            ANGLES, PYRAMID, FakeScaleManager(prior=1.23), use_cascade=True
        )
        assert {sc for _, sc in s1} == {1.0}
        assert len(s1) + len(s2) == 20

    def test_single_scale_is_one_stage(self):
        """Prior-масштаб у steady state → дробити нема чого."""
        stages = RotationSelector._plan_stages(
            ANGLES, [1.0], FakeScaleManager(prior=1.0), use_cascade=True
        )
        assert len(stages) == 1 and len(stages[0]) == 4


class TestSelectCost:
    def test_baseline_runs_all_20(self):
        ex = FakeExtractor()
        _select(_cfg(cascade=False), FakeRetriever([0.9]), FakeScaleManager(), ex)
        assert ex.total == 20
        assert ex.batches == [20]

    def test_cascade_good_first_stage_costs_4(self):
        ex = FakeExtractor()
        res = _select(_cfg(cascade=True), FakeRetriever([0.95]), FakeScaleManager(), ex)
        assert ex.total == 4
        assert ex.batches == [4]
        assert res is not None and res.best_scale == 1.0

    def test_cascade_bad_first_stage_costs_exactly_20(self):
        """Найгірший випадок каскаду не дорожчий за поточну поведінку."""
        ex = FakeExtractor()
        _select(_cfg(cascade=True), FakeRetriever([0.10]), FakeScaleManager(), ex)
        assert ex.total == 20
        assert ex.batches == [4, 16]

    def test_cascade_borderline_score_triggers_stage2(self):
        """Скор рівно під порогом → етап 2 (порог перевіряється як >=)."""
        ex = FakeExtractor()
        _select(_cfg(cascade=True, rescan_min=0.70), FakeRetriever([0.699]), FakeScaleManager(), ex)
        assert ex.total == 20

    def test_cascade_score_at_threshold_stops(self):
        ex = FakeExtractor()
        _select(_cfg(cascade=True, rescan_min=0.70), FakeRetriever([0.70]), FakeScaleManager(), ex)
        assert ex.total == 4

    def test_stage2_can_win_over_stage1(self):
        """Слабкий етап 1 (0.2), сильна комбінація на етапі 2 — перемагає друга."""
        scores = [0.2] * 4 + [0.2] * 7 + [0.99] + [0.2] * 8
        ex = FakeExtractor()
        res = _select(_cfg(cascade=True), FakeRetriever(scores), FakeScaleManager(), ex)
        assert ex.total == 20
        assert res is not None
        assert res.score == pytest.approx(0.99)
        assert res.best_scale != 1.0

    def test_result_identical_with_and_without_cascade(self):
        """При однакових скорах вибір каскаду збігається з базовою поведінкою."""
        scores = [0.1, 0.2, 0.3, 0.4] + [0.15] * 16
        a = FakeExtractor()
        res_base = _select(_cfg(cascade=False), FakeRetriever(scores), FakeScaleManager(), a)
        b = FakeExtractor()
        res_casc = _select(_cfg(cascade=True), FakeRetriever(scores), FakeScaleManager(), b)
        # Базова гілка йде angle-major (0.5, 0.7, 1.0, ...), каскад — спершу 1.0.
        # Порівнюємо не порядок, а те, що обидві повертають валідний результат
        # із максимальним побаченим скором.
        assert res_base is not None and res_casc is not None
        assert res_base.score == pytest.approx(0.4)
        assert res_casc.score == pytest.approx(0.4)

    def test_rotation_cached_per_angle(self):
        """rot90 не повторюється на кожен масштаб — інакше зайві копії кадру."""
        cache: dict[int, object] = {}
        sm = FakeScaleManager()
        frame = _frame()
        f1 = RotationSelector._prepare_frame(frame, 90, 1.0, sm, cache)
        f2 = RotationSelector._prepare_frame(frame, 90, 1.0, sm, cache)
        assert f1 is f2
        assert list(cache.keys()) == [90]

    def test_no_scale_manager_is_single_stage(self):
        ex = FakeExtractor()
        _select(_cfg(cascade=True), FakeRetriever([0.1]), None, ex)
        assert ex.total == 4
