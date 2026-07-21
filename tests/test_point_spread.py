"""ADDENDUM 1.1: просторовий розкид інлаєрів (spatial collapse, OrthoTrack §3.4).

Чисті numpy-функції — тест без torch/Qt/БД.
"""

import numpy as np
import pytest

from src.geometry.point_spread import (
    UNIFORM_SPREAD,
    inlier_spread,
    spread_confidence_factor,
    spread_weight_factor,
)

W, H = 1920, 1080


def _uniform(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)])


class TestInlierSpread:
    def test_uniform_coverage_matches_theory(self):
        """Рівномірне покриття кадру → σ/L = 1/√12 ≈ 0.289, незалежно від аспекту."""
        s = inlier_spread(_uniform(), W, H)
        assert s == pytest.approx(UNIFORM_SPREAD, abs=0.02)

    def test_aspect_invariance(self):
        """Той самий розподіл у квадратному кадрі дає той самий розкид."""
        rng = np.random.default_rng(1)
        square = np.column_stack([rng.uniform(0, 1000, 4000), rng.uniform(0, 1000, 4000)])
        assert inlier_spread(square, 1000, 1000) == pytest.approx(UNIFORM_SPREAD, abs=0.02)

    def test_resolution_invariance(self):
        """Масштабування кадру і точок разом не змінює метрику."""
        pts = _uniform(seed=2)
        assert inlier_spread(pts, W, H) == pytest.approx(
            inlier_spread(pts * 2.0, W * 2, H * 2), abs=1e-9
        )

    def test_cluster_in_corner(self):
        """Кластер у ~5% сторони кадру — головний режим відмови — < 0.05."""
        rng = np.random.default_rng(3)
        pts = np.column_stack([rng.uniform(0, 0.05 * W, 300), rng.uniform(0, 0.05 * H, 300)])
        assert inlier_spread(pts, W, H) < 0.05

    def test_collinear_is_zero_not_none(self):
        """Точки на одній прямій = вироджена оцінка. Це 0.0, а НЕ «немає сигналу»:
        плутанина цих двох випадків знімала б штраф із найгіршого сценарію."""
        rng = np.random.default_rng(4)
        line = np.column_stack([rng.uniform(0, W, 200), np.full(200, 500.0)])
        s = inlier_spread(line, W, H)
        assert s == 0.0
        assert s is not None
        assert spread_confidence_factor(s) < 1.0

    def test_monotonic_in_cloud_size(self):
        rng = np.random.default_rng(5)
        prev = -1.0
        for frac in (0.05, 0.15, 0.35, 0.7, 1.0):
            pts = np.column_stack([rng.uniform(0, frac * W, 2000), rng.uniform(0, frac * H, 2000)])
            cur = inlier_spread(pts, W, H)
            assert cur > prev
            prev = cur

    @pytest.mark.parametrize(
        "pts",
        [None, np.empty((0, 2)), np.array([[10.0, 20.0]]), np.array([1.0, 2.0])],
    )
    def test_unavailable_returns_none(self, pts):
        assert inlier_spread(pts, W, H) is None

    @pytest.mark.parametrize("w,h", [(0, H), (W, 0), (-5, H), (float("nan"), H)])
    def test_bad_frame_size_returns_none(self, w, h):
        assert inlier_spread(_uniform(100), w, h) is None

    def test_nan_points_dropped_not_propagated(self):
        pts = np.vstack([_uniform(500, seed=6), [[np.nan, 1.0], [2.0, np.inf]]])
        s = inlier_spread(pts, W, H)
        assert s is not None and np.isfinite(s) and s > 0.1

    def test_all_nan_returns_none(self):
        assert inlier_spread(np.full((10, 2), np.nan), W, H) is None


class TestConfidenceFactor:
    def test_healthy_frame_no_penalty(self):
        assert spread_confidence_factor(UNIFORM_SPREAD) == 1.0

    def test_at_reference_no_penalty(self):
        assert spread_confidence_factor(0.15, spread_ref=0.15) == 1.0

    def test_none_means_no_penalty(self):
        """Сигнал недоступний ≠ поганий кадр."""
        assert spread_confidence_factor(None) == 1.0

    def test_collapse_clipped_to_floor(self):
        assert spread_confidence_factor(0.0, floor=0.35) == pytest.approx(0.35)
        assert spread_confidence_factor(0.001, floor=0.35) == pytest.approx(0.35)

    def test_linear_between_floor_and_one(self):
        assert spread_confidence_factor(0.075, spread_ref=0.15, floor=0.1) == pytest.approx(0.5)

    def test_monotonic(self):
        vals = [spread_confidence_factor(s) for s in (0.0, 0.05, 0.10, 0.15, 0.30)]
        assert vals == sorted(vals)


class TestWeightFactor:
    def test_no_penalty_above_reference(self):
        assert spread_weight_factor(0.15) == 1.0
        assert spread_weight_factor(0.30) == 1.0

    def test_none_means_no_penalty(self):
        assert spread_weight_factor(None) == 1.0

    def test_known_values(self):
        """k=10, ref=0.15: дефіцит 0.10 → ×0.50; повний колапс → ×0.40."""
        assert spread_weight_factor(0.05, 0.15, 10.0) == pytest.approx(0.5)
        assert spread_weight_factor(0.0, 0.15, 10.0) == pytest.approx(1 / 2.5)

    def test_never_zero_or_negative(self):
        """Вага глушиться, але ребро не зникає — це не гейт."""
        for s in (0.0, 0.01, 0.1):
            f = spread_weight_factor(s, 0.15, 1000.0)
            assert 0.0 < f <= 1.0

    def test_monotonic(self):
        vals = [spread_weight_factor(s) for s in (0.0, 0.05, 0.10, 0.15)]
        assert vals == sorted(vals)
