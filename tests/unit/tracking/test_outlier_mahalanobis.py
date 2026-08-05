"""Mahalanobis-гейт: χ² від коваріації інновації KF замість Z-score за швидкістю.

Чому взагалі інша метрика. Z-score рахує швидкість проти вікна ПРИЙНЯТИХ
швидкостей, тобто поріг формується тим самим потоком, який гейт фільтрує —
самопідтримна петля (заміряно на місії 2026-07-31: mean_speed падав до 11.9 при
реальних 89.2 м/с). Mahalanobis нормує інновацію коваріацією самого фільтра:
нормалізація приходить з моделі, а не з відфільтрованої вибірки.

Тести описують КОНТРАКТ гейта, а не значення конфігу: пороги задаються явно.
Гілка дефолтом вимкнена (``tracking.outlier_mahalanobis_enabled=False``).
"""

import numpy as np

from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector

CHI2_P999_2DOF = 13.816


def _steady_filter(n=12, step=10.0, dt=1.0):
    """KF, що проїхав прямою зі сталою швидкістю ``step`` м за крок."""
    kf = TrajectoryFilter(process_noise=2.0, measurement_noise=5.0, dt=dt)
    for i in range(n):
        kf.update((i * step, 0.0), dt=dt)
    return kf


class TestMahalanobisProbeIsPure:
    def test_returns_none_before_initialisation(self):
        kf = TrajectoryFilter()
        assert kf.mahalanobis_sq((0.0, 0.0)) is None

    def test_does_not_mutate_filter_state(self):
        kf = _steady_filter()
        x_before = kf.kf.x.copy()
        P_before = kf.kf.P.copy()
        R_before = kf.kf.R.copy()

        kf.mahalanobis_sq((10_000.0, 10_000.0), dt=1.0)

        # Це головна вимога до гейта: відкинуте вимірювання не має зсунути
        # фільтр. filterpy kf.predict() мутує, тому probe рахує на копіях.
        assert np.array_equal(kf.kf.x, x_before)
        assert np.array_equal(kf.kf.P, P_before)
        assert np.array_equal(kf.kf.R, R_before)

    def test_repeated_calls_give_identical_result(self):
        kf = _steady_filter()
        first = kf.mahalanobis_sq((200.0, 50.0), dt=1.0)
        second = kf.mahalanobis_sq((200.0, 50.0), dt=1.0)
        assert first == second


class TestMahalanobisDistanceOrdering:
    def test_on_track_measurement_is_below_threshold(self):
        kf = _steady_filter(n=12, step=10.0)
        # Наступний крок точно по прогнозу: d² має бути малим
        d2 = kf.mahalanobis_sq((120.0, 0.0), dt=1.0)
        assert d2 is not None
        assert d2 < CHI2_P999_2DOF

    def test_gross_jump_is_above_threshold(self):
        kf = _steady_filter(n=12, step=10.0)
        d2 = kf.mahalanobis_sq((5000.0, 5000.0), dt=1.0)
        assert d2 is not None
        assert d2 > CHI2_P999_2DOF

    def test_distance_grows_with_deviation(self):
        kf = _steady_filter(n=12, step=10.0)
        near = kf.mahalanobis_sq((125.0, 0.0), dt=1.0)
        far = kf.mahalanobis_sq((400.0, 0.0), dt=1.0)
        assert near < far

    def test_larger_noise_scale_softens_the_gate(self):
        kf = _steady_filter(n=12, step=10.0)
        strict = kf.mahalanobis_sq((300.0, 0.0), dt=1.0, noise_scale=1.0)
        soft = kf.mahalanobis_sq((300.0, 0.0), dt=1.0, noise_scale=10.0)
        assert soft < strict


class TestDetectorBranch:
    def _detector(self, **kw):
        params = dict(
            window_size=10,
            threshold_std=3.0,
            max_speed_mps=1e9,  # фізичний гейт свідомо вимкнено
            max_consecutive=5,
            zscore_enabled=False,
            mahalanobis_enabled=True,
            chi2_threshold=CHI2_P999_2DOF,
        )
        params.update(kw)
        return OutlierDetector(**params)

    def _fill(self, det, n=5):
        for i in range(n):
            det.add_position((i * 10.0, 0.0), dt=1.0)

    def test_high_d2_is_rejected(self):
        det = self._detector()
        self._fill(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=99.0) is True

    def test_low_d2_is_accepted(self):
        det = self._detector()
        self._fill(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=1.0) is False

    def test_disabled_branch_ignores_d2(self):
        det = self._detector(mahalanobis_enabled=False)
        self._fill(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=99.0) is False

    def test_missing_d2_does_not_reject(self):
        # Probe повертає None (фільтр не готовий) → гейт має пропустити вимір,
        # а не відкинути його: невідомо ≠ погано.
        det = self._detector()
        self._fill(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=None) is False

    def test_works_before_window_is_filled(self):
        # Вікно швидкостей ще коротке (<3), але коваріація фільтра вже щось
        # знає — гілка не має мовчати саме тоді, коли фільтр щойно скинули.
        det = self._detector()
        det.add_position((0.0, 0.0), dt=1.0)
        assert det.is_outlier((10.0, 0.0), dt=1.0, maha_d2=99.0) is True

    def test_consecutive_outliers_force_acceptance(self):
        # Дрон реально перемістився: після max_consecutive підряд гейт здається,
        # інакше трекінг залипає назавжди. Контракт спільний з іншими гілками.
        det = self._detector(max_consecutive=3)
        self._fill(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=99.0) is True
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=99.0) is True
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=99.0) is False
        assert len(det.window) == 0

    def test_zscore_and_mahalanobis_are_independent(self):
        # Z-score увімкнений, Mahalanobis ні: d² не має впливати.
        det = self._detector(zscore_enabled=True, mahalanobis_enabled=False)
        self._fill(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0, maha_d2=1e6) is False


class TestEndToEndAgainstFilter:
    def test_real_filter_gate_accepts_track_and_rejects_jump(self):
        kf = _steady_filter(n=12, step=10.0)
        det = OutlierDetector(
            window_size=10,
            max_speed_mps=1e9,
            zscore_enabled=False,
            mahalanobis_enabled=True,
            chi2_threshold=CHI2_P999_2DOF,
        )
        for i in range(5):
            det.add_position((i * 10.0, 0.0), dt=1.0)

        on_track = (120.0, 0.0)
        jump = (5000.0, 5000.0)
        assert det.is_outlier(on_track, dt=1.0, maha_d2=kf.mahalanobis_sq(on_track)) is False
        assert det.is_outlier(jump, dt=1.0, maha_d2=kf.mahalanobis_sq(jump)) is True
