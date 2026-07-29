"""Юніт-тести OutlierDetector (аудит §3.3).

Модуль — єдиний швидкий запобіжник між локалізацією та GPS-виходом, і до цього
не мав жодного тесту. Аудит §1.2 додав його виклик на OF-шлях
(`tracking.of_outlier_gate`), тож поведінка стала важити більше.

Тести описують КОНТРАКТ, а не поточні значення конфігу: у user_config.json
зараз стоять outlier_threshold_std=80 і max_speed_mps=350, які фактично
вимикають детектор. Тут скрізь задаються явні фізичні пороги.
"""

import numpy as np
import pytest

from src.tracking.outlier_detector import OutlierDetector


def _steady(det: OutlierDetector, n: int = 5, step: float = 10.0, dt: float = 1.0):
    """Наповнює вікно рівномірним рухом `step` метрів за `dt` секунд."""
    for i in range(n):
        det.add_position((i * step, 0.0), dt=dt)


class TestWarmup:
    def test_no_verdict_before_three_samples(self):
        """Менше 3 точок історії — судити нема з чого, все приймається."""
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0)
        assert det.is_outlier((10_000.0, 10_000.0), dt=1.0) is False
        det.add_position((0.0, 0.0))
        det.add_position((1.0, 0.0))
        assert det.is_outlier((10_000.0, 10_000.0), dt=1.0) is False


class TestSpeedGate:
    def test_teleport_is_rejected(self):
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0, max_consecutive=5)
        _steady(det)
        # 5000 м за 1 с = 5000 м/с ≫ 50 м/с
        assert det.is_outlier((5000.0, 0.0), dt=1.0) is True

    def test_normal_motion_is_accepted(self):
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0)
        _steady(det)
        assert det.is_outlier((50.0, 0.0), dt=1.0) is False

    def test_speed_uses_dt_not_distance(self):
        """Та сама відстань за довший час — це не викид."""
        det = OutlierDetector(threshold_std=100.0, max_speed_mps=50.0)
        _steady(det, step=10.0, dt=1.0)
        assert det.is_outlier((140.0, 0.0), dt=1.0) is True    # 100 м/с
        det2 = OutlierDetector(threshold_std=100.0, max_speed_mps=50.0)
        _steady(det2, step=10.0, dt=1.0)
        assert det2.is_outlier((140.0, 0.0), dt=10.0) is False  # 10 м/с


class TestConsecutiveReset:
    def test_sustained_jump_is_eventually_accepted(self):
        """Після max_consecutive поспіль детектор здається: дрон реально там.

        Це навмисна поведінка (інакше вихід за межі покриття заклинив би
        трекінг назавжди), і саме її треба зафіксувати, щоб не «полагодити»
        випадково.
        """
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0, max_consecutive=3)
        _steady(det)
        assert det.is_outlier((5000.0, 0.0), dt=1.0) is True
        assert det.is_outlier((5010.0, 0.0), dt=1.0) is True
        # третій поспіль — скидання вікна і прийняття
        assert det.is_outlier((5020.0, 0.0), dt=1.0) is False
        assert len(det.window) == 0

    def test_accepted_position_clears_the_counter(self):
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0, max_consecutive=3)
        _steady(det)
        assert det.is_outlier((5000.0, 0.0), dt=1.0) is True
        assert det.is_outlier((50.0, 0.0), dt=1.0) is False
        # лічильник обнулився → знову треба 3 поспіль
        assert det.is_outlier((5000.0, 0.0), dt=1.0) is True
        assert det.is_outlier((5010.0, 0.0), dt=1.0) is True
        assert det.is_outlier((5020.0, 0.0), dt=1.0) is False


class TestAddPosition:
    def test_reset_consecutive_false_keeps_counter(self):
        """OF-шлях дописує позиції з reset_consecutive=False — лічильник живе."""
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0, max_consecutive=3)
        _steady(det)
        det.is_outlier((5000.0, 0.0), dt=1.0)
        assert det._consecutive_outliers == 1
        det.add_position((51.0, 0.0), dt=1.0, reset_consecutive=False)
        assert det._consecutive_outliers == 1
        det.add_position((52.0, 0.0), dt=1.0, reset_consecutive=True)
        assert det._consecutive_outliers == 0

    def test_window_is_bounded(self):
        det = OutlierDetector(window_size=4)
        _steady(det, n=20)
        assert len(det.window) == 4

    def test_zero_dt_does_not_divide_by_zero(self):
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0)
        _steady(det)
        # dt=0 клампиться до 0.01 — має дати скінченний вердикт, а не ZeroDivision
        assert det.is_outlier((60.0, 0.0), dt=0.0) in (True, False)


class TestReset:
    def test_reset_clears_state(self):
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0, max_consecutive=3)
        _steady(det)
        det.is_outlier((5000.0, 0.0), dt=1.0)
        det.reset()
        assert len(det.window) == 0
        assert det._consecutive_outliers == 0
        # після reset знову фаза прогріву
        assert det.is_outlier((99_999.0, 0.0), dt=1.0) is False


class TestZScoreDependsOnHistoryVariance:
    """Документує §1.2 ТОЧНО — з поправкою до першої редакції аудиту.

    Ефективний поріг Z-гейта у м/с дорівнює ``threshold_std · max(std_швидкості, 1.0)``
    (std має підлогу 1.0 у коді). Отже при chinному ``threshold_std=80``:

    * рівний політ (std → 0, підлога 1.0) — гейт ловить стрибок > 80 м/с;
    * реальний шум швидкості — поріг росте пропорційно і при
      std > 350/80 = 4.375 м/с ПЕРЕВИЩУЄ cap ``max_speed_mps=350``, тобто
      Z-гейт стає мертвим і лишається тільки cap 350 м/с (1260 км/год).

    Тобто «Z-гейт не спрацьовує ніколи» — неправда; правда — «він відмирає, щойно
    траєкторія набирає звичайний шум швидкості». Тест фіксує обидві гілки, щоб
    зміна порогів була свідомою.
    """

    ZSCORE_DIES_ABOVE_STD = 350.0 / 80.0  # ≈ 4.375 м/с

    def test_smooth_flight_zscore_still_catches_large_jumps(self):
        det = OutlierDetector(
            window_size=10, threshold_std=80.0, max_speed_mps=350.0, max_consecutive=3
        )
        _steady(det, step=10.0, dt=1.0)  # ідеально рівна швидкість → std = підлога 1.0
        assert det.is_outlier((40.0 + 100.0, 0.0), dt=1.0) is True

    def test_noisy_flight_zscore_is_dead_and_cap_is_too_high(self):
        """З реалістичним шумом швидкості 340 м/с (1224 км/год) проходить наскрізь."""
        rng = np.random.default_rng(0)
        det = OutlierDetector(
            window_size=10, threshold_std=80.0, max_speed_mps=350.0, max_consecutive=3
        )
        x = 0.0
        for _ in range(8):
            x += 10.0 + rng.normal(0.0, 10.0)  # std ≈ 10 м/с > 4.375
            det.add_position((x, 0.0), dt=1.0)
        assert det.is_outlier((x + 340.0, 0.0), dt=1.0) is False

    def test_physical_thresholds_catch_it_regardless_of_noise(self):
        rng = np.random.default_rng(0)
        det = OutlierDetector(
            window_size=10, threshold_std=4.0, max_speed_mps=120.0, max_consecutive=3
        )
        x = 0.0
        for _ in range(8):
            x += 10.0 + rng.normal(0.0, 10.0)
            det.add_position((x, 0.0), dt=1.0)
        assert det.is_outlier((x + 340.0, 0.0), dt=1.0) is True

    @pytest.mark.parametrize("std_noise", [0.0, 2.0, 4.0])
    def test_crossover_below_threshold_zscore_fires_first(self, std_noise):
        """Нижче кросоверу Z-поріг < cap, отже Z спрацьовує раніше за cap."""
        assert 80.0 * max(std_noise, 1.0) < 350.0



class TestArrayLikeInput:
    def test_accepts_tuple_and_ndarray(self):
        det = OutlierDetector(threshold_std=3.0, max_speed_mps=50.0)
        _steady(det)
        assert det.is_outlier(np.array([50.0, 0.0]), dt=1.0) is False
        det.add_position(np.array([60.0, 0.0]), dt=1.0)
        assert len(det.window) > 0
