"""Етап 6: аутлаєр-гейт має міряти СПРАВЖНІ наземні метри, не проєкційні.

WebMercator (EPSG:3857) розтягує відстані у 1/cos(lat). На широті місії
2026-07-31 (47.83°) це ×1.4895, тож max_speed_mps=120 реально гейтив на
80.6 м/с, і 28 із 71 speed-відсіювань у тому лозі були хибними.

Числа в тестах узяті прямо з того прогону.
"""

import math

import pytest

from src.tracking.outlier_detector import OutlierDetector

MISSION_LAT = 47.83
K = math.cos(math.radians(MISSION_LAT))  # ≈ 0.6713326, тобто інфляція 1.48957×


def _det(**kw):
    """Детектор з ізольованою гілкою max_speed (z-score вимкнено порогом)."""
    kw.setdefault("window_size", 10)
    kw.setdefault("threshold_std", 1e9)
    kw.setdefault("max_speed_mps", 120.0)
    kw.setdefault("max_consecutive", 10**6)
    return OutlierDetector(**kw)


def _seed(det, n=3, step=5.0, dt=1.0):
    """Заповнює вікно рівномірним повільним рухом (потрібно >= 3 точки)."""
    for i in range(n):
        det.add_position((0.0, i * step), dt=dt)
    return (0.0, (n - 1) * step)


def test_scale_factor_matches_mercator_inflation():
    """Звірка константи: Mercator на 47.83° завищує відстані у ~1.49×."""
    assert K == pytest.approx(0.6713326, abs=1e-7)
    assert 1.0 / K == pytest.approx(1.4895746, abs=1e-7)


def test_default_is_bit_exact_old_behaviour():
    """Дефолт (1.0) не змінює жодного рішення — стара поведінка побітово."""
    for dist in (5.0, 12.4, 17.9, 20.0, 100.3):
        a, b = _det(), _det(ground_scale=1.0)
        last_a, last_b = _seed(a), _seed(b)
        assert a.is_outlier((0.0, last_a[1] + dist), dt=0.1) == b.is_outlier(
            (0.0, last_b[1] + dist), dt=0.1
        )


def test_logged_false_positive_is_cleared_by_correction():
    """distance=12.4 м @ dt=0.1 — рядок з лога: 123.8 м/с «перевищення».

    Справжня швидкість 83.1 м/с, тобто добре в межах 120. Без корекції —
    відсіювався; з корекцією — проходить.
    """
    raw = _det()
    last = _seed(raw)
    assert raw.is_outlier((0.0, last[1] + 12.4), dt=0.1) is True

    fixed = _det(ground_scale=K)
    last = _seed(fixed)
    assert fixed.is_outlier((0.0, last[1] + 12.4), dt=0.1) is False


def test_genuine_outlier_still_rejected():
    """distance=20.0 і 100.3 м @ dt=0.1 — теж з лога, 134 і 673 м/с справжніх.

    Корекція НЕ повинна їх пропустити, інакше вона просто вимикає гейт.
    """
    for dist in (20.0, 100.3):
        det = _det(ground_scale=K)
        last = _seed(det)
        assert det.is_outlier((0.0, last[1] + dist), dt=0.1) is True


def test_threshold_boundary_is_in_true_metres():
    """Поріг спрацьовує рівно на 120 справжніх м/с, а не на 120/cos(lat)."""
    d_at_limit = 120.0 * 0.1 / K  # ≈ 17.879 проєкційних метрів

    below = _det(ground_scale=K)
    last = _seed(below)
    assert below.is_outlier((0.0, last[1] + d_at_limit * 0.99), dt=0.1) is False

    above = _det(ground_scale=K)
    last = _seed(above)
    assert above.is_outlier((0.0, last[1] + d_at_limit * 1.01), dt=0.1) is True


def test_set_ground_scale_ignores_nonpositive():
    """Нуль/від'ємне занулило б усі швидкості й вимкнуло гейт — ігноруємо."""
    det = _det(ground_scale=K)
    det.set_ground_scale(0.0)
    det.set_ground_scale(-1.0)
    last = _seed(det)
    assert det.is_outlier((0.0, last[1] + 12.4), dt=0.1) is False  # K лишився


def test_utm_scale_one_is_noop():
    """UTM-режим повертає 1.0 — гейт має поводитись як до Етапу 6."""
    det = _det(ground_scale=1.0)
    last = _seed(det)
    assert det.is_outlier((0.0, last[1] + 12.4), dt=0.1) is True


def test_characterization_reset_grants_two_free_passes():
    """ХАРАКТЕРИЗАЦІЯ відомого дефекту — НЕ виправлено, зафіксовано.

    Після max_consecutive відсіювань детектор робить window.clear() і приймає
    позицію. Але `if len(self.window) < 3: return False` означає, що наступні
    ДВА вимірювання проходять узагалі без перевірки (третє вже бачить повне
    вікно). Разом 3 неперевірені позиції на кожен RESET; у лозі 2026-07-31
    було 9 таких RESET → 27 позицій повз гейт.

    Якщо цей тест впаде — дефект полагоджено, оновіть його.
    """
    det = OutlierDetector(window_size=10, threshold_std=1e9, max_speed_mps=120.0, max_consecutive=3)
    _seed(det)
    huge = 1000.0  # свідомо абсурдний стрибок

    y = 10.0
    assert det.is_outlier((0.0, y + huge), dt=0.1) is True
    assert det.is_outlier((0.0, y + huge), dt=0.1) is True
    # третє поспіль → RESET, позиція приймається, вікно очищено
    assert det.is_outlier((0.0, y + huge), dt=0.1) is False

    # і тепер РІВНО два безкоштовні проходи (вікно < 3)
    for i in range(2):
        det.add_position((0.0, y + huge * (i + 1)), dt=0.1)
        assert det.is_outlier((0.0, 1e9), dt=0.1) is False, f"pass {i}: гейт не мав спрацювати"

    # третій уже перевіряється — вікно набрало 3 точки
    det.add_position((0.0, y + huge * 3), dt=0.1)
    assert det.is_outlier((0.0, 1e9), dt=0.1) is True
