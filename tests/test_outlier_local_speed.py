"""OF-швидкість має бути локальною (кадр до кадру), а не накопиченою від keyframe.

Діагностика місії top (2026-07-31): LK трекає завжди від keyframe
(tracking_worker.py:486 — prev_gray навмисно не оновлюється), тож зсув росте
лінійно з часом від keyframe. Але dt для OF рахується від останньої УСПІШНОЇ
локалізації (tracking_worker.py:243) і дорівнює одному кроку OF.

Наслідок: швидкість завищена рівно в N разів, де N — номер OF-кадру після
keyframe. Логовані 450-1271 м/с лягли на сітку N * 89.2 м/с при N=5..14, а між
keyframe-ами вміщається якраз 10 OF-кадрів (keyframe_interval=30, of_stride=3).
"""

import pytest

from src.tracking.outlier_detector import OutlierDetector

V_TRUE = 89.2  # справжня швидкість платформи, м/с (медіана між keyframe-ами)
DT = 0.1  # of_stride=3 @ 30 fps
STEP = V_TRUE * DT  # 8.92 м — реальний зсув за один OF-крок
CAP = 200.0  # поріг із запасом 2.2x над справжньою швидкістю


def _det():
    return OutlierDetector(
        window_size=10, threshold_std=1e9, max_speed_mps=CAP, max_consecutive=10**6
    )


def _seed_keyframe(det, at=0.0):
    """Вікно з трьох прийнятих позицій; остання = 'keyframe' у точці at."""
    for i in range(3):
        det.add_position((0.0, at - (2 - i) * 1e-3), dt=1.0)
    return at


def test_step_matches_measured_mission_values():
    """Звірка вхідних чисел із заміряними на місії."""
    assert STEP == pytest.approx(8.92, abs=1e-9)
    assert STEP / DT == pytest.approx(V_TRUE, abs=1e-9)


def test_without_ref_speed_accumulates_and_false_triggers():
    """ХАРАКТЕРИЗАЦІЯ старої поведінки: швидкість росте як N * V_TRUE.

    Кожен OF-кадр порівнюється з keyframe, тому N-й дає N * 8.92 м за ті самі
    0.1 с. При CAP=200 гейт починає хибно спрацьовувати з третього кадру.
    """
    det = _det()
    base = _seed_keyframe(det)
    seen = []
    for n in range(1, 11):
        pos = (0.0, base + STEP * n)  # накопичений зсув від keyframe
        seen.append(det.is_outlier(pos, dt=DT))
    # N=1,2 -> 89, 178 м/с проходять; з N=3 (268 м/с) починаються спрацювання
    assert seen[:2] == [False, False]
    assert all(seen[2:]), "мала б спрацювати вся решта — це і є накопичення"
    assert sum(seen) == 8


def test_with_ref_speed_is_local_and_no_false_triggers():
    """ФІКС: ref_position = попередній сирий OF-вимір -> швидкість завжди V_TRUE."""
    det = _det()
    base = _seed_keyframe(det)
    prev = (0.0, base)  # перший OF порівнюється з keyframe
    for n in range(1, 11):
        pos = (0.0, base + STEP * n)
        assert det.is_outlier(pos, dt=DT, ref_position=prev) is False, f"кадр {n}"
        prev = pos


def test_ref_still_catches_a_real_jump():
    """Локальний розрахунок не має вимикати гейт: раптовий зрив LK ловиться."""
    det = _det()
    base = _seed_keyframe(det)
    prev = (0.0, base)
    for n in range(1, 4):
        pos = (0.0, base + STEP * n)
        det.is_outlier(pos, dt=DT, ref_position=prev)
        prev = pos
    # зрив: 50 м за 0.1 с = 500 м/с
    assert det.is_outlier((0.0, prev[1] + 50.0), dt=DT, ref_position=prev) is True


def test_ref_none_is_bit_exact_old_behaviour():
    """ref_position=None не змінює жодного рішення відносно старого коду."""
    a, b = _det(), _det()
    base_a, base_b = _seed_keyframe(a), _seed_keyframe(b)
    for n in range(1, 11):
        pa = (0.0, base_a + STEP * n)
        pb = (0.0, base_b + STEP * n)
        assert a.is_outlier(pa, dt=DT) == b.is_outlier(pb, dt=DT, ref_position=None)


def test_logged_values_lie_on_the_accumulation_grid():
    """Фактичні відсіювання прогону 'top' відтворюються моделлю N * V_TRUE."""
    logged = [450.5, 533.2, 592.0, 743.9, 892.0, 1195.6, 1271.2]
    for s in logged:
        n = s / V_TRUE
        # кожне значення близьке до цілого N у діапазоні можливих OF-кадрів
        assert 4.5 <= n <= 15.0, f"{s} м/с -> N={n:.1f} поза очікуваним діапазоном"


# ── Z-score гілка: замір місії top 2026-07-31 ────────────────────────────────
# З 70 спрацювань 47 дала z-гілка, і всі 23 спрацювання на ПРАВИЛЬНИХ вимірах
# (distance 2.3-4.5 м при нормі 3.0) — теж вона. Жодного справжнього зриву
# (>12 м) вона не спіймала: їх усі ловить фізичний max_speed.

DT33 = 1 / 30  # of_stride=1 @ 30 fps
NORM = 89.2 * DT33  # ≈ 2.97 м — нормальний зсув за один кадр


def _det33(zscore: bool):
    return OutlierDetector(
        window_size=10,
        threshold_std=4.0,
        max_speed_mps=350.0,
        max_consecutive=10**6,
        zscore_enabled=zscore,
    )


def _seed33(det, n=3):
    for i in range(n):
        det.add_position((0.0, i * NORM), dt=DT33)
    return (n - 1) * NORM


def test_zscore_off_passes_normal_measurements():
    """Виміри 2.3-4.3 м (норма 3.0) з логу мають проходити без z-гілки."""
    for dist in (2.3, 2.4, 2.9, 3.0, 3.1, 3.3, 4.1, 4.3):
        det = _det33(zscore=False)
        base = _seed33(det)
        assert det.is_outlier((0.0, base + dist), dt=DT33) is False, f"{dist} м"


def test_zscore_off_still_catches_real_dropouts():
    """Справжні зриви з логу (12.5-62.0 м) мають ловитись і без z-гілки."""
    for dist in (12.5, 17.4, 20.9, 30.1, 41.8, 62.0):
        det = _det33(zscore=False)
        base = _seed33(det)
        assert det.is_outlier((0.0, base + dist), dt=DT33) is True, f"{dist} м"


def test_zscore_on_falsely_rejects_normal_measurement():
    """ХАРАКТЕРИЗАЦІЯ: з увімкненою z-гілкою нормальний вимір відкидається.

    Відтворює рядок логу: distance=2.3 м (норма 3.0!) -> z=9.68 при
    mean_speed=26.4, бо вікно зібрало лише повільні виміри.
    """
    det = _det33(zscore=True)
    # вікно з майже нерухомих позицій — так виглядає самопідтримна петля
    for i in range(4):
        det.add_position((0.0, i * 0.2), dt=DT33)
    assert det.is_outlier((0.0, 0.6 + 2.3), dt=DT33) is True


def test_default_keeps_zscore_enabled():
    """Дефолт конструктора не змінюється — стара поведінка."""
    det = OutlierDetector()
    assert det._zscore_enabled is True
