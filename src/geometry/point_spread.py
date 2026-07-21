"""Просторовий розкид відповідностей — обумовленість геометричної оцінки.

Мотивація (OrthoTrack §3.4, `docs/RESEARCH_ADDENDUM_2026-07.md` п.1): інлаєрів
може бути формально досить, але якщо всі вони скупчилися в одному куті кадру,
гомографія/афінна оцінка ill-conditioned — модель екстраполюється на решту
кадру, а саме центр кадру далі йде в координату. Кількість інлаєрів цього не
бачить; RANSAC теж — локально узгоджений кластер є валідним консенсусом.

Чисті функції (лише numpy) — тестуються в будь-якому середовищі і викликаються
з обох сторін: live (`ResultBuilder.compute_confidence`) і offline
(`PropagationPipeline._match_and_build_edge`).

ВАЖЛИВО про семантику: низький розкид — це НЕ автоматично помилка. На межі
покриття БД query-кадр легітимно перетинається з референсом лише кутом, і
скупчення там правильне. Тому метрика входить у пайплайн неперервно (множник
до confidence / до ваги ребра), а жорсткий гейт лишається тільки на екстремумі.

ВАЖЛИВО про None: ``None`` означає «сигнал недоступний» (мало точок, битий
кадр), а ``0.0`` — «розкид справді нульовий» (всі точки на одній прямій —
найгірший можливий випадок). Плутати їх не можна: перше не має штрафуватись,
друге має штрафуватись максимально.
"""

from __future__ import annotations

import numpy as np

# Розкид рівномірного розподілу по стороні L: σ = L/√12 ≈ 0.2887·L.
# Тобто здоровий надирний кадр з рівномірним покриттям дає spread ≈ 0.29.
UNIFORM_SPREAD = float(1.0 / np.sqrt(12.0))


def inlier_spread(points: np.ndarray, frame_w: float, frame_h: float) -> float | None:
    """min(σx, σy) / min(W, H) — безрозмірний просторовий розкид точок.

    Нормування на ``min(W, H)`` робить метрику інваріантною і до роздільності,
    і до співвідношення сторін: для рівномірного покриття σx = 0.2887·W,
    σy = 0.2887·H, тож min(σx, σy) = 0.2887·min(W, H) → spread ≈ 0.289 за
    будь-якого аспекту.

    Береться саме ``min`` двох осей, а не площа хмари: типовий режим відмови —
    точки вздовж однієї борозни/лінії, де один із σ великий, а другий ≈ 0.
    Добуток σx·σy теж це ловить, але ``min`` дає лінійну шкалу в тих самих
    одиницях, що й сторона кадру, і легше калібрується порогами.

    Args:
        points: (N, 2) координати у пікселях кадру (query-сторона).
        frame_w: ширина кадру в тих самих пікселях, що й ``points``.
        frame_h: висота кадру.

    Returns:
        Розкид у [0, ~0.5], або ``None``, якщо метрику неможливо порахувати
        (< 2 валідних точок, нульовий/нечисловий розмір кадру). Нуль — це
        валідне значення «повністю вироджена хмара», а не відсутність сигналу.
    """
    if points is None:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None

    denom = min(float(frame_w), float(frame_h))
    if not np.isfinite(denom) or denom <= 0.0:
        return None

    xy = pts[:, :2]
    if not np.all(np.isfinite(xy)):
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if xy.shape[0] < 2:
            return None

    sigma = xy.std(axis=0)
    return float(min(sigma[0], sigma[1]) / denom)


def spread_confidence_factor(
    spread: float | None, spread_ref: float = 0.15, floor: float = 0.35
) -> float:
    """Множник до confidence живої локалізації: clip(spread/ref, floor, 1.0).

    ``spread_ref`` = 0.15 — приблизно половина рівномірного покриття (0.289):
    вище нього штрафу нема взагалі. ``floor`` не дає confidence обнулитись —
    скупчений фікс лишається вимірюванням із більшим R у Калмані, а не
    відкинутим кадром (це і є різниця з жорстким порогом OrthoTrack).

    ``None`` (сигнал недоступний) → 1.0, без штрафу. ``0.0`` (вироджена
    хмара) → ``floor``, тобто максимальний штраф.
    """
    if spread is None or not np.isfinite(spread):
        return 1.0
    ref = max(float(spread_ref), 1e-6)
    return float(np.clip(max(0.0, float(spread)) / ref, float(floor), 1.0))


def spread_weight_factor(spread: float | None, spread_ref: float = 0.15, k: float = 10.0) -> float:
    """Множник до ваги ребра графа: 1 / (1 + k·max(0, ref − spread)).

    Та сама форма, що вже вживається для якості афінного фіту
    (``temporal_weight_use_fit_quality``), тож ваги лишаються в одній шкалі.
    Дефіцит розкиду обмежений зверху величиною ``ref`` (0.15), тому k має бути
    порядку 10, щоб штраф був відчутним: spread 0.05 → ×0.50, spread 0 → ×0.40.

    ``None`` (сигнал недоступний) → 1.0, без штрафу.
    """
    if spread is None or not np.isfinite(spread):
        return 1.0
    deficit = max(0.0, float(spread_ref) - max(0.0, float(spread)))
    return float(1.0 / (1.0 + float(k) * deficit))
