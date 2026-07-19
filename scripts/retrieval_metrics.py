"""Retrieval-quality метрики для UAV absolute visual localization (AVL).

Призначення (`RESEARCH_INTEGRATION_PLAN.md` §3.2, `REMAINING_WORK_PLAN.md`
Трек 6): дешевий офлайн-гейт, що зв'язує якість *retrieval* з *фінальною*
похибкою локалізації і рахується з уже логованих величин — ранжований список
кандидатів + істинна позиція запиту + розмір реферного футпринта. Використання:
A/B будь-якої зміни retrieval (VLAD vs CLS, GIM-ваги, навчена агрегаційна
голова, ...) — порахувати метрики на ТОМУ Ж реферному відео для кожного
варіанта і порівняти.

ВАЖЛИВО: абсолютні числа порівнянні лише між прогонами з ОДНАКОВИМИ параметрами
(alpha, lam, l, s, K). Гейт живе на відносному порівнянні, не на збігу з
абсолютними цифрами AnyVisLoc (інший датасет).

Визначення метрик — за бенчмарком AnyVisLoc (Ye et al., "Exploring the best way
for UAV visual localization under Low-altitude Multi-view Observation Condition:
a Benchmark", arXiv:2503.10692, CVPR 2026 Findings; код:
github.com/UAV-AVL/Benchmark), §5.2 "Image Retrieval Metric".

Модель даних. На один запит (один кадр БпЛА) retriever повертає Top-K реферних
кандидатів у порядку similarity-рангу. Для кандидата i (i = 1..K, 1 = найкращий
ранг):
    d_i  — відстань по землі (метри) від істинної позиції запиту до центру
           реферного тайла кандидата i;
    W_i  — ширина реферного футпринта кандидата i в метрах (= w_i_px · GSD, де
           GSD — метри-на-піксель реферної карти). Це член `w_i · r` статті.

Три метрики, усі усереднюються по запитах (для всіх трьох: більше = краще):

Recall@K
    1, якщо хоч один із Top-K кандидатів у межах `recall_thresh_m` метрів від
    істини, інакше 0. Бінарна; ігнорує, на якому саме ранзі стався хіт.

SDM@K  — Spatial Distance Metric (AnyVisLoc Eq. 2, з DenseUAV [Zhu et al.])
    ранг-зважене середнє загасання сирої відстані:
        SDM@K = Σ_{i=1..K} (K-i+1) · exp(-s · d_i)  /  Σ_{i=1..K} (K-i+1)
    `s` (1/метр) задає масштаб відстані; він чутливий до GSD карти — саме та
    вада, яку виправляє PDM@K.

PDM@K  — Proportional Distance Metric (AnyVisLoc §5.2, рекомендована метрика)
    те саме ранг-зважування, але загасання сирої відстані замінене на скор-
    функцію f(R_i) від *нормалізованої* відстані
        R_i = d_i / W_i                          (стаття: R_i = d_i / (w_i · r))
    отже скор інваріантний до розміру зображення й GSD карти:
        PDM@K = Σ_{i=1..K} (K-i+1) · f(R_i)  /  Σ_{i=1..K} (K-i+1)
    R_i — зсув реферного футпринта відносно істини як частка ширини футпринта,
    тобто проксі перекриття пари зображень: малий R_i → велике перекриття →
    легкий матчинг → точна локалізація; щойно R_i перевищує нормалізовану
    діагональ `l`, футпринти не перекриваються і скор = 0. Стаття наводить
    R_i≈0.27 → похибка 1.4 м, 0.35 → 2.1 м, 1.37 → провал (649 м).

РЕКОНСТРУЙОВАНО (регістр c — закриту форму f(R_i) прочитати не вдалося: у статті
вона рендериться як рисунок, не текст). f(R_i) реалізовано як логістику, що
однозначно лягає на словесну специфікацію статті ("alpha задає поріг R_i, на
якому скор падає; lambda керує різкістю загасання", f: монотонно 1→0):
        f(R) = 1 / (1 + exp(lambda · (R - alpha)))          , 0 для R ≥ l.
Звірено з трьома прикладами статті вище (див. tests/test_retrieval_metrics.py).
Рекомендовані параметри статті (її зйомка 4:3): l = 1.67, alpha трохи вище l/2
(≈ 0.85), lambda ∈ [4, 8] — це дефолти тут. УВАГА: l і alpha залежать від
співвідношення сторін; для кадрів 16:9 (1920×1080) цього проєкту перерахуйте l
і задайте alpha ≈ l/2, перш ніж довіряти абсолютним значенням PDM@K. Для гейта
важливо лиш одне: обидва варіанти A/B рахуються з ІДЕНТИЧНИМИ (alpha, lam, l, s,
K).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

# ── Рекомендовані дефолти AnyVisLoc §5.2 (для її зйомки 4:3) ──────────────────
PDM_ALPHA = 0.85  # поріг R_i, на якому f(R) падає (стаття: трохи вище l/2)
PDM_LAMBDA = 6.0  # різкість загасання (діапазон статті 4–8)
PDM_L = 1.67  # нормалізована діагональ кадру; f(R)=0 для R ≥ l (стаття: 4:3)
RECALL_THRESH_M = 5.0  # радіус хіта Recall@K, м (узгоджено з родиною A@5m проєкту)
SDM_S = 1.0 / RECALL_THRESH_M  # масштаб відстані SDM@K, 1/м; ЗАЛЕЖИТЬ ВІД GSD карти

_EXP_CLIP = 500.0  # межа аргументу exp, щоб уникнути overflow-попереджень


def r_i(d_m: float | np.ndarray, footprint_width_m: float | np.ndarray) -> np.ndarray:
    """R_i = d_i / W_i — позиційна похибка, нормалізована шириною футпринта.

    Безрозмірна. `footprint_width_m` = w_i_px · GSD (ширина реферного футпринта
    в метрах). Обидва аргументи скаляр або масив (broadcast).
    """
    d = np.asarray(d_m, dtype=np.float64)
    w = np.asarray(footprint_width_m, dtype=np.float64)
    if np.any(w <= 0.0):
        raise ValueError("footprint_width_m має бути додатним (метри на футпринт)")
    return d / w


def pdm_score(
    r: float | np.ndarray,
    alpha: float = PDM_ALPHA,
    lam: float = PDM_LAMBDA,
    l_diag: float | None = PDM_L,
) -> np.ndarray:
    """f(R): логістичний скор перекриття, 1 (повне перекриття) → 0 (немає).

    РЕКОНСТРУЙОВАНА закрита форма (див. docstring модуля). За наявності `l_diag`
    жорстко зануляється для R ≥ l_diag (футпринти не перетинаються). f(0) ≈ 0.994,
    а не рівно 1 — логістика не досягає 1.
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.clip(lam * (r - alpha), -_EXP_CLIP, _EXP_CLIP)
    score = 1.0 / (1.0 + np.exp(z))
    if l_diag is not None:
        score = np.where(r >= l_diag, 0.0, score)
    return score


def _rank_weights(k: int) -> np.ndarray:
    """Ваги рангу (K - i + 1) для i=1..K → [K, K-1, ..., 1]."""
    return np.arange(k, 0, -1, dtype=np.float64)


def _topk(d: np.ndarray, k: int | None) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).ravel()
    return d if k is None else d[:k]


def recall_at_k(
    distances_m: Sequence[float] | np.ndarray,
    k: int | None = None,
    thresh_m: float = RECALL_THRESH_M,
) -> float:
    """1.0, якщо хоч один із перших K кандидатів у межах `thresh_m`, інакше 0.0."""
    d = _topk(distances_m, k)
    if d.size == 0:
        return 0.0
    return float(np.any(d <= thresh_m))


def sdm_at_k(
    distances_m: Sequence[float] | np.ndarray,
    k: int | None = None,
    s: float = SDM_S,
) -> float:
    """SDM@K (AnyVisLoc Eq. 2). `distances_m` — у порядку рангу retrieval."""
    d = _topk(distances_m, k)
    if d.size == 0:
        return 0.0
    w = _rank_weights(d.size)
    z = np.clip(-s * d, -_EXP_CLIP, _EXP_CLIP)
    return float(np.sum(w * np.exp(z)) / np.sum(w))


def pdm_at_k(
    distances_m: Sequence[float] | np.ndarray,
    footprint_width_m: float | Sequence[float] | np.ndarray,
    k: int | None = None,
    alpha: float = PDM_ALPHA,
    lam: float = PDM_LAMBDA,
    l_diag: float | None = PDM_L,
) -> float:
    """PDM@K (AnyVisLoc §5.2). `distances_m` — у порядку рангу retrieval;
    `footprint_width_m` — скаляр (одна GSD/розмір) або пер-кандидатний масив."""
    d = np.asarray(distances_m, dtype=np.float64).ravel()
    w = np.broadcast_to(np.asarray(footprint_width_m, dtype=np.float64), d.shape)
    if k is not None:
        d = d[:k]
        w = w[:k]
    if d.size == 0:
        return 0.0
    r = r_i(d, w)
    f = pdm_score(r, alpha=alpha, lam=lam, l_diag=l_diag)
    wt = _rank_weights(d.size)
    return float(np.sum(wt * f) / np.sum(wt))


def query_from_positions(
    gt_xy: Sequence[float] | np.ndarray,
    candidate_xy: Sequence[Sequence[float]] | np.ndarray,
    footprint_width_m: float | Sequence[float] | np.ndarray,
) -> dict:
    """Побудувати запис запиту з логованих позицій.

    `gt_xy` — істинна позиція запиту (x, y) в метрах (напр. UTM з
    ground_truth.json); `candidate_xy` — центри реферних тайлів Top-K кандидатів
    (K, 2) У ПОРЯДКУ РАНГУ retrieval; `footprint_width_m` — ширина футпринта
    (метри), скаляр або пер-кандидат. d_i = евклідова відстань GT ↔ центр.
    """
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(2)
    cand = np.asarray(candidate_xy, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(cand - gt[None, :], axis=1)
    return {"distances_m": d, "footprint_width_m": footprint_width_m}


def compute_retrieval_metrics(
    queries: Iterable[Mapping],
    k: int,
    *,
    alpha: float = PDM_ALPHA,
    lam: float = PDM_LAMBDA,
    l_diag: float | None = PDM_L,
    s: float = SDM_S,
    recall_thresh_m: float = RECALL_THRESH_M,
) -> dict:
    """Усереднити Recall@K, SDM@K, PDM@K по багатьох запитах.

    `queries` — ітеровані пер-запитні маппінги, кожен з ключами:
        "distances_m"      — ранжовані Top-K GT-відстані (метри);
        "footprint_width_m" — скаляр або пер-кандидатні ширини (метри).
    Повертає dict: {"recall@k", "sdm@k", "pdm@k", "n_queries", "k"}. Порожній
    вхід → нулі й n_queries=0.
    """
    recalls: list[float] = []
    sdms: list[float] = []
    pdms: list[float] = []
    for q in queries:
        d = q["distances_m"]
        w = q["footprint_width_m"]
        recalls.append(recall_at_k(d, k=k, thresh_m=recall_thresh_m))
        sdms.append(sdm_at_k(d, k=k, s=s))
        pdms.append(pdm_at_k(d, w, k=k, alpha=alpha, lam=lam, l_diag=l_diag))
    n = len(recalls)
    return {
        "recall@k": float(np.mean(recalls)) if n else 0.0,
        "sdm@k": float(np.mean(sdms)) if n else 0.0,
        "pdm@k": float(np.mean(pdms)) if n else 0.0,
        "n_queries": n,
        "k": k,
    }
