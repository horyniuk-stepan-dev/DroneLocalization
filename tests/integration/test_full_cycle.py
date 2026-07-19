"""Integration flow test — повний цикл пайплайну (IMPROVEMENT_PLAN п.4.5,
REMAINING_WORK_PLAN Трек 1).

Наскрізний acceptance для ланцюга: відео → build → 2 якорі → пропагація →
локалізація (поза кожного keyframe, яку дає граф) → медіанна похибка центру
кадру проти GT < GSD-порогу. Ловить розриви на швах build↔propagation↔pose:
якщо будь-яка ланка ламається, пропаговані пози розповзаються > порогу.

ЩО ДЕ ЗАПУСКАЄТЬСЯ
- Чисті хелпери (`gsd_from_affines`, `gsd_threshold_m`, `select_two_anchors`) і
  сама логіка гейта (compute_metrics + поріг на синтетичному GT) — тестуються
  БУДЬ-ДЕ, зокрема в Linux-пісочниці/CI. Це «прохід у CI» з гейта плану.
- Наскрізний `test_full_cycle_two_anchor_propagation` — ЛИШЕ на Windows/GPU з
  датасетом і torch; інакше skip. Рушій пропагації — `scripts/benchmark_
  propagation.py` (перевикористання, щоб не дублювати API DatabaseBuilder/
  CalibrationPropagationWorker).

ЗАПУСК (Windows/GPU):
    set DRONELOC_E2E_DATASET=D:\\шлях\\до\\датасету
    python -m pytest tests/integration/test_full_cycle.py -q -s
Опційно: DRONELOC_E2E_TOL_M — абсолютний поріг у метрах замість GSD-похідного.

ПЕРЕДУМОВА (build). Збудова БД потребує GPU/моделей і робиться ОКРЕМО (як і в
`scripts/benchmark_propagation.py`): у датасеті має вже лежати `benchmark.h5`,
збудований білдером проєкту з того самого відео. Без нього — skip з інструкцією.

СХЕМА `ground_truth.json` — та сама, що в `benchmark_propagation.py`
(`load_ground_truth`): верхньорівневі `frame_width`/`frame_height`, `anchors`:
[frame_id...], `frames`: [{`frame_id`, `affine` — 2×3 px→metric}]. Це НЕ
slots/center_mercator-схема з `test_scale_invariance.py`.

РОЗПОДІЛ ПРАЦІ. Локалізацію ЖИВИХ query-кадрів через `localize_frame` на готовій
БД уже покриває `test_scale_invariance.py`. Цей файл покриває інший шов —
build→propagation і точність пропагованих поз від мінімальної (2 якорі)
калібровки. Разом вони закривають увесь ланцюг без дублювання.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

import scripts.benchmark_propagation as B

# Синтетичний GT для чистих тестів гейта (px→metric, як у test_benchmark_metrics)
_W, _H = 1920, 1080
_CX, _CY = _W / 2.0, _H / 2.0


# ── Чисті хелпери (тестуються без GPU) ───────────────────────────────────────


def gsd_from_affines(affines) -> float:
    """Медіанний GSD (метри/піксель) = медіана √|det| афінних px→metric матриць.

    √|det| 2×2-блоку — ізотропний масштаб-проксі (той самий, що `_iso_scale` у
    benchmark_propagation), тобто метри на піксель. Береться з GT — окремого
    входу не треба.
    """
    scales = [
        math.sqrt(abs(np.linalg.det(np.asarray(a, dtype=np.float64)[:2, :2]))) for a in affines
    ]
    if not scales:
        raise ValueError("порожній набір афінних матриць — GSD не визначений")
    return float(np.median(scales))


def gsd_threshold_m(
    gsd_m_per_px: float,
    frame_w_px: int,
    frac: float = 0.05,
    floor_m: float = 3.0,
) -> float:
    """Поріг медіанної похибки в метрах, похідний від GSD.

    `frac` частка ширини футпринта (frame_w_px · GSD) — «потік працює» гейт, не
    прецизійний; floor_m — підлога проти іграшково малих сцен. Це не гейт
    точності методу (той — mission benchmark), а страховка, що ланцюг не
    розвалюється (пози не розповзаються на сотні метрів).
    """
    return max(frac * gsd_m_per_px * frame_w_px, floor_m)


def select_two_anchors(frame_ids) -> list[int]:
    """Перший і останній із відсортованих унікальних frame_id — мінімальна
    2-якірна калібровка, що охоплює рейс."""
    uniq = sorted({int(f) for f in frame_ids})
    if len(uniq) < 2:
        raise ValueError(f"потрібно ≥2 кадри для 2 якорів, є {len(uniq)}")
    return [uniq[0], uniq[-1]]


def _affine(
    cx_m: float, cy_m: float, s: float = 0.1, sign: float = -1.0, angle: float = 0.0
) -> np.ndarray:
    """Синтетична px→metric афінна: центр кадру лягає в (cx_m, cy_m), GSD = s."""
    from src.geometry.affine_utils import compose_affine_5dof

    M = compose_affine_5dof(0.0, 0.0, s, s, angle, sign=sign)
    M[0, 2] = cx_m - (M[0, 0] * _CX + M[0, 1] * _CY)
    M[1, 2] = cy_m - (M[1, 0] * _CX + M[1, 1] * _CY)
    return M


def test_gsd_from_affines_recovers_scale():
    gt = [_affine(0.0, 0.0, s=0.1), _affine(500.0, 0.0, s=0.1)]
    assert gsd_from_affines(gt) == pytest.approx(0.1, abs=1e-9)


def test_gsd_from_affines_empty_raises():
    with pytest.raises(ValueError):
        gsd_from_affines([])


def test_gsd_threshold_fraction_and_floor():
    assert gsd_threshold_m(0.1, 1920) == pytest.approx(0.05 * 0.1 * 1920)  # 9.6 м
    assert gsd_threshold_m(0.1, 1920, frac=0.1) == pytest.approx(19.2)
    assert gsd_threshold_m(1e-4, 100) == 3.0  # підлога


def test_select_two_anchors_spans_range():
    assert select_two_anchors([5, 1, 3, 9, 3]) == [1, 9]


def test_select_two_anchors_needs_two():
    with pytest.raises(ValueError):
        select_two_anchors([7])


def test_gate_logic_on_synthetic_gt():
    """Сама acceptance-логіка (compute_metrics + GSD-поріг) без GPU: хороший
    прогін проходить гейт, зламаний (пози на 50 м) — валить його."""
    gt = {i: _affine(i * 50.0, 0.0, s=0.1) for i in range(6)}
    tol = gsd_threshold_m(gsd_from_affines(gt.values()), _W)  # 9.6 м

    good = {i: gt[i].copy() for i in gt}
    good[2][0, 2] += 4.0  # один кадр на 4 м < порогу
    m = B.compute_metrics(good, gt, _W, _H)
    assert m["median_err_m"] <= tol
    assert m["det_sign_ok"] == 1.0

    bad = {i: gt[i].copy() for i in gt}
    for i in bad:
        bad[i][0, 2] += 50.0  # усі на 50 м >> порогу
    assert B.compute_metrics(bad, gt, _W, _H)["median_err_m"] > tol


# ── Наскрізний прогін (Windows/GPU) ──────────────────────────────────────────


def _env() -> Path:
    ds = os.environ.get("DRONELOC_E2E_DATASET")
    if not ds:
        pytest.skip(
            "set DRONELOC_E2E_DATASET=<тека з flight .mp4 + ground_truth.json + "
            "benchmark.h5> (Windows/GPU e2e)"
        )
    d = Path(ds)
    if not any(d.glob("*.mp4")):
        pytest.skip(f"немає *.mp4 у {d}")
    if not (d / "ground_truth.json").is_file():
        pytest.skip(f"немає ground_truth.json у {d}")
    if not (d / "benchmark.h5").is_file():
        pytest.skip(
            f"немає benchmark.h5 у {d} — спершу збудуй БД із відео білдером "
            "проєкту (окремий GPU-крок, як для scripts/benchmark_propagation.py)"
        )
    pytest.importorskip("torch")
    return d


def test_full_cycle_two_anchor_propagation():
    """build(передумова) → 2 якорі → пропагація → медіанна похибка < GSD-порогу."""
    dataset = _env()

    gt = B.load_ground_truth(dataset / "ground_truth.json")
    anchors = select_two_anchors(gt["anchors"] or sorted(gt["gt_affines"]))
    gt_two = {**gt, "anchors": anchors}

    # реальний ланцюг: DatabaseLoader(benchmark.h5) → add_anchor×2 →
    # CalibrationPropagationWorker._propagate() → пропаговані афінні пози
    pred = B.run_pipeline(dataset, gt_two)

    m = B.compute_metrics(pred, gt["gt_affines"], gt["frame_width"], gt["frame_height"])
    gsd = gsd_from_affines(gt["gt_affines"].values())
    tol = float(os.environ.get("DRONELOC_E2E_TOL_M", gsd_threshold_m(gsd, gt["frame_width"])))

    assert m["coverage"] > 0.5, f"пропагація покрила лише {m['coverage']:.0%} GT-кадрів"
    assert m["det_sign_ok"] == 1.0, "переворот дзеркала/орієнтації в циклі (знак det)"
    assert m["median_err_m"] <= tol, (
        f"наскрізна медіанна похибка {m['median_err_m']:.1f} м > GSD-поріг {tol:.1f} м "
        f"(gsd {gsd:.3f} м/px, 2 якорі {anchors})"
    )
