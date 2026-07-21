# Статус оптимізації пайплайна — 2026-07-21

> Знімок стану до `docs/PIPELINE_OPTIMIZATION_PLAN_2026-07.md`.
> Ціль: машини класу **Shtepsill** (GTX 1650 4 GB). Гейти: worst keyframe < 1000 мс, трекінг < 33 мс, вміститись у 4 GB VRAM.

---

## Зроблено — код у репозиторії, задеплоєний і перевірений

- **A1 темпоральний prior** — `src/localization/localizer.py` (`_try_temporal_prior`, `_tp_neighbour_ids`, `_prepare_and_extract` + гілка в `localize_frame`), `src/localization/geometric_verifier.py` (винесений `mnn_counts`, `verify(..., ref_cache=)`)
- **B1 `of_stride` + B2 `of_half_res`** — `src/workers/tracking_worker.py`
- **A3** — `max_local_edge` доданий у pydantic (доти ключ у `user_config.json` мовчки ігнорувався)
- **Крок 0 бенча** — OF@200 у реальній конфігурації, half-res, YOLO, препроцес, нові агрегати `video_second_ms` / `realtime_factor`
- `user_config.example.json` синхронізовано, `docs/PIPELINE_OPTIMIZATION_PLAN_2026-07.md` написано
- **Перевірка:** ruff чистий (три зауваження в бенчі були вже в HEAD), 34 config-тести, 24 функціональні перевірки нових методів

Усе flag-gated: дефолти дорівнюють старій поведінці.

---

## Не зроблено — чекає на тебе

- прогнати import-перевірку, `pytest`, `benchmark_hardware.py` на Windows
- вмикати конфіг покроково з бенчем між кроками; **A2, A4, C2 коду не потребують** — реалізовані раніше, лише вимкнені
- A/B на реальній місії: збіг `best_candidate_id` з/без A1, інлаєри, RMS траєкторії для B2
- `git add` + коміт

```
python -c "import src.localization.localizer, src.localization.geometric_verifier, src.workers.tracking_worker"
python -m pytest tests -q
python scripts/benchmark_hardware.py
```

```
git add config/localization.py src/localization/localizer.py src/localization/geometric_verifier.py \
        src/workers/tracking_worker.py scripts/benchmark_hardware.py user_config.example.json \
        docs/PIPELINE_OPTIMIZATION_PLAN_2026-07.md docs/PIPELINE_OPTIMIZATION_STATUS_2026-07-21.md
```

---

## Валідація якості на Stozhar (2026-07-21, A/B по app.log)

Інструмент: `scripts/ab_localization.py` (Jaccard matched-frame + агрегати; парсить app.log). Місія: `TEST/newZap.mp4`, 186 keyframe-успіхів, 7 outlier-невдач в обох прогонах.

- **candidate_prefilter** (baseline off → on): Jaccard **1.000**, success/inliers ідентичні, зсув позиції median 0.00 м / max 1.22 м. Розбіжність «frame по парах 89%» — доброякісне переставляння порядку кандидатів (той самий набір), не втрата якості. → **quality-neutral, увімкнено.**
- **temporal_candidate_prior** (prefilter-on baseline → +A1): hit-rate **99%** (149/150), Jaccard **1.000**, frame по парах **100%**, зсув позиції median 0.00 / **max 0.13 м**. → **quality-neutral на цій місії, увімкнено.**
  - Застереження: 99% — best-case (добре покриття); на recovery hit-rate падає, A1 відкочується на повний шлях (safe by design). Виграш часу тут мізерний (DINOv3 17 мс на Stozhar) — реальна економія 470 мс лише на Shtepsill.

Ще не валідовано на Stozhar: `of_stride` / `of_half_res` (трекінг між keyframe — компаратор їх не бачить, потрібна траєкторна звірка), `max_local_edge` (якісний A/B), швидкість усього набору (лише на Shtepsill).

Спостереження, що коригує план: на реальній ріллі inliers median = **2048** (стеля ref-точок) — тобто query-точок значно більше за 2048, і важіль зрізання keypoints на Shtepsill **живий** (на синтетиці був no-op).

---

## Тести (2026-07-21, Stozhar — основна dev-машина)

Набір був **червоний ще до цієї сесії**: 4 падіння, усі під старі сигнатури або старий стек, не через мій код (`git log`: тести востаннє чіпані в `39bf749`/`8be2d8c`, до модульного рефактора).

Полагоджено (перевірено ruff + перерахунком):

- `tests/test_localization.py::test_compute_confidence` — стара сигнатура `(inliers, max_inliers, rmse, features)`. Переписано під `(candidate_id, inliers, total_matches, rmse_val)` з DB-QA через `frame_rmse`/`frame_disagreement`; очікувані значення перераховані незалежно (high ≈ 0.956 > 0.8; low ≈ 0.095 < 0.4).
- `tests/test_localization.py::test_localize_optical_flow` — бракувало `rot_width`/`rot_height`; кликало неіснуючий `localizer.converter`. Виправлено на `calibration.converter.metric_to_gps` + розміри кадру.
- `tests/benchmarks/test_benchmark_tracking.py` — `DummyDatabase` без `get_frame_size` (потрібен `result_builder.fallback`). Додано `get_frame_size` + `frame_rmse=None`/`frame_disagreement=None`.

Лишилось (потребує torch, лише на Windows):

- `tests/test_localization_characterization.py` — golden-снапшот. Різниться рівно один сценарій (`after_reset`, шлях scale≠1 з cv2-resize), решта збігається. Мій код на цьому шляху **побайтово ідентичний** старому (temporal prior дефолтом off), тож це дрейф стеку (cv2 4.13 / poselib 2.0.5 новіші за момент зняття базлайна), а не регресія. Перезняти базлайн на HEAD:

```
git stash
$env:CAPTURE=1; python -m pytest tests/test_localization_characterization.py -q; $env:CAPTURE=""
git stash pop
python -m pytest tests/test_localization_characterization.py -q   # має пройти з моїми змінами
```

Якщо крок 2 (з моїми змінами) впаде — це вже реальна регресія, скинь `-vv` diff.

---

## Не зроблено — код, свідомо відкладено

| Захід | Чому не зараз |
|---|---|
| A6 препроцес (CLAHE після даунскейлу, uint8 H2D, один ресайз на кадр) | −10–15 мс steady; після A1 бекбон і так рахується рідко |
| A5 ONNX/TRT для DINOv3 | 1–2 дні; пріоритет упав з тієї ж причини |
| C3 вивантаження Depth-Anything-V2 | важіль по VRAM, вартий діла лише якщо крок 1 дасть OOM |
| `depth_confidence` / `width_confidence` LightGlue | мертві ключі конфігу — ручка є в моделі, але в конструктор не передається |
| YOLO TRT-engine з `half=True` | на TU117 fp16 у 4× повільніший — треба перезібрати у fp32 і заміряти |
| `max_keypoints` 4096 → 2048 | заблоковано: на синтетиці кап не спрацював (1825 точок), потрібен реальний розподіл |
| D1 `input_size` 224 → 160, D2 XFeat | ребілд БД + різносезонна валідація; за планом — лише якщо кроки 1–5 не візьмуть гейти |

---

## Найближча дія

Один прогін бенча на Shtepsill **без жодного ввімкненого прапорця** — він дасть чесний baseline із YOLO та препроцесом, яких у старих 945 мс не було. Від нього залежить порядок усього решти.
