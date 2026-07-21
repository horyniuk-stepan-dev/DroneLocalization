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
