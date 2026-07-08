# Патч-план: розбиття `localizer.py` (IMPROVEMENT_PLAN п.1.1)

Мета — розбити `Localizer` (935 рядків) на пакет фокусованих компонентів **без зміни
поведінки** (bit-identical результати `localize_frame` / `localize_optical_flow`).
`localize_frame` (~450 рядків) стає ~80 рядків послідовних викликів.

> ⚠️ Найвищий ризик у §1 плану. Тому: **спершу характеризаційні тести**, потім
> рефакторинг як чистий move — жодної зміни формул чи порядку side-effect'ів.
> Прогін — на машині з torch/faiss (у пісочниці `localizer` не імпортується:
> `matcher.py` тягне `torch`+`faiss` на рівні модуля).

---

## 1. Карта поточного коду

`localize_frame` (рядки 222–672) — 9 фаз:

| Фаза | Рядки | Що робить | Мутує стан |
|---|---|---|---|
| 1. Out-of-coverage guard | 226–241 | скидання лічильника, `out_of_coverage` | `_consecutive_failures`, `_last_best_angle` |
| 2. Нормалізація | 243–249 | resolution normalize | `_last_scale` |
| 3. Вибір ротації (A3 prior + A2 батч-скан) | 251–341 | 1 або 4 DINOv2-forward, retrieval | читає `_last_best_angle`, `_consecutive_failures` |
| 4. Перемикання multi-source | 343–349 | вибір активної БД/калібр. | `self.database`, `self.calibration`, `_active_source_id` |
| 5. Патчифай + local features | 351–390 | rotate frame, patchify-merge, ALIKED/RDD | — |
| 6. Цикл кандидатів (match+RANSAC+early-stop) | 392–448 | LightGlue match, гомографія, RMSE | — |
| 7. Fallback-и (мало inliers / нема affine) | 450–499 | retrieval-only fallback | `_consecutive_failures` |
| 8. Стан→метрика→GPS→outlier→confidence→Kalman | 501–574 | координатні перетворення, фільтри | `_last_state`, `_consecutive_failures`, `trajectory_filter`, `outlier_detector` |
| 9. FOV з exploded-guard + return | 576–672 | розрахунок FOV, GPS-кути | `_last_best_angle` |

Інші методи: `localize_optical_flow` (676–806), `_merge_candidates` (810–846),
`_compute_confidence` (848–882), `_localize_by_reference_frame` (884–921),
`_log_failure` (923–935). Модульний рівень: `FAILURE_TYPES` (17–24),
`_ROTATION_VEC` (38–44), `_rotate_point_np90` (47–59).

---

## 2. Критичний ризик — спільний мутабельний стан

`localize_optical_flow` і `TrackingWorker` (через `@property last_state`) читають
session-стан, встановлений `localize_frame`. НЕ МОЖНА розривати цей контракт.

Session-стан, що ЗАЛИШАЄТЬСЯ у `Localizer`:
`_last_state`, `_last_scale`, `_last_best_angle`, `_consecutive_failures`,
`_active_source_id`, `trajectory_filter`, `outlier_detector`, і — у multi-режимі —
`self.database` / `self.calibration` (мутуються у фазі 4, відновлюються в OF 705–709).

**Золоте правило:** усі витягнуті компоненти — **stateless**. Приймають явні входи
+ context, повертають dataclass-результати. Оркестратор (`Localizer.localize_frame`)
володіє УСІМ мутабельним станом і оновлює його МІЖ викликами, зберігаючи точний
порядок side-effect'ів (напр. `_consecutive_failures += 1` саме там, де зараз;
`_consecutive_failures = 0` саме у рядку 559, ДО confidence).

---

## 3. Цільова структура пакета `src/localization/`

| Новий модуль | Переносимо (рядки) | Публічний контракт |
|---|---|---|
| `failure_log.py` | `FAILURE_TYPES`, `_log_failure` (17–24, 923–935) | `FailureLogger.log(error_type, inliers=0, details="")` |
| `rotation_geometry.py` | `_ROTATION_VEC`, `_rotate_point_np90` (38–59) | чисті функції (вже stateless) |
| `rotation_selector.py` | фази 2–3 (243–341) | `RotationSelector.select(frame, mask, prior_angle, allow_prior) -> RotationResult \| None` |
| `candidate_retriever.py` | `_retrieve_candidates` (194–200), патчифай-розширення (358–385), `_merge_candidates` (810–846) | `CandidateRetriever.retrieve(desc, top_k)`, `.expand(rotated_frame, base_candidates, top_k)` |
| `geometric_verifier.py` | фаза 6 (392–448) | `GeometricVerifier.verify(query_features, candidates) -> VerificationResult \| None` |
| `result_builder.py` | фази 8–9 (501–574, 576–672), `_compute_confidence` (848–882), `_localize_by_reference_frame` (884–921) | `ResultBuilder.build(ver, ctx, tracking) -> dict`, `.fallback(frame_id, score) -> dict \| None` |
| `localizer.py` | `__init__`, `last_state`, `reset_session`, `localize_frame` (оркестрація ~80 р.), `localize_optical_flow` | без змін контракту |

`localize_optical_flow` ЛИШАЄТЬСЯ у `Localizer` — щільно зав'язаний на `_last_state`,
`_last_scale`, `trajectory_filter`. Використовує `rotation_geometry` (спільні функції).

---

## 4. Контракти між компонентами (dataclasses)

```python
# rotation_selector.py
@dataclass
class RotationResult:
    angle: int
    score: float
    candidates: list[tuple[int, float]]
    source_id: str | None
    rotated_frame: np.ndarray
    rotated_mask: np.ndarray | None
    rot_width: int          # розміри повернутого нормалізованого кадру
    rot_height: int

# geometric_verifier.py
@dataclass
class VerificationResult:
    candidate_id: int
    H_query_to_ref: np.ndarray
    inliers: int
    mkpts_q_in: np.ndarray
    mkpts_r_in: np.ndarray
    total_matches: int
    rmse: float
```

DI: `RotationSelector` отримує `feature_extractor` + `retrieve: Callable[[np.ndarray,int],
tuple[str|None,list]]` (оркестратор передає `candidate_retriever.retrieve`); власного
стану не тримає. `CandidateRetriever` отримує `db_manager`/`retriever`/`patchify_retrieval`
явно. `GeometricVerifier` — `matcher`, `database`, пороги. `ResultBuilder` — `calibration`,
`config`, `_compute_confidence`.

---

## 5. Порядок виконання (інкрементальний, тести зелені після КОЖНОГО кроку)

0. **Характеризаційні тести (СПЕРШУ).** Харнес із fakes (§6) → snapshot результатів
   `localize_frame`/`localize_optical_flow` на синтетиці ДО будь-яких змін.
1. `failure_log.py` — найбезпечніше (чистий CSV-лог, нуль числової логіки).
2. `rotation_geometry.py` — `_ROTATION_VEC` + `_rotate_point_np90` (вже stateless; є тест
   `test_localization`? — додати параметричний проти `np.rot90`).
3. `candidate_retriever.py` — `_retrieve_candidates`, `_merge_candidates`, патчифай.
4. `geometric_verifier.py` — цикл кандидатів (найбільша числова частина; VerificationResult).
5. `result_builder.py` — FOV/confidence/metric-GPS/fallback.
6. `rotation_selector.py` — A3/A2 (лічильник викликів екстрактора: prior → 1 forward).
7. `localize_frame` → оркестрація: guard → normalize → select → expand → extract →
   verify → (fallback|build) → оновити session-стан.

Дублювання формул НЕ вводити; координатні перетворення переносити дослівно.

---

## 6. Характеризаційний харнес

`tests/fixtures/localizer_fakes.py` (додано цим планом) — pure-numpy fakes, що
реалізують Protocol'и з `src/interfaces.py`:

- `FakeGlobalExtractor` (GlobalDescriptorExtractor) — детерміністичний дескриптор per-frame;
  `extract_global_descriptors_multi` для перевірки A2-батчу.
- `FakeLocalExtractor` (LocalFeatureExtractor) — сітка keypoints + дескриптори.
- `FakeMatcher` — `match(q, r)` повертає відповідні точки під відому гомографію
  (identity → багато inliers → успішна локалізація).
- `FakeDatabase` (FrameDatabase) — in-memory; `global_descriptors`, `lance_table=None`,
  `frame_rmse`, `frame_disagreement`, `get_frame_affine/size/local_features`.
- `FakeCalibration` — `.converter.metric_to_gps`, `.set_gsd_calculator`.

Сценарії для snapshot:
1. Успішна локалізація (identity-гомографія) → `success=True`, `lat/lon`, `_last_state`.
2. Out-of-coverage guard після N невдач.
3. Retrieval-only fallback (score вище/нижче `retrieval_only_min_score`).
4. Outlier відфільтрований.
5. A3-prior: при високому score НЕ робиться повний скан (лічильник викликів екстрактора).
6. `localize_optical_flow` після успішного keyframe (bit-identical GPS).

Харнес зберігає `json.dumps(result, sort_keys=True)` + `_last_state` у baseline; після
кожного кроку рефакторингу — `assert` ідентичності (як bit-exact probe для pose_graph).

---

## 7. Верифікація (на машині з torch/faiss)

```bash
# 0) до рефакторингу — зафіксувати baseline
pytest tests/test_localization_characterization.py --snapshot-update   # або run+save

# після кожного кроку 1..7:
pytest tests/test_localization_characterization.py -q                  # має бути зелено
ruff check src/localization/
mypy src/localization/localizer.py        # опційно, gradual

# наскрізний прогін (якщо є synth-відео / реальна БД):
pytest tests/integration/ -m "not gpu" -q
```

Ключова умова прийняття: snapshot результатів `localize_frame`/`localize_optical_flow`
**байт-ідентичний** до і після кожного кроку.

---

## 8. Реєстр ризиків

| Ризик | Мітигація |
|---|---|
| Розрив контракту `last_state` (OF/TrackingWorker) | `_last_state` лишається у `Localizer`; тест OF-сценарію (§6.6) |
| Зміна порядку side-effect'ів (`_consecutive_failures`, reset у 559) | оркестратор володіє станом; переносити «як є», тест guard/outlier |
| Мутація `self.database`/`self.calibration` у multi-режимі | передавати active-context явно; тест multi-source (за наявності db_manager fake) |
| `hasattr`-гілки (`lance_table`, `extract_global_descriptors_multi`) | НЕ прибирати в цьому кроці; Protocol'и з п.3.1 зроблять контракт явним пізніше |
| Числові розбіжності (гомографія/RMSE/FOV) | чистий move формул; bit-identical snapshot; `estimate_homography` не чіпати |

---

## Посилання
- Поточний код: `src/localization/localizer.py`
- Контракти колабораторів: `src/interfaces.py` (Protocol'и)
- Fakes: `tests/fixtures/localizer_fakes.py`
- Стиль bit-exact перевірки: як у `docs/POSE_GRAPH_MATH.md` / pose_graph-рефакторингу
