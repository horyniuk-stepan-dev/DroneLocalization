# Розв'язання scale-проблеми: локалізація на висоті, відмінній від висоти збудови БД

> Постановка: БД збудована з обльоту на ~100 м; локалізація має працювати на 50–200+ м. Телеметрії (висота/курс) НЕМАЄ. Дата: 2026-07-07.

> **Статус (2026-07-08):** Етап 1 (ScaleManager: prior + піраміда) — ✅ реалізовано (`src/localization/scale_manager.py`, інтеграція в `rotation_selector.py`/`localizer.py`, конфіг-ключі в `config/localization.py`). Етап 2 (depth-hint) — 🔨 у робочому дереві (uncommitted). Етапи 3–5 — відкриті. Живий статус: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## 1. Діагноз: де саме ламається пайплайн

Позначимо **r = висота_запиту / висота_БД** (r=2 → GSD запиту вдвічі грубший, кадр покриває вчетверо більшу площу).

| Стадія | Що відбувається при r≠1 | Поріг зламу |
|---|---|---|
| Retrieval (DINO CLS) | Дескриптор кадру з іншим FOV/GSD "не схожий" на дескриптори БД → score падає нижче `rotation_rescan_min_score=0.70` і `retrieval_only_min_score=0.90` → неправильні кандидати або їх немає | r ≳ 1.3–1.5 (плавна деградація) |
| Matching (ALIKED/RDD + LightGlue) | Локальні дескриптори не scale-інваріантні; LightGlue навчений на обмежених scale-гепах → matches < `min_matches=12` → "Not enough valid inliers" | r ≳ 1.5–2 |
| Гомографія → координати | **НЕ ламається.** H поглинає масштаб; якщо інлаєри є — координати коректні | — |
| `ResolutionNormalizer` | Нормалізує лише роздільну здатність (пікселі), не GSD — при r≠1 не допомагає | — |
| Kalman/outlier | Метричні пороги — не залежать від r | — |

Висновок: математика координат справна; ламаються дві перцептивні стадії. Отже задача зводиться до **оцінки r без телеметрії** і **нормалізації зображення до GSD БД** до extraction. Це підтверджує література: [Altitude-Adaptive Vision-Only Geo-Localization](https://arxiv.org/abs/2602.23872) (2026) показує, що сама лише висотна нормалізація запиту піднімає R@1 retrieval на **+41.5 п.п.** при змінах висоти.

## 2. Джерела оцінки r (без телеметрії)

**(a) З самого трекінгу — безкоштовно.** Після кожної успішної локалізації `H_query→ref` містить масштаб: `decompose_affine_5dof(homography_to_affine(H))` → (sx, sy) — уже реалізовано в `affine_utils`/`pose_graph_optimizer`. Висота не стрибає вдвічі між keyframes (1 с) → **темпоральний prior масштабу**, точний аналог кутового prior A3. Покриває 95% кадрів польоту.

**(b) Скан-піраміда — bootstrap і відновлення.** Коли prior немає (старт, out_of_coverage): прогнати retrieval на пірамідi версій запиту, вибрати рівень з найкращим score — точний аналог батч-скану 4 ротацій (A2). Октавна сітка `[0.5, 0.7, 1.0, 1.4, 2.0]` покриває 50–200 м навколо висоти БД; батчований DINO-forward 5 варіантів (у комбінації з 4 ротаціями — 20 зображень одним forward, для DINOv3\@224 це прийнятно).

**(c) Depth-моделі — незалежна оцінка, вже наполовину в коді.** `DatabaseBuilder` ВЖЕ зберігає `depth_scales` (Depth-Anything-V2, `get_relative_scale`) на кожен кадр БД — але `Localizer` їх не читає. Порахувати ту саму величину для запиту → `r ≈ scale_query / scale_db_frame`. Застереження з літератури: zero-shot metric-depth на надирних знімках з висоти деградує через domain shift ([MovingDrone-оцінка](https://arxiv.org/pdf/2509.14839); краще тримаються [UniDepth v2](https://arxiv.org/html/2502.20110v2), MoGe v2) — тому використовувати як **hint/арбітр** (валідація prior, вибір стартового рівня піраміди), а не як єдине джерело.

**(d) Оцінка scale-ratio по парі** — [Scale-Net](https://arxiv.org/abs/2112.10485): CNN оцінює r між двома зображеннями, потім обидва ресайзяться до спільного масштабу (SDAIM) перед matching. Застосовно точково: retrieval знайшов кандидата, а matching впав → оцінити r по парі (query, top-1), ресайзнути, повторити.

**(e) Навчений висотний естіматор** — [Altitude-Adaptive](https://arxiv.org/abs/2602.23872): частотна область + regression-as-classification, 13.3 fps, R@1 +41.5 п.п.; [HE-VPR](https://arxiv.org/pdf/2603.04050): оцінка висоти як retrieval-задача в спільному бекбоні → вибір під-БД потрібного масштабу. Потребує навчального датасету — далекий горизонт.

## 3. Нормалізація: два асиметричні випадки

**r > 1 (летимо ВИЩЕ, ніж БД):** кадр запиту містить кілька кадрів БД. Нормалізація на боці запиту: **центральний кроп зі стороною 1/r + upscale** ≈ вигляд з висоти БД (так робить Altitude-Adaptive: "crop query to canonical scale"). Для повного покриття FOV — 4–5 кропів (центр + квадранти), батчем. Для matching — кропнуту версію подаємо в ALIKED.

**r < 1 (летимо НИЖЧЕ):** кадр запиту — фрагмент кадру БД; "домалювати" оточення не можна. Нормалізація на боці БД: дескриптори субрегіонів кадрів БД — **це буквально ваш patchify** (сітки 2×2/3×3 ≈ r від 1/3 до 1). Патчифай — правильний механізм саме для цього випадку; для matching — downscale запиту в GSD БД (кількість пікселів зменшиться — це нормально).

**r > ~2.5 (значно вище):** кропи втрачають деталі; надійніше — **мультирівнева БД**: з наявних кадрів + пропагованих аффін рендериться ортомозаїка (панорамний воркер уже вміє суміщати кадри), з неї — tile-піраміда рівнів GSD ×1, ×2, ×4 (як zoom-рівні супутникових карт), дескриптори кожного рівня в LanceDB з колонкою `level`. Retrieval по всіх рівнях; знайдений рівень ⇒ r відомий одразу. Це стандартна практика satellite-map локалізації і найнадійніше рішення для великих r, ціна — офлайн-крок після пропагації.

**Matching-fallback для екстремальних r:** dense-матчери з coarse-стадією на низькій роздільності толерантні до великих scale-гепів — [RoMa v2](https://arxiv.org/pdf/2511.15706) (див. SOTA_RESEARCH.md §2); епізодичні 300–600 мс на keyframe при зриві sparse — прийнятно.

## 4. План впровадження (ранжовано за ефект/зусилля)

### Етап 1 — ScaleManager: prior + піраміда (ядро рішення, ~тиждень) — ✅ РЕАЛІЗОВАНО

Новий `src/localization/scale_manager.py`, дзеркало логіки A2/A3 для масштабу:

```python
class ScaleManager:
    """Оцінка і трекінг GSD-відношення запиту до БД (без телеметрії)."""
    def __init__(self, cfg):
        self.prior: float | None = None          # r останньої успішної локалізації
        self.pyramid = cfg.scale_pyramid          # [0.5, 0.7, 1.0, 1.4, 2.0]
        self.rescan_min = cfg.scale_rescan_min_score

    def candidates(self) -> list[float]:
        """Prior валідний → [prior]; інакше повна піраміда (батч-скан)."""
        return [self.prior] if self.prior is not None else list(self.pyramid)

    def normalize(self, frame, r) -> np.ndarray:
        """r>1: центр-кроп 1/r + resize up; r<1: resize down до GSD БД."""
        ...

    def update_from_homography(self, H, frame_shape):
        """Після успіху: r з decompose_affine_5dof(homography_to_affine(H))."""
        sx, sy = ...
        self.prior = self._ema(np.sqrt(sx * sy))   # EMA, кліп у [0.3, 3.0]

    def invalidate(self):                          # out_of_coverage / N невдач
        self.prior = None
```

Інтеграція в `Localizer.localize_frame` (Крок 1): цикл вибору тепер по (кут × масштаб) з батчованим forward; у steady-state prior кута + prior масштабу = як і зараз, 1 forward. Matching (Крок 2): екстракція з `normalize(frame, r_best)`; координати центру перераховуються назад через відомий кроп/resize (розширення `ResolutionNormalizer`).

Конфіг: `localization.scale_pyramid`, `scale_rescan_min_score` (аналог rotation_rescan_min_score), `scale_prior_ema`.

### Етап 2 — задіяти наявні depth_scales (~2-3 дні) — 🔨 У РОБОТІ (uncommitted)

`Localizer` читає `depth_scales` з HDF5 (уже пишуться), рахує depth_scale запиту раз на keyframe (інференс уже є в кодовій базі): (а) стартовий рівень піраміди = найближчий до depth-оцінки (скорочення скану 5→2 рівні); (б) гейт узгодженості: H-масштаб і depth-ratio розходяться >1.5× → знизити confidence.

### Етап 3 — patchify за замовчуванням для r<1 + крос-масштабні тести (~тиждень)

Увімкнути patchify (у варіанті grid-пулінгу патч-токенів з 1 forward — див. IMPROVEMENT_PLAN.md п.5/SOTA_RESEARCH.md §1) — це штатний механізм низьких висот. Тести: взяти еталонне відео БД, синтетично кропнути/ресайзнути кадри під r ∈ {0.5, 0.7, 1.4, 2.0} → загнати в `localize_frame` → медіанна помилка ≤ 2× базової. Це і є acceptance-критерій усього епіка.

### Етап 4 — ортомозаїка + tile-піраміда рівнів (r до ×4, ~2-3 тижні)

Офлайн-крок після пропагації: рендер ортомозаїки з кадрів+аффін → рівні GSD ×2, ×4 → дескриптори з `level` у LanceDB → retrieval по рівнях (знайдений level задає r і одразу правильний реф для matching). Бонус: мозаїка корисна і для GUI-карти.

### Етап 5 — fallback-матчер великих r (опційно)

RoMa v2 (або Scale-Net перед LightGlue) при зриві sparse-matching на валідному retrieval-кандидаті.

## 5. Очікуване покриття по r

| Діапазон r | Механізм | Очікування |
|---|---|---|
| 0.8–1.3 | уже працює (толерантність ALIKED/LightGlue) | без змін |
| 1.3–2.5 | Етап 1 (prior+піраміда) + Етап 2 | штатна робота |
| 0.35–0.8 | Етап 1 (downscale) + Етап 3 (patchify) | штатна робота |
| 2.5–4+ | Етап 4 (tile-піраміда) ± Етап 5 | робоча, повільніша |
| Різка зміна висоти в польоті | prior інвалідується score-гейтом → рескан піраміди | 1 batched forward |

Ризики: (1) кроп 1/r при r>2 різко звужує FOV retrieval → потрібні мультикропи (закладено в Етап 1) або Етап 4; (2) EMA-prior може "залипнути" при плавному наборі висоти — score-гейт на кожному keyframe обов'язковий; (3) depth-оцінки на воді/однорідних полях ненадійні — тому лише hint (Етап 2), ніколи не єдине джерело.

## Джерела

[Altitude-Adaptive Vision-Only Geo-Localization for UAVs](https://arxiv.org/abs/2602.23872) (2026, найближча постановка: оцінка висоти з одного кадру без сенсорів → нормалізація масштабу, R@1 +41.5 п.п.) · [HE-VPR: Height Estimation Enabled Aerial VPR](https://arxiv.org/pdf/2603.04050) (2026) · [Scale-adaptive UAV Geo-Localization via Height-aware Partition Learning](https://arxiv.org/abs/2412.11535) · [Scale-Aware UAV-to-Satellite CVGL](https://arxiv.org/pdf/2603.07535) · [Scale-Net: Learning to Reduce Scale Differences](https://arxiv.org/abs/2112.10485) · [UniDepth v2](https://arxiv.org/html/2502.20110v2) · [MMDE на аерознімках: оцінка обмежень](https://arxiv.org/pdf/2509.14839) · [RoMa v2](https://arxiv.org/pdf/2511.15706) · [OrthoTrack: прив'язка до ортофото](https://arxiv.org/pdf/2606.25245) · [VPR for Large-Scale UAV Applications](https://arxiv.org/html/2507.15089v1)
