# XFeat swap — план і handoff (раунд 2: VRAM + швидкість + спрощення)

> Дата: **2026-07-22**. Ціль (з відповідей): менше VRAM, ще швидше, спростити стек. Ризик-стеля: агресивно (ребілд БД + A/B). Тренування досі заборонене — XFeat це pretrained-ваги, не навчання.
> Стек зараз: DINOv3-sat ViT-L (retrieval) + **ALIKED** (локальні, 128-dim) + **LightGlue** (матчинг). XFeat замінює два останні одним.

---

## Чому XFeat (а не інше)

Одна заміна б'є по всіх трьох цілях:

- **VRAM (головне).** ALIKED має пік ~3.5 ГБ (найтісніший гейт на 4 ГБ), LightGlue тримає ще ~0.8 ГБ резидентно. XFeat — ~0.3–0.5 ГБ і **прибирає LightGlue повністю** (64-dim дескриптори маршрутизуються в numpy-MNN, не в LightGlue). Це найбільший разовий виграш VRAM у всьому проєкті.
- **Швидкість.** XFeat — до 5× легший за SuperPoint (CVPR 2024), real-time на CPU. Матчинг: MNN ~46 мс (CPU) проти LightGlue 128 мс на Shtepsill. Плюс легший екстрактор.
- **Спрощення.** Один екстрактор замість зв'язки ALIKED+LightGlue; менше точок відмови, менше моделей на диску й у VRAM.

Прецедент прямий: SatLoc-Fusion тримає realtime на edge саме на DINOv2-retrieval + XFeat (`RELATED_WORK_MAP §1.3`).

**2026-альтернативи (перевірено веб-пошуком).** XFeat досі еталон легкого екстрактора. Новіше — CLIDD (deformable descr., arXiv січ-2026) і AsymLoc (асиметричний матчинг, 2026) — існує, але без aerial-валідації й свіже; під «агресивно, але rebuild+A/B» XFeat надійніший. CLIDD/AsymLoc — кандидати на майбутній A/B, не на зараз.

---

## Що вже було в репо, і що бракувало

XFeat був **~80% вбудований**: завантаження (`model_manager.load_xfeat`), batch-екстракція з true batching (`extract_features_batch`, гілка `is_xfeat` → `detectAndCompute`), маршрутизація 64-dim→MNN у матчері (`matcher.match`), і db_builder знає про `local_descriptor_dim=64`. **fp16-зберігання дескрипторів уже зроблено** (`database_builder.py:724`, `dtype="float16"`) — цей VRAM-важіль знято з дошки, він давно в коді.

Бракувало рівно двох речей у **онлайн-шляху локалізації** (не в збудові БД) — реалізовано цією сесією, flag-gated за `models.local_extractor` (дефолт `aliked`):

1. `model_manager.load_local_extractor` — не маршрутизував `"xfeat"`, тихо падав на ALIKED. **Виправлено** → `load_xfeat()`.
2. `feature_extractor.extract_local_features` (single-frame) — був суто ALIKED (`self.local_model({"image": tensor})`). Додано **XFeat-гілку** (`detectAndCompute`), що дзеркалить batch-шлях. ALIKED-гілка побайтово та сама, коли екстрактор не xfeat.

Змінені файли: `src/models/model_manager.py`, `src/models/wrappers/feature_extractor.py`. Ruff чистий, AST/null-байти ок. **Онлайн XFeat-шлях у пісочниці не виконати** (нема torch/XFeat) — import-тест на Windows обов'язковий.

---

## Кроки на твоєму боці

**1. Import-тест (перш ніж будувати БД).**
```
python -c "import src.models.model_manager, src.models.wrappers.feature_extractor"
```

**2. Перемкнути екстрактор.** У `user_config.json`:
```json
"models": { "local_extractor": "xfeat" }
```
Опційно `"models": {"xfeat": {"top_k": 4096}}` (дефолт 2048) — більше точок, якщо на ріллі бракуватиме.

**3. Ребілд БД.** Збудувати базу з того самого відео (апка → database build). Отримаєш нову `database.h5` з 64-dim XFeat-дескрипторами (fp16). **Стара ALIKED-база несумісна** — dim 128 vs 64, матчер це побачить і попередить. Тримай стару окремо для A/B.

**4. Пропагація калібрації** на новій БД (як завжди після ребілду), інакше локалізація не матиме affine.

**5. A/B — той самий відеофайл, дві бази.**
- baseline: стара ALIKED-база (лог → `app_aliked.log`);
- кандидат: нова XFeat-база (лог → `app_xfeat.log`);
- `python scripts/ab_localization.py logs/app_aliked.log logs/app_xfeat.log`.

**6. Швидкість/VRAM:** `python scripts/benchmark_hardware.py` з `local_extractor:xfeat` — стадія `local_features` заміряє XFeat, `match_*` піде через MNN. Порівняй пік VRAM проти ALIKED-прогону.

---

## Як читати A/B (тут інакше, ніж для A1)

XFeat і ALIKED — **різні матчери**, тож абсолютні `inliers` НЕ порівнювані (XFeat дає інші числа матчів). Дивись:

- **success-rate** — головний сигнал: чи XFeat не колапсує на ріллі (його слабке місце). Має бути не гіршим за ALIKED.
- **Jaccard matched-frame** — чи локалізує на ті самі кадри БД (нумерація та сама, бо те саме відео).
- **позиція проти ground_truth** — якщо є `ground_truth.json`, це фінальний критерій точності (компаратор дає зсув між прогонами, але істина — GT).

Ігноруй `Δinliers` — він тут шум через різні матчери.

---

## Ризик (той, що ти свідомо прийняв)

XFeat **слабший на кросдоменних парах** і на однорідних текстурах (рілля, ліс) — саме твій сценарій. `RESEARCH_ADDENDUM §2` уже фіксував колапс матчингу на ріллі як ризик. XFeat його може загострити: менше, але рідших матчів → на бідних текстурах success-rate може просісти. Тому **A/B саме на ріллі-місії обов'язковий**, а не на текстурному місті. Якщо success-rate падає — два ходи без відкату: (а) `xfeat.top_k` 2048→4096; (б) XFeat-semi-dense (`match_xfeat_star`) замість sparse — це вже наступний крок, semi-dense дорожчий, але для ріллі якраз показаний (`RESEARCH_ADDENDUM §2`).

Якщо success-rate тримається — виграв найбільше зі всього плану: −0.8 ГБ (LightGlue) плюс легший екстрактор, і мінус ціла модель зі стеку.

## Коміт (за тобою)
```
git add src/models/model_manager.py src/models/wrappers/feature_extractor.py docs/XFEAT_PLAN.md
git commit -m "feat: enable XFeat in online localization path (loader route + single-frame detectAndCompute)"
```
