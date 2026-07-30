# План покращення швидкості та точності

Дата: 2026-07-04. Базується на повному аудиті кодової бази (hot path локалізації, побудова БД, моделі, трекінг, GUI, мережа). Виправлення калібрування з BUGREPORT_CALIBRATION.md вже зроблені й тут не повторюються.

Пріоритет: **P0** = максимальний ефект/малий ризик, **P1** = значний ефект, **P2** = корисно, але після P0/P1.

## ✅ СТАТУС ВПРОВАДЖЕННЯ (2026-07-05)

| Пункт | Статус | Примітка |
|-------|--------|----------|
| A1 empty_cache з гарячих циклів | ✅ | tracking_worker (лише при винятку), builder (раз на 500 кадрів) |
| A2 батч 4 ротацій в 1 forward | ✅ | `extract_global_descriptors_multi` у feature_extractor |
| A3 темпоральний prior кута | ✅ | `localization.rotation_rescan_min_score = 0.70` |
| A4 TensorRT дефолтом | ⬜ | відкладено (потрібна компіляція двигунів на цільовому GPU) |
| A5 батчі збудови БД | ⬜ | відкладено (реструктуризація циклу builder) |
| A6 depth кожен K-й кадр | ✅ | `database.depth_every_n = 10` |
| A7 GUI-конвеєр | ✅ | Format_BGR888 без cvtColor, без зайвої копії, fitInView лише при зміні розміру |
| A8 decimation маркерів | ✅ | `step = max(1, num_frames // 600)` |
| A9 логи/deepcopy | ✅ | FOV-діагностика, early-stop, object-log → DEBUG; deepcopy прибрано |
| B1 outlier config | ✅ | 4.0 σ / 120 м/с (+ синхронізовано `user_config.json`, який ПЕРЕКРИВАЄ дефолти!) |
| B2 адаптивний Kalman R | ✅ | `noise_scale = 1/max(conf, 0.25)`; чесний OF-confidence з inlier ratio |
| B3 robust loss у pose graph | ❌ ВІДКОЧЕНО | `soft_l1, f_scale=3.0` погіршив реальну калібрацію: residuals зважені на cx=960, тож f_scale=3.0 «зрізав» УСІ ребра, а не лише викиди. Плюс файли обрізало гремліном запису. Повернуто до d25aa7a (pure L2). Ідея валідна, але f_scale треба узгодити з масштабом зважених residuals (~сотні), не 3.0 |
| B4 OF з обертанням | ✅ | `estimateAffinePartial2D` на flow-точках, `flow_affine` у localize_optical_flow |
| B5 мертві фічі | 🟡 | `histogram_matching` вимкнено чесно (не реалізований); depth-корекція та GSD — досі не застосовуються (TODO) |
| B3a нормалізація ваг s_ref | ❌ ВІДКОЧЕНО | w/sx_i — індивідуальна вага вузла коректно переводить метричний резидуал у піксельний для КОЖНОГО масштабу; фіксований s_ref завищував вагу далеких вузлів і занижував близьких → деформація траєкторії. Роздування масштабу і так стримує w_reg=200 (square-pixel constraint) |
| B3b ребра графа | ✅ (користувач) | homography_to_similarity → alias на homography_to_affine (estimateAffine2D, 6-DoF): ребра зберігають анізотропію sx≠sy, яку очікує 5-DoF модель оптимізатора; partial2D (sx==sy) конфліктував із w_reg |
| B6 LOO QA якорів | ✅ | у `on_anchor_added`: медіана + поріг 3×med для підозрілих точок |
| B7 гео-гейтинг single-mode | ⬜ | відкладено |
| B8 дрібне | 🟡 | частково (одиниці RMSE) |

**Важливе відкриття:** `user_config.json` перекриває дефолти `config.py` (`AppConfig(**data)`). Старі значення (outlier 150σ, швидкість 1000 м/с, ratio 0.85, WEB_MERCATOR, hist_match=true) сиділи саме там — оновлено і файл, і дефолти. Також відремонтовано `config/config.py` (470 хвостових null-байтів — артефакт конкурентного запису).

**Верифікація (оновлено після відкату):** всі файли компілюються; регресійні тести (відбиття det<0, стабільність масштабу) проходять на робочій версії (w/sx_i + чистий L2 + estimateAffine2D-ребра); механізм robust loss повністю прибрано з коду й конфіга.

---

## A. ШВИДКІСТЬ

### A1. 🔴 P0 — Прибрати `torch.cuda.empty_cache()` з гарячих циклів
- `tracking_worker.py:266-269` — після КОЖНОГО keyframe; `database_builder.py:510-513` — після КОЖНОГО кадру збудови; `tracking_worker.py:206` — ок (тільки при винятку).
- `empty_cache()` синхронізує GPU і повертає блоки драйверу → наступні алокації йдуть через повільний `cudaMalloc`. Це «податок» 10–100 мс на кожен кадр/keyframe і головний вбивця FPS збудови БД.
- **Фікс:** викликати тільки при OOM (обгортка try/except RuntimeError) або раз на N сотень кадрів. Очікування: **+20–40% швидкості збудови БД**, стабільніший FPS трекінгу.

### A2. 🔴 P0 — Батчити 4 повороти DINOv2/v3 в один forward
- `localizer.py:217-241` — auto_rotation робить **4 окремі** `extract_global_descriptor` (4 H2D-копії + 4 ViT-L forward) на кожен keyframe.
- **Фікс:** зібрати batch (4, 3, S, S) і один forward (`feature_extractor` вже вміє батчі — `extract_features_batch`); скор по кожному ряду. Очікування: **розвідка ракурсу ~3× швидше**, keyframe-латентність −30–50%.

### A3. 🔴 P0 — Темпоральний prior на кут повороту
- Дрон не обертається на 90° між keyframe-ами. Тримати `last_best_angle`, пробувати ТІЛЬКИ його; повний скан 4 кутів — лише якщо score < поріг (напр. 0.75×останній) або N невдач поспіль.
- Разом з A2: у сталому польоті — **1 DINO-forward замість 4** на keyframe. Також прибирає «фліп» кута на слабких сценах (бонус до точності).

### A4. 🟠 P1 — TensorRT для DINOv2/v3 (інфраструктура вже є)
- `scripts/compile_dinov2_trt.py` і `trt_dinov2_wrapper.py` існують; для LightGlue/YOLO backend-и torchscript/tensorrt теж передбачені (`model_manager.py:368-405`).
- **Фікс:** зробити TRT-двигун дефолтом за наявності (`models.performance.use_tensorrt_*`), задокументувати компіляцію в README. ViT-L у TRT FP16 ≈ 2–3× швидше за PyTorch.

**Уточнення 2026-07-21** (звірка коду під `docs/RESEARCH_ADDENDUM_2026-07.md`; деталі й порядок робіт — `docs/ADDENDUM_IMPLEMENTATION_PLAN.md` §1.3):

1. **INT8 не робити — зупинитись на FP16.** У TRT для ViT INT8 не швидший за FP16: LayerNorm/GELU/attention лишаються у вищій точності, а reformat-и з'їдають виграш ([issue NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT/actions/runs/12362619431/job/34502302378)). Це запобіжник на майбутнє: у поточному коді INT8 і не було — `compile_dinov2_trt.py` компілює жорстко з `--fp16`.
2. **Блокер: скрипт компілює НЕ ТУ модель.** `compile_dinov2_trt.py` робить `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")` @336 → `dinov2_vitl14_fp16.engine`. Активний бекенд — DINOv3 sat493m @224 через HF (`user_config.json: global_descriptor.backend = "dinov3"`). Тобто A4 у теперішньому вигляді дасть двигун для моделі, яка не використовується. Переписати під готові ONNX-експорти DINOv3 (посилання — `docs/EFFICIENCY_OPTIONS.md` §2.1).
3. **Блокер (перевірити перед роботою): API-розрив TensorRT.** `trt_dinov2_wrapper.py:101` викликає `context.execute_async_v2(bindings=[...])` — binding-index API TRT ≤ 8, тоді як `pyproject.toml` вимагає `tensorrt-cu12>=10.0.0`. У TRT 10 цей шлях, схоже, замінено на `set_tensor_address` + `execute_async_v3`. Перевірка:
   ```
   .venv\Scripts\python -c "import tensorrt as trt; print(trt.__version__); print(hasattr(trt.IExecutionContext,'execute_async_v2'), hasattr(trt.IExecutionContext,'execute_async_v3'))"
   ```
   Якщо перший `hasattr` → `False`, wrapper треба переписати (плюс прибрати хардкоди `input_size=336` і `output_shape=(1,1024)` у `_load_engine`), і оцінку «~0.5 дня + компіляція» піднімати.

### A5. 🟠 P1 — Збудова БД: справжні батчі
- `database_builder.py` обробляє кадри по одному (`extract_features` single), хоча `FeatureExtractor.extract_features_batch` з CUDA-streams вже написаний; `yolo_batch_size: 1` у конфігу.
- **Фікс:** прогнати екстракцію батчами 4–8 (yolo_batch_size=4, features batch), ALIKED лишити ітеративним всередині батчу (він там так і зроблений). Очікування: **+30–60%** пропускної здатності збудови.

### A6. 🟠 P1 — Depth-Anything не на кожен кадр збудови
- `database_builder.py:415-423` — `get_relative_scale` (повний інференс Depth-Anything-V2) на КОЖНОМУ семпльованому кадрі, а використовується лише скалярний масштаб, який змінюється повільно.
- **Фікс:** рахувати кожен K-й кадр (K=10–20) + інтерполяція, або лише на keyframes. Очікування: −20–35% часу збудови при depth увімкненому.

### A7. 🟠 P1 — GUI: конвеєр відображення кадрів
- `tracking_mixin.py:310-312` + `image_utils.py`: на КОЖЕН кадр (30 fps) у GUI-потоці: BGR→RGB `cvtColor` повного кадру + `ascontiguousarray` + `QImage` + `q_img.copy()` + `QPixmap.fromImage` (2 повні копії), потім `video_widget.display_frame` → `fitInView` щокадру.
- **Фікси:**
  1. Конвертувати/зменшувати у воркері (emit уже готовий зменшений RGB під розмір віджета).
  2. `QImage.Format_BGR888` — прибирає cvtColor взагалі.
  3. `fitInView` викликати лише при resize, не щокадру (`video_widget.py:37`).
  4. Прибрати одну з копій (`q_img.copy()` зайвий, якщо тримати посилання на буфер).
- Очікування: розвантаження GUI-потоку в рази на 4K-відео, зникнення «фризів» карти.

### A8. 🟡 P2 — Верифікаційні маркери на карті
- `calibration_mixin.py on_verify_propagation`: `step = 1` — на карту вивалюються ТИСЯЧІ Leaflet-маркерів одним JSON → WebEngine підвисає.
- **Фікс:** decimation (step = max(1, num_frames//500)) + полілінія замість маркерів для траєкторії; конфіг `gui.verify_display_mode` вже існує — використати.

### A9. 🟡 P2 — Дрібне в hot path
- `localizer.py:522-547` — багато `logger.info` (FOV-діагностика) на кожен keyframe → перевести в DEBUG.
- `tracking_worker.py:236-246` — `deepcopy` кожного трекнутого об'єкта на keyframe → скопіювати лише 2 поля вручну.
- OF: `goodFeaturesToTrack`/`calcOpticalFlowPyrLK` на повній роздільній здатності — можна на 0.5× сірому (± однакова якість медіанного зсуву, 4× менше пікселів).
- `resolution_normalizer.py:57` — LANCZOS4 для upscale → INTER_CUBIC (швидше, різниці для фіч немає).
- `matcher._fast_numpy_match` — ок (argpartition вже є).

---

## B. ТОЧНІСТЬ

### B1. 🔴 P0 — Outlier-фільтр фактично вимкнений конфігом
- `config.py: outlier_threshold_std = 150.0` — Z-score тест `z > 150` не спрацює ніколи (при floor std=1.0). Реально фільтрує лише `max_speed_mps = 1000` (тобто майже ніщо).
- **Фікс:** `outlier_threshold_std: 3.5–5.0`, `max_speed_mps` під реальну платформу (напр. 60 м/с). Це прямий вплив на «стрибки» траєкторії.

### B2. 🔴 P0 — Адаптивний шум вимірювання в Kalman
- `kalman_filter.py:35` — R фіксований (5 м) незалежно від якості локалізації; `localize_optical_flow` повертає хардкод `confidence: 0.8`, OF-inliers = `0.8 × KF-inliers` (вигадані).
- **Фікс:** масштабувати R від confidence/inliers: `R = R_base / max(confidence, 0.2)`; для OF — R більший (він відносний). Чесний OF-confidence: від розкиду flow-векторів (MAD) і кількості трекнутих точок. Ефект: плавна траєкторія без «прилипання» до поганих вимірювань.

### B3. 🔴 P0 — Robust loss у графовій оптимізації пропагації
- `pose_graph_optimizer.py:259-270` — `least_squares(...)` з чистим L2: один хибний loop closure (сильна візуальна схожість різних місць — поля, посадки) тягне всю траєкторію.
- **Фікс:** `loss='soft_l1'` (або 'huber'), `f_scale≈2–3` px — один рядок. Опційно двопрохідно: оптимізація → відкинути ребра з residual > 3σ → повторна оптимізація. Ефект: помітно стабільніша пропагація на маршрутах з повторюваними текстурами.

### B4. 🟠 P1 — OF з урахуванням обертання
- `tracking_worker.py:284-296` — з flow-векторів береться лише медіанний зсув (dx, dy); обертання/зміна масштабу між keyframe-ами ігноруються → дрейф на віражах.
- **Фікс:** `cv2.estimateAffinePartial2D(good_old, good_new)` (дешево, точки вже є) → застосовувати повну симілярність до центру кадру в `localize_optical_flow`. Ефект: точніший трекінг між keyframe-ами, можна збільшити `keyframe_interval` (бонус до швидкості).

### B5. 🟠 P1 — Мертві «фічі точності»: реалізувати або видалити
1. `config.py: histogram_matching: True` + `reference_image_path` — **ніде не реалізовано** (`image_preprocessor.py` робить лише CLAHE). Реалізація (histogram matching L-каналу до reference-кадру з БД) підвищить стійкість ALIKED-матчингу до сезону/освітлення — це прямий вплив на inliers.
2. `multi_anchor_calibration.get_metric_position_with_depth` — `correction` обчислюється, кліпається, логується і **не застосовується** до (mx, my).
3. `set_gsd_calculator` — GSD зберігається й логується, але ніде не використовується. Використати як sanity-check: масштаб propagated affine vs фізичний GSD (розбіжність > 30% → warning у звіті пропагації).

### B6. 🟠 P1 — Leave-one-out QA для якорів
- У `calibration_mixin.on_anchor_added` після LSQ-фіту: для кожної точки перефітити матрицю без неї і показати, наскільки зсувається прогноз цієї точки. Одразу видно «криву» точку (неправильний клік/координата) — головне джерело похибки калібрування користувачем. Дешево: 4–8 LSQ 6×6.

### B7. 🟠 P1 — Гейтинг кандидатів retrieval за GPS-контекстом
- `GeoAwareRetriever` + `SpatialIndex` вже є і перебудовують FAISS-підмножину по позиції — переконатися, що вони активні і в single-DB режимі (зараз `Localizer` у single-mode використовує повний `FastRetrieval` без гео-звуження, `localizer.py:90-96`). Звуження простору пошуку = менше хибних кандидатів на однорідних текстурах + швидше.

### B8. 🟡 P2 — Дрібне
- `_compute_confidence`: `frame_rmse` тепер px, а `rmse_norm_m = 10.0` «в метрах» — перейменувати параметр/узгодити одиниці (після сьогоднішнього фіксу підписів).
- Retrieval-only fallback ставить позицію в ЦЕНТР опорного кадру з conf 0.3 — додати розсіювання/більший R у Kalman для таких вимірювань (зараз він фільтрується як звичайне).
- `min_matches=12` і `min_inliers_accept=10` близькі — для 8-DoF гомографії з 12 матчів RANSAC легко «знаходить» випадкову модель; підняти `min_inliers_accept` до 15 або вимагати inlier_ratio ≥ 0.35.
- Loop closure у пропагації: `lc_min_similarity 0.75` + `lc_min_inliers 15` — додати перевірку узгодженості масштабу ребра з сусідніми temporal-ребрами (відсікає false loop closures до оптимізації).

---

## C. ПОРЯДОК ВПРОВАДЖЕННЯ (рекомендація)

| Крок | Пункти | Зусилля | Ефект |
|------|--------|---------|-------|
| 1 | A1 (empty_cache) + B1 (outlier config) + B3 (soft_l1) | ~1 год | Великий, нульовий ризик |
| 2 | A2+A3 (батч ротацій + кутовий prior) | ~0.5 дня | Keyframe 2–4× швидше |
| 3 | B2 (адаптивний Kalman R) + B4 (OF з ротацією) | ~0.5 дня | Плавність і точність траєкторії |
| 4 | A7 (GUI-конвеєр) + A8 (маркери) | ~0.5 дня | Чуйний UI на 4K |
| 5 | A5+A6 (батчі збудови + depth кожен K-й) | ~1 день | Збудова БД у ~2× швидше |
| 6 | B5 (histogram matching + depth/GSD) + B6 (LOO QA) | ~1–2 дні | Стійкість до сезону/світла, якість якорів |
| 7 | A4 (TensorRT дефолтом) | ~0.5 дня + компіляція | 2–3× DINO |
| 8 | B7, B8, A9 | за залишковим принципом | Полірування |

**Як міряти:** `Telemetry.profile` вже інст