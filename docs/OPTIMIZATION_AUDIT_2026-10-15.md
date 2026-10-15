# Аудит оптимізації DroneLocalization — 15.10.2026

Найперспективніша оптимізація — зменшити кількість дорогих проходів DINO/Patchify, тримати ознаки на GPU між extraction і matching та відкласти обчислення даних БД до рішення зберегти keyframe. Заміна моделей — наступний етап, після усунення цих витрат.

## Межі та перевірка

- Інвентаризація й AST-перевірка всіх 325 Python-файлів у `src`, `config`, `scripts` (без згенерованого `scripts/allFiles`), `tests` та `main.py`: 63 574 рядки; синтаксичних помилок немає.
- Детальне читання основних шляхів: моделі, extraction, VLAD, Patchify, localization, verification, DB build/load/write, propagation, sparse optimizers, smoothing, video, tracking worker, GUI delivery, конфігурація і телеметрія. AST-обхід усіх файлів не означає ручної перевірки кожного рядка.
- Сторонні ваги, `.venv`, build/dist і згенерований код не аналізувалися як власний код. Інтерпретатор `.venv` використано для перевірок.
- Середовище перевірки: Python 3.11.9, torch 2.12.0.dev20260329+cu128, NumPy 2.4.3, CUDA доступна, NVIDIA GeForce RTX 5070 Ti.
- 56 вибраних unit-тестів пройшли; 3 попередження FAISS/SWIG. Наприкінці pytest є окрема помилка Loguru через закритий capture stream; тести успішні.
- Додатково відтворено дві помилки batching Patchify і перевірено приклад ненормованого Patchify score на синтетичних даних без завантаження моделей.
- Новий benchmark на реальному відео не проводився. Нижче часи зі збереженого звіту до запуску тестів, а очікувані прискорення не видаються за виміряні.
- Код застосунку та `user_config.json` не змінювалися.

## Що показує наявна телеметрія

Знімок `logs/telemetry_report.json`, прочитаний перед перевірками:

| Етап | Виклики | Сумарно, с | Середнє, мс | Максимум, мс |
|---|---:|---:|---:|---:|
| view_prepare | 279 | 1.8686 | 6.7 | 38.9 |
| dinov2 | 778 | 30.0740 | 38.7 | 352.6 |
| retrieval | 226 | 1.7043 | 7.5 | 22.7 |
| local_preprocess | 215 | 1.2833 | 6.0 | 83.0 |
| local_extractor | 215 | 6.7942 | 31.6 | 3144.3 |
| patchify_retrieval | 51 | 24.9910 | 490.0 | 636.5 |
| prefilter | 275 | 2.8631 | 10.4 | 91.7 |
| match | 453 | 9.6134 | 21.2 | 88.1 |
| ransac_homography | 373 | 1.4400 | 3.9 | 15.9 |

`dinov2` — історична назва етапу; за поточним конфігом виконується DINOv3. Етапи вкладені: час DINO входить у Patchify, тому складати всі рядки як незалежний час чи виводити з них відсотки загального навантаження некоректно. Середні округлені самим звітом.

Найсильніший практичний сигнал — близько 490 мс на Patchify-виклик. Retrieval і RANSAC значно дешевші на виклик у цьому записі. Максимум local_extractor 3.14 с потребує окремого розділення cold start, compilation і steady state; причина цього максимуму зі звіту не встановлюється.

Старий `telemetry_report_baseline.json` не підходить для прямого порівняння: до виправлення instrumentation DINO-таймер міг не включати весь forward, і конфігурація змінилася.

## P0: спочатку виправити шлях, який заважає прискоренню

### 1. Patchify batching не еквівалентний послідовному шляху

Код: `src/localization/patchify.py:84–140`.

При `batch_size=1` кожен crop проходить через `FeatureExtractor.extract_global_descriptor`: застосовуються активні VLAD/CESP та CPU resize. При `batch_size>1` окремий `_extract_batch_descriptors`:

1. Складає crops у `torch.stack` **до** вирівнювання розмірів. Batch може містити повний кадр і crop 2×2; навіть crops однієї сітки можуть мати різний розмір через залишок від ділення.
2. Викликає `fe.global_model(...)` без VLAD/CESP. За поточного конфігу вихід — CLS 1024, але очікуються VLAD 256.
3. Обходить активний `dino_cpu_resize=true`, тобто змінює preprocessing.

Відтворення з fake extractor і входом 192×192:

- batch=1, сітки 1×1/2×2/3×3: `(14,256)`, успіх;
- batch=4, ті самі сітки: `RuntimeError: stack expects each tensor to be equal size`;
- batch=2, лише 2×2: `ValueError: could not broadcast input array from shape (2,1024) into shape (2,256)`.

**Рішення:** прибрати дубльований inference-код і направити batches через `extract_global_descriptors_multi`. Він уже підтримує preprocessing, VLAD/CESP і `global_batch_max`. Після цього A/B для batch=1/2/4/8. Перевірити числову близькість дескрипторів, однакову обробку масок, top-k recall і геометричний результат. Зараз просто збільшувати `patchify_batch_size` небезпечно.

### 2. Patchify score не є coverage у межах 0…1

Код: `src/localization/patchify.py:205–236`, merge — `candidate_retriever.py:114`.

`hits` рахує всі знайдені reference patches, включно з кількома hits того самого кадру для одного query patch. `coverage=hits/num_query_patches` може бути >1. Формула `coverage * avg_score` фактично дорівнює сумі всіх scores, поділеній на кількість query patches. Це не нормований coverage score і не має тієї самої шкали, що звичайна cosine similarity.

Перевірений приклад: один reference frame із 14 однаковими patches і 14 такими самими query patches дає score **14.0**, а не 1.0. Це підтверджує шкалу; саме по собі не доводить поганий ranking на реальних сценах.

**Рішення:** для кожної пари `(query_patch, reference_frame)` брати найкращий hit, потім агрегувати через унікальне coverage. Альтернатива — калібрувати окрему шкалу score чи використовувати rank fusion. Зміна впливає на ranking; потрібен replay, а не механічне коригування порогів.

## P1: найбільші можливості зменшити обчислення

### 3. Викликати Patchify як розширення після невдалої стандартної перевірки

Код: `src/localization/localizer.py:1048–1064`, `patchify.py:94–110`.

У звичайному шляху Patchify додає кандидатів до того, як стандартний список пройде геометричну перевірку. Сітки 1×1 + 2×2 + 3×3 означають 14 глобальних дескрипторів. При поточному batch=1 це 14 окремих викликів extraction на один Patchify-прохід.

**Рішення:** temporal/spatial prior → стандартний retrieval → verification → Patchify, якщо результат не пройшов визначені критерії якості. У recovery залишити повне покриття. Для joint multi-layer ranking потрібна окрема перевірка: рання зупинка може пропустити кращий шар.

Додатково: 1×1 crop повторює весь кадр. Його дескриптор можна перевикористовувати лише коли view, preprocessing і valid-mask semantics однакові. Для VLAD rotated view з маскою і Patchify без маски це не гарантовано.

Успіх: менше Patchify-викликів на підтверджений fix, нижчий p95, без погіршення false-fix rate та recovery recall. Не просто зменшення середнього часу.

### 4. Обчислювати глобальні й patch descriptors тільки для кадрів, які зберігаються

Код: `src/database/frame_processor.py:106–128`, рішення keyframe — `139–184`, запис — `185–202`.

Зараз `extract_features` одразу виконує local + global extraction. Далі обчислюються 14 Patchify descriptors і depth, і лише потім визначається `save_this_frame`. Кадр, відкинутий keyframe selector, уже оплатив глобальний forward та Patchify.

**Рішення:** local features → inter-frame matching/pose → keyframe decision → global/Patchify для збереженого кадру. Local features залишаються для побудови ланцюга руху. SIFT уже правильно обчислюється тільки для збережених кадрів.

Перевірка перед зміною: вимоги injected matcher до `global_desc`, сумісність characterization-тестів, forced calibration frames. Depth можна відкладати лише якщо не потрібно зберегти семантику cadence/carry-forward; це окреме рішення.

Економія залежить від частки відкинутих кадрів. Якщо зберігаються всі кадри, цей крок майже не допоможе — потрібен лічильник processed/saved.

### 5. VLAD + PCA виконувати на GPU

Код: `src/models/wrappers/feature_extractor.py:217–233`, `vlad_aggregator.py:150–208`.

DINO patch tokens переводяться в FP32, копіюються на CPU, після чого NumPy робить assignment, цикл по clusters, normalization, PCA й whitening. Для поточного input 336 і patch size 16 виходить 21×21=441 patch tokens. За 1024 каналів — 1 806 336 байтів токенів на зображення; готовий VLAD 256 FP32 — 1024 байти. Співвідношення 1764:1 описує обсяг цього перенесення, **не прискорення всього pipeline**.

**Рішення:** centers/PCA/mean/eigenvalues як кешовані GPU tensors; assignment через matmul; накопичення residuals через scatter/index_add; normalization і whitening на GPU; CPU отримує лише готовий descriptor. Спочатку FP32 aggregation, окремо тестувати reduced precision для backbone.

Контроль: порожні clusters, valid tokens, MIN_VALID_TOKENS, low-norm filter, cosine deviation, retrieval recall. Паралельна редукція може змінити порядок додавання; побітова тотожність не гарантується. Доцільність перевіряти на batches реальних сцен, включно з конкуренцією за GPU з ALIKED/LightGlue.

### 6. Прибрати повторні CPU↔GPU переміщення local features

Код: `feature_extractor.py:366–380`, `src/localization/matcher.py:369–403`, `geometric_verifier.py:337–402`.

Local extraction повертає NumPy arrays через `.cpu().numpy()`. Для кожного LightGlue candidate query і reference знову створюються як CUDA tensors. MNN prefilter окремо завантажує reference descriptors і синхронізується на `.item()` для кожного кандидата.

**Рішення:** тримати GPU-представлення query протягом одного localize call; завести GPU LRU для reference descriptors із лімітом у байтах; CPU cache БД залишити для I/O. Ключ кешу: database identity/revision, frame id, device, dtype, feature budget. Invalidate на reload та close.

MNN counts можна збирати на GPU й переносити разом, але batch similarity tensors мають бути обмежені за пам'яттю. Не створювати необмежений `candidates × keypoints²` tensor.

### 7. Налаштувати inference під активний DINOv3, а не лише DINOv2

Код: `src/models/model_manager.py:566–655`, `dinov3_wrapper.py:42–108`, `feature_extractor.py:99–107`.

Поточний конфіг: DINOv3 SAT ViT-L/16, input=336, VLAD=256, `fp16_enabled=false`, LightGlue `mixed_precision=false`, `torch_compile=true`.

- DINOv3 loader повертає модель до гілки, де застосовуються torch.compile/TRT для DINOv2. Отже, `torch_compile=true` не компілює активний DINOv3. ALIKED має власну гілку compile.
- `fp16_enabled` керує autocast глобального extractor, але local ALIKED у цьому wrapper не загорнутий в autocast. YOLO має окрему half-логіку. Один прапорець не описує precision всіх моделей.
- `ModelManager.prewarm` завантажує моделі, але не виконує representative forward. Він не гарантує прогрівання kernels/compile до першого кадру.

**Експерименти в порядку:** FP32 baseline → DINO FP16/BF16 → LightGlue mp → inference_mode → compile DINOv3 для стабільних shapes. Кожен окремо. Для compile вимірювати cold start, steady p50/p95, recompiles і peak VRAM. Не знижувати точність геометрії/координат разом із neural inference.

LightGlue вже має `flash=true`, adaptive depth та pruning: це не нові можливості, які треба просто ввімкнути. Спочатку тестувати keypoint budgets, потім пороги адаптивного виконання. Офіційний проєкт описує mp, compilation і компроміс pruning: https://github.com/cvg/LightGlue .

`torch.inference_mode` усуває додатковий autograd overhead, але має обмеження використання отриманих tensors; очікуваний ефект треба вимірювати: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html .

## P2: загальна швидкодія й масштабування

### 8. Обмеження keypoint budget вже існує — почати з нього

Код: `src/localization/keypoint_budget.py`, `localizer.py:2071–2093`.

ALIKED добуває до 4096 точок, БД зберігає до 2048. Це не обов'язково помилка: query може вимагати більше coverage. Але temporal prior зараз має `temporal_prior_max_keypoints=0`, тобто використовує повний набір.

Спробувати 1024 і 2048 **лише для temporal prior**, залишаючи повний набір у recovery. Механізм strongest-first slicing уже реалізований, його не потрібно створювати вдруге. Вимірювати hit rate prior і вартість подальшого fallback: дешевший prior, який частіше помиляється, може зробити весь pipeline дорожчим.

### 9. RAM і I/O бази мають масштабуватися за байтами

Код: `database_loader.py:229–237,467–507`, `localizer.py:584–604`, `patchify.py:145–166`, `propagation_pipeline.py:1112–1171`.

- Loader читає весь patch tensor у RAM; FAISS build створює FP32 copy, нормовану copy та власний index. Peak RAM більший за розмір одного tensor.
- CPU feature LRU обмежений 200 записами **на базу**, не сумарними байтами.
- Propagation завантажує всі local features у dict. Витрати лінійні за кількістю кадрів, а не обмежені LRU loader.

Для 10 000 slots × 14 patches × 256 FP32 один patch tensor займає 143 360 000 байтів, приблизно 136.7 MiB. Кілька баз і проміжні copies множать витрати; це оцінка прикладу, не вимір RAM поточного проєкту.

**Рішення:** lazy patch loading при першому використанні, block-wise index creation, shared byte budget для CPU/GPU caches; propagation — локальне sliding window та bounded cache для loop closures. Переходити з exact FAISS до ANN лише якщо профіль великих баз доведе потребу; перевіряти recall і exact reranking.

### 10. Prefetch відео прив'язати до обсягу пам'яті

Код: `src/database/video_frame_source.py:120–171`, `src/utils/hardware_profile.py:340–366`.

Queue зберігає BGR та RGB. За 64 кадрів 1920×1080 це 64×2×1920×1080×3 = 796 262 400 байтів, приблизно 759.4 MiB лише queued arrays. Decord batch, оброблювані кадри й tensors — додатково. Для 4K такий самий queue потребує приблизно 2.97 GiB.

**Рішення:** розраховувати queue/decode batch за resolution, available RAM і максимальним byte budget; BGR створювати за потреби для visualization. Збільшення queue допомагає лише якщо GPU реально чекає декодування. Live source уже має drop-late reader — повторно реалізовувати його не потрібно.

### 11. Реальний VRAM budget замість номінального ratio

Код: `model_manager.py:124–135,144–149`, `src/models/vram.py:78–108`.

`max_vram_ratio` зчитується і налаштовується auto_tune, але не використовується у callback доступної пам'яті чи політиці eviction. Перевірка йде лише за physical free VRAM та оцінкою required_mb. Це не забезпечує заявлену частку ліміту.

Потрібні measured peak allocations за mode/batch/shape, запас під activations, явний bounded retry після OOM. DepthEstimator живе окремо від ModelManager і теж має враховуватися в бюджеті. Наявність Python references у wrappers/consumers також слід перевіряти перед обіцянкою, що unload звільнить VRAM.

`global_batch_max` розбиває вже підготовлений GPU batch: він обмежує forward activations, але не всю пам'ять input/preprocessing. За великих batches нарізати images до upload, не після `torch.stack`.

### 12. Propagation: прибрати O(E²) допоміжну перевірку

Код: `propagation_pipeline.py:1660–1679`.

`_cluster_consistency_factors` робить вкладений цикл по loop closures. Для великих графів замінити його на індекс пар `(min(i,j), max(i,j))` із перевіркою сусідніх buckets або range query. Зберегти точну семантику двох орієнтацій і Manhattan distance. Пріоритет залежить від реальної кількості closures.

Основні optimizers уже використовують sparse структури й аналітичні похідні. Не варто переписувати їх лише через наявність Python loops. Windowed projective optimization/warm start оцінювати на великих БД окремо: це може змінити global consistency.

### 13. GUI та smoothing — нижчий пріоритет

`tracking_worker.py:209–210` надсилає кожен raw frame у Qt; `tracking_mixin.py:342–345` конвертує його в pixmap. Debug channels уже мають backpressure, raw video — ні. Якщо GUI не встигає, latest-frame mailbox і незалежний refresh timer можуть зменшити чергу. Перевіряти presentation latency, а не лише processing FPS.

`tracking/smoother.py:359–411` розв'язує dense систему на кожній IRLS-ітерації. При поточному window=60 це нижчий пріоритет. Якщо вікно росте, суміжні odometry edges дають banded структуру; її можна використати, зберігши priors та missing-edge semantics.

`extract_features_batch` у `feature_extractor.py:420` не має викликів у перевіреному application source. Там є недосяжний дубль після `return` на рядку 583. Видалення дубля покращить підтримку, але саме по собі не прискорить активний pipeline. Device-wide synchronize у цьому неактивному шляху не слід називати нинішнім основним bottleneck.

## Вимірювання, без яких оптимізація буде ненадійною

`src/utils/telemetry.py:27–39` використовує CPU perf_counter; для асинхронних CUDA операцій він може виміряти enqueue, а синхронізація потрапить в інший етап. Додати CUDA Events для GPU-time та залишити wall time для end-to-end. Не вставляти глобальний synchronize після кожної операції у production. Офіційна довідка: https://docs.pytorch.org/docs/stable/notes/cuda.html .

`non_blocking=True` не гарантує overlap сам по собі. Використовувати reusable pinned buffers із чітким lifetime тільки після вимірювання; виклик pin_memory на кожному кадрі теж може коштувати дорого. Довідка: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html .

Потрібні окремі серії:

1. Cold start/loading/compile та прогрітий steady state.
2. Prior hit, full retrieval, Patchify recovery, layer handoff, long loss.
3. Localization keyframes, optical-flow frames і presentation latency окремо.
4. DB build: processed/saved, frames/s, час global/Patchify/depth, peak RAM/VRAM.
5. Propagation: кількість features/edges, matching, optimization, holdout residual.

Метрики: p50/p95/p99/max wall latency; peak RAM/allocated/reserved VRAM; retrieval recall@k; verified-fix recall; false fixes; GPS median/p95 error; recovery time. Не приймати зміну лише за FPS чи кількістю inliers.

Наявні інструменти для цього: `scripts/benchmark_hardware.py`, `scripts/replay_localization_ground_truth.py`, `scripts/compare_replays.py`, `scripts/ab_localization.py`, `scripts/retrieval_metrics.py`, `scripts/benchmark_propagation.py`.

Окремий недолік діагностики: singleton Telemetry на atexit записує один спільний `logs/telemetry_report.json`. Під час цього аудиту unit-тести перезаписали попередній application report тестовими даними; початковий знімок відновлено. Потрібні reports із run-id/config-hash/process-kind і окремий test output directory, щоб не втрачати baseline.

## Заміна моделей: лише після виміряних змін вище

- Менший DINO input або менший backbone потребує перевірки retrieval accuracy. Поточний VLAD vocab має provenance під backbone/preprocessing/resolution; не можна просто замінити модель чи 336→224 і залишити бази без перевірки сумісності та переобчислення descriptors/vocabulary.
- XFeat уже підтримується, але його 64D descriptors не сумісні з ALIKED databases; matching route інший. Порівнювати як окрему повну конфігурацію з перебудованими БД, а не як hot swap.
- TensorRT export для активного DINOv3 повинен повертати patch tokens для VLAD, мати перевірені dynamic-batch profiles і той самий preprocessing. Наявний DINOv2 TRT wrapper не є готовим прискорювачем поточного DINOv3+VLAD pipeline.
- Depth scale у поточному online localization hint вимкнений. Depth однаково обчислюється при build; відключати build depth можна тільки після перевірки використання depth_scales у проєкті. Sparse cadence уже існує.
- YOLO вже nano segmentation і має TRT-гілку. Спочатку виміряти фактичний backend та latency, а не міняти модель наосліп.

## Рекомендована послідовність

1. Виправити й покрити перевірками єдиний Patchify batch path; окремо визначити бажану шкалу patch score.
2. Зняти ізольований baseline на реальному replay з фіксованим config/vocab/weights.
3. Patchify batch=4 та fallback-only search — окремими A/B.
4. Відкласти global/Patchify extraction до keyframe-save decision у DB build.
5. GPU VLAD/PCA та per-query GPU features/reference LRU — окремими змінами.
6. Temporal prior budget 1024/2048, DINO precision і LightGlue mp — незалежні експерименти.
7. Реальний warmup, compile DINOv3; потім bounded memory queues/cache і масштабування propagation.
8. Лише за недостатнього результату — інший backbone/input/XFeat/TRT із перевіркою сумісності БД.

Оцінки на кшталт «прискорить усе у 2–5 разів» тут не обґрунтовані: виграш залежить від частки recovery, відкинутих keyframes, розміру БД і вимог до точності. Виявлені зайві обчислення та помилки підтверджені кодом; числовий виграш потребує A/B на ваших відео.
