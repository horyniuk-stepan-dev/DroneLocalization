# План модернізації Drone Topometric Localization System

> **Статус:** Планування  
> **Дата створення:** 2026-03-25  
> **Горизонт виконання:** 3–4 місяці

---

## Зміст

- [Горизонт 1 — Quick Wins (1–3 тижні)](#горизонт-1--quick-wins-13-тижні)
- [Горизонт 2 — Структурні зміни (1–2 місяці)](#горизонт-2--структурні-зміни-12-місяці)
- [Горизонт 3 — Фундаментальна трансформація (2–4 місяці)](#горизонт-3--фундаментальна-трансформація-24-місяці)

---

## Горизонт 1 — Quick Wins (1–3 тижні)

### Фаза 5: Оптимізація підсистеми сегментації

#### 5.1 Пониження YOLO до nano-версії
- [ ] **5.1.1** Завантажити `yolo11n-seg.pt` та додати в корінь проєкту
- [ ] **5.1.2** Оновити `YoloConfig.model_path` у `config/config.py`:
  - `model_path: str = "yolo11n-seg.pt"`
  - `vram_required_mb: float = 200.0`
  - `description: str = "YOLOv11n-seg (Nano) for dynamic object masking"`
- [ ] **5.1.3** Провести порівняльне тестування якості маскування:
  - Вибрати 10 репрезентативних кадрів із наявного відео
  - Порівняти маски `yolo11x-seg` vs `yolo11n-seg` (IoU ≥ 0.85 допустимо)
  - Зафіксувати результати у таблиці (кадр, IoU, час інференсу)
- [ ] **5.1.4** Виміряти VRAM та швидкість після заміни:
  - `torch.cuda.max_memory_allocated()` до і після
  - Час побудови бази (80 кадрів) з x-seg vs n-seg

#### 5.2 TensorRT INT8 компіляція YOLO
- [ ] **5.2.1** Дослідити сумісність `ultralytics` з TensorRT INT8 PTQ
- [ ] **5.2.2** Створити скрипт `scripts/compile_yolo_trt.py`:
  - Завантажити `yolo11n-seg.pt`
  - Викликати `model.export(format="engine", int8=True, data="calibration_images/")`
  - Зберегти як `yolo11n-seg.engine`
- [ ] **5.2.3** Оновити `YOLOWrapper` для підтримки `.engine` файлів:
  - Спробувати завантажити `.engine` файл; якщо немає — fallback на `.pt`
  - Логувати `logger.info("TensorRT INT8 engine loaded")` при успіху
- [ ] **5.2.4** Перевірити коректність сегментації після INT8 квантування

#### 5.3 Полімерний інтерфейс маскування (підготовка до EfficientViT-SAM)
- [ ] **5.3.1** Створити абстрактний базовий клас `MaskingStrategy`:
  - Файл: `src/models/wrappers/masking_strategy.py`
  - Метод: `get_mask(frame_rgb: np.ndarray) -> np.ndarray`
- [ ] **5.3.2** Реалізувати `YOLOMaskingStrategy(MaskingStrategy)`:
  - Перемістити логіку з `YOLOWrapper` у стратегію
  - Зберегти існуючий інтерфейс `detect_and_mask_batch()`
- [ ] **5.3.3** Додати `masking_strategy: str = "yolo"` у `PreprocessingConfig`
- [ ] **5.3.4** Оновити `DatabaseBuilder` для використання `MaskingStrategy` замість прямого `YOLOWrapper`

#### 5.4 EfficientViT-SAM (опціонально, після валідації 5.1–5.3)
- [ ] **5.4.1** Встановити `efficient-sam` або відповідну бібліотеку
- [ ] **5.4.2** Реалізувати `EfficientViTSAMMaskingStrategy(MaskingStrategy)`
- [ ] **5.4.3** Порівняти якість маскування з YOLO на складних сценаріях (нетипове обладнання, тварини)
- [ ] **5.4.4** Додати прапорець `masking_strategy: str = "efficient_sam"` у конфіг

---

### Фаза 4: Оптимізація локального зіставлення (PoseLib)

#### 4.1 Заміна OpenCV гомографії на PoseLib
- [ ] **4.1.1** Встановити `poselib` через pip
- [ ] **4.1.2** Дослідити API `poselib`:
  - `poselib.estimate_homography()` — LO-RANSAC з 4-точковим розв'язувачем
  - `poselib.estimate_absolute_pose()` — P3P/P4P для 3D-точок
  - Порівняти параметри з `cv2.findHomography(method=USAC_MAGSAC)`
- [ ] **4.1.3** Оновити `GeometryTransforms.estimate_homography()` у `src/geometry/transformations.py`:
  - Замінити виклик `cv2.findHomography` на `poselib.estimate_homography`
  - Зберегти fallback на OpenCV якщо `poselib` не встановлений
  - Параметри: `ransac_threshold=3.0`, `min_inliers=10` (з конфігу)
- [ ] **4.1.4** Оновити `GeometryTransforms.estimate_affine_partial()`:
  - Дослідити чи PoseLib пропонує кращий RANSAC для afine
  - Якщо ні — залишити `cv2.estimateAffinePartial2D`
- [ ] **4.1.5** Тестування:
  - Запустити `python -m pytest tests/test_geometry_utils.py -v`
  - Порівняти inlier count та RMSE між OpenCV і PoseLib на тих самих парах точок
  - Виміряти час виконання (benchmark 1000 ітерацій)

#### 4.2 Підготовка до 3D-PnP (дослідження)
- [ ] **4.2.1** Дослідити доступність DEM (Digital Elevation Model) для тестових регіонів
- [ ] **4.2.2** Визначити формат зберігання 3D-координат ключових точок у LanceDB:
  - `kpt_gps_coords: list<float64, 3>` — (lat, lon, altitude_m)
- [ ] **4.2.3** Описати алгоритм заповнення z-координати:
  - При побудові БД: pixel → metric (через affine) → GPS (через pyproj) → lookup DEM
  - Fallback: altitude = 0.0 (плоский рельєф)
- [ ] **4.2.4** Створити прототип PnP-локалізації:
  - `poselib.estimate_absolute_pose(kp_2d_query, kp_3d_ref, camera_intrinsics)`
  - Порівняти точність з гомографією на тестових даних

#### 4.3 RDD як fallback extractor (дослідження)
- [ ] **4.3.1** Дослідити доступність pretrained моделей RDD
- [ ] **4.3.2** Описати інтерфейс інтеграції з `FeatureExtractor`
- [ ] **4.3.3** Оцінити VRAM-витрати та швидкість

---

### Фаза 6: TensorRT для DINOv2

#### 6.1 Компіляція DINOv2 → TensorRT FP16
- [ ] **6.1.1** Створити скрипт `scripts/compile_dinov2_trt.py`:
  - Завантажити `dinov2_vitl14` через `torch.hub`
  - `torch.onnx.export(model, dummy_input, "dinov2_vitl14.onnx", opset_version=17)`
  - Фіксований вхід: `(1, 3, 336, 336)`
- [ ] **6.1.2** Компілювати ONNX → TensorRT engine:
  - `trtexec --onnx=dinov2_vitl14.onnx --saveEngine=dinov2_vitl14_fp16.engine --fp16`
  - Зафіксувати час компіляції та розмір engine-файлу
- [ ] **6.1.3** Створити `TensorRTDINOv2Wrapper`:
  - Файл: `src/models/wrappers/trt_dinov2_wrapper.py`
  - Метод: `forward(image_tensor) -> np.ndarray` (1024-dim)
  - Зберігати engine-файл у `models/engines/`
- [ ] **6.1.4** Інтегрувати з `ModelManager.load_dinov2()`:
  - Спробувати завантажити TRT engine; якщо немає — fallback на PyTorch
  - Логувати рівень оптимізації: `logger.info("DINOv2 TensorRT FP16 engine loaded")`
- [ ] **6.1.5** Валідація:
  - Порівняти дескриптори PyTorch vs TRT (cosine similarity ≥ 0.999)
  - Виміряти speedup (очікуваний: 2–3x)

#### 6.2 ONNX Runtime для LightGlue
- [ ] **6.2.1** Дослідити `onnxruntime-gpu` сумісність з LightGlue:
  - Проблема: динамічні осі (`num_keypoints_0`, `num_keypoints_1`)
  - Рішення: `dynamic_axes={"keypoints0": {1: "num_kp0"}, "keypoints1": {1: "num_kp1"}}`
- [ ] **6.2.2** Створити ONNX-експорт LightGlue з ALIKED вагами:
  - Фіксувати `depth_confidence=-1`, `width_confidence=-1`
  - Тестувати на парах з різною кількістю kp (100, 500, 2000, 4096)
- [ ] **6.2.3** Створити `OnnxLightGlueWrapper`:
  - Файл: `src/models/wrappers/onnx_lightglue_wrapper.py`
  - Зберегти інтерфейс `match(query_features, ref_features) -> (mkpts_q, mkpts_r)`
- [ ] **6.2.4** Інтегрувати з `FeatureMatcher`:
  - Пріоритет: TRT > ONNX > PyTorch > Numpy L2

#### 6.3 Кешування engine-файлів
- [ ] **6.3.1** Додати `ModelsCacheConfig` у `config/config.py`:
  - `engine_cache_dir: str = "models/engines/"`
  - `engine_hash_gpu: bool = True`
- [ ] **6.3.2** Реалізувати логіку іменування engine-файлів:
  - `{model_name}_{gpu_name}_{trt_version}_{precision}.engine`
  - Автоматична перекомпіляція при зміні GPU або версії TensorRT
- [ ] **6.3.3** Додати команду CLI для попередньої компіляції:
  - `python -m scripts.compile_engines --all`

#### 6.4 ALIKED C++ API (дослідження)
- [ ] **6.4.1** Дослідити проєкт `aliked_cpp` та його Python-біндінги
- [ ] **6.4.2** Оцінити складність інтеграції через `nanobind`
- [ ] **6.4.3** Створити бенчмарк: Python ALIKED vs C++ ALIKED (latency, throughput)

#### 6.5 Адаптивне управління живленням (Jetson-specific)
- [ ] **6.5.1** Додати `EdgeConfig` у `config/config.py`:
  - `enable_nvpmodel: bool = False`
  - `battery_saver_threshold: float = 0.2`
- [ ] **6.5.2** Реалізувати `PowerManager`:
  - Перевірка `nvpmodel` при старті
  - Перемикання режимів: MAXN ↔ 10W
  - Fallback маскування (frame differencing) при низькому заряді

---

## Горизонт 2 — Структурні зміни (1–2 місяці)

### Фаза 1: Заміна HDF5 → LanceDB

#### 1.1 Проєктування схеми LanceDB
- [ ] **1.1.1** Встановити `lancedb` через pip
- [ ] **1.1.2** Дослідити API LanceDB:
  - Створення таблиць з Apache Arrow схемою
  - Векторний пошук: `table.search(query_vec).limit(k).to_list()`
  - Scalar filtering: `table.search(query_vec).where("altitude_m BETWEEN 50 AND 100")`
  - IVF_PQ індексація: `table.create_index(metric="cosine", num_partitions=..., num_sub_vectors=64)`
- [ ] **1.1.3** Визначити фінальну схему таблиці:
  ```
  frame_id:        int32          — оригінальний індекс кадру
  global_desc:     vector[1024]   — DINOv2 глобальний дескриптор
  local_kpts:      list<float32>  — ALIKED keypoints (N×2, flattened)
  local_desc:      list<float32>  — ALIKED descriptors (N×128, flattened)
  kp_count:        int32          — кількість keypoints
  pose_matrix:     list<float64>  — кумулятивна поза (3×3, flattened 9)
  lat:             float64        — GPS широта (nullable, after calibration)
  lon:             float64        — GPS довгота (nullable, after calibration)
  altitude_m:      float32        — висота БПЛА (nullable)
  affine_matrix:   list<float32>  — pixel→metric (2×3, flattened 6, nullable)
  frame_valid:     bool           — чи калібрований кадр
  frame_rmse:      float32        — QA RMSE (nullable)
  frame_disagree:  float32        — QA disagreement (nullable)
  ```
- [ ] **1.1.4** Створити прототип: записати 80 кадрів, порівняти з HDF5 (розмір, швидкість)

#### 1.2 Рефакторинг DatabaseBuilder
- [ ] **1.2.1** Створити `src/database/lancedb_builder.py`:
  - Клас `LanceDBBuilder` з тим самим публічним API, що й `DatabaseBuilder`
  - `build_from_video()` — основний метод
- [ ] **1.2.2** Замінити `create_hdf5_structure()` → `create_lancedb_schema()`:
  - Ініціалізація LanceDB з'єднання: `db = lancedb.connect(output_dir)`
  - Створення таблиці: `db.create_table("frames", schema=...)`
- [ ] **1.2.3** Замінити `save_frame_data()` → батчевий append:
  - Накопичувати дані в `list[dict]` по 64 записи
  - `table.add(batch)` — Apache Arrow нативний batch insert
  - flush при завершенні або по `finally`
- [ ] **1.2.4** Зберегти HDF5-версію для зворотної сумісності:
  - Фабричний паттерн: `get_builder(format="lancedb"|"hdf5")`
  - Конфіг: `DatabaseConfig.storage_format: str = "lancedb"`
- [ ] **1.2.5** Тести:
  - Запустити `python -m pytest tests/integration/test_pipeline.py -v`
  - Порівняти `DatabaseLoader` vs `LanceDBLoader` на тих самих вхідних даних

#### 1.3 Рефакторинг DatabaseLoader
- [ ] **1.3.1** Створити `src/database/lancedb_loader.py`:
  - Клас `LanceDBLoader` з тим самим публічним API, що й `DatabaseLoader`
- [ ] **1.3.2** Реалізувати ключові методи:
  - `get_num_frames()` → `table.count_rows()`
  - `get_local_features(frame_id)` → `table.search().where(f"frame_id = {fid}").limit(1)`
  - `get_frame_affine(frame_id)` → scalar query
  - `get_global_descriptors()` → `table.to_pandas()["global_desc"]`
- [ ] **1.3.3** Реалізувати RAM-hot кешування:
  - При ініціалізації: завантажити `global_desc` і `pose_matrix` у numpy-масиви
  - `self.global_descriptors`, `self.frame_poses` — аналогічно до HDF5 loader
- [ ] **1.3.4** Забезпечити зворотну сумісність:
  - Фабрика `get_loader(path)`: якщо `path.endswith(".h5")` → HDF5, інакше → LanceDB

#### 1.4 Адаптація Localizer до LanceDB
- [ ] **1.4.1** Замінити `FastRetrieval(FAISS)` на `LanceDBRetrieval`:
  - `table.search(query_desc).metric("cosine").limit(top_k).to_list()`
  - Один запит повертає frame_id, score, affine_matrix, lat, lon
- [ ] **1.4.2** Додати геопросторову фільтрацію (scalar pre-filter):
  - Якщо доступна приблизна GPS → `where("lat BETWEEN ... AND ... AND lon BETWEEN ... AND ...")`
  - Зменшення простору пошуку у 10–100x
- [ ] **1.4.3** Створення IVF_PQ індексу (для бази > 500K кадрів):
  - Після побудови БД: `table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=64)`
  - Конфіг: `DatabaseConfig.lancedb_index_threshold: int = 500_000`
  - Для баз < 500K — brute-force пошук (оптимізований SIMD)
- [ ] **1.4.4** Оновити `CalibrationPropagationWorker`:
  - Збереження пропагованих afine назад у LanceDB: `table.update(where=..., values=...)`

#### 1.5 Міграція існуючих баз
- [ ] **1.5.1** Створити скрипт `scripts/migrate_hdf5_to_lancedb.py`:
  - Читає `database.h5` через `DatabaseLoader`
  - Записує в LanceDB через `LanceDBBuilder`
  - CLI: `python -m scripts.migrate_hdf5_to_lancedb --input path/to/database.h5`
- [ ] **1.5.2** Тестувати міграцію на реальній БД (80 кадрів)

---

### Фаза 3: Модернізація VPR (SALAD + HE-VPR)

#### 3.1 SALAD-голова поверх DINOv2
- [ ] **3.1.1** Дослідити відкриті реалізації SALAD:
  - Оригінальний репозиторій, pretrained ваги
  - Розмір моделі (~2–5М параметрів)
- [ ] **3.1.2** Створити `src/models/wrappers/salad_head.py`:
  - Клас `SALADHead(nn.Module)`:
    - `__init__(dim=1024, num_clusters=64, epsilon=0.05)` — Sinkhorn параметри
    - `dustbin_bias` — навчаємий скаляр
    - `forward(patch_tokens, h_patches, w_patches) -> global_descriptor`
- [ ] **3.1.3** Алгоритм Сінкхорна:
  - Вхід: `patch_tokens (576, 1024)`, cluster centers `(64+1, 1024)`
  - Обчислити матрицю спорідненості `(576, 65)`
  - Ітерації Сінкхорна (3–5 разів): рядкова norm → стовпцева norm
  - Soft-assignment → зважена агрегація → L2-нормалізація
- [ ] **3.1.4** Механізм dustbin:
  - Стовпець 65 (dustbin) поглинає неінформативні токени
  - Токени неба, монотонних поверхонь → dustbin → виключаються з дескриптора
- [ ] **3.1.5** Інтеграція з `FeatureExtractor`:
  - Замінити гілку `if self.cesp_module is not None` на `if self.salad_head is not None`
  - `CespConfig` → `SaladConfig(enabled=True, num_clusters=64, sinkhorn_iters=3)`
  - Зберегти fallback на CLS-токен

#### 3.2 Донавчання SALAD
- [ ] **3.2.1** Підготувати датасет:
  - Зібрати пари (anchor, positive, negative) з аерофотознімків
  - Формат: triplet або contrastive loss
  - Мінімум 10K трійок для початкового донавчання
- [ ] **3.2.2** Визначити стратегію заморожування:
  - Перші 22 блоки DINOv2 — заморожені (`requires_grad=False`)
  - Останні 2 блоки + SALAD-голова — навчаються
- [ ] **3.2.3** Створити скрипт навчання `scripts/train_salad.py`:
  - Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
  - Loss: Multi-Similarity Loss або Contrastive Loss
  - Epochs: 20–50, batch_size: 32
- [ ] **3.2.4** Зберегти ваги SALAD окремо: `models/salad_head.pth`
- [ ] **3.2.5** Валідація:
  - Recall@1, Recall@5 на тестовому наборі аерофотознімків
  - Порівняти з baseline CLS-токен DINOv2

#### 3.3 HE-VPR адаптери
- [ ] **3.3.1** Реалізувати MLP оцінки висоти:
  - Файл: `src/models/wrappers/height_estimator.py`
  - Вхід: CLS-токен DINOv2 (1024-dim)
  - Вихід: altitude_m (float32)
  - Архітектура: Linear(1024, 256) → ReLU → Linear(256, 1)
- [ ] **3.3.2** Реалізувати центрально-зважене маскування:
  - Гауссове ядро `G(i,j)` з σ пропорційним оціненій висоті
  - Зважити `patch_tokens *= G` перед SALAD агрегацією
- [ ] **3.3.3** Висотна підбаза в LanceDB:
  - 5 висотних шарів: 0–50, 50–100, 100–200, 200–400, 400+ м
  - Scalar filter: `table.search(q).where("altitude_m BETWEEN 50 AND 100")`
- [ ] **3.3.4** Навчання height estimator:
  - Датасет: аерофотознімки з відомою висотою (EXIF, PX4 logs)
  - Loss: MSE
  - Epochs: 30, lr=1e-3

---

## Горизонт 3 — Фундаментальна трансформація (2–4 місяці)

### Фаза 2: Factor Graph Optimization (GTSAM iSAM2)

#### 2.1 Встановлення та налаштування GTSAM
- [ ] **2.1.1** Встановити `gtsam` через pip:
  - `pip install gtsam` (Python wrapper)
  - Перевірити: `import gtsam; print(gtsam.__version__)`
- [ ] **2.1.2** Дослідити API GTSAM:
  - `gtsam.Pose3`, `gtsam.Point3` — типи даних
  - `gtsam.NonlinearFactorGraph` — фактор-граф
  - `gtsam.ISAM2` — інкрементальний оптимізатор
  - `gtsam.noiseModel` — моделі шуму (Gaussian, Robust Cauchy)

#### 2.2 Визначення типів факторів
- [ ] **2.2.1** Створити `src/tracking/factor_graph.py`:
  - Клас `DroneFactorGraph`:
    - `isam = gtsam.ISAM2(params)`
    - `graph = gtsam.NonlinearFactorGraph()`
    - `initial_estimate = gtsam.Values()`
- [ ] **2.2.2** Реалізувати GPSFactor:
  - `gtsam.PriorFactorPose3(key, pose, noise_model)`
  - Noise: `gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Cauchy(5.0), base_noise)`
  - base_noise: `gtsam.noiseModel.Diagonal.Sigmas([gps_noise_m]*6)`
  - Confidence від VPR → масштабує noise: `sigma = base / confidence`
- [ ] **2.2.3** Реалізувати BetweenFactor (одометрія):
  - `gtsam.BetweenFactorPose3(key_prev, key_curr, relative_pose, noise_odom)`
  - `relative_pose` обчислюється з гомографії ALIKED/LightGlue
  - noise: `gtsam.noiseModel.Diagonal.Sigmas([odom_noise_m]*6)`
- [ ] **2.2.4** Реалізувати ImuFactor (якщо доступний IMU):
  - `gtsam.CombinedImuFactor(key_pose, key_vel, key_bias, ...)`
  - `PreintegratedCombinedMeasurements` для преінтеграції
  - Noise: `accel_noise`, `gyro_noise` із конфігу

#### 2.3 Інкрементальне оновлення (iSAM2)
- [ ] **2.3.1** Реалізувати метод `add_measurement(timestamp, gps_pos, confidence)`:
  - Створити змінну `X(t)` — Pose3
  - Додати GPSFactor з коваріацією на основі confidence
  - Додати BetweenFactor від попереднього кадру
  - `isam.update(graph, initial_estimate)`
  - `result = isam.calculateEstimate()`
  - Повернути оптимізовану позицію: `result.atPose3(X(t))`
- [ ] **2.3.2** Реалізувати sliding window:
  - Зберігати останні N вузлів (N=100)
  - Маргіналізувати старі вузли для запобігання зростанню графа
- [ ] **2.3.3** Обробка outliers:
  - Cauchy loss автоматично знижує вагу GPS-факторів з великою похибкою
  - Логувати: `logger.warning(f"GPS factor weight reduced: residual {res:.1f}m")`

#### 2.4 Рефакторинг TrackingConfig та TrackingWorker
- [ ] **2.4.1** Додати `GTSAMConfig` у `config/config.py`:
  ```python
  class GTSAMConfig(BaseModel):
      enabled: bool = False  # False = Kalman (зворотна сумісність)
      gps_noise_m: float = 5.0
      odom_noise_m: float = 1.0
      imu_noise_accel: float = 0.1
      imu_noise_gyro: float = 0.01
      cauchy_param: float = 5.0
      sliding_window_size: int = 100
  ```
- [ ] **2.4.2** Оновити `TrackingConfig`:
  - Додати поле `gtsam: GTSAMConfig = GTSAMConfig()`
  - Зберегти `kalman_*` параметри для fallback
- [ ] **2.4.3** Оновити `TrackingWorker.run()`:
  - Якщо `gtsam.enabled`: використовувати `DroneFactorGraph`
  - Інакше: використовувати існуючий `TrajectoryFilter` + `OutlierDetector`
- [ ] **2.4.4** Тестування:
  - Порівняти траєкторії Kalman vs GTSAM на записаному відео
  - Метрики: ATE (Absolute Trajectory Error), RPE (Relative Pose Error)
  - Simulation: додати штучні outliers та виміряти recovery time

#### 2.5 IMU Preintegration (якщо доступний IMU-сигнал)
- [ ] **2.5.1** Визначити формат вхідних IMU-даних:
  - CSV або ROS bag: timestamp, accel_x/y/z, gyro_x/y/z
- [ ] **2.5.2** Реалізувати `ImuPreintegrator`:
  - Між кадрами: накопичити IMU-вимірювання
  - `PreintegratedCombinedMeasurements.integrateMeasurement(accel, gyro, dt)`
- [ ] **2.5.3** Інтегрувати з `DroneFactorGraph.add_measurement()`:
  - Додати ImuFactor тільки якщо IMU-дані присутні
  - Fallback: тільки GPS + Odometry factors

---

### Фаза 7: PyOpenGL замість QWebEngine

#### 7.1 Архітектура нової карти
- [ ] **7.1.1** Створити `src/gui/widgets/gl_map_widget.py`:
  - Клас `GLMapWidget(QOpenGLWidget)`:
    - `initializeGL()`, `resizeGL()`, `paintGL()`
    - Координатна система: Web Mercator (EPSG:3857)
- [ ] **7.1.2** Визначити шейдерну програму:
  - Vertex shader: GPS → screen coordinates
  - Fragment shader: колір точки на основі confidence
  - Geometry shader: FOV-конус (трикутний фан)
- [ ] **7.1.3** Управління камерою карти:
  - Pan (перетягування мишкою)
  - Zoom (колесико → зміна zoom level)
  - Center on target (подвійний клік → auto-pan)

#### 7.2 Тайловий менеджер
- [ ] **7.2.1** Створити `src/gui/tile_manager.py`:
  - Клас `TileManager`:
    - `get_tile(z, x, y) -> QImage | None` — з кешу або мережі
    - Асинхронне завантаження: `QNetworkAccessManager`
    - Кеш на диску: `~/.cache/drone_localization/tiles/{z}/{x}/{y}.png`
- [ ] **7.2.2** Реалізувати Mercator проєкцію:
  - `gps_to_tile_xy(lat, lon, zoom) -> (tile_x, tile_y, pixel_x, pixel_y)`
  - `tile_xy_to_gps(tile_x, tile_y, zoom) -> (lat, lon)`
- [ ] **7.2.3** Рендеринг тайлів як OpenGL текстур:
  - `glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)`
  - Texture atlas або per-tile текстури
- [ ] **7.2.4** Offline-режим:
  - Попереднє завантаження регіону: `scripts/download_tiles.py --bbox lat1,lon1,lat2,lon2 --zoom 10-18`

#### 7.3 Рендеринг траєкторії
- [ ] **7.3.1** Створити VAO/VBO для траєкторії:
  - Vertex: `(x_metric, y_metric, confidence, timestamp)`
  - `glDrawArrays(GL_LINE_STRIP, 0, N)` — один виклик для всієї траєкторії
- [ ] **7.3.2** Динамічне дописування точок:
  - `glBufferSubData(GL_ARRAY_BUFFER, offset, size, data)` — без пересоздання
  - Maximal pre-allocated buffer: 100K точок
- [ ] **7.3.3** Колоризація на основі confidence:
  - Fragment shader: `color = mix(red, green, confidence)` — низька → червона, висока → зелена
  - Анімація: поточна позиція — пульсуючий маркер

#### 7.4 FOV-конус через геометричний шейдер
- [ ] **7.4.1** Реалізувати geometry shader:
  - Вхід: `(position, yaw_angle, fov_distance)`
  - Вихід: трикутний фан з 8–16 сегментів
  - Напівпрозорий полігон: `alpha=0.3`
- [ ] **7.4.2** Синхронізація з `TrackingWorker`:
  - Signal `fov_updated(corners: list[tuple])` → оновити GPU-буфер FOV

#### 7.5 CUDA-OpenGL interop (опціонально)
- [ ] **7.5.1** Дослідити `cudaGraphicsGLRegisterBuffer`:
  - Передача YOLO-маски безпосередньо з CUDA в OpenGL текстуру
  - Zero-copy відображення маски поверх відео
- [ ] **7.5.2** Реалізувати як опціональну функцію:
  - Конфіг: `GuiConfig.cuda_gl_interop: bool = False`

#### 7.6 Інтеграція з MainWindow
- [ ] **7.6.1** Замінити `MapWidget(QWebEngineView)` на `GLMapWidget(QOpenGLWidget)`:
  - Зберегти існуючий `MapWidget` як fallback
  - Конфіг: `GuiConfig.map_renderer: str = "opengl"` або `"webengine"`
- [ ] **7.6.2** Адаптувати сигнали:
  - `map_clicked(lat, lon)` — для калібрування
  - `trajectory_updated(points)` — для відображення
  - `clear_trajectory()` — очищення
- [ ] **7.6.3** Тестування:
  - FPS benchmark: 60 FPS target
  - Stress test: 50K точок траєкторії
  - Порівняння з WebEngine: latency, memory

---

## Залежності між фазами

```
Фаза 5 (YOLO nano) ──────────────────────────────────┐
Фаза 4 (PoseLib) ─────────────────────────────────────┤
Фаза 6.1 (DINOv2 TRT) ────────────────────────────────┤ Горизонт 1
                                                       │ (незалежні)
                                                       │
Фаза 1 (LanceDB) ─────────┬───────────────────────────┤ Горизонт 2
                           │                           │
Фаза 3 (SALAD) ───────────┤  ← SALAD інтегрується     │
                           │    в LanceDB retrieval     │
Фаза 3.3 (HE-VPR) ────────┘  ← потребує altitude_m    │
                              у LanceDB                │
                                                       │
Фаза 2 (GTSAM) ───────────────────────────────────────┤ Горизонт 3
Фаза 7 (PyOpenGL) ─────────────────────────────────────┤ (незалежні
Фаза 6.4 (ALIKED C++) ─────────────────────────────────┘  між собою)
```

---

## Ключові ризики

| Ризик | Фаза | Імовірність | Вплив | Мітігація |
|-------|------|-------------|-------|-----------|
| DINOv2 INT8 деградація | 6.1 | Низька | Високий | Суворо FP16, НІКОЛИ INT8 для ViT |
| LightGlue динамічні осі в TRT | 6.2 | Висока | Середній | ONNX Runtime замість TensorRT |
| LanceDB не підтримує ACID-запис | 1.2 | Низька | Середній | Batch-append + журнал відновлення |
| GTSAM Python wrapper повільний | 2.1 | Середня | Середній | Профілювання; C++ API через pybind |
| SALAD потребує аеродатасет | 3.2 | Висока | Високий | Використати GSV-Cities або MSLS |
| OpenGL некросплатформний | 7.1 | Середня | Середній | Тестувати на Windows + Linux |
| PoseLib відсутній на PyPI | 4.1 | Низька | Низький | `pip install poselib` або build from source |

---

## Метрики успіху

| Метрика | Поточне значення | Цільове значення | Фаза |
|---------|-----------------|-----------------|------|
| VRAM (YOLO) | ~1200 MB | ~200 MB | 5.1 |
| Localization latency | ~200 ms | ~50 ms | 1.4, 6.1 |
| Retrieval (ANN top-12) | ~5 ms (80 frames) | <20 ms (1M frames) | 1.4 |
| Recall@1 (VPR) | Baseline CLS | +10–15% (SALAD) | 3.2 |
| Trajectory ATE | Baseline Kalman | -30–50% (GTSAM) | 2.4 |
| Map FPS | ~15 FPS (WebEngine) | 60 FPS (OpenGL) | 7.3 |
| DINOv2 inference | ~30 ms | ~10 ms (TRT FP16) | 6.1 |
