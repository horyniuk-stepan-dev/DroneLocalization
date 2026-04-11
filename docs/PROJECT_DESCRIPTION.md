# Drone Topometric Localization System

## 1. Загальний опис

**Drone Topometric Localization System** — це десктопний застосунок для визначення GPS-координат дрона в реальному часі на основі візуальних ознак відеопотоку. Система порівнює поточний кадр камери дрона з попередньо побудованою базою даних референсних кадрів, прив'язаних до географічних координат, і визначає положення дрона без використання GNSS-приймача.

### Ключові можливості

- **Побудова бази даних** з референсного відео з екстракцією локальних (ALIKED) та глобальних (DINOv2) ознак
- **Калібрування** — прив'язка кадрів бази до GPS координат через афінні трансформації
- **Пропагація GPS** — розповсюдження координат від якірних кадрів на всю базу через гомографічні ланцюги
- **Локалізація** — визначення GPS-позиції дрона за поточним кадром камери
- **Відстеження** — безперервне відстеження траєкторії з фільтрацією Калмана
- **Візуалізація** — інтерактивна карта OpenStreetMap з траєкторією та FOV
- **Експорт** — збереження результатів у CSV, GeoJSON, KML

### Стек технологій

| Категорія | Технології |
|-----------|-----------|
| Мова | Python 3.11 |
| GUI | PyQt6, PyQt6-WebEngine (карта) |
| Deep Learning | PyTorch 2.2+, CUDA |
| Computer Vision | OpenCV 4.9+, Kornia |
| Neural Networks | ALIKED, DINOv2 (ViT-L/14), LightGlue, YOLOv11n-seg |
| База даних | HDF5 (h5py) |
| Геопроекція | pyproj (WGS84 → UTM / Web Mercator) |
| Конфігурація | Pydantic v2 |
| Тестування | pytest, pytest-qt |
| Code Quality | Ruff (лінтинг + форматування), pre-commit |

---

## 2. Архітектура системи

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main.py                                     │
│                     (точка входу програми)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                      MainWindow                                      │
│   ┌─────────────┬───────────────┬──────────────┬─────────────────┐   │
│   │ DatabaseMixin│CalibrationMixin│TrackingMixin │ PanoramaMixin   │   │
│   └──────┬──────┴───────┬───────┴──────┬───────┴────────┬────────┘   │
│          │              │              │                │             │
│   ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐ ┌──────▼───────┐    │
│   │ VideoWidget │ │ MapWidget │ │ControlPanel│ │  Dialogs     │    │
│   └─────────────┘ └───────────┘ └────────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────────────┐
        │                  │                          │
┌───────▼───────┐  ┌───────▼────────┐  ┌──────────────▼──────────────┐
│  Workers      │  │  Core Logic    │  │  Models & Wrappers          │
│  (QThread)    │  │                │  │                             │
│ ─────────────── │ ──────────────── │ ────────────────────────────── │
│ DatabaseWorker│  │ DatabaseBuilder│  │ ModelManager                │
│ TrackingWorker│  │ DatabaseLoader │  │   ├── YOLO (segmentation)   │
│ PropagWorker  │  │ Localizer      │  │   ├── ALIKED (keypoints)    │
│ PanoramaWorker│  │ Matcher        │  │   ├── DINOv2 (global desc)  │
│               │  │ Calibration    │  │   ├── LightGlue (matching)  │
│               │  │ Coordinates    │  │   ├── XFeat (fallback)      │
│               │  │ Transformations│  │   └── CESP (optional)       │
└───────────────┘  └────────────────┘  └─────────────────────────────┘
```

---

## 3. Структура проєкту

```
DroneLocalization/
├── main.py                     # Точка входу
├── config/
│   └── config.py               # Pydantic-конфігурація (AppConfig)
├── src/
│   ├── calibration/
│   │   └── multi_anchor_calibration.py  # Мультиякірне калібрування + PCHIP
│   ├── core/
│   │   ├── project.py           # ProjectManager — створення/завантаження проєктів
│   │   ├── project_registry.py  # Реєстр останніх проєктів
│   │   └── export_results.py    # Експорт у CSV/GeoJSON/KML
│   ├── database/
│   │   ├── database_builder.py  # Побудова HDF5 бази з відео
│   │   └── database_loader.py   # Читання HDF5 бази (v1 + v2)
│   ├── geometry/
│   │   ├── coordinates.py       # Конвертер GPS ↔ метрична система
│   │   ├── transformations.py   # Гомографії, афінні трансформації (MAGSAC++)
│   │   ├── affine_utils.py      # Decompose/compose афінних матриць
│   │   └── pose_graph_optimizer.py # Оптимізація траєкторії (iSAM2/GTSAM)
│   ├── gui/
│   │   ├── main_window.py       # Головне вікно (QMainWindow + mixins)
│   │   ├── dialogs/
│   │   │   ├── calibration_dialog.py   # Діалог калібрування (5+ GCP)
│   │   │   ├── new_mission_dialog.py   # Діалог нової місії
│   │   │   └── open_project_dialog.py  # Діалог відкриття проєкту
│   │   ├── widgets/
│   │   │   ├── control_panel.py  # Панель інструментів (ліва бічна)
│   │   │   ├── map_widget.py     # Leaflet/OSM карта (QWebEngineView)
│   │   │   └── video_widget.py   # Відображення відео/кадрів (QGraphicsView)
│   │   └── mixins/
│   │       ├── database_mixin.py      # Логіка генерації/завантаження БД
│   │       ├── calibration_mixin.py   # Логіка калібрування та пропагації
│   │       ├── tracking_mixin.py      # Логіка відстеження
│   │       └── panorama_mixin.py      # Логіка панорами
│   ├── localization/
│   │   ├── localizer.py         # Основний алгоритм локалізації
│   │   └── matcher.py           # FeatureMatcher + FastRetrieval
│   ├── models/
│   │   ├── model_manager.py     # Lifecycle моделей (VRAM, lazy loading)
│   │   └── wrappers/
│   │       ├── aliked_wrapper.py      # Обгортка ALIKED
│   │       ├── feature_extractor.py   # Єдиний екстрактор (ALIKED + DINOv2)
│   │       ├── yolo_wrapper.py        # YOLOv11n-seg з micro-batching
│   │       ├── cesp_module.py         # CESP для покращення DINOv2 (opt.)
│   │       ├── masking_strategy.py    # Стратегії маскування (YOLO/SAM)
│   │       └── trt_dinov2_wrapper.py  # TensorRT прискорення для DINOv2
│   ├── tracking/
│   │   ├── kalman_filter.py     # Фільтр Калмана для GPS-траєкторії
│   │   └── outlier_detector.py  # Детектор аномалій (стрибки координат)
│   ├── utils/
│   │   ├── image_preprocessor.py  # CLAHE preprocessing
│   │   ├── image_utils.py         # OpenCV ↔ QPixmap конвертери
│   │   └── logging_utils.py       # Loguru setup
│   └── workers/
│       ├── database_worker.py              # QThread для побудови БД
│       ├── tracking_worker.py              # QThread для live tracking
│       ├── calibration_propagation_worker.py # QThread для пропагації GPS
│       ├── panorama_worker.py              # QThread для генерації панорами
│       └── panorama_overlay_worker.py      # QThread для overlay на карту
├── tests/                       # pytest (52 тести)
├── docs/                        # Документація
├── build/                       # PyInstaller артефакти
└── pyproject.toml               # Конфігурація проєкту, ruff, pytest
```

---

## 4. Конвеєр обробки (Pipeline)

### 4.1 Побудова бази даних

```
Вхідне відео (MP4)
    │
    ├── [1] Декодування кадрів (OpenCV, кожен N-й кадр)
    │        └── frame_step=3 (конфігурується)
    │
    ├── [2] YOLO v11n-seg (сегментація динамічних об'єктів)
    │        └── Генерація бінарної маски (авто, людей, тварин)
    │        └── Micro-batching (yolo_batch_size=2)
    │
    ├── [3] ALIKED (локальні ключові точки + дескриптори)
    │        └── До 4096 keypoints, 128-dim descriptors
    │        └── FP16 mixed precision для VRAM-ефективності
    │
    ├── [4] DINOv2 ViT-L/14 (глобальний дескриптор)
    │        └── 1024-dim descriptor для place recognition
    │        └── Input: 336×336 (центральний crop)
    │
    ├── [5] Inter-frame Homography (накопичення пози)
    │        └── Матчинг сусідніх кадрів → H_step → current_pose
    │
    ├── [6] Adaptive Keyframe Selection (П4)
    │        └── Пропускає кадри без значного руху
    │        └── min_translation=15px, min_rotation=1.5°
    │
    └── [7] Збереження у HDF5 v2
             └── Pre-allocated масиви з LZF компресією
             └── float16 для дескрипторів (50% менше RAM)
             └── Chunk-based I/O (64 кадри на chunk)
```

### 4.2 Калібрування

```
Користувач відкриває діалог калібрування
    │
    ├── [1] Вибір якірного кадру (слайдер або spinbox)
    │
    ├── [2] Розставлення ≥5 GCP (Ground Control Points)
    │        └── Клік на відео → координати пікселя
    │        └── Клік на карті → GPS координати
    │
    ├── [3] Обчислення афінної матриці (2×3)
    │        └── Partial Affine (4 DoF) — базовий
    │        └── Full Affine (6 DoF) — якщо покращує RMSE >15%
    │        └── QA: RMSE, медіанна та макс. похибка в метрах
    │
    ├── [4] Збереження у calibration.json
    │        └── orjson serialization (v2.2, PCHIP-ready)
    │
    └── [5] Пропагація GPS на всю базу
             └── Гомографічні ланцюги через збережені pose-матриці
             └── Блендель якщо кадр між двома якорями
             └── Паралельна обробка сегментів (ThreadPoolExecutor)
             └── Результат: frame_affine (2×3) для кожного кадру
```

### 4.3 Локалізація

```
Вхідний кадр камери дрона
    │
    ├── [1] DINOv2 Global Retrieval
    │        └── Порівняння з усіма кадрами бази (cosine similarity)
    │        └── Top-K=12 кандидатів
    │        └── Авто-ротація: перебираємо 0°, 90°, 180°, 270°
    │
    ├── [2] ALIKED + LightGlue Local Matching
    │        └── Для кожного кандидата: keypoint matching
    │        └── Homography estimation (MAGSAC++)
    │        └── Early stop при ≥40 inliers
    │
    ├── [3] Coordinate Projection
    │        └── Query center → Ref frame (via Homography)
    │        └── Ref pixels → Metric coords (via Affine)
    │        └── Metric → GPS (via pyproj)
    │
    ├── [4] Outlier Detection
    │        └── Перевірка стрибків координат
    │        └── max_speed=60 м/с, поліноміальний детектор
    │
    ├── [5] Kalman Filter
    │        └── Згладжування GPS-траєкторії
    │        └── process_noise=2.0, measurement_noise=5.0
    │
    └── [6] FOV Projection
             └── 4 кути кадру → GPS полігон
             └── Відображення на карті як overlay
```

---

## 5. Детальна логіка роботи

### 5.1 Побудова бази даних (`DatabaseBuilder`)

#### Producer–Consumer архітектура

Побудова БД використовує двопотокову модель з чергою (`Queue`):

```
┌────────────────┐     Queue(32)      ┌────────────────────┐
│  Reader Thread │ ──────────────────► │  Main Thread       │
│  (декодування) │   (idx, frame)     │  (YOLO + features) │
└────────────────┘                    └────────────────────┘
```

1. **Reader Thread** — декодує кадри з відео через OpenCV з кроком `frame_step` (кожен 3-й кадр). Кадри потрапляють у чергу розміром `prefetch_queue_size=32`, що дозволяє GPU обробляти поки CPU декодує наступні.

2. **Main Thread** — споживає кадри з черги і обробляє їх у такому порядку:

#### Крок 1: YOLO сегментація (micro-batching)

Кадри накопичуються в масів `pending_frames` до досягнення `yolo_batch_size=2`. Потім:

```python
# YOLOWrapper.detect_and_mask_batch()
results = model.predict(frames_batch, classes=[0,1,2,...])  # авто, люди, тварини
for result in results:
    binary_mask = об'єднання всіх полігонів сегментації
    static_mask = 255 - binary_mask  # інвертовано: 255 = статичний фон
```

**Навіщо:** маска видаляє ключові точки з динамічних об'єктів (автомобілі, пішоходи), щоб вони не заважали при matching.

#### Крок 2: Екстракція ознак (`FeatureExtractor`)

Для кожного кадру паралельно витягуються два типи ознак:

**Локальні (ALIKED):**
```
image (RGB) → CLAHE preprocessing → ALIKED CNN
    → keypoints: (N, 2) — координати точок
    → descriptors: (N, 128) — локальні дескриптори (float16)
```

Після ALIKED застосовується маска YOLO — ключові точки на динамічних об'єктах відкидаються векторизованою операцією:
```python
valid = static_mask[iy, ix] > 128  # True = статичний фон
keypoints = keypoints[valid]       # залишаємо лише "чисті" точки
```

**Глобальний (DINOv2 ViT-L/14):**
```
image → Resize(336×336) → ImageNet normalize → DINOv2
    → CLS token: (1024,) — один вектор на весь кадр
```

Глобальний дескриптор використовується для швидкого пошуку (place recognition), локальні — для точної прив'язки (geometric verification).

#### Крок 3: Накопичення пози (Inter-frame Homography)

Між сусідніми кадрами обчислюється гомографія `H_step` через ALIKED matching:

```
H_0 = I (одинична 3×3)
H_1 = H_0 @ H_step(0→1)
H_2 = H_1 @ H_step(1→2)
...
H_n = H_{n-1} @ H_step(n-1→n)
```

Результат — матриця `frame_pose` (3×3) для кожного кадру, яка кодує кумулятивне переміщення камери відносно першого кадру.

#### Крок 4: Adaptive Keyframe Selection

Перед збереженням кожного кадру система аналізує `H_step` і вирішує, чи був значний рух:

```python
# Декомпозиція гомографії H_step:
translation = sqrt(tx² + ty²)  # зсув у пікселях
rotation = |atan2(H[1,0], H[0,0])| * (180/π)  # кут обертання

save_this_frame = (translation > 15px) or (rotation > 1.5°)
```

Кадри без значного руху пропускаються — вони ідентичні попередніму і лише витрачають місце в БД.

#### Крок 5: Збереження у HDF5 v2

Дані зберігаються за оригінальним індексом кадру (`p_idx`), а не послідовною нумерацією. Це критично для коректної роботи калібрування/пропагації:

```python
self.save_frame_data(p_idx, features, current_pose)
# frame_id=79 → slot 79 в масиві
# Пропущені кадри мають kp_counts=0
```

---

### 5.2 Калібрування (`MultiAnchorCalibration`)

#### Процес розстановки якорів

Користувач вибирає кадр і виставляє мінімум 5 пар точок:
- **Pixel coordinates** — клік на відео (x, y у пікселях)
- **GPS coordinates** — клік на карті (lat, lon)

GPS координати конвертуються у метричну систему через `CoordinateConverter`:

```python
# WGS84 (lat, lon) → Web Mercator EPSG:3857 (mx, my) у метрах
metric_x, metric_y = converter.gps_to_metric(lat, lon)
```

#### Обчислення афінної матриці

Система обчислює два типи трансформацій і вибирає найкращу:

| Тип | DoF | Мін. точок | Опис |
|-----|-----|-----------|------|
| Partial Affine | 4 | 4 | scale + rotation + translation |
| Full Affine | 6 | 5 | + shear + non-uniform scale |

```python
# Partial Affine (4 DoF) — завжди обчислюється першою
M_partial, _ = cv2.estimateAffinePartial2D(pts_pixel, pts_metric, method=cv2.RANSAC)

# Full Affine (6 DoF) — лише якщо є ≥5 точок і покращує RMSE на >15%
M_full, _ = cv2.estimateAffine2D(pts_pixel, pts_metric, method=cv2.RANSAC)
if rmse_full < rmse_partial * 0.85:
    best_M = M_full  # вибираємо повний Affine
```

**QA метрики** (показуються користувачеві):
- **RMSE** — середньоквадратична похибка (метри)
- **Медіанна похибка** — стійкіша до викидів
- **Макс. похибка** — найгірша точка

#### PCHIP інтерполяція між якорями

При наявності ≥2 якорів система будує **PCHIP-інтерполятор** (Piecewise Cubic Hermite Interpolating Polynomial) для C¹-гладкого переходу між афінними матрицями:

```python
# anchors: frame_id=[10, 50, 120], affine=[M_10, M_50, M_120]
# Кожна матриця 2×3 розгортається у вектор (6,)
interp = PchipInterpolator(ids, matrices_flat)  # scipy

# Для кадру 30: інтерполюємо між M_10 і M_50
M_30 = interp(30).reshape(2, 3)  # C¹-гладкий перехід
```

**Навіщо PCHIP:** Лінійна інтерполяція створює "зламані" переходи на стиках якорів. PCHIP забезпечує монотонність і неперервність першої похідної, що дає плавний рух GPS-координат.

---

### 5.3 Пропагація GPS (`CalibrationPropagationWorker`)

Мета: присвоїти кожному кадру бази афінну матрицю pixel→metric, використовуючи лише кілька вручну каліброваних якорів.

#### Крок 1: Сегментація на інтервали

```
Кадри:    0 ... 10 ... 50 ... 120 ... 200
Якорі:         [10]    [50]     [120]
Сегменти:
  tail_left:  [9, 8, 7, ..., 0]         ← від якоря 10 вліво
  between_1:  [11, 12, ..., 49]          ← між 10 та 50
  between_2:  [51, 52, ..., 119]         ← між 50 та 120
  tail_right: [121, 122, ..., 200]       ← від якоря 120 вправо
```

Сегменти `between` обробляються **паралельно** через `ThreadPoolExecutor(max_workers=4)`, оскільки вони пишуть у різні діапазони масиву і не мають race conditions (перевіряється `assert`).

#### Крок 2: Побудова гомографічних ланцюгів

Для кожного якоря будується ланцюг гомографій до кожного кадру сегмента:

```python
# Від якоря (frame_id=10) до кадру 15:
H_10→11 = match(features[10], features[11])  # ALIKED LightGlue
H_10→12 = H_10→11 @ H_11→12
H_10→13 = H_10→12 @ H_12→13
H_10→14 = H_10→13 @ H_13→14
H_10→15 = H_10→14 @ H_14→15
```

Для кожного `H` обчислюється кількість інлаєрів (matches) — показник надійності.

#### Крок 3: Проєкція та блендінг (для `between` сегментів)

Для кадру між двома якорями (наприклад, кадр 30 між якорями 10 і 50) будуються два ланцюги:
- **Зліва**: `H_10→30` (від якоря 10)
- **Справа**: `H_50→30` (від якоря 50, зворотний ланцюг)

Обидва ланцюги проєктують сітку 4×4 точок у метричний простір:

```python
grid_points = meshgrid(0..W, 0..H, 4×4)           # 16 точок у pixel
ref_points = apply_homography(grid_points, H)       # → pixel якоря
metric_pts = apply_affine(ref_points, anchor.M)     # → метри
```

Фінальна позиція — **зважений блендінг** на основі відстані до якорів:

```python
weight_left = dist_to_right / (dist_to_left + dist_to_right)
weight_right = 1 - weight_left
final_pts = metric_pts_left * weight_left + metric_pts_right * weight_right
```

З фінальних 16 метричних точок апроксимується афінна матриця 2×3 (через LSQ), яка зберігається в HDF5:

```python
M_frame, _ = cv2.estimateAffinePartial2D(grid_points, final_metric_pts)
frame_affine[frame_id] = M_frame
frame_valid[frame_id] = True
```

#### QA метрики пропагації

Для кожного кадру обчислюється:
- **RMSE** — похибка апроксимації afine від повної сітки
- **Disagreement** — розбіжність між лівим та правим ланцюгами (drift)
- **Matches** — кількість інлаєрів у гомографічному ланцюзі

---

### 5.4 Локалізація (`Localizer`)

#### Крок 1: Глобальний пошук (place recognition)

FAISS Inner Product index для мільйонного пошуку за ~1мс:

```python
# Побудова індексу (один раз при ініціалізації)
index = faiss.IndexFlatIP(1024)
index.add(normalize(all_global_descriptors))  # N × 1024

# Запит
query_desc = DINOv2(query_frame)             # 1024-dim
scores, ids = index.search(normalize(query_desc), top_k=12)
# → candidates = [(frame_id, cosine_score), ...]
```

**Авто-ротація:** Якщо камера дрона повернута, система перебирає 4 кути (0°, 90°, 180°, 270°), для кожного обчислює глобальний дескриптор, і обирає ракурс з найвищим cosine similarity.

#### Крок 2: Локальний matching (geometric verification)

Для top-K кандидатів виконується точний matching:

**LightGlue (основний):**
```python
# Нейронний матчер LightGlue з вагами для ALIKED (128-dim)
data = {"image0": {"keypoints": kp_q, "descriptors": desc_q},
        "image1": {"keypoints": kp_r, "descriptors": desc_r}}
matches = lightglue(data)["matches"]  # (M, 2) — індекси пар
```

**Numpy L2 + MNN (fallback):** Якщо LightGlue недоступний:

```python
# 1. Нормалізація дескрипторів
# 2. Cosine similarity через матричне множення: sim = Q_norm @ R_norm.T
# 3. Lowe's Ratio Test: best_dist / second_best_dist < 0.85
# 4. Mutual Nearest Neighbor: якщо A→B найкращий І B→A найкращий
```

Для найкращого матчу обчислюється **гомографія** (8 DoF):

```python
H, mask = cv2.findHomography(mkpts_query, mkpts_ref, cv2.USAC_MAGSAC, 3.0)
inliers = mask.sum()  # кількість коректних пар
```

**Early stop:** якщо знайдено ≥40 inliers — решту кандидатів не перевіряємо.

#### Крок 3: Проєкція координат

```python
# 1. Центр query → reference через Homography (8 DoF)
center_query = [W/2, H/2]                              # центр кадру
center_ref = apply_homography(center_query, H)          # → позиція в ref кадрі

# 2. Reference pixel → Metric через Affine (2×3)
metric_pt = apply_affine(center_ref, frame_affine[ref_id])  # → метри (x, y)

# 3. Metric → GPS через pyproj
lat, lon = converter.metric_to_gps(metric_x, metric_y)  # EPSG:3857 → WGS84
```

#### Крок 4: Фільтрація аномалій (`OutlierDetector`)

Двоступеневий детектор:

1. **Speed check:** `distance / dt > max_speed_mps` (60 м/с) → outlier
2. **Z-score test:** обчислюється стандартне відхилення відстаней у вікні останніх 10 позицій. Якщо `|d - mean| / std > 25` і `|d - mean| > 50m` → outlier

**Self-reset:** Якщо 5 підряд вимірювань відкинуті як outliers — система вважає, що дрон справді перемістився, і скидає вікно.

#### Крок 5: Фільтр Калмана (`TrajectoryFilter`)

4-вимірний лінійний фільтр Калмана зі станом `[x, y, vx, vy]`:

```
Модель руху:         x' = x + vx·dt,   y' = y + vy·dt
Вимірювання:         z = [x_measured, y_measured]

Матриці:
  F = [1 0 dt 0]     # перехід стану
      [0 1 0  dt]
      [0 0 1  0 ]
      [0 0 0  1 ]

  H = [1 0 0 0]      # модель вимірювання
      [0 1 0 0]

  R = diag(5.0, 5.0)  # шум вимірювання
  Q = discrete_white_noise(dim=2, dt, var=2.0)  # шум процесу
```

Фільтр оновлюється при кожному новому вимірюванні: `predict() → update(z)`. dt адаптується до реальної частоти кадрів.

#### Крок 6: Проєкція FOV (Field of View)

4 кути кадру проєктуються через ту саму Homography+Affine ланцюжок:

```python
corners = [[0,0], [W,0], [W,H], [0,H]]
ref_corners = apply_homography(corners, H)
metric_corners = apply_affine(ref_corners, frame_affine[ref_id])
gps_corners = [metric_to_gps(mx, my) for mx, my in metric_corners]
```

Результат — GPS-полігон, який відображається на карті як напівпрозорий прямокутник, показуючи що саме "бачить" камера.

**Захист від "вибуху" гомографії:** якщо |ref_corners| > 50000 пікселів (перспективне спотворення), система fallback-ує на bounding box інлаєрів замість повного кадру.

#### Формула впевненості (Confidence)

```python
confidence = (stability_score × 0.3) + (inlier_score × 0.4) + (match_score × 0.3)

де:
  inlier_score = min(1.0, inliers / 80)                    # кількість
  stability_score = 1 - (rmse/10 × 0.5 + disag/5 × 0.5)   # якість БД
  match_score = (inlier_ratio × 0.5) + (rmse_score × 0.5)  # якість матчу
```

---

### 5.5 Конвертація координат (`CoordinateConverter`)

Система підтримує дві проєкції:

| Проєкція | EPSG | Характеристика |
|----------|------|----------------|
| **Web Mercator** | 3857 | Глобальна, масштаб спотворюється з широтою, стандарт для веб-карт |
| **UTM** | залежить від зони | Локальна, точна в межах зони (6° довготи), потребує reference point |

Конвертер використовує **pyproj** (`Transformer`) для точних перетворень між WGS84 (GPS) і метричною системою. Обидва напрямки: `gps_to_metric(lat, lon) → (x, y)` та `metric_to_gps(x, y) → (lat, lon)`.

Для обчислення фізичної відстані між GPS точками використовується формула **Хаверсіна** з радіусом Землі R=6371 км.

---

## 6. Нейромережеві моделі

| Модель | Тип | Розмірність | Призначення | VRAM |
|--------|-----|-------------|-------------|------|
| **ALIKED** | Keypoint Extractor | 128-dim descriptors | Локальні ознаки для точного matching | ~400 MB |
| **DINOv2 ViT-L/14** | Vision Transformer | 1024-dim descriptor | Глобальне place recognition (retrieval) | ~1600 MB |
| **LightGlue** | Matcher | — | Точний matching ALIKED keypoints | ~1000 MB |
| **YOLOv11n-seg** | Segmentation | — | Маскування динамічних об'єктів (авто, люди) | ~200 MB |
| **XFeat** | Keypoint Extractor | 64-dim | Fallback extractor | ~300 MB |
| **SuperPoint** | Keypoint Extractor | 256-dim | Legacy extractor (LightGlue compatibility) | ~500 MB |
| **CESP** | Feature Enhancement | 1024-dim | Покращення DINOv2 descriptors (опціонально) | Мінімум |

### Управління VRAM

`ModelManager` реалізує lazy loading та automatic eviction:
- Моделі завантажуються тільки при першому використанні
- Якщо не вистачає VRAM — автоматично вивантажує найменш використовувану модель
- Порогове значення `max_vram_ratio=0.8` для запобігання OOM

---

## 7. Формат бази даних (HDF5 v2)

```
database.h5
├── metadata/
│   ├── attrs: num_frames, frame_width, frame_height, fps, hdf5_schema="v2"
│   ├── attrs: actual_num_frames (кількість збережених keyframes)
│   └── frame_index_map (dataset: int32) — відповідність db_index → frame_id
│
├── local_features/
│   ├── attrs: frame_height, frame_width, max_kps
│   ├── keypoints       (num_frames × max_kps × 2, float32, lzf)
│   ├── descriptors      (num_frames × max_kps × 128, float16, lzf)
│   ├── coords_2d        (num_frames × max_kps × 2, float32, lzf)
│   ├── kp_counts        (num_frames, int32) — кількість keypoints на кадр
│   └── scores           (num_frames × max_kps, float32, lzf)
│
├── global_features/
│   └── descriptors      (num_frames × 1024, float32, lzf)
│
├── frame_poses/         (num_frames × 3 × 3, float64) — накопичені гомографії
│
└── calibration/         (створюється після пропагації)
    ├── attrs: version="2.1", num_anchors, anchors_json, projection_json
    ├── frame_affine      (num_frames × 2 × 3, float32, gzip)
    ├── frame_valid       (num_frames, uint8, gzip)
    ├── frame_rmse        (num_frames, float32, gzip)
    ├── frame_disagreement(num_frames, float32, gzip)
    └── frame_matches     (num_frames, int32, gzip)
```

Ключові оптимізації v2:
- **Pre-allocated масиви** замість per-frame груп → O(1) запис замість O(N) створення груп
- **LZF компресія** — вбудована в h5py, без додаткових залежностей, висока швидкість
- **float16 дескриптори** — 50% менший footprint, конвертація в float32 при читанні
- **Chunked I/O** — оптимальний розмір chunk=64 фрейми для послідовного доступу

---

## 8. Конфігурація

Вся конфігурація зосереджена у `config/config.py` через Pydantic v2:

```python
AppConfig
├── Dinov2Config         # descriptor_dim=1024, input_size=336
├── DatabaseConfig       # frame_step, compression, max_keypoints, keyframe thresholds
├── LocalizationConfig   # min_matches, retrieval_top_k, auto_rotation, confidence
├── TrackingConfig       # kalman_process_noise, outlier_window, max_speed
├── PreprocessingConfig  # clahe_clip_limit, histogram_matching
├── GuiConfig            # video_fps
├── ModelsConfig         # use_cuda, VRAM limits, model paths, CESP settings
└── ProjectionConfig     # default_mode, RMSE thresholds, disagreement limits
```

Доступ до конфігу через `get_cfg(config, "section.key", default)` — підтримує як dict, так і Pydantic.

---

## 9. Тестування

- **21 тестовий файл** (pytest + pytest-qt + pytest-benchmark)
- Структура: `tests/unit/`, `tests/integration/`, `tests/benchmarks/`, `tests/` (верхній рівень)
- Покриття:
  - Конфігурація (структура, типи, ключі, синхронізація)
  - Геометрія (гомографії, афінні, валідація матриць)
  - Координати (UTM, Web Mercator, Haversine, проєкції)
  - Графова оптимізація (PoseGraphOptimizer: BFS, LM, GeoJSON)
  - База даних (створення, запит)
  - GUI (MainWindow)
  - Моделі (YOLO wrapper, MaskingStrategy, Feature extractor)
  - Калібрування (pixel-to-metric round trip, інтерполяція)
  - Локалізація (пайплайн)
  - Повний pipeline (integration)
  - Бенчмарки (побудова БД, трекінг)

---

## 10. Системні вимоги

| Компонент | Мінімум | Рекомендовано |
|-----------|---------|---------------|
| Python | 3.10 | 3.11 |
| GPU | NVIDIA GTX 1650 (4 GB VRAM) | RTX 3060+ (8+ GB VRAM) |
| RAM | 8 GB | 16 GB |
| CUDA | 11.8+ | 12.1+ |
| ОС | Windows 10 64-bit | Windows 11 |
| Диск | SSD (для HDF5 I/O) | NVMe SSD |

---

## 11. Запуск

```bash
# Створення та активація середовища
python -m venv .venv
.venv\Scripts\activate

# Встановлення залежностей
pip install -e ".[dev]"

# Запуск
python main.py

# Тестування
python -m pytest tests/ -v

# Лінтинг
ruff check --fix src/ tests/
ruff format src/ tests/
```

---

## 12. Ліцензія

MIT License — див. файл `LICENSE` у кореневій директорії.
