# Детальний план імплементації покращень
**Drone Topometric Localization System — Фази 2 та 3**

| | |
|---|---|
| Задач | 8 |
| Core-файли | `database_builder.py`, `database_loader.py`, `multi_anchor_calibration.py`, `config.py` |
| Рекомендований старт | П7 → П5 → П1+П2+П3 → П6 → П4 → П8 |

---

## Зміст

1. [П1 — Pre-allocated HDF5 + lzf (×15 швидкість запису)](#п1)
2. [П2 — float16 для локальних дескрипторів (−50% розмір)](#п2)
3. [П3 — Оновлення DatabaseLoader під нову схему](#п3)
4. [П4 — Adaptive Keyframe Selection (−60% кадрів)](#п4)
5. [П5 — orjson для AnchorCalibration (×8 серіалізація)](#п5)
6. [П6 — PCHIP замість linear lerp](#п6)
7. [П7 — Виправити подвійний APP\_CONFIG](#п7)
8. [П8 — YOLO micro-batching](#п8)
9. [Зведена таблиця](#зведена-таблиця)

---

## П1 — Pre-allocated HDF5 + lzf {#п1}

**Файл:** `database/database_builder.py` | **Фаза:** 1 | **Складність:** 2 дні | **Ефект:** ×15 швидкість запису

Поточний `save_frame_data` (рядок 1070) викликає `create_group` + `create_dataset` × 3 на кожен кадр. Для 10 000 кадрів — 30 000 HDF5-операцій із записами у B-tree. Нова схема: один `create_dataset` на весь масив при ініціалізації, потім запис через slice-assignment.

> **⚠️ Критично:** `DatabaseLoader.get_local_features` (рядок 1243) і `get_frame_size` (рядок 1220) читають зі структури `local_features/frame_N/...`. При зміні схеми HDF5 їх **обов'язково** треба оновити одночасно (П3).

### Крок 1 — нові ключі у config.py

```python
# config/config.py — додати до DatabaseConfig або відповідного розділу
hdf5_compression: str = "lzf"       # "gzip" | "lzf" | None  (lzf = без pip)
hdf5_chunk_frames: int = 64         # розмір chunk по осі кадрів
max_keypoints_stored: int = 2048    # MAX_KPS — фіксований розмір другої осі
```

> **Вибір компресора:** `zstd` потребує `hdf5plugin` як залежності. Щоб не додавати нову залежність — використовуйте вбудований `lzf` (HDF5 native, без pip). Він у 8–10× швидший за gzip при стисненні ~80% від gzip. Якщо `hdf5plugin` додається — тоді `zstd`.

### Крок 2 — переписати `create_hdf5_structure` (рядки 1043–1068)

```python
def create_hdf5_structure(self, num_frames: int, width: int, height: int):
    compression   = get_cfg(self.config, "database.hdf5_compression", "lzf")
    chunk_f       = get_cfg(self.config, "database.hdf5_chunk_frames", 64)
    max_kps       = get_cfg(self.config, "database.max_keypoints_stored", 2048)
    local_desc_dim = 128             # ALIKED descriptor dim

    with h5py.File(self.output_path, "w") as f:
        # --- global_descriptors: без змін, тільки додаємо chunks ---
        g1 = f.create_group("global_descriptors")
        g1.create_dataset(
            "descriptors",
            shape=(num_frames, self.descriptor_dim),
            dtype="float32",
            compression=compression,
            chunks=(256, self.descriptor_dim),
        )
        g1.create_dataset(
            "frame_poses",
            shape=(num_frames, 3, 3),
            dtype="float64",
            compression=compression,
            chunks=(256, 3, 3),
        )

        # --- local_features: PRE-ALLOCATED chunked arrays (НОВА СХЕМА) ---
        lf = f.create_group("local_features")
        lf.create_dataset(
            "keypoints",
            shape=(num_frames, max_kps, 2),
            dtype="float32",
            compression=compression,
            chunks=(chunk_f, max_kps, 2),
            fillvalue=0.0,
        )
        lf.create_dataset(
            "descriptors",
            shape=(num_frames, max_kps, local_desc_dim),
            dtype="float16",          # ← float16: -50% розміру (деталі в П2)
            compression=compression,
            chunks=(chunk_f, max_kps, local_desc_dim),
            fillvalue=0.0,
        )
        lf.create_dataset(
            "coords_2d",
            shape=(num_frames, max_kps, 2),
            dtype="float32",
            compression=compression,
            chunks=(chunk_f, max_kps, 2),
            fillvalue=0.0,
        )
        lf.create_dataset(
            "kp_counts",              # скільки keypoints у кожному кадрі
            shape=(num_frames,),
            dtype="int16",
            compression=compression,
            chunks=(min(num_frames, 4096),),
            fillvalue=0,
        )
        # Розміри кадру — зберігаємо ОДИН РАЗ у групі, не для кожного кадру
        lf.attrs["frame_width"]  = width
        lf.attrs["frame_height"] = height

        g3 = f.create_group("metadata")
        g3.attrs["num_frames"]    = num_frames
        g3.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        g3.attrs["frame_width"]   = width
        g3.attrs["frame_height"]  = height
        g3.attrs["descriptor_dim"] = self.descriptor_dim
        g3.attrs["hdf5_schema"]   = "v2"    # версія схеми для зворотньої сумісності
        g3.attrs["max_keypoints"] = max_kps
```

### Крок 3 — переписати `save_frame_data` (рядки 1070–1084)

```python
def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
    # global — без змін
    self.db_file["global_descriptors"]["descriptors"][frame_id] = features["global_desc"]
    self.db_file["global_descriptors"]["frame_poses"][frame_id] = pose_2d

    # local — slice assignment замість create_group + create_dataset
    kps   = features["keypoints"]
    descs = features["descriptors"]
    c2d   = features["coords_2d"]

    max_kps = self.db_file["local_features"]["keypoints"].shape[1]
    n = min(len(kps), max_kps)

    lf = self.db_file["local_features"]
    lf["keypoints"][frame_id, :n]   = kps[:n]
    lf["descriptors"][frame_id, :n] = descs[:n].astype("float16")
    lf["coords_2d"][frame_id, :n]   = c2d[:n]
    lf["kp_counts"][frame_id]        = n

    # ВИДАЛИТИ весь старий блок:
    # frame_group = self.db_file["local_features"].create_group(f"frame_{frame_id}")
    # frame_group.create_dataset("keypoints",   ...)
    # frame_group.create_dataset("descriptors", ...)
    # frame_group.create_dataset("coords_2d",   ...)
```

---

## П2 — float16 для локальних дескрипторів {#п2}

**Файл:** `database/database_builder.py` | **Фаза:** 1 | **Складність:** 1 год | **Ефект:** −50% розмір БД

Дескриптори ALIKED (float32, dim=128) займають 2048 × 128 × 4 = 1 MB на кадр. При 10 000 кадрів — 10.5 GB. Float16 дає рівно вдвічі менше. У `create_hdf5_structure` (П1) вже вказано `dtype="float16"` і `descs[:n].astype("float16")` у `save_frame_data`.

Єдина додаткова зміна — читання з конвертацією назад у float32, бо PyTorch вимагає float32 для inference.

> **Де саме:** у `get_local_features` (П3 нижче) при зчитуванні додати `.astype("float32")`.

---

## П3 — Оновлення DatabaseLoader {#п3}

**Файл:** `database/database_loader.py` | **Фаза:** 1 | **Складність:** 3 год | **Блокує П1**

DatabaseLoader читає `local_features/frame_N/keypoints` (стара схема v1). Також `get_frame_size` читає `g.attrs["height"]` з кожної групи — в новій схемі розміри збережені один раз у `local_features.attrs`.

> **⚠️ Зверніть увагу:** у поточному коді `save_frame_data` (рядок 1070) атрибути `height`/`width` для груп `frame_N` взагалі не записуються, тому `get_frame_size` завжди падав на fallback до metadata. В новій схемі це виправлено.

### Зміна 1 — `get_frame_size` (рядки 1220–1241)

```python
def get_frame_size(self, frame_id: int) -> tuple[int, int]:
    if frame_id in self._size_cache:
        return self._size_cache[frame_id]
    if self.db_file is None:
        return 1080, 1920

    # Нова схема v2: розміри збережені один раз в local_features.attrs
    schema = self.metadata.get("hdf5_schema", "v1")
    if schema == "v2" and "local_features" in self.db_file:
        lf_attrs = self.db_file["local_features"].attrs
        h = int(lf_attrs.get("frame_height", self.metadata.get("frame_height", 1080)))
        w = int(lf_attrs.get("frame_width",  self.metadata.get("frame_width",  1920)))
        self._size_cache[frame_id] = (h, w)
        return h, w

    # Стара схема v1: fallback — читаємо з групи кадру (зворотня сумісність)
    group_name = f"local_features/frame_{frame_id}"
    h, w = 1080, 1920
    if group_name in self.db_file:
        g = self.db_file[group_name]
        if "height" in g.attrs and "width" in g.attrs:
            h, w = int(g.attrs["height"]), int(g.attrs["width"])
        else:
            h = self.metadata.get("frame_height") or 1080
            w = self.metadata.get("frame_width")  or 1920
    res = (h, w)
    self._size_cache[frame_id] = res
    return res
```

### Зміна 2 — `get_local_features` (рядки 1243–1267)

```python
def get_local_features(self, frame_id: int) -> dict[str, np.ndarray]:
    if frame_id in self._feature_cache:
        return self._feature_cache[frame_id]
    if self.db_file is None:
        raise RuntimeError("Database not opened")

    schema = self.metadata.get("hdf5_schema", "v1")
    if schema == "v2":
        lf = self.db_file["local_features"]
        n  = int(lf["kp_counts"][frame_id])
        if n == 0:
            raise ValueError(f"Кадр {frame_id} не має keypoints (kp_count=0).")
        res = {
            "keypoints":   lf["keypoints"][frame_id, :n],
            "descriptors": lf["descriptors"][frame_id, :n].astype("float32"),  # float16→32
            "coords_2d":   lf["coords_2d"][frame_id, :n],
        }
    else:
        # Стара схема v1 — зворотня сумісність
        group_name = f"local_features/frame_{frame_id}"
        if group_name not in self.db_file:
            raise ValueError(f"Кадр {frame_id} не знайдено у базі даних.")
        g = self.db_file[group_name]
        res = {
            "keypoints":   g["keypoints"][:],
            "descriptors": g["descriptors"][:],
            "coords_2d":   g["coords_2d"][:],
        }

    # LRU-витіснення (без змін)
    if len(self._feature_cache) > 200:
        self._feature_cache.pop(next(iter(self._feature_cache)))
    self._feature_cache[frame_id] = res
    return res
```

---

## П4 — Adaptive Keyframe Selection {#п4}

**Файл:** `database/database_builder.py` + `config` | **Фаза:** 2 | **Складність:** 1.5 дні | **Ефект:** −60% кадрів у БД

Зараз кожен `frame_step`-й кадр обробляється беззаперечно. Гомографія між сусідніми кадрами вже обчислюється у `_compute_inter_frame_H` (рядок 1015). Порівнюємо shift і кут з порогами: якщо дрон не рухнувся — пропускаємо inference.

> **⚠️ Архітектурна складність:** `num_frames` використовується для pre-allocation HDF5-масиву *до* циклу. Якщо ми пропускаємо кадри, масив залишається більшим за фактичну кількість. Вирішення: зберігати `actual_num_frames` і `frame_index_map` у metadata.

### Крок 1 — нові пороги у конфігу

```python
# config/config.py
keyframe_min_translation_px: float = 15.0
keyframe_min_rotation_deg:   float = 1.5
keyframe_always_save_first:  bool  = True   # перший кадр завжди зберігається
```

### Крок 2 — допоміжна функція аналізу руху

```python
def _is_significant_motion(self, H: np.ndarray) -> bool:
    """
    Повертає True якщо гомографія H відповідає значному руху.
    H: (3,3) float32 — матриця з frame_b до frame_a.
    """
    min_t = get_cfg(self.config, "database.keyframe_min_translation_px", 15.0)
    min_r = get_cfg(self.config, "database.keyframe_min_rotation_deg", 1.5)

    # Трансляція: зсув центру кадру через H
    cx, cy = self.frame_w / 2.0, self.frame_h / 2.0
    p_src = np.array([cx, cy, 1.0], dtype=np.float64)
    p_dst = H.astype(np.float64) @ p_src
    p_dst /= p_dst[2]
    translation = np.linalg.norm(p_dst[:2] - np.array([cx, cy]))

    if translation >= min_t:
        return True

    # Кут: з лінійної частини H (2×2 зліва вгорі)
    A   = H[:2, :2].astype(np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return True               # вироджена матриця → вважаємо рухом
    angle_rad = np.arctan2(A[1, 0], A[0, 0])
    angle_deg = abs(np.degrees(angle_rad))
    return angle_deg >= min_r
```

### Крок 3 — вбудувати перевірку у головний цикл `build_from_video`

```python
# Ініціалізація перед циклом (після рядка ~852)
current_pose = np.eye(3, dtype=np.float32)
prev_features = None
db_index      = 0                           # лічильник РЕАЛЬНО записаних кадрів
frame_index_map: list[int] = []             # db_index → original_frame_id
use_keyframe_selection = get_cfg(
    self.config, "database.keyframe_min_translation_px", 0.0
) > 0

# --- У циклі обробки кадрів ---
while True:
    idx, data = frame_queue.get()
    if idx == -1:
        break
    frame, frame_rgb = data

    static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)
    features = feature_extractor.extract_features(frame_rgb, static_mask)
    features["coords_2d"] = features["keypoints"]

    if idx == 0 or prev_features is None:
        H_step = None
        current_pose = np.eye(3, dtype=np.float64)
        save_this_frame = True                   # перший кадр завжди
    else:
        H_step = self._compute_inter_frame_H(prev_features, features)
        if H_step is not None:
            current_pose = current_pose @ H_step.astype(np.float64)
        # Keyframe selection
        if use_keyframe_selection and H_step is not None:
            save_this_frame = self._is_significant_motion(H_step)
        else:
            save_this_frame = True

    prev_features = features

    if not save_this_frame:
        continue                                 # пропускаємо — YOLO/ALIKED вже відпрацювали

    frame_index_map.append(idx)                  # зберігаємо відповідність індексів
    self.save_frame_data(db_index, features, current_pose)
    db_index += 1

    # progress без змін ...

# --- Після циклу: зберегти frame_index_map у metadata ---
with h5py.File(self.output_path, "a") as f:
    f["metadata"].attrs["actual_num_frames"] = db_index
    f["metadata"].create_dataset(
        "frame_index_map",
        data=np.array(frame_index_map, dtype=np.int32),
    )
```

> **Потребує змін у DatabaseLoader:** `get_num_frames()` має повертати `actual_num_frames` якщо воно є в metadata. `frame_index_map` завантажується у `_load_hot_data` для зворотного пошуку (який оригінальний кадр відповідає db_index).

---

## П5 — orjson для AnchorCalibration {#п5}

**Файл:** `calibration/multi_anchor_calibration.py` | **Фаза:** 2 | **Складність:** 30 хв | **Ефект:** ×8 серіалізація

`save()` (рядок 173) використовує стандартний `json.dump(indent=2)`. `orjson` у 5–10× швидший, підтримує `numpy.ndarray` нативно (не потрібен `.tolist()`).

### Зміна 1 — додати orjson до pyproject.toml

```toml
# [project.dependencies]
"orjson>=3.9.0",
```

### Зміна 2 — переписати `save()` і `load()`

```python
# Замінити "import json" на:
try:
    import orjson as _json_lib
    _USE_ORJSON = True
except ImportError:
    import json as _json_lib    # fallback якщо orjson не встановлено
    _USE_ORJSON = False

# ---

def save(self, path: str):
    if not self.is_calibrated:
        raise RuntimeError("Немає даних для збереження")

    data = {
        "version":    self.VERSION,
        "projection": CoordinateConverter.export_projection_metadata(),
        "anchors":    [a.to_dict() for a in self.anchors],
    }

    if _USE_ORJSON:
        raw = _json_lib.dumps(
            data,
            option=_json_lib.OPT_INDENT_2 | _json_lib.OPT_NON_STR_KEYS,
        )
        with open(path, "wb") as f:
            f.write(raw)
    else:
        with open(path, "w", encoding="utf-8") as f:
            _json_lib.dump(data, f, indent=2, ensure_ascii=False)
    logger.success(
        f"MultiAnchorCalibration saved: {path} (v{self.VERSION}, {len(self.anchors)} anchors)"
    )

def load(self, path: str):
    logger.info(f"Loading MultiAnchorCalibration from: {path}")
    with open(path, "rb") as f:    # rb — сумісно з обома форматами
        content = f.read()
    if _USE_ORJSON:
        data = _json_lib.loads(content)
    else:
        import json
        data = json.loads(content)
    # ... далі без змін ...
```

---

## П6 — PCHIP замість linear lerp {#п6}

**Файл:** `calibration/multi_anchor_calibration.py` | **Фаза:** 2 | **Складність:** 1 день

Поточний `get_metric_position` (рядок 150) — C⁰ неперервність (кути при переходах між якорями). PCHIP — C¹ та монотонний, тобто без "підстрибів". scipy вже є залежністю проєкту.

> Коли PCHIP не застосовується: при 0 або 1 якорі — fall back на поточну логіку. PCHIP потребує мінімум 2 точки.

### Крок 1 — нове поле і `_rebuild_interpolators`

```python
from scipy.interpolate import PchipInterpolator

class MultiAnchorCalibration:
    VERSION = "2.2"

    def __init__(self):
        self.anchors: list[AnchorCalibration] = []
        self._interp: PchipInterpolator | None = None   # кешований інтерполятор

    def _rebuild_interpolators(self):
        """Перебудовує PCHIP-інтерполятор. Викликати після кожної зміни anchors."""
        if len(self.anchors) < 2:
            self._interp = None
            return

        ids      = np.array([a.frame_id for a in self.anchors], dtype=np.float64)
        matrices = np.stack([a.affine_matrix.ravel() for a in self.anchors])  # (N, 6)
        # PchipInterpolator обробляє multi-column масиви нативно
        self._interp = PchipInterpolator(ids, matrices, extrapolate=False)
```

### Крок 2 — додати виклик `_rebuild_interpolators` у 3 методи

```python
def add_anchor(self, frame_id, affine_matrix, qa_data=None):
    # ... існуюча логіка без змін ...
    self.anchors.sort(key=lambda a: a.frame_id)
    self._rebuild_interpolators()                   # ← ДОДАТИ

def remove_anchor(self, frame_id: int) -> bool:
    # ... існуюча логіка без змін ...
    if success:
        self._rebuild_interpolators()               # ← ДОДАТИ
        logger.info(f"Removed anchor for frame {frame_id}")
    return success

def load(self, path: str):
    # ... існуюча логіка завантаження ...
    self.anchors.sort(key=lambda a: a.frame_id)
    self._rebuild_interpolators()                   # ← ДОДАТИ
    logger.success(f"Loaded {len(self.anchors)} anchors (file version: {version})")
```

### Крок 3 — переписати `get_metric_position`

```python
def get_metric_position(self, frame_id: int, x: float, y: float) -> tuple | None:
    if not self.is_calibrated:
        return None

    # Граничні умови — без змін
    if len(self.anchors) == 1 or frame_id <= self.anchors[0].frame_id:
        return self.anchors[0].pixel_to_metric(x, y)
    if frame_id >= self.anchors[-1].frame_id:
        return self.anchors[-1].pixel_to_metric(x, y)

    # PCHIP: інтерполяція матриці → застосування до точки
    if self._interp is not None:
        flat = self._interp(float(frame_id))        # (6,) float64
        if flat is not None and not np.any(np.isnan(flat)):
            M      = flat.reshape(2, 3).astype(np.float32)
            pt     = np.array([[x, y]], dtype=np.float32)
            result = GeometryTransforms.apply_affine(pt, M)[0]
            return float(result[0]), float(result[1])

    # Fallback — лінійна інтерполяція (якщо PCHIP недоступний або NaN)
    for i in range(len(self.anchors) - 1):
        a1, a2 = self.anchors[i], self.anchors[i + 1]
        if a1.frame_id <= frame_id <= a2.frame_id:
            dist_1 = abs(frame_id - a1.frame_id)
            dist_2 = abs(frame_id - a2.frame_id)
            total  = dist_1 + dist_2
            if total == 0:
                return a1.pixel_to_metric(x, y)
            w2 = dist_1 / total
            m1 = a1.pixel_to_metric(x, y)
            m2 = a2.pixel_to_metric(x, y)
            return m1[0] * (1 - w2) + m2[0] * w2, m1[1] * (1 - w2) + m2[1] * w2
    return None
```

---

## П7 — Виправити подвійний APP\_CONFIG {#п7}

**Файл:** `config/config.py` | **Фаза:** 3 | **Складність:** 5 хв

У `config.py` два незалежних об'єкти конфігурації — зміна одного не відображається в іншому. Runtime-зміна параметрів через GUI не буде видна модулям, що читають `APP_CONFIG`.

```python
# config/config.py — кінець файлу

# ВИДАЛИТИ:
# APP_CONFIG   = AppConfig().model_dump()   # ← незалежний об'єкт
# APP_SETTINGS = AppConfig()                # ← незалежний об'єкт

# ЗАМІНИТИ НА:
APP_SETTINGS = AppConfig()                  # єдине джерело правди (Pydantic)
APP_CONFIG   = APP_SETTINGS.model_dump()   # dict-представлення того самого об'єкта
```

> Якщо потрібна реактивна синхронізація (GUI змінює параметри в runtime), передавати `APP_SETTINGS` скрізь і викликати `.model_dump()` лінійно замість зберігання dict.

---

## П8 — YOLO micro-batching {#п8}

**Файл:** `database/database_builder.py`, `models/wrappers/yolo_wrapper.py` | **Фаза:** 3 | **Складність:** 1 день | **Ефект:** ×1.5 швидкість маскування

`detect_and_mask` обробляє кожен кадр окремо. Ultralytics приймає список зображень в один виклик, що дозволяє GPU обробити кілька кадрів за один forward pass.

> **⚠️ Обмеження VRAM:** при batch=2–3 і VRAM 6 GB — безпечно. Batch=4+ ризикований при одночасно завантажених ALIKED + DINOv2. Розмір виносити у конфіг.

### Крок 1 — новий ключ у конфігу

```python
yolo_batch_size: int = 2    # у DatabaseConfig
```

### Крок 2 — batch-метод у YOLOWrapper

```python
def detect_and_mask_batch(self, images: list[np.ndarray]) -> list[tuple]:
    """
    Обробляє список зображень одним викликом YOLO.
    Повертає list[(static_mask, detections)] того самого порядку.
    """
    if not images:
        return []

    results = self.model(images, verbose=False, half=self.use_half, conf=0.50)

    output = []
    for result, image in zip(results, images):
        height, width = image.shape[:2]
        static_mask = np.ones((height, width), dtype=np.uint8) * 255
        detections  = []

        if result.masks is not None:
            # Скопіювати логіку фільтрації масок з detect_and_mask без змін
            masks      = result.masks.data.cpu().numpy()
            boxes      = result.boxes.data.cpu().numpy()
            classes    = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            total_pixels = height * width

            dynamic_mask_indices = [
                i for i, cls in enumerate(classes) if cls in self.dynamic_classes
            ]
            for i, (cls, conf, box) in enumerate(zip(classes, confidences, boxes)):
                detections.append(
                    {"class_id": int(cls), "confidence": float(conf), "bbox": box[:4].tolist()}
                )
            if dynamic_mask_indices:
                combined_dynamic = np.zeros((height, width), dtype=np.float32)
                for idx in dynamic_mask_indices:
                    mask_resized = cv2.resize(
                        masks[idx], (width, height), interpolation=cv2.INTER_NEAREST
                    )
                    if np.sum(mask_resized > 0.5) / total_pixels <= 0.40:
                        combined_dynamic = np.maximum(combined_dynamic, mask_resized)
                if np.sum(combined_dynamic > 0.5) / total_pixels < 0.70:
                    static_mask[combined_dynamic > 0.5] = 0

        output.append((static_mask, detections))
    return output
```

### Крок 3 — головний цикл `build_from_video` з мікро-батчами

```python
# Ініціалізація перед циклом
yolo_batch_size  = get_cfg(self.config, "database.yolo_batch_size", 2)
pending_frames: list[tuple] = []    # буфер (idx, frame, frame_rgb)

def _flush_batch(batch: list) -> list:
    """Обробляє батч через YOLO, повертає (idx, frame, frame_rgb, static_mask)."""
    images_rgb = [b[2] for b in batch]
    masks_list = yolo_wrapper.detect_and_mask_batch(images_rgb)
    return [(b[0], b[1], b[2], m[0]) for b, m in zip(batch, masks_list)]

# --- У циклі ---
while True:
    idx, data = frame_queue.get()

    if idx != -1:
        frame, frame_rgb = data
        pending_frames.append((idx, frame, frame_rgb))
        if len(pending_frames) < yolo_batch_size:
            continue               # накопичуємо батч

    # Якщо EOF або батч повний — обробляємо все накопичене
    if not pending_frames:
        break
    processed = _flush_batch(pending_frames)
    pending_frames = []

    for pidx, pframe, pframe_rgb, static_mask in processed:
        features = feature_extractor.extract_features(pframe_rgb, static_mask)
        # ... решта логіки (pose, keyframe selection, save_frame_data, progress) без змін ...

    if idx == -1:
        break
```

---

## Зведена таблиця {#зведена-таблиця}

| Задача | Файл(и) | Фаза | Складність | Залежності |
|--------|---------|------|-----------|------------|
| П1 — Pre-alloc HDF5 + lzf | `database_builder.py` | 1 | 2 дні | — |
| П2 — float16 descriptors | `database_builder.py` | 1 | 1 год | П1 |
| П3 — DatabaseLoader оновлення | `database_loader.py` | 1 | 3 год | П1 (блокує) |
| П4 — Keyframe Selection | `database_builder.py`, config | 2 | 1.5 дні | П1, П3 |
| П5 — orjson | `multi_anchor_calibration.py` | 2 | 30 хв | — |
| П6 — PCHIP інтерполяція | `multi_anchor_calibration.py` | 2 | 1 день | — |
| П7 — Виправити APP\_CONFIG | `config.py` | 3 | 5 хв | — |
| П8 — YOLO micro-batching | `database_builder.py`, `yolo_wrapper.py` | 3 | 1 день | П1 |

> **Рекомендований порядок:** П7 (5 хвилин, нульовий ризик) → П5 (30 хв, ізольований) → **П1+П2+П3 разом** (один PR, нова схема HDF5) → П6 (ізольований) → П4 → П8.
> П1, П2, П3 **повинні іти одним коммітом** — вони змінюють формат файлу, і якщо `DatabaseLoader` не оновлений, старі бази стануть нечитабельними.

---

## 🚀 ФАЗА 4: Екстремальна оптимізація під Blackwell (RTX 5070 Ti)

Враховуючи, що RTX 5070 Ti має шалену пропускну здатність та новітню архітектуру (sm_120), стандартний послідовний підхід завантажує її ледве на 10-15%. Ось план, як "вичавити" максимум і пришвидшити генерацію бази ще у 3-5 разів:

### П9 — Повний Батчинг (YOLO + ALIKED + DINOv2)
Зараз пропонувалося зробити мікро-батчинг лише для YOLO (П8). Але **ALIKED і DINOv2 приймають тензори (B, C, H, W)**! 
Замість передачі по 1 кадру (що спричиняє overhead на виклики CUDA-ядер), ми можемо передавати весь батч (напр. розміром 4-8 кадрів).
- **Як:** Обновити `FeatureExtractor.extract_features_batch()`, щоб він обробляв список кадрів одним викликом нейромережі.

### П10 — Асинхронність нейромереж (CUDA Streams)
DINOv2 (Vision Transformer) і ALIKED (CNN) повністю незалежні одна від одної.
- **Як:** Змусити їх працювати **одночасно** за допомогою `torch.cuda.Stream()`. Поки половина SM-блоків (Streaming Multiprocessors) відеокарти рахує DINOv2, інша половина паралельно витягує локальні точки через ALIKED. 
```python
stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
with torch.cuda.stream(stream1):
    global_desc = self.extract_global_descriptor_batch(images)
with torch.cuda.stream(stream2):
    local_feats = self.extract_local_features_batch(images)
torch.cuda.synchronize()
```

### П11 — Компіляція графів (torch.compile)
PyTorch 2.6/2.7 має безпрецедентну оптимізацію індуктивних графів під 50-ту серію RTX.
- **Як:** Додати `self.global_model = torch.compile(self.global_model, mode="max-autotune")`. Це зіллє дрібні операції (fusions) і використання тензорних ядер (Tensor Cores) стане майже 100-відсотковим.

### П12 — Апаратний NVDEC (GPU Video Decoding)
Зараз `cv2.VideoCapture` розшифровує відео на процесорі (CPU), потім кадр по шині PCIe 4.0 копіюється у відеопам'ять. Для 30 FPS це нормально, але для 200 FPS відеокарта чекатиме на CPU.
- **Як:** Використати `torchvision.io.VideoReader(..., stream="video:hw")` (або `decord`), які розшифровують H.264 прямо на апаратному декодері відеокарти (NVDEC) одразу в `torch.Tensor` у пам'ять GPU. CPU взагалі вимикається з процесу!

### П13 — FP8 / BFloat16 Precision
RTX 5070 Ti створена для `bfloat16` та нового `float8`. 
- **Як:** Перевести `torch.cuda.amp.autocast(dtype=torch.float16)` на `torch.bfloat16` (він набагато стабільніший за fp16 і швидший на нових архітектурах).

Ці оптимізації перетворять ваш пайплайн із "просто швидкого" (20-30 fps) на "індустріальний" (100-150+ fps обробки на одному GPU).
