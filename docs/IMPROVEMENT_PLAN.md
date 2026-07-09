# План покращення DroneLocalization

> Аналіз кодової бази станом на 2026-07-07. Порядок пунктів — за співвідношенням цінність/ризик (перший — найбільше цінності при найменшому ризику).

> **Статус виконання (звірено з кодом 2026-07-08):** ✅ виконано — розбиття `localizer.py` (1.1), пакет `pose_graph/` (1.2), пакет `config/` (2.2), Protocol-інтерфейси `src/interfaces.py` (3.1), `src/exceptions.py` (3.3), характеризаційні тести + fakes (4), `architecture.md`/`POSE_GRAPH_MATH.md` (6). 🔲 відкрито — `database_builder.py` (1.3, досі 886 рядків), `PropagationPipeline` (1.4), `ModelManager`→registry/vram (1.6), mixins→контролери (2.1/1.5), performance (5), повна get_cfg-міграція (2.3). Живий, звірений план — у [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Резюме та розбіжності з ТЗ

Перед планом — факти, які відрізняються від припущень у ТЗ:

- **Rotation scan уже оптимізований.** У `localizer.py` повний скан 4 кутів виконується ОДНИМ батчованим forward-пасом (`extract_global_descriptors_multi`, коментар A2), плюс темпоральний prior кута (A3) зводить типовий keyframe до 1 forward-пасу. "Nested loops з multiple DINOv2 forward passes" — уже вирішено; залишкові резерви — у патчифаї та матчингу кандидатів (див. п.5).
- **Стиль type hints уже уніфікований**: `Optional[]` у src/ не використовується взагалі, скрізь `X | None`. Уніфікація не потрібна.
- **Bare `except:` не знайдено.** Є ~60 `except Exception` різного ступеня обґрунтованості — це реальна проблема, але інша (див. п.3).
- **Міксини менші, ніж заявлено**: calibration_mixin 699 рядків, database_mixin 695, tracking_mixin 383, panorama_mixin 245. `main_window.py` — лише 148 рядків. Проблема god-object реальна, але масштаб помірний.
- `calibration_propagation_worker.py` — 870 рядків, але вже добре методизований (17 приватних методів) — виділення pipeline-класу майже механічне.

**Рекомендований порядок виконання:** 4 (тести-страховка) → 1 (рефакторинг монолітів) → 2 (архітектура) → 3 (типізація) → 5 (performance) → 6 (документація). Тести першими: без страховки будь-який рефакторинг Localizer/DatabaseBuilder — гра в рулетку. Але для зручності зіставлення з ТЗ нижче нумерація збережена.

---

## 1. Рефакторинг монолітних файлів

### 1.1 localizer.py (935 рядків) — ✅ ВИКОНАНО

**Аналіз.** Клас `Localizer` — оркестратор, що робить усе сам: вибір кута (prior + batch scan), retrieval (single/multi-db/patchify + merge), цикл матчинг→RANSAC з early stop, 3 варіанти fallback, координатні перетворення, Kalman + outlier, обчислення FOV з захистом від "вибуху" гомографії, confidence, CSV-лог невдач. `localize_frame()` — ~450 рядків. Побічний ефект: неможливо юніт-тестити вибір кута окремо від RANSAC, а FOV — окремо від Kalman.

**План.** Створити пакет `src/localization/` (він уже є) з модулями:

| Новий модуль | Що переносимо | Рядків |
|---|---|---|
| `rotation_selector.py` | `RotationSelector`: A3-prior, батч-скан, `_ROTATION_VEC`, `_rotate_point_np90` | ~120 |
| `candidate_retriever.py` | `CandidateRetriever`: `_retrieve_candidates`, patchify-розширення, `_merge_candidates` | ~120 |
| `geometric_verifier.py` | `GeometricVerifier`: цикл кандидатів, матчинг, RANSAC, early stop, RMSE | ~120 |
| `result_builder.py` | FOV-розрахунок + захист від exploded homography, `_compute_confidence`, формування dict-результату | ~150 |
| `failure_log.py` | `FailureLogger`: `FAILURE_TYPES`, `_log_failure` (CSV) | ~40 |
| `localizer.py` | Оркестрація: `localize_frame` стає ~80 рядків послідовних викликів; `localize_optical_flow` лишається (залежить від `_last_state`) | ~250 |

**Приклад коду (ядро):**

```python
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

class GeometricVerifier:
    def __init__(self, matcher, database, cfg: LocalizationConfig, hcfg: HomographyConfig):
        ...
    def verify(self, query_features: dict,
               candidates: list[tuple[int, float]]) -> VerificationResult | None:
        best: VerificationResult | None = None
        for cand_id, _score in candidates:
            ref = self.database.get_local_features(cand_id)
            mq, mr = self.matcher.match(query_features, ref)
            if len(mq) < self.cfg.min_matches:
                continue
            H, mask = GeometryTransforms.estimate_homography(mq, mr, ...)
            ...
            if best and best.inliers >= self.cfg.early_stop_inliers:
                break
        return best
```

```python
# localizer.py (після рефакторингу)
def localize_frame(self, frame, static_mask=None, dt=1.0) -> dict:
    if self._guard_out_of_coverage():
        return self._fail("out_of_coverage")
    frame, scale = self.normalizer.normalize(frame)
    rot = self.rotation_selector.select(frame, self.retriever)      # кут + кандидати
    if rot is None:
        return self._fail("no_retrieval_candidates")
    cands = self.candidate_retriever.expand(rot)                    # patchify merge
    feats = self.feature_extractor.extract_local_features(rot.frame, rot.mask)
    ver = self.geometric_verifier.verify(feats, cands)
    if ver is None:
        return self._fallback_or_fail(cands, rot)
    return self.result_builder.build(ver, rot, self._track(ver, rot))
```

**Ризики.** (а) `_last_state` використовується `localize_optical_flow` і TrackingWorker через property `last_state` — контракт зберегти дослівно. (б) Порядок side-effect'ів (`_consecutive_failures`, `_last_best_angle`, перемикання `self.database` у мульти-режимі) впливає на поведінку — переносити разом із станом, не "чистити" попутно. (в) Перемикання database/calibration у мульти-режимі мутує self — у новій схемі передавати active-контекст явно. Мітигація: спершу characterization-тести (п.4), рефакторинг без зміни жодної формули.

### 1.2 pose_graph_optimizer.py (915 рядків) — ✅ ВИКОНАНО

**Аналіз.** Стан кращий, ніж підказує розмір: residuals/jacobian уже векторизовані і локалізовані в `_residuals_vec`/`_jacobian_vec`, prune і діагностика — окремі групи методів. Головні проблеми: (а) дублювання формули резидуала у 3 місцях (`_residuals_vec`, `_single_edge_residual`, `_predict_forward`) — зміна моделі вимагає синхронного редагування; (б) GeoJSON-експорт і текстова діагностика — не справа оптимізатора; (в) `optimize()` рекурсивно викликає себе для two-stage prune.

**План.**

- `pose_graph/model_5dof.py` — ЄДИНЕ місце формул: `edge_residual()`, `edge_jacobian_blocks()`, `predict_forward/inverse`, `_affine_to_state/_state_to_affine`. `_residuals_vec` і `_single_edge_residual` викликають одне ядро.
- `pose_graph/optimizer.py` — граф, BFS/warm start, `optimize()` (без рекурсії: цикл `for stage in range(2)`).
- `pose_graph/pruning.py` — `_prune_bad_spatial_edges`, `_anchor_reachable`.
- `pose_graph/diagnostics.py` — residual stats, anchor stress, `format_diagnostics`, `export_graph_geojson`.
- Реекспорт у `src/geometry/pose_graph_optimizer.py` для сумісності: `from src.geometry.pose_graph.optimizer import PoseGraphOptimizer, GraphEdge, homography_to_affine`.

**Ризики.** Найвищі в усьому п.1 — це чисельне ядро. Мітигація вже існує: `test_pose_graph_jacobian.py` звіряє аналітичний якобіан з FD, `test_pose_graph_regression.py` фіксує розв'язки. Додатково: зафіксувати bit-exact еталон (cost, x) на синтетичному графі до/після. Дублювання формул усувати ОСТАННІМ кроком, окремим комітом.

### 1.3 database_builder.py (886 рядків)

**Аналіз.** `build_from_video()` — ~520 рядків з вкладеними замиканнями (`prefetch_frames`, `_flush_mask_batch`, `_process_single_frame`), що тягнуть 10+ змінних із зовнішнього скоупу. Змішано: декодування (decord/cv2 + prefetch-потік), завантаження моделей, детекція розмірностей, keyframe selection, depth-кешування, малювання kp-відео, HDF5 + LanceDB. `_temp_model_manager` — прихована залежність (див. п.2.4).

**План.**

| Модуль | Відповідальність |
|---|---|
| `database/video_frame_source.py` | `FrameSource` (Protocol) + `DecordFrameSource` / `Cv2FrameSource` + prefetch Queue. Ітератор `(idx, bgr, rgb)` |
| `database/keyframe_selector.py` | `_is_significant_motion`, `_compute_inter_frame_H`, накопичення pose chain |
| `database/frame_processor.py` | features + patchify + depth (з кешем depth_every_n) для одного кадру/батчу |
| `database/db_writer.py` | `create_hdf5_structure`, `save_frame_data`, LanceDB batch/index, frame_index_map |
| `database/keypoint_video_writer.py` | вибір кодека, `_draw_keypoints_frame` |
| `database_builder.py` | оркестратор ~150 рядків |

**Приклад (розрив замикань):**

```python
class DatabaseBuilder:
    def __init__(self, output_path, model_manager, matcher=None, config=None):  # DI, п.2.4
        ...
    def build_from_video(self, video_path, progress_callback=None, ...):
        source = create_frame_source(video_path, self.config)      # decord | cv2
        writer = DbWriter(self.output_path, source.meta, self.config)
        processor = FrameProcessor(self.feature_extractor, self.config)
        selector = KeyframeSelector(self.matcher, self.config, source.meta)
        for batch in source.batches():
            masks = self.masking.get_mask_batch([f.rgb for f in batch])
            for f, mask in zip(batch, masks):
                feats = processor.process(f, mask)
                pose, is_key = selector.step(feats)
                writer.write_pose(f.idx, pose)
                if is_key:
                    writer.write_frame(f.idx, feats, pose)
        writer.finalize()
```

**Ризики.** (а) Семантика "pose пишеться ЗАВЖДИ, keyframe — вибірково" критична для пропагації — зафіксувати тестом. (б) frame_id ↔ slot identity (збереження за оригінальним p_idx) ламати не можна — калібрування конвертує номери кадрів у слоти через `frame_step`. (в) SWMR/finally-блок (закриття HDF5, LanceDB flush + index) — перенести у `writer.finalize()` з тим самим try/finally.

### 1.4 calibration_propagation_worker.py (870 рядків)

**Аналіз.** QThread, але бізнес-логіка (temporal edges, loop closures, фізичні гейти, cluster consistency, збереження, інтерполяція прогалин) уже розкладена по методах — у потоковій частині лише `run()`, `stop()` і progress-сигнали.

**План.** Майже механічний move:
- `src/calibration/propagation_pipeline.py` — клас `PropagationPipeline` з чистими методами (`build_temporal_edges`, `detect_loop_closures`, `save_to_hdf5`, `fill_gaps_by_interpolation`...). Прогрес — через callback `Callable[[str, int], None]`, скасування — через `threading.Event`.
- `workers/calibration_propagation_worker.py` — тонкий QThread (~80 рядків): створює pipeline, транслює callback у сигнали, `stop()` ставить Event.

**Вигода:** pipeline тестується без Qt і працює у headless-режимі без QThread. **Ризики:** мінімальні; головне — сигнали прогресу з тими самими текстами (GUI парсить/показує їх).

### 1.5 calibration_dialog.py (891) + calibration_mixin.py (699)

**Аналіз.** Розподіл відповідальності зараз: dialog = введення якорів/точок, mixin = життєвий цикл калібрування у MainWindow + запуск propagation worker + оновлення карти. Дублюються стани ("яке калібрування активне", "чи йде пропагація").

**План.** Не консолідувати в один файл (він стане 1600 рядків), а розділити за MVP:
- `src/gui/controllers/calibration_controller.py` — увесь non-widget стан і логіка з mixin: створення/запуск worker, реакції на результат, синхронізація MultiCalibrationManager. Спілкується з GUI через Qt-сигнали.
- dialog лишається view: збирає точки, віддає `CalibrationRequest` (dataclass), нічого не знає про worker.
- mixin зводиться до ~50 рядків glue або зникає (див. п.2.1).

**Ризики:** середні — регресії в UI-флоу видно лише вручну. Робити після 1.4 (controller тоді працює з чистим pipeline).

### 1.6 model_manager.py (736 рядків)

**Аналіз.** Три ролі в одному класі: (а) VRAM-облік/евікшн/pinning; (б) реєстр моделей (dict + usage timestamps); (в) 8 методів `load_*` по ~60-100 рядків з повторюваним шаблоном lock→check→ensure_vram→try/log→register. Плюс деталі експорту (TRT для YOLO, TorchScript для LightGlue) вшиті в лоадери.

**План.**

```python
# models/vram.py
class VramBudget:
    def available_mb(self) -> float: ...
    def ensure(self, required_mb: float, evictable: dict[str, float],
               pinned: set[str]) -> list[str]:  # повертає кого вивантажити

# models/registry.py
@dataclass
class ModelSpec:
    name: str
    vram_mb: float
    loader: Callable[[], Any]        # замикання з конфігом

class ModelRegistry:
    """LRU-реєстр: get() = load-if-missing + evict-if-needed + touch."""
    def get(self, name: str) -> Any: ...
    def pin(self, names: list[str]) -> None: ...
    def unload(self, name: str) -> None: ...

# models/loaders/  — yolo.py, dinov2.py, dinov3.py, aliked.py, rdd.py,
#                    lightglue.py (+ export у lightglue_export.py), cesp.py
```

`ModelManager` лишається фасадом зі старими методами `load_dinov2()` тощо, які делегують у `registry.get("dinov2")` — жоден викликач не змінюється.

**Ризики:** низькі. Увага на: (а) `_model_lock` має охоплювати load+evict атомарно, як зараз; (б) DINOv3 живе під іменем "dinov2" у реєстрі — зберегти (на цьому сидить `load_dinov2()` контракт).

---

## 2. Архітектурні покращення

### 2.1 MainWindow: mixins → контролери

**Аналіз.** `MainWindow(CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin, QMainWindow)` — MRO з 5 баз, ~2000 рядків сумарної поведінки на одному екземплярі, спільний неявний стан через self.*. Класичний випадок, коли міксини використані не для перевикористання, а для розпилу одного класу.

**План.** Composition + контролери (Presenter-стиль, без важких фреймворків):

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_ui()
        ctx = AppContext(project_manager, model_manager, config)   # п.2.4
        self.database_ctrl = DatabaseController(ctx, parent=self)
        self.calibration_ctrl = CalibrationController(ctx, parent=self)
        self.tracking_ctrl = TrackingController(ctx, parent=self)
        self.panorama_ctrl = PanoramaController(ctx, parent=self)
        # зв'язки — тільки сигналами:
        self.control_panel.build_requested.connect(self.database_ctrl.start_build)
        self.database_ctrl.progress.connect(self.control_panel.set_progress)
        self.tracking_ctrl.position_updated.connect(self.map_widget.update_position)
```

Кожен контролер — `QObject` (живе в GUI-потоці, володіє своїми worker'ами), НЕ віджет. Міграція інкрементальна: по одному міксину за крок, mixin-клас лишається порожнім аліасом до завершення.

**Ризики:** середні. Найтонше — прихована взаємодія міксинів між собою через self (наприклад, tracking читає стан calibration). Перед міграцією — grep усіх `self.<attr>` кожного міксина і явний перелік спільного стану: він і переїде в `AppContext`.

### 2.2 Розбити config.py по доменах — ✅ ВИКОНАНО

**Аналіз.** 450 рядків, ~24 моделі — ще не критично, але файл росте, і кожна правка конфлікт-магніт.

**План.** Пакет `config/` зі збереженням усіх імпортів:

```
config/
├── __init__.py        # re-export: AppConfig, get_cfg, APP_SETTINGS, APP_CONFIG, ...
├── models.py          # Dinov2/Dinov3/GlobalDescriptor/Yolo/ModelSettings/Cesp/Vram/Performance
├── localization.py    # LocalizationConfig, ConfidenceConfig, TrackingConfig, HomographyConfig
├── database.py        # DatabaseConfig
├── graph.py           # GraphOptimizationConfig, ProjectionConfig
├── app.py             # AppConfig, GuiConfig, NetworkApiConfig, LiveStreamConfig, Preprocessing
└── access.py          # get_cfg, get_active_descriptor_cfg, load/save_user_config
```

`config/config.py` лишити на 1 реліз як `from config import *` + DeprecationWarning. Ризики: майже нульові (чистий move + реекспорт); єдина пастка — `import config.config` у PyInstaller spec/hooks, перевірити `DroneLocalization.spec`.

### 2.3 get_cfg(): чи можна замінити на нативний Pydantic-доступ?

**Аналіз.** Так, і варто — але поетапно. `get_cfg` існує з трьох причин: (а) одні місця отримують `APP_SETTINGS` (Pydantic), інші — `APP_CONFIG` (dict-дамп) або довільні dict; (б) дефолти продубльовані в кожному виклику (`get_cfg(cfg, "localization.min_matches", 12)` — дефолт 12 живе і тут, і в `LocalizationConfig`, ризик розсинхрону — і він уже траплявся, судячи з коментаря про ratio_threshold); (в) усередину вшито PyInstaller-хак підміни шляхів `models/*` — прихований side-effect у функції доступу до конфігу.

**План.**
1. Витягти PyInstaller-хак у `utils/paths.py: resolve_model_path(p)` і викликати явно в лоадерах моделей. `get_cfg` стає чистим.
2. Прибрати дуальність: усі конструктори приймають типізований під-конфіг (`LocalizationConfig`), а не весь config-dict. `self.min_matches = cfg.min_matches` — дефолт живе лише в Pydantic.
3. `APP_CONFIG` (dict) — deprecated; лишити тільки для serialization boundary (user_config.json).
4. `get_cfg` залишити тимчасово як shim; видалити, коли grep покаже нуль викликів.

**Ризики:** великий обсяг дрібних правок (147 викликів get_cfg). Робити модуль за модулем разом з рефакторингом п.1, не окремою "великою заміною". Тест `test_config_sync.py` уже існує — розширити.

### 2.4 self._temp_model_manager → DI

**Аналіз.** Єдине місце — `DatabaseBuilder`: `build_from_video()` кладе model_manager у `self._temp_model_manager`, щоб `_compute_inter_frame_H` міг ліниво створити FeatureMatcher. Також `Localizer` дістає manager і project_manager з config-dict через магічні ключі `config["_model_manager"]`, `config["_project_manager"]` — той самий антипатерн у профіль.

**План.** Конструкторна ін'єкція:

```python
class DatabaseBuilder:
    def __init__(self, output_path, model_manager: ModelManager,
                 matcher: FeatureMatcher | None = None, config: AppConfig = ...):
        self.model_manager = model_manager
        self.matcher = matcher or FeatureMatcher(model_manager, config)

class Localizer:
    def __init__(self, ..., model_manager: ModelManager | None = None,
                 project_settings: ProjectSettings | None = None):
```

Магічні ключі `_model_manager`/`_project_manager` підтримати 1 реліз з DeprecationWarning. Service locator не потрібен — граф залежностей малий і збирається в MainWindow/HeadlessRunner. **Ризики:** низькі; знайти всіх викликачів конструкторів (GUI mixins, headless_runner, тести).

---

## 3. Типізація та якість коду

### 3.1 Protocol-інтерфейси — ✅ ВИКОНАНО (src/interfaces.py)

**Аналіз.** `Localizer` приймає `database`, `feature_extractor`, `matcher`, `calibration` без типів; сумісність тримається на duck typing і `hasattr`-перевірках (`hasattr(self.database, "lance_table")`, `hasattr(fe, "extract_global_descriptors_multi")`). Реальні "інтерфейси" вже існують неявно: два ретрівери (FastRetrieval, LanceDBRetrieval), стратегії маскування (masking_strategy.py з фабрикою), два екстрактори (ALIKED/RDD через wrappers).

**План.** `src/interfaces.py` (або по одному Protocol поруч із реалізаціями):

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Retriever(Protocol):
    def find_similar_frames(self, query_desc: np.ndarray,
                            top_k: int = 5) -> list[tuple[int, float]]: ...
    def add_descriptor(self, query_desc: np.ndarray, frame_id: int) -> None: ...

class GlobalDescriptorExtractor(Protocol):
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray: ...
    def extract_global_descriptors_multi(
        self, images: list[np.ndarray]) -> np.ndarray: ...

class LocalFeatureExtractor(Protocol):
    def extract_local_features(self, image: np.ndarray,
                               static_mask: np.ndarray | None = None) -> dict: ...

class MaskingStrategy(Protocol):
    def get_mask_batch(self, images_rgb: list[np.ndarray]) -> list[np.ndarray | None]: ...

class FrameDatabase(Protocol):
    def get_local_features(self, frame_id: int) -> dict: ...
    def get_frame_affine(self, frame_id: int) -> np.ndarray | None: ...
    def get_frame_size(self, frame_id: int) -> tuple[int, int]: ...
    def get_num_frames(self) -> int: ...
```

Protocol, а не ABC: нуль змін у реалізаціях, mypy перевіряє структурно, `runtime_checkable` дає isinstance для тестів. Побічний бонус: `extract_global_descriptors_multi` стає обов'язковим контрактом — hasattr-гілку в localizer можна прибрати.

### 3.2 dict config → типізований Pydantic

Це п.2.3, кроки 2-3. Ключове правило: **дефолт існує лише в Pydantic-моделі**. Патерн `config=None` → `config or {}` замінити на обов'язковий типізований параметр.

### 3.3 Консистентний error handling

**Аналіз.** Bare except відсутній. Проблема інша: `except Exception` як універсальна відповідь — ~100 місць, три різні реакції (warning+fallback, error+raise, мовчазний `pass` у FOV gps-corners). Частина обґрунтована (optional-фічі: patchify, depth, CESP, TRT), частина маскує баги (`except Exception` навколо координатних перетворень).

**План.**
1. `src/exceptions.py`: `DroneLocError(Exception)` → `ModelLoadError`, `DatabaseFormatError`, `CalibrationError`, `VideoDecodeError`.
2. Правило: **optional-фіча** → `except (ImportError, RuntimeError) as e: logger.warning(...)`, максимально вузький тип; **обов'язковий шлях** → ловити лише очікувані типи, загорнути в доменний exception, re-raise; **`except Exception: pass`** — заборонити (Ruff: `S110`), мінімум `logger.debug`.
3. Увімкнути Ruff-правила `BLE001` (blind except) + `TRY` як warning, чистити поступово.

**Ризики:** звуження типів може "розгерметизувати" місця, де широкий except ховав реальний баг — саме тому робити після тестів п.4, по модулю за раз.

### 3.4 Type hints

Стиль уже уніфікований (`X | None`). Залишок роботи: (а) анотувати публічні сигнатури `Localizer`, `DatabaseBuilder`, `ModelManager` (зараз конструктори без типів); (б) додати `mypy --strict` для `src/geometry`, `src/localization`, `config` у CI (де типізація вже майже повна), решту — поступово через per-module overrides.

---

## 4. Тести та CI (робити ПЕРШИМ)

**Аналіз.** ~24 test-файли, сильне покриття geometry/pose_graph (jacobian, regression, two-stage, warmstart, diagnostics — зразкове), є unit/ (project, calibration, image utils) і один integration/test_pipeline.py. Дірки: Localizer (test_localization.py — мінімальний), DatabaseBuilder, ModelManager, CalibrationPropagation, matcher.

**План.**

1. **MockModelManager** (`tests/fixtures/mock_models.py`) — ключ до всього без-GPU тестування:

```python
class FakeGlobalExtractor:
    """Детерміністичні дескриптори: hash кадру → напрямок у R^d."""
    def __init__(self, dim=1024): self.dim = dim
    def extract_global_descriptor(self, image):
        rng = np.random.default_rng(abs(hash(image.tobytes())) % 2**32)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)
    def extract_global_descriptors_multi(self, images):
        return np.stack([self.extract_global_descriptor(i) for i in images])

class MockModelManager:
    device = "cpu"
    def load_dinov2(self): return FakeGlobalExtractor()
    def load_local_extractor(self): return FakeLocalExtractor()  # grid keypoints
    def load_yolo(self): return None
    def pin(self, m): ...
    def get_available_vram_mb(self): return float("inf")
```

2. **Localizer unit-тести** (з fake extractor + in-memory database + identity matcher): успішна локалізація на синтетиці; out_of_coverage guard після N невдач; retrieval-only fallback (score вище/нижче 0.90); outlier відфільтрований; A3-prior — що при high score НЕ робиться повний скан (лічильник викликів екстрактора); rotation math — параметризований тест `_rotate_point_np90` проти np.rot90 на реальних масивах.
3. **DatabaseBuilder**: синтетичне відео (30 кадрів, рухомий патерн, cv2.VideoWriter у tmp) → build → перевірити HDF5 схему v2, frame_index_map, "pose пишеться завжди, keyframe вибірково", frame_step attrs.
4. **ModelManager**: евікшн-логіка з fake-моделями і підміненим `get_available_vram_mb` (monkeypatch): LRU-порядок, pinned не вивантажується, warning при all-pinned.
5. **Integration flow** `tests/integration/test_full_cycle.py`: synth video → build DB → 2 anchors → PropagationPipeline (після 1.4 — без Qt) → localize кадри з того ж відео → медіанна помилка < GSD-порога. Це САМЕ той тест, що дозволяє все інше рефакторити сміливо.
6. **CI**: GitHub Actions, matrix py3.10/3.11, `ruff check` + `pytest -m "not gpu"`; маркер `@pytest.mark.gpu` для реальних моделей.

**Ризики:** немає — тільки час. Це передумова, а не ризик.

---

## 5. Performance

### 5.1 localizer: rotation scan — УЖЕ вирішено; справжні резерви

Батчований скан (A2) + кутовий prior (A3) уже дають 1 forward на типовий keyframe. Що лишилось:

1. **Патчифай — 14 forward-пасів** на keyframe при use_patchify=True (grids 1+4+9, `patchify_batch_size` дефолт 1 — послідовно!). Дешевий виграш: дефолт `patchify_batch_size: 7` (два батчі) або 14 (один), це лише зміна конфігу + бенчмарк VRAM. Стратегічно: патч-дескриптори можна знімати з ОДНОГО forward DINOv2 через пулінг patch-tokens по регіонах сітки замість окремих пасів на кожен кроп — ~14× менше обчислень, потребує перевірки якості retrieval на еталонному відео.
2. **Цикл кандидатів** — до `top_k*2 = 24` послідовних LightGlue-матчингів у гіршому разі. LightGlue підтримує батчинг пар; батч по 4-8 кандидатів до early-stop-перевірки дасть ~2-3× на цьому етапі. Складність: різна кількість keypoints → падінг.

### 5.2 database_builder: паралелізація YOLO + DINOv2

**Аналіз.** Конвеєр уже непоганий: decord batch decode + prefetch-потік, YOLO micro-batching (yolo_batch_size), depth_every_n, рідкий empty_cache (A1). Але у межах кадру YOLO → ALIKED → DINOv2 — строго послідовно на одному GPU.

**План (за віддачею):**
1. **Батчинг DINOv2 по кадрах** (не streams): збирати 8-16 кадрів і робити один forward — простіше і швидше за streams, ViT відмінно масштабується по batch dim. Вимагає перебудови `_process_single_frame` на батчі (природно після 1.3).
2. **CUDA streams** для overlaps YOLO(TRT)/DINOv2 — реальний виграш зазвичай 10-20% і додає складності синхронізації; робити лише після 1 і бенчмарка. Копіювання H2D → окремий stream + pinned memory — дешевша половина цього виграшу.
3. Профілювання вже є (`Telemetry.profile`) — почати зі звіту, де реально час: video_read/yolo/extract/hdf5_write.

### 5.3 VRAM: чи є витоки при unload/reload?

**Аналіз коду:** `_unload_model_unsafe` робить del + empty_cache + gc — коректно. Знайдені реальні ризики:
1. **torch.compile:** скомпільовані DINOv2/ALIKED тримають inductor-кеші поза життєвим циклом модуля; після unload частина VRAM не повертається. Мітигація: `torch._dynamo.reset()` при unload скомпільованої моделі (задокументувати, що це скидає ВСІ compiled-кеші).
2. **ONNX Runtime LightGlue:** `ort.InferenceSession` звільняє GPU-пам'ять лише при знищенні сесії; del з існуючими посиланнями (worker тримає model) — витік. Переконатися, що воркери не кешують посилання на моделі довше за задачу.
3. **Цикли посилань:** FeatureExtractor тримає local+global model; якщо Localizer/DatabaseBuilder живуть довше — del у менеджері не звільняє пам'ять (посилання живе). Рецепт: воркери отримують моделі на час задачі і зануляють у finally; додати діагностику `torch.cuda.memory_allocated()` до/після unload у DEBUG.
4. Тест: цикл load→infer→unload ×20 з assert стабільності `memory_allocated` (gpu-маркер).

---

## 6. Документація

1. **Docstrings (Google style).** Пріоритет: публічні API `Localizer`, `DatabaseBuilder`, `PoseGraphOptimizer`, `ModelManager`, protocols з п.3.1. Увімкнути Ruff `D` (pydocstyle, google convention) лише для цих модулів через per-file-ignores, розширювати поступово. Мова: українська для пояснень, англійська для термінів — як уже склалось у коді.
2. **ARCHITECTURE.md** з mermaid-діаграмами трьох потоків: build (video → decode → mask → features → keyframe → HDF5/LanceDB), calibration (anchors → temporal/spatial edges → 5-DoF LM → frame_affine), localization (frame → rotation select → retrieval → match/RANSAC → affine → GPS → Kalman). Плюс таблиця модулів і діаграма VRAM-життєвого циклу моделей.
3. **Математика pose_graph.** Найкраще місце — не inline-коментарі, а `docs/POSE_GRAPH_MATH.md`: модель стану [cx, cy, log sx, log sy, θ], чому лог-масштаби, формула резидуала і повний вивід якобіана (у коді j0_lxi тощо зараз без пояснень), зважування w/sx проти w·cx, регуляризатор ізотропії 200·cx, чому L2 без robust loss (коментар з config.py — перенести сюди). У коді — короткі посилання на розділи документа.

---

## Зведена таблиця пріоритетів

| # | Задача | Цінність | Ризик | Залежить від |
|---|---|---|---|---|
| 1 | ✅ MockModelManager + Localizer/Builder тести (п.4) | висока | ~0 | — |
| 2 | ◑ Integration flow test (п.4.5) | висока | ~0 | 1, частково 1.4 |
| 3 | 🔲 PropagationPipeline із worker (п.1.4) | висока | низький | 1 |
| 4 | 🔲 ModelManager → registry/vram (п.1.6) | середня | низький | 1 |
| 5 | 🔲 database_builder розбиття (п.1.3) + DI (п.2.4) | висока | середній | 1, 2 |
| 6 | ✅ localizer розбиття (п.1.1) | висока | середній | 1, 2 |
| 7 | ✅ config/ пакет (п.2.2) + get_cfg-міграція (п.2.3) | середня | низький | поступово, разом з 5-6 |
| 8 | ✅ pose_graph пакет (п.1.2) | середня | середній* | існуючі тести |
| 9 | 🔲 Mixins → контролери (п.2.1, включно з 1.5) | середня | середній | 3 |
| 10 | 🔲 Performance: patchify batch, candidate batch, DINOv2 frame batch (п.5) | середня | середній | 5, 6, бенчмарки |
| 11 | ◑ Error handling + Protocols + mypy (п.3) | середня | низький | розмазано по 5-9 |
| 12 | ✅ ARCHITECTURE.md + docstrings (п.6) | середня | 0 | після 5-9 |

\* ризик чисельного ядра, але покритий найкращими тестами в проєкті.
