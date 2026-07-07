# Архітектура DroneLocalization

Топометрична локалізація дрона за відео: будуємо базу кадрів із карти-відео,
калібруємо її під GPS кількома якорями, потім локалізуємо live-кадри відносно бази
й проєктуємо в GPS. Три незалежні потоки — **build**, **calibration**,
**localization** — плюс менеджер моделей (VRAM) під ними.

---

## Компоненти

```mermaid
graph TB
    subgraph GUI["GUI - PyQt6"]
        MW["MainWindow + controllers/mixins"]
        CP["ControlPanel"]
        MAP["MapWidget"]
        VID["VideoWidget"]
    end
    subgraph WK["Workers - QThread"]
        DBW["DatabaseWorker"]
        CPW["CalibrationPropagationWorker"]
        TW["TrackingWorker"]
    end
    subgraph CORE["Core"]
        DB["DatabaseBuilder"]
        LOC["Localizer"]
        PGO["PoseGraphOptimizer"]
        MM["ModelManager - VRAM"]
    end
    subgraph STORE["Storage"]
        HDF["HDF5 - features, poses, frame_affine"]
        LDB["LanceDB - global descriptors"]
    end
    MW --> DBW & CPW & TW
    DBW --> DB
    CPW --> PGO
    TW --> LOC
    DB --> MM & HDF & LDB
    LOC --> MM & HDF & LDB
    CPW --> HDF
    LOC --> MAP
```

---

## Потік 1 — Побудова бази (build)

`DatabaseBuilder.build_from_video` (`src/database/database_builder.py`).

```mermaid
flowchart TD
    A["Відкрити відео - decord або cv2"] --> B["prefetch-потік - Queue RGB кадрів"]
    B --> C["Батч кадрів"]
    C --> D["YOLO-seg маска динаміки - masking_strategy"]
    D --> E["Локальні фічі - ALIKED або RDD"]
    E --> F["Глобальний дескриптор - DINOv2 або v3"]
    F --> G["Inter-frame H - matcher.match до попереднього"]
    G --> H{"Значний рух? - _is_significant_motion"}
    H -->|так, keyframe| I["Зберегти фічі у HDF5 + додати у LanceDB-батч"]
    H -->|ні| J["Записати лише позу"]
    I --> K["Наступний кадр"]
    J --> K
    K --> C
    K --> L["Finalize: LanceDB add + create_index, close HDF5"]
```

Ключове: **поза пишеться ЗАВЖДИ**, keyframe (з фічами) — вибірково за рухом; глибина
(Depth-Anything) рахується раз на `depth_every_n` кадрів (масштаб змінюється повільно).

---

## Потік 2 — Калібрувальна пропагація (calibration)

`CalibrationPropagationWorker.run` (`src/workers/calibration_propagation_worker.py`)
→ будує й розв'язує 5-DoF pose graph (див. `POSE_GRAPH_MATH.md`).

```mermaid
flowchart TD
    A["Передзавантажити фічі у RAM"] --> B["PoseGraphOptimizer - frame_w, frame_h"]
    B --> C["Часові ребра - sequential match, homography_to_similarity, add_edge temporal"]
    C --> D["Просторові замикання - retrieval top-k, match, гейти, add_edge spatial"]
    D --> E["Фіксація GPS-якорів - fix_node, Local Origin"]
    E --> F["Warm start або BFS-ініціалізація"]
    F --> G["optimize - Levenberg-Marquardt / TRF, two_stage_prune"]
    G --> H["diagnostics_report - резидуали, anchor stress"]
    H --> I["Зберегти frame_affine у HDF5"]
    I --> J["Заповнити прогалини інтерполяцією"]
```

Прогрес транслюється в GUI через Qt-сигнали (`progress`, `completed`, `error`).
Гейти ребер (`edge_gate_*`) і two-stage prune — за прапорцями конфігу
(`GraphOptimizationConfig`), дефолти off = поточна поведінка.

---

## Потік 3 — Локалізація (localization)

`Localizer.localize_frame` (`src/localization/localizer.py`).

```mermaid
flowchart TD
    A["Кадр"] --> B["Guard покриття + нормалізація роздільності"]
    B --> C{"A3: prior кута успішний?"}
    C -->|так| D["1 forward DINOv2 на prior-кут"]
    C -->|ні| E["A2: батчований скан 4 кутів - extract_global_descriptors_multi"]
    D --> F["Retrieval кандидатів - retriever.find_similar_frames"]
    E --> F
    F --> G["Patchify-розширення + merge - опційно"]
    G --> H["Локальні фічі кадру - extract_local_features"]
    H --> I["Цикл кандидатів: match -> estimate_homography -> RANSAC/inliers, early stop"]
    I --> J{"Достатньо inliers?"}
    J -->|ні| K["Fallback: retrieval-only - _localize_by_reference_frame"]
    J -->|так| L["Affine -> метрична точка"]
    L --> M{"Outlier? - OutlierDetector"}
    M -->|так| N["Відкинути стрибок"]
    M -->|ні| O["Confidence + Kalman - TrajectoryFilter"]
    O --> P["FOV - із захистом від exploded homography"]
    P --> Q["Результат -> GPS -> MapWidget"]
    K --> Q
```

Мультиджерельний режим (`db_manager`, `calib_manager`) перемикає активну базу/калібрування
під час скану кандидатів. `localize_optical_flow` — легкий шлях між keyframe-ами.

---

## Модулі

| Пакет | Відповідальність |
|---|---|
| `config/` | Pydantic-конфіг по доменах: `models`, `database`, `localization`, `graph`, `app`, `access` |
| `src/interfaces.py` | Structural Protocols: `Retriever`, `GlobalDescriptorExtractor`, `LocalFeatureExtractor`, `FrameDatabase` |
| `src/database/` | `DatabaseBuilder`, `DatabaseLoader`, multi-db manager, spatial index |
| `src/localization/` | `Localizer`, `matcher` (retrieval + LightGlue), `patchify`, geo-aware retriever |
| `src/geometry/` | `pose_graph/` (5-DoF LM), `coordinates` (UTM/WebMercator), `affine_utils`, `gsd_calculator`, `transformations` |
| `src/models/` | `ModelManager` (VRAM), `wrappers/` (DINOv2/v3, ALIKED, RDD, YOLO, LightGlue, masking, CESP) |
| `src/calibration/` | multi-anchor calibration, multi-calibration manager |
| `src/tracking/` | Kalman `TrajectoryFilter`, `OutlierDetector`, object tracker/projector |
| `src/workers/` | QThread-обгортки: database, calibration propagation, tracking, panorama, video decode |
| `src/gui/` | `MainWindow`, mixins/controllers, widgets (map/video/control), dialogs |
| `src/network/` | REST + WebSocket сервери телеметрії, coordinates broker |
| `src/depth/`, `src/utils/`, `src/video/` | глибина, утиліти (I/O, логи, нормалізація, телеметрія), джерела відео |

---

## Життєвий цикл моделей (VRAM)

`ModelManager` (`src/models/model_manager.py`) — LRU-реєстр із бюджетом VRAM,
евікшном і pinning'ом. Моделі великі (DINOv2 ~1.6GB, LightGlue ~0.8GB), тож на 8GB-картах
одночасно тримається лише потрібний набір.

```mermaid
stateDiagram-v2
    [*] --> Missing
    Missing --> Loading: load_X()
    Loading --> Resident: ensure_vram (evict LRU якщо треба)
    Resident --> Resident: touch (оновити timestamp)
    Resident --> Pinned: pin()
    Pinned --> Resident: unpin
    Resident --> Missing: evict LRU / unload (del + empty_cache + gc)
    Pinned --> Pinned: не витісняється
```

Ризики витоків (див. `IMPROVEMENT_PLAN.md` п.5.3): `torch.compile` inductor-кеші,
ONNX Runtime сесії LightGlue, цикли посилань у воркерах — воркери мають брати моделі на
час задачі й занулювати у `finally`.

---

## Посилання

- Математика графа: `docs/POSE_GRAPH_MATH.md`
- План рефакторингу: `docs/IMPROVEMENT_PLAN.md`
- Потоки: `database_builder.py`, `calibration_propagation_worker.py`, `localizer.py`
