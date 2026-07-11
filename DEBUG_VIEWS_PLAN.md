# DEBUG_VIEWS_PLAN — вікна візуалізації моделей у режимі локалізації

**Мета:** окремі вікна (QDockWidget), що показують покадрово «очима моделей»: YOLO, Depth Anything, DINO (v2/v3), точки/матчі ALIKED/RDD. Вмикаються/вимикаються з меню «Вигляд», працюють під час локалізації (tracking). Все off за замовчуванням, нуль overhead при вимкнених вікнах.

---

## 1. Архітектура даних

### 1.1 Канал worker → GUI

- Новий сигнал у `RealtimeTrackingWorker` (`src/workers/tracking_worker.py`):
  `debug_view_ready = pyqtSignal(str, np.ndarray)` — (ім'я каналу, готове BGR-зображення).
- Worker тримає thread-safe set активних каналів (`threading.Lock` + `set`);
  GUI оновлює його методом `set_debug_channels(channels: set[str])`.
- Канал не активний → нічого не рендериться і не emit-иться.
- Рендер overlay (cv2) виконується у worker-потоці, downscale до `max_width` (~640 px),
  копія масиву при emit (масив перетинає межу потоків).

### 1.2 DebugCollector у Localizer

`localize_frame` зараз не віддає назовні keypoints/matches/retrieval. Додати opt-in
колектор (`src/localization/localizer.py`, за замовчуванням `None` — нульовий кошт):

- повернутий/GSD-нормалізований кадр (`best_rotated_frame`);
- `best_query_features` (keypoints ALIKED/RDD);
- inlier-матчі (`best_mkpts_q_inliers`, `best_mkpts_r_inliers`), `total_matches`, `rmse`;
- `candidate_id` + retrieval top-k scores, обраний кут (`best_global_angle`) і масштаб (`best_scale`);
- patch-токени DINO (наповнює `FeatureExtractor`, лише коли collector активний).

Worker після `localize_frame` читає collector і рендерить активні канали.

---

## 2. Чотири канали (вікна)

### 2.1 YOLO
Кадр + bbox/сегментаційні маски з класом і confidence + напівпрозорий `static_mask`.
Дані (`frame_rgb`, `detections`, `static_mask`) вже є в keyframe-циклі tracking_worker —
найпростіший канал, реалізується без DebugCollector.

### 2.2 Depth (Depth Anything)
Colormap depth-мапи (`cv2.applyColorMap`) + значення relative scale.
Зараз depth рахується лише кожні `depth_hint_every_n=30` keyframes і лише якщо БД має
`median_depth_scale`. Для вікна — рахувати on-demand, коли вікно відкрите
(окремий параметр `debug_views.depth_every_n_keyframes`, бо це додатковий GPU-прохід).

### 2.3 DINO (v2/v3)
- PCA патч-токенів: 3 головні компоненти → RGB heatmap поверх кадру (стандартна DINO-візуалізація).
- Панель retrieval: top-k кандидатів (id, score), обраний кут і масштаб.
- Обмеження: HDF5-БД зберігає лише фічі, без зображень — кандидат показується як id/score, не картинкою.

### 2.4 Точки/Матчі (ALIKED/RDD)
- Всі keypoints — сірим, inliers — зеленим, вектори зміщення через H.
- Лічильники: kpts / total_matches / inliers / RMSE.
- Через відсутність реф-зображень у БД — тільки query-бік. Side-by-side з реф-кадром —
  окремий майбутній етап (збереження thumbnails у БД = зміна формату БД).

---

## 3. GUI

- Новий файл `src/gui/widgets/debug_view.py`: клас `DebugViewDock`
  (QDockWidget + QLabel зі scaled pixmap, метод `update_frame(np.ndarray)`).
- `MainWindow._init_ui`: 4 доки, tabified у нижній/правій зоні, приховані за замовчуванням.
- Меню «Вигляд» → підменю «Вікна моделей» з `toggleViewAction()` кожного дока
  (той самий патерн, що вже застосовано для доків карти/панелі управління).
- `visibilityChanged` кожного дока → перерахунок enabled-set у worker.
- `tracking_mixin._start_tracking_worker`: підключити `debug_view_ready` → маршрутизація
  по імені каналу; при старті передати worker-у поточний стан видимості вікон.
- Стан видимості зберігати в `user_config.json` (секція `debug_views`).

---

## 4. Конфіг

Секція `debug_views`:
- `max_width: 640` — ширина зображень у вікнах;
- `depth_every_n_keyframes` — частота depth-інференсу для вікна;
- `dino_pca_enabled` — вмикає обчислення PCA токенів.

Все off за замовчуванням.

---

## 5. Етапи впровадження

1. **Каркас:** `DebugViewDock`, 4 доки, меню, маршрутизація сигналу — вікна показують сирий keyframe.
2. **YOLO-канал** (дані вже є у worker).
3. **DebugCollector** у localizer + вікно точок/матчів.
4. **DINO:** PCA токенів + retrieval-панель.
5. **Depth:** on-demand estimate + colormap.
6. **Перф-перевірка:** FPS локалізації з закритими вікнами == поточному;
   з відкритими — throttle (рендер не частіше keyframe-рейту, drop замість черги).

---

## 6. Ризики / обмеження

1. Оновлення лише на keyframes (кожен `keyframe_interval`-й кадр, типово 5-й) —
   вікна оновлюються з частотою ~6 Гц; OF-кадри цих даних не мають, це очікувано.
2. Depth і DINO-PCA — реальний GPU-кошт → обчислюються строго при відкритому вікні.
3. Без зображень у БД match/retrieval-вікна показують лише query-бік;
   повний side-by-side потребує thumbnails у БД (окремий етап, перезбірка БД).
