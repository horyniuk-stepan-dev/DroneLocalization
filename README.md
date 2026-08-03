# Drone Topometric Localization System

Професійна система топометричної локалізації та візуальної навігації дронів у середовищах без стабільного сигналу GPS. Завдяки використанню сучасних Foundation Models (базових моделей) та семантичного аналізу, система забезпечує високоточне визначення координат, стійке до жорстких змін освітлення, тіней, погодних умов та пори року.

## 🎯 Основні можливості

- **Desktop GUI (PyQt6)**: Багатопотоковий графічний інтерфейс для Windows, оптимізований для роботи в реальному часі.
- **Headless режим**: Серверне розгортання з WebSocket + REST API для зовнішніх інтеграцій.
- **Семантична глобальна локалізація (DINOv3 / DINOv2)**: Foundation Model від Meta для глобальних дескрипторів, стійких до змін освітлення, тіней та пір року. За замовчуванням — **DINOv3 ViT-L/16**, попередньо навчена на 493M супутникових знімків (`models.global_descriptor.backend = "dinov3"`); DINOv2 ViT-L/14 залишається як альтернативний бекенд.
- **VLAD-агрегація (AnyLoc, опційно)**: Ненавчена VLAD-агрегація патч-токенів DINOv3 замість CLS-дескриптора — вмикається у конфізі, потребує словника (`scripts/build_vlad_vocab.py`) і перебудови бази.
- **Фільтрація динамічних об'єктів (YOLOv11-Seg)**: Автоматичне нейромережеве маскування рухомих об'єктів (машин, людей) для прив'язки лише до стабільної геометрії.
- **Адаптивний препроцесинг (CLAHE)**: Локальне вирівнювання контрасту для витягування текстур із глибоких тіней.
- **Гібридний матчинг ознак**: Використання **ALIKED + LightGlue** для високоточного зіставлення ключових точок у складних сценаріях; **RDD** доступний як альтернативний детектор локальних ознак.
- **Оцінка глибини (Depth-Anything-V2)**: Монокулярна оцінка глибини для масштабно-усвідомленої локалізації.
- **Мульти-якірне калібрування**: Інтерактивне задання GPS-якорів з автоматичною оптимізацією через граф поз (5-DoF Levenberg-Marquardt).
- **Інтелектуальний трекінг**: Згладжування траєкторії через **Kalman Filter**, fixed-lag згладжувач (servo-режим) та детекція аномалій за Z-score.
- **Трекінг об'єктів**: Детекція та проєкція рухомих об'єктів із кадру в GPS-координати.
- **Мульти-база / мульти-калібрування**: Одночасне завантаження кількох баз та калібрувань із перемиканням активної під час локалізації.
- **Інтерактивна карта**: Візуалізація Field of View (FOV) та маршруту на Leaflet карті у реальному часі.
- **Експорт результатів**: CSV, GeoJSON та KML формати для збереження локалізаційних даних.
- **Шифрування проєкту (at-rest)**: Опційне парольне шифрування файлів проєкту (бази, калібрування, відео) з дешифруванням у пам'ять під час завантаження.
- **Автопідбір під залізо**: Автоматичне визначення GPU/CPU-профілю та тюнінг батчів, потоків і бюджету VRAM (`models.performance.auto_tune`).

## 📋 Вимоги до системи

### Апаратні вимоги
- **GPU**: NVIDIA з підтримкою CUDA (мінімум **4GB VRAM**, наприклад, GTX 1650 або краще)
- **CPU**: 6+ ядер
- **RAM**: 16 GB (рекомендовано 32 GB для великих баз даних)
- **Накопичувач**: SSD (рекомендовано)

### Програмні вимоги
- Python 3.10–3.11
- PyTorch ≥ 2.2.0 (CUDA 12.x)
- Windows 10/11 (основна платформа)

## 🚀 Швидкий старт

### 1. Встановлення

```powershell
# Клонування репозиторію
git clone https://github.com/horyniuk-stepan-dev/DroneLocalization.git
cd DroneLocalization

# Створення віртуального середовища
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Встановлення проєкту та залежностей (editable mode)
pip install -e .

# (Опціонально) Встановлення dev-залежностей (pytest, ruff, ...)
pip install -e ".[dev]"

# (Опціонально) Встановлення TensorRT прискорення для YOLO
pip install -e ".[tensorrt]"
```

### 1.1 Встановлення RDD (Robust Deformable Detector)
Вихідний код RDD вже включено до репозиторію у `third_party/rdd/`. Рекомендується скомпілювати кастомні CUDA оператори для швидкодії:

```powershell
# Встановлення залежностей RDD
pip install -r third_party/rdd/requirements.txt

# Компіляція кастомних CUDA-операторів (рекомендовано для швидкості)
cd third_party/rdd/RDD/models/ops
pip install -e . --no-build-isolation
```

### 1.2 Встановлення Depth-Anything-V2
Модуль оцінки глибини потребує клонування Depth-Anything-V2 у папку `third_party`:

```powershell
# Клонування Depth-Anything-V2 в third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2

# Встановлення залежностей
pip install -r third_party/Depth-Anything-V2/requirements.txt
```

### 1.3 Завантаження ваг моделей

**Основний спосіб — архів `models.zip` з Google Drive.** Автоматичне завантаження
покриває не всі моделі: `scripts/download_models.py` тягне лише `yolo11n-seg.pt` і
`depth_anything_v2_vits.pth`, а ваги RDD та **DINOv3** так завантажити не вийде —
`facebook/dinov3-vitl16-pretrain-sat493m` є gated-репозиторієм HuggingFace і потребує
окремо схваленого доступу.

```powershell
# Розпакувати архів у корінь репозиторію — має утворитися папка models/
Expand-Archive models.zip -DestinationPath .
```

Архів [за цим посиланням](https://drive.google.com/drive/folders/1qyO9AtUNkmkHXswvCbNkYcoTv8ChBbyP?usp=sharing).
Після розпакування структура має бути такою:

```
models/
├── yolo11n-seg.pt / .onnx / .engine   # маскування динаміки
├── depth_anything_v2_vits.pth         # оцінка глибини
├── RDD-v*.pth, RDD_lg-v*.pth          # альтернативний детектор ознак
├── vlad_vocab_c32_p256.npz            # словник VLAD (опційно)
└── .cache/                            # torch.hub + HuggingFace кеш (DINOv2/DINOv3)
```

> `models/` — **єдиний** корінь зберігання ваг: `config.paths.ensure_model_cache_env`
> перенаправляє `TORCH_HOME`, `HF_HOME` і `HUGGINGFACE_HUB_CACHE` у `models/.cache/`,
> тож нічого не осідає у `C:\Users\<you>\.cache`.

**Якщо ви завантажуєте DINOv3 самостійно** (замість готового кешу з архіву):

1. Запросіть доступ до `facebook/dinov3-vitl16-pretrain-sat493m` на HuggingFace і дочекайтесь схвалення.
2. Залогіньтесь так, щоб токен ліг у кеш проєкту, інакше буде 401:

```powershell
$env:HF_HOME = "$PWD\models\.cache\huggingface"
hf auth login          # старіші версії huggingface_hub: huggingface-cli login
```

> **Безпека:** DINOv3 вантажиться з `trust_remote_code=True`. Зафіксуйте
> `models.global_descriptor.dinov3.hf_revision` (commit hash), щоб зміна upstream-коду
> не перетворилась на виконання чужого коду на вашій машині. Порожнє значення = latest.

Часткове довантаження (YOLO + Depth-Anything) залишається доступним:

```powershell
python scripts/download_models.py
```

> **Примітка:** PyTorch з підтримкою CUDA встановлюється окремо згідно з
> [офіційною інструкцією](https://pytorch.org/get-started/locally/), оскільки
> потрібна версія залежить від вашого GPU та драйвера.

### 2. Запуск програми

```powershell
# Запуск GUI
python main.py

# Запуск у headless режимі (WebSocket + REST API)
python main.py --headless --project /шлях/до/проєкту --source /шлях/до/відео.mp4
python main.py --headless --project /шлях/до/проєкту --source rtsp://drone-ip/stream
```

**Параметри headless режиму:**

| Прапор | За замовчуванням | Опис |
|---|---|---|
| `--project` | обов'язково | Шлях до директорії проєкту |
| `--source` | обов'язково | Шлях до відеофайлу або RTSP/HTTP потік |
| `--ws-port` | з конфігу (`network_api.ws_port`) | Порт WebSocket сервера |
| `--rest-port` | з конфігу (`network_api.rest_port`) | Порт REST API сервера |
| `--supervise` | вимкнено | Запуск пайплайна в дочірньому процесі з автоперезапуском після падіння |
| `--max-restarts` | 0 (без ліміту) | Зупинитись після N перезапусків (лише зі `--supervise`) |

## 📖 Workflow

### Етап 1: Створення бази даних

1. Запустити програму.
2. Вибрати "Створити нову місію".
3. Завантажити еталонне відео.
4. Встановити висоту польоту (для масштабування).
5. Почати обробку.
6. Зачекати завершення (індикатор прогресу).

### Етап 2: GPS Калібрування

1. Відкрити створену базу даних.
2. Вибрати меню "Калібрування" → "Додати якір...".
3. Діалог автоматично відкриє відео `*_keypoints.mp4` (його нумерація кадрів збігається з базою даних; оригінальне відео польоту конвертується автоматично). Знайти початковий кадр маршруту, клікнути на 4+ орієнтири (рекомендовано 5–6, розкидані по всьому кадру) та ввести їхні реальні GPS-координати. Додати якір.
4. Повторити процедуру для кадру в середині маршруту та для фінального кадру.
5. Натиснути "Готово — запустити пропагацію". Програма автоматично розрахує координати для всіх тисяч проміжних кадрів за допомогою нейромережі LightGlue та оптимізації графу поз.

### Етап 3: Real-time локалізація

1. Підключити дрон або завантажити тестове відео польоту.
2. Завантажити калібровану базу даних (з уже виконаною пропагацією).
3. Натиснути "Почати відстеження".
4. Спостерігати точну локалізацію на карті в реальному часі зі згладжуванням траєкторії.

### Етап 4: Панорами (Опціонально)

1. Натиснути "Згенерувати панораму з відео", щоб створити широке зображення місцевості.
2. Натиснути "Накласти панораму на карту". Система розріже панораму, знайде її координати через нейромережі та відобразить поверх супутникової карти.

## 🏗️ Архітектура

```
src/
├── core/             # Lifecycle проєкту, headless runner, реєстр проєктів, експорт
├── models/           # AI обгортки (DINOv3/DINOv2, ALIKED, RDD, YOLOv11, LightGlue, VLAD, TensorRT)
├── database/         # HDF5 v2 + LanceDB (Builder, Loader, keyframe selector, spatial index)
├── localization/     # Основний пайплайн (Localizer, matcher, retrieval, patchify, верифікація)
├── geometry/         # Математика (Affine, Homography, pose_graph 5-DoF, UTM координати, GSD)
├── calibration/      # Управління GPS-якорями та пропагація координат
├── tracking/         # Kalman Filter, fixed-lag smoother, Outlier Detector, трекер об'єктів
├── depth/            # Монокулярна оцінка глибини (Depth-Anything-V2)
├── network/          # WebSocket сервер, REST API, брокер координат
├── security/         # Шифрування at-rest, сканування проєкту
├── video/            # Декодування відео кадрів
├── workers/          # QThread фонові потоки (трекінг, БД, пропагація, панорами, шифрування)
├── gui/              # PyQt6 інтерфейс (MainWindow, mixins, widgets, dialogs)
└── utils/            # Логування (Loguru), CLAHE, hardware profile, телеметрія, fault injection

config/               # Pydantic-конфіг по доменах: models, database, localization, graph, app, access
```

> Детальні діаграми потоків (build / calibration / localization) — у `docs/architecture.md`,
> математика графа поз — у `docs/POSE_GRAPH_MATH.md`.

## 🔧 Розробка

### Запуск тестів

```powershell
# Потрібен pip install -e ".[dev]"
pytest tests/ -v
```

### Лінтинг

```powershell
ruff check src/
ruff format src/
```

### Компіляція TensorRT движка

```powershell
python scripts/compile_dinov2_trt.py
```

### Корисні скрипти

```powershell
python scripts/benchmark_hardware.py     # профіль заліза та рекомендовані параметри
python scripts/build_vlad_vocab.py       # словник VLAD на референсних кадрах
python scripts/encrypt_project.py        # шифрована копія проєкту
python scripts/soak_test.py              # тривалий стрес-тест зі штучними збоями
python scripts/validate_vs_ground_truth.py  # порівняння з ground truth симулятора
```

### Компіляція у .exe

```powershell
# PyInstaller
python scripts/build_executable.py
```

## 📧 Контакти

Для питань та підтримки відкрийте Issue на GitHub.
