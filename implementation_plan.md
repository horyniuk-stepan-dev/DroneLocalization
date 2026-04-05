# Sprint 2: Video I/O та GUI

Спринт зосереджений на усуненні синхронного відео-декодування з головного (GUI) потоку, оптимізації слайдера та загальному покращенні плавності інтерфейсу під час калібрування.

## User Review Required

- Будь ласка, перегляньте деталі архітектури `VideoDecodeWorker`. Він буде використовувати чергу команд (`queue.Queue()`) для передачі команд `load`, `seek`, `play`, `pause`, `stop` від GUI до фонового потоку, щоб GUI потік ніколи не викликав `cv2.VideoCapture` напряму. Це влаштовує?

## Proposed Changes

---

### UI & Video Decoupling

#### [NEW] [video_decode_worker.py](file:///e:/Dip/gsdfg/New/DroneLocalization/src/workers/video_decode_worker.py)
Створення класу `VideoDecodeWorker(QThread)`.
- Відкриття відео (`cv2.VideoCapture`) перенесено у внутрішній цикл (`run()`).
- GUI буде передавати команди (`seek`, `play`, `pause`) через потокобезпечну чергу.
- Воркер емітуватиме сигнал `frame_ready(int, np.ndarray)`, який головний потік перехоплюватиме для відображення.

#### [MODIFY] [calibration_dialog.py](file:///e:/Dip/gsdfg/New/DroneLocalization/src/gui/dialogs/calibration_dialog.py)
- **Інтеграція VideoDecodeWorker**: Заміна прямих викликів `cv2.VideoCapture` на спілкування з новим воркером.
- **LRU Cache (`collections.OrderedDict`)**: Додавання `self._frame_cache` з обмеженням у `maxsize=32`. Зберігатиме розпарсені `np.ndarray` або вже підготовлені `QPixmap`, щоб миттєво відтворювати їх при поверненні на той самий кадр.
- **Slider Debounce**:
  - `slider.valueChanged` (під час drag) — якщо кадр є у кеші, миттєво повертаємо його. Якщо немає, очікуємо.
  - `sliderReleased` — надсилаємо команду `seek(target_frame)` у VideoDecodeWorker для повноцінного рендеру.

---

### Дрібні UI фікси

#### [MODIFY] [video_widget.py](file:///e:/Dip/gsdfg/New/DroneLocalization/src/gui/widgets/video_widget.py)
- Заміна `logger.warning("CLICK DIAG: ...")` на `logger.debug` у методі `mousePressEvent`, щоб логер не спамив консоль при кожному кліку по відео.

#### [MODIFY] [image_utils.py](file:///e:/Dip/gsdfg/New/DroneLocalization/src/utils/image_utils.py)
- Оптимізація `opencv_to_qpixmap`. Виклик `np.ascontiguousarray` перед `QImage(cv_rgb.data, ...)` буде видалено. `cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)` і так гарантує С-contiguous пам'ять. Головним залишається використання `.copy()` об'єкту `q_img` під час `QPixmap.fromImage(q_img.copy())`, щоб запобігти segmentation fault.

## Verification Plan

### Manual Verification
- Відкрити `CalibrationDialog`, завантажити відео бази даних.
- Протестувати переміщення повзунка (drag & release) — GUI більше не повинен "замерзати" під час скролінгу великих MP4.
- Перевірити кешування — повернення назад на ті ж самі кадри під час drag має відображати зображення миттєво.
- Перевірити консольні логи — переконатися що `CLICK DIAG` зник із рівня WARNING.
