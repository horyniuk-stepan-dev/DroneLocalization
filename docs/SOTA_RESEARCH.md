# Пайплайн DroneLocalization проти SOTA (GPS-denied локалізація)

> Огляд найкращих досліджень 2024–2026 по кожній стадії пайплайна + конкретні рекомендації. Дата аналізу: 2026-07-07.

## Резюме

Задача проєкту в літературі називається **Absolute Visual Localization (AVL)**: визначення позиції БпЛА зіставленням бортового зображення з геоприв'язаною реферною картою, без дрейфу (на відміну від SLAM/VIO). Головна знахідка огляду — **бенчмарк AnyVisLoc** ([arXiv 2503.10692](https://arxiv.org/abs/2503.10692), 18 000 реальних знімків, 7 дронів, висоти 30–300 м), який систематично перебрав комбінації retrieval × matching × стратегія — тобто провів саме той експеримент, який потрібен цьому проєкту для валідації архітектурних рішень.

**Що в пайплайні вже відповідає найкращим практикам:**

1. **Стратегія "retrieval top-K → матчинг → вибір за інлаєрами"** — AnyVisLoc показав, що Top-N re-rank за інлаєрами — найкращий баланс точності/швидкості (A@10m: 82.4 проти 74.3 у Top-1, і лише 0.8 с/кадр проти 10.2 с у повного перебору). Цикл кандидатів з early-stop у `localizer.py` — це саме воно.
2. **Реферна карта з власного аерознімання** (БД з відео) — на аерокартах базлайн AnyVisLoc дає 74.1% у межах 5 м проти **18.5%** на супутникових картах. Вибір джерела карти — найбільший фактор точності, і проєкт за замовчуванням у сильній позиції.
3. **DINOv3 sat493m як бекбон** — DINOv3 ([arXiv 2508.10104](https://arxiv.org/pdf/2508.10104), [Meta AI](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)) — SOTA на 12/15 Earth-observation бенчмарків frozen-фічами; супутниковий варіант у конфігу проєкту — правильний вибір.
4. **ALIKED/RDD + LightGlue** — sparse-матчинг цього класу досі рекомендований для real-time (AnyVisLoc: dense точніший, але у 6–45× повільніший). RDD (CVPR 2025, deformable transformer, тренований зокрема на Air-to-Ground) — сучасний вибір.

**Де відставання від SOTA (за впливом):**

1. **Retrieval — сирий CLS-токен.** Плоский DINOv2/v3-дескриптор без навченої агрегації — це рівень AnyLoc (2023). SOTA — навчені агрегатори поверх патч-токенів: SALAD, BoQ, **MegaLoc** ([arXiv 2502.17237](https://arxiv.org/abs/2502.17237), CVPR 2025 W). Свіжі роботи ([VFM-Loc](https://arxiv.org/html/2603.13855), [DINO-GFSA](https://arxiv.org/pdf/2606.00784)) прямо показують: плоский пулінг DINOv3 у cross-view retrieval слабкий, з навченою головою — R@1 95%+.
2. **Немає dense-матчера для складних випадків.** RoMa / **RoMa v2** ([arXiv 2511.15706](https://arxiv.org/pdf/2511.15706)) — найточніший матчинг в AnyVisLoc (A@5m 70.1 проти 55.8 у SP+LG) та основа переможців Image Matching Challenge (у [IMC 2025](https://note.com/jdsc/n/na05e02b25b46) виграли на MASt3R). Для офлайн-стадій (пропагація калібрування, loop closures) час не критичний — там dense дав би прямий приріст якості графа.
3. **Ваги матчера не адаптовані до домену.** В AnyVisLoc LightGlue з вагами **GIM** (навчання на інтернет-відео) стабільно кращий за стандартні: +3.7 п.п. A@5m без зміни архітектури.
4. **Одногіпотезний трекінг.** KF + евристичний outlier-гейт ламається при виході за покриття (у коді це видно: `out_of_coverage` guard). Дослідження сходяться на **мультигіпотезності**: particle filter поверх абсолютних фіксів + одометрії — так працює переможець SPRIN-D challenge ([arXiv 2510.01348](https://arxiv.org/abs/2510.01348), 9 км GNSS-denied, RMSE < 11 м, real-time на CPU); факторний граф — [MASt3R-Fusion](https://arxiv.org/pdf/2509.20757).
5. **Гомографія на похилих ракурсах.** Планарне припущення деградує при oblique-зйомці на малих висотах (AnyVisLoc, §6.2). SOTA-нaпрямок — 2.5D-карти (DSM) + PnP ([OrthoLoC](https://arxiv.org/pdf/2509.18350), UAVD4L). У проєкті вже є Depth-Anything-V2 — половина шляху.

---

## 1. Стадія retrieval (глобальні дескриптори)

**У проєкті:** DINOv2/v3 CLS-токен → cosine у LanceDB; patchify (1+2²+3² кропів, 14 forward-пасів) як мультимасштабне розширення; A3-prior кута + батч-скан 4 ротацій.

**SOTA 2024–2026:**

- **MegaLoc** — один retrieval-звід для VPR/landmark/localization задач, SOTA на більшості VPR-датасетів ([CVPR 2025 W](https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Berton_MegaLoc_One_Retrieval_to_Place_Them_All_CVPRW_2025_paper.html)). Навчений переважно на наземних сценах — для nadir-аерознімків потрібна перевірка, але як drop-in замінник CLS вартий експерименту.
- **SALAD-стиль агрегації** (optimal transport поверх патч-токенів DINOv2) — за оцінками VPR-оглядів ([arXiv 2603.13917](https://arxiv.org/abs/2603.13917)) саме fine-tuned агрегація, а не бекбон, дає основний приріст.
- **Доменні cross-view моделі**: CAMP — найкращий retrieval в AnyVisLoc (R@1 62.4, +31 п.п. проти NetVLAD); Sample4Geo (ICCV 2023) — приріст головно від hard-negative семплінгу. Тобто: контрастивне донавчання на власних парах "кадр ↔ сусідні кадри БД" — дешевий і перевірений шлях.
- **HE-VPR** ([arXiv 2603.04050](https://arxiv.org/pdf/2603.04050)) — оцінка висоти для боротьби зі scale-variance в аеро-VPR; та сама проблема, яку в проєкті вирішує patchify.

**Рекомендації:**

1. *(малий ризик, швидко)* Замінити patchify-кропи на **пулінг патч-токенів з одного forward-пасу** (grid-пулінг = той самий мультимасштаб за 1 інференс замість 14). Це узгоджується з тим, як побудовані SALAD/GFSA.
2. *(середній)* Донавчити легку агрегаційну голову (SALAD-стиль, ~1 GPU-день) на власних БД: позитиви — сусідні keyframes, негативи — далекі. Бекбон DINOv3 frozen.
3. *(експеримент)* Прогнати MegaLoc проти поточного CLS на еталонному відео — 1 день роботи, одразу видно PDM@K/R@1.
4. Якщо в телеметрії є курс (магнітометр/INS) — використовувати його як prior кута замість 4-кутового скану: AnyVisLoc показує, що yaw-prior з шумом до 10° майже не втрачає точності (A@5m −1.9 п.п.), а прибирає ¾ обчислень ротацій.

## 2. Стадія matching (локальні фічі)

**У проєкті:** ALIKED або RDD + LightGlue; fast-numpy fallback; MAD-уточнення RANSAC-порогу.

**SOTA 2024–2026:**

| Метод | Клас | AnyVisLoc A@5m (Top-1) | Час | Коментар |
|---|---|---|---|---|
| SP+LightGlue | sparse | 52.1 | 92 мс | ≈ поточний рівень проєкту |
| SP+LightGlue-**GIM** | sparse | 55.8 | 75 мс | ті самі ваги → [GIM](https://github.com/ericzzj1989/Awesome-Image-Matching)-навчені |
| LoFTR-GIM | semi-dense | 59.5 | 165 мс | |
| **RoMa** | dense | **70.1** | 659 мс | RoMa v2 (11.2025): швидша й точніша |
| MASt3R | 3D-aware | — | ~сек | переможці IMC 2025 |

- **MINIMA** ([CVPR 2025](https://arxiv.org/abs/2412.19412), [код](https://github.com/LSXI7/MINIMA)) — modality-invariant матчинг (RGB↔IR/night/sketch/satellite), 19 крос-модальних сценаріїв. Прямий кандидат, якщо з'явиться тепловізор або матчинг проти супутникових тайлів.
- **DeDoDe v2 / DaD** — альтернативні детектори; в Air-to-Ground оцінках RDD виглядає сильно — заміна не пріоритетна.
- **UASTHN** ([arXiv 2502.01035](https://arxiv.org/pdf/2502.01035)) — uncertainty-aware гомографія для thermal↔satellite: патерн "оцінюй невизначеність трансформації, а не лише інлаєри" вартий запозичення в confidence-формулу.

**Рекомендації:**

1. *(майже безкоштовно)* Спробувати **GIM-ваги для LightGlue** — заміна файлу ваг, очікувано +3–4 п.п. точності на крос-доменних парах (запасний план: без змін, якщо на власних даних гірше).
2. *(середній)* **RoMa v2 як офлайн-матчер** для `CalibrationPropagation` (temporal/spatial ребра) і loop closures: там 300–600 мс/пара прийнятні, а якість ребер безпосередньо визначає точність усієї пропагації. Real-time шлях лишається на LightGlue.
3. *(стратегічно)* MINIMA-ваги як опція `matcher.backend` для майбутніх модальностей.

## 3. Геометрична оцінка (гомографія → координати)

**У проєкті:** OpenCV/PoseLib RANSAC + самописний MAD-refinement; гомографія → аффінна (5-DoF) → метрика → GPS.

**SOTA:**

- **SupeRANSAC** ([arXiv 2506.04803](https://arxiv.org/html/2506.04803v1)) — уніфікований робастний естиматор, mAA 0.51 проти 0.44 у MAGSAC++/PoseLib-LO-RANSAC. Важливо: той самий бенчмарк показує, що **PoseLib слабкий саме на гомографіях** — а він у конфігу проєкту як опційний backend. Мінімум: за замовчуванням тримати OpenCV MAGSAC++ (`cv2.USAC_MAGSAC`), PoseLib для H не рекомендувати.
- ["RANSAC Scoring Functions: Reality Check"](https://arxiv.org/html/2512.19850v1) (2025) — навчені скорингові функції не кращі за класичні; тобто самописний MAD-refinement не є відставанням, але і не дасть виграшу проти MAGSAC++.
- Планарне припущення: AnyVisLoc §6 — на похилих ракурсах і при перепадах рельєфу гомографія деградує; розв'язок — **2.5D reference (DSM < 1 м) + PnP** (OrthoLoC, UAVD4L). Для nadir-зйомки з 100+ м поточна схема адекватна.

**Рекомендації:** (1) дефолт `homography.backend` → OpenCV USAC_MAGSAC, звірити з MAD-варіантом на еталоні; (2) SupeRANSAC — спостерігати, з'явиться стабільний реліз — заміряти; (3) PnP-режим з DSM — тільки якщо потрібні польоти на малих висотах/oblique.

## 4. Трекінг і ф'южн (Kalman, викиди, optical flow)

**У проєкті:** 2D-KF (filterpy) + Z-score/швидкісний outlier-гейт + OF між keyframes + `out_of_coverage` лічильник.

**SOTA:**

- **Particle filter поверх абсолютних фіксів + одометрії** — стандарт для kidnapped/ambiguous випадків: класика [PF+VO](https://arxiv.org/pdf/1910.12121), і головне — переможець SPRIN-D Challenge 2025 ([arXiv 2510.01348](https://arxiv.org/abs/2510.01348)): кластеризований PF, що ф'юзить одометрію з template-matching висотної карти, км-масштаби, CPU-only. Патерн 1:1 лягає на цей проєкт: замініть "heightmap template score" на "retrieval/inlier score".
- **Факторний граф (sliding window smoothing)** — [MASt3R-Fusion](https://arxiv.org/pdf/2509.20757) (feed-forward візуальна модель + IMU/GNSS у факторному графі), [UAV-UGV fusion](https://www.mdpi.com/2504-446X/10/3/175) з Mahalanobis-гейтом і адаптивними вагами. Абсолютний фікс = унарний фактор з вагою від confidence; OF/VO = бінарні фактори. GTSAM/g2o — зрілі бібліотеки.
- Феномен, який це лікує: зараз одиночний хибний фікс або серія відхилених вимірювань → жорсткий reset. Мультигіпотезність деградує м'яко і сама відновлюється після виходу з-за меж покриття.

**Рекомендації:** (1) *(мала зміна)* у `OutlierDetector` додати Mahalanobis-гейт від KF-коваріації замість Z-score по вікну — це безкоштовне уточнення в межах поточної архітектури; (2) *(середня)* сесійний particle filter (кількасот частинок, CPU) як опційний `tracking.backend: "pf"` — вирішує out-of-coverage відновлення і неоднозначні retrieval-кластери; (3) факторний граф — лише якщо додасться IMU/компас у телеметрію.

## 5. Джерело карти (стратегічне)

- Поточний режим (БД з реферного відео) = "aerial map" в AnyVisLoc — найточніший клас, але вимагає попереднього обльоту. Це обмеження місій.
- **Супутниковий режим** — окремий дослідницький напрям (cross-view geo-localization): CAMP/Sample4Geo retrieval + GIM/MINIMA/RoMa матчинг; свіже — [Beyond Matching to Tiles](https://arxiv.org/pdf/2603.22153) (незіставлені аеро/супутникові в'ю), [контекстні методи self-positioning](https://arxiv.org/pdf/2502.11408). Точність нижча (AnyVisLoc: 18.5% A@5m проти 74.1%), але місія можлива без обльоту. Архітектурно `MultiDatabaseManager` вже готовий тримати "супутникову БД" як ще одне джерело — потрібен лише offline-конвертер тайлів у БД (дескриптори + аффінні з georef тайлів, без пропагації).
- **Рельєфний канал** (SPRIN-D winner): у рівнинних/лісових зонах, де фічі бідні, template-matching висот проти DEM — незалежний сигнал. Далекий горизонт, потребує LiDAR/стерео.

## 6. Що НЕ варто міняти зараз

- **LanceDB** — на 10K–100K дескрипторів вузьким місцем не є (мілісекунди); міграція на FAISS/Milvus не окупиться.
- **Своя 5-DoF пропагація** — літературний аналог (pose graph + loop closures) уже реалізований коректно, з аналітичним якобіаном і діагностикою; GTSAM дав би той самий розв'язок.
- **filterpy → інша KF-бібліотека** — заміна шила на мило; якщо міняти, то модель (PF/факторний граф), а не бібліотеку.
- **ONNX/TensorRT-міграція всього** — точкові TRT-двигуни (уже є для YOLO/DINOv2) достатні.

## Пріоритезований roadmap

| # | Дія | Очікуваний ефект | Зусилля |
|---|---|---|---|
| 1 | GIM-ваги для LightGlue (A/B на еталоні) | +3–4 п.п. точності матчингу | години |
| 2 | Патч-токен пулінг замість 14-кропного patchify | той самий recall, −13 forward-пасів | 1–2 дні |
| 3 | Yaw-prior з телеметрії (якщо є курс) | −75% ротаційних обчислень | 1 день |
| 4 | `USAC_MAGSAC` дефолтом; PoseLib для H — off | точність/стабільність H | години |
| 5 | RoMa v2 в CalibrationPropagation (офлайн) | якісніші ребра → точніша пропагація | ~тиждень |
| 6 | Навчена агрегаційна голова (SALAD-стиль) на власних БД | головний приріст retrieval | 1–2 тижні |
| 7 | Particle filter як опційний трекінг-бекенд | м'яка деградація, відновлення після втрати | 1–2 тижні |
| 8 | Супутниковий режим (satellite-DB конвертер + MINIMA) | місії без попереднього обльоту | місяць+ |
| 9 | DSM + PnP для oblique/малих висот | точність поза nadir | місяць+ |

## Джерела

**Бенчмарки і задача:** [AnyVisLoc benchmark](https://arxiv.org/abs/2503.10692) ([код](https://github.com/UAV-AVL/Benchmark)) · [OrthoLoC](https://arxiv.org/pdf/2509.18350) · [UAVD4L](https://arxiv.org/pdf/2401.05971) · [From GPS to AI: огляд UAV-локалізації](https://www.researchgate.net/publication/398170980_From_GPS_to_AI_A_comprehensive_review_of_Unmanned_Aerial_Vehicle_UAV_localization_solutions) · [SPRIN-D heightmap winner](https://arxiv.org/abs/2510.01348)
**Retrieval/VPR:** [MegaLoc](https://arxiv.org/abs/2502.17237) · [DINOv3](https://arxiv.org/pdf/2508.10104) ([Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)) · [VFM-Loc](https://arxiv.org/html/2603.13855) · [DINO-GFSA](https://arxiv.org/pdf/2606.00784) · [HE-VPR](https://arxiv.org/pdf/2603.04050) · [VPR evaluation 2026](https://arxiv.org/abs/2603.13917)
**Matching:** [RoMa v2](https://arxiv.org/pdf/2511.15706) · [MINIMA](https://arxiv.org/abs/2412.19412) ([код](https://github.com/LSXI7/MINIMA)) · [IMC 2025 (MASt3R)](https://note.com/jdsc/n/na05e02b25b46) · [DeDoDe v2](https://www.researchgate.net/publication/384417282_DeDoDe_v2_Analyzing_and_Improving_the_DeDoDe_Keypoint_Detector) · [Awesome Image Matching](https://github.com/ericzzj1989/Awesome-Image-Matching) · [UASTHN](https://arxiv.org/pdf/2502.01035)
**Робастна геометрія:** [SupeRANSAC](https://arxiv.org/html/2506.04803v1) · [RANSAC Scoring Reality Check](https://arxiv.org/html/2512.19850v1) · [MAGSAC](https://github.com/danini/magsac)
**Ф'южн/одометрія:** [MASt3R-Fusion](https://arxiv.org/pdf/2509.20757) · [PF+VO](https://arxiv.org/pdf/1910.12121) · [UAV-UGV factor-graph fusion](https://www.mdpi.com/2504-446X/10/3/175) · [MASt3R-Nav](https://arxiv.org/pdf/2605.24111)
**Cross-view/супутниковий напрям:** [Beyond Matching to Tiles](https://arxiv.org/pdf/2603.22153) · [Context-enhanced self-positioning](https://arxiv.org/pdf/2502.11408) · [Hierarchical semantic+structural matching](https://arxiv.org/html/2506.09748v1)
