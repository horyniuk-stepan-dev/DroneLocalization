# Дослідницький аддендум — липень 2026

**Що це.** Дельта поверх `RELATED_WORK_MAP.md`, `SOTA_RESEARCH.md` і `EFFICIENCY_OPTIONS.md` (усі три вже покривають архітектурних близнюків і основні осі прискорення). Тут — лише те, чого в тих документах немає. Фокус: швидкість і якість локалізації **без донавчання**. Пошук проведено 2026-07-21; твердження про методи взяті з абстрактів/текстів статей за посиланнями, власних замірів немає.

---

## 1. Retrieval: zero-shot re-ranking замість «сліпого» top-K → LightGlue

**EffoVPR** ([arXiv:2405.18065](https://arxiv.org/abs/2405.18065), ICLR 2025) показує, що фічі з self-attention шарів замороженого DINOv2 працюють як потужний re-ranker top-K кандидатів retrieval **у чистому zero-shot, без жодного тренування**. Це прямо лягає в наш вузький хвіст: `retrieval_top_k = 12` → до 12 повних прогонів LightGlue у поганому кадрі (`EFFICIENCY_OPTIONS.md` §1).

**Куди в пайплайн.** Той самий слот, що й MNN-передфільтр (EFFICIENCY §1.1), але дешевше: патч-токени DINOv3 для query **вже пораховані** у тому ж forward, що дає CLS — re-ranking через local-feature similarity кандидатів не потребує додаткового проходу бекбона (дескриптори патчів рефів треба один раз додати в БД → перебудова або окрема таблиця). Ранжуємо 12 кандидатів → LightGlue лише на 1–2 верхніх.

**Стосунок до MNN-передфільтра.** Це не конкуренти, а два рівні одного фільтра: EffoVPR-style re-rank працює до екстракції локальних фіч (зовсім дешево), MNN — після (точніше). Можна почати з будь-якого; якщо MNN-фільтр уже дав скорочення хвоста, re-rank додасть небагато швидкості, але може підняти якість вибору кандидата (у EffoVPR zero-shot re-rank топ-100 дає SOTA-рівень без тренування).

## 2. Matching: легші матчери та semi-dense варіант проти колапсу на ріллі

- **LighterGlue** ([xfeat_lightglue_onnx](https://github.com/noahzhy/xfeat_lightglue_onnx), [verlab](https://github.com/verlab/accelerated_features)) — офіційна полегшена версія LightGlue під XFeat: менше параметрів, ~3× швидша за оригінальний LightGlue. Підсилює кейс «XFeat замість ALIKED» (EFFICIENCY #9): міграція дає не лише легший екстрактор, а й легший матчер, обидва з готовими ONNX-експортами.
- **EfficientLoFTR** ([HF docs](https://huggingface.co/docs/transformers/model_doc/efficientloftr), CVPR 2024) — detector-free semi-dense матчер, ~2.5× швидший за LoFTR і за заявою авторів швидший навіть за SuperPoint+LightGlue. Головна цінність для нас не швидкість, а **режим відмови**: detector-free матчинг будує щільні відповідності по всій площині й не залежить від повторюваності кейпоінтів — саме та властивість, якої бракує на низькотекстурній ріллі, де LightGlue колапсує (lasttest, звідки виросли anchor-gap guards). Кандидат на **fallback-гілку** замість/поряд із `match_mnn`: вмикати лише коли інлаєри нижче порога, щоб не платити semi-dense ціну на кожному кадрі.
- **MapGlue** ([arXiv:2503.16185](https://arxiv.org/abs/2503.16185)) — матчер, навчений спеціально під мультимодальні пари remote sensing (у т.ч. крос-сорсні відмінності аеро↔супутник). Готові ваги; кандидат в A/B для сателітної гілки, коли вона зʼявиться.

## 3. OrthoTrack §3.3–3.4: конкретні механізми, які варто портувати

`RELATED_WORK_MAP.md` фіксує OrthoTrack як найближчий аналог, але без деталей. Розбір методу (прочитано повний текст §3.3–3.4, [arXiv:2606.25245](https://arxiv.org/abs/2606.25245)) дає чотири механізми, що майже 1:1 лягають на наш `keyframe_selector` / PropagationPipeline / anchor-gap guards:

1. **Visibility-aware crop selection.** Замість матчити всю карту — кроп навколо очікуваної позиції; для oblique-видів (θ>15°) центр кропа = inverse-depth-weighted середнє видимих DSM-точок (зміщує кроп до ближньої землі, що займає найбільшу площу кадру), розмір кропа пропорційний просторовому розкиду видимих точок. Наш гео-гейтинг (B7) — це та сама ідея на рівні retrieval; тут — рецепт, як робити її pose-aware, коли камера не надир.
2. **Two-phase matching із multi-scale fallback.** Fast path: PnP на одному кропі; якщо інлаєрів мало — матчити додаткові кропи зростаючого масштабу з центрами на перших збігах, змержити відповідності, повторити PnP. Структурно ідентично нашому rotation/scale-pyramid recovery, але тригериться за якістю, а не запускає всю піраміду одразу — це і є пропозиція «recovery спершу 4 кути на масштабі 1.0» з EFFICIENCY §2, підтверджена чужою абляцією.
3. **Чотири умови тригера keyframe:** (а) точок < N_min; (б) точок < α·(кількість на keyframe); (в) **spatial collapse**: min(σx, σy) розкиду точок нижче порога — точки скупчились у малій зоні кадру, PnP/гомографія ill-conditioned; (г) reprojection error вище адаптивного порога. Умову (в) наші guards, наскільки видно з плану стадії 8, не перевіряють — а це саме сценарій ріллі: інлаєрів формально досить, але всі в одному куті. Дешева перевірка, вартує додати до anchor-gap guards.
4. **Подвійний адаптивний поріг reprojection error.** Фіксований поріг або зациклює ре-тригер (крос-сорс матчинг дає різну базову помилку на кожному keyframe), або пропускає повільний дрейф. Рішення: абсолютний поріг з warmup-послабленням (e_base + δ·max(0, 1−Δt/G)) ловить швидкий дрейф + відносний поріг e_t > f̄·e_k, що затягується з часом, ловить повільний. Прямий кандидат для порогів у наших propagation guards замість фіксованих констант.

Також: forward-backward перевірка optical flow з відкиданням точок, чий round-trip зсув > τ_fb — якщо в нашому OF-трекінгу її нема, це стандартний дешевий фільтр.

## 4. Інференс: уточнення до A4 (TRT-двигун DINOv3)

- Порядок виграшу від TensorRT FP16 для ViT підтверджується практикою: DINOv2 ViT-B 224² ~25–30 мс у PyTorch → ~10–12 мс у TRT FP16 ([приклад оптимізації](https://medium.com/@testth02/accelerating-vision-ai-inference-with-tensorrt-yolov8-and-dinov2-optimization-in-practice-287acd4c73e1); числа чужі, наш ViT-L буде повільніший, пропорція схожа).
- **INT8 для DINOv2 у TRT не швидший за FP16** ([issue у NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT/actions/runs/12362619431/job/34502302378)) — LayerNorm/GELU/attention лишаються у вищій точності, а reformat-и зʼїдають виграш. Висновок для A4: **зупинитись на FP16, INT8-калібрування не робити** — це зекономить день-два експериментів із передбачувано нульовим результатом.

## 5. Depth: streaming-режим без тренування (на майбутнє)

**Video Depth Anything** (CVPR 2025 highlight, [сайт](https://videodepthanything.github.io/), [код](https://github.com/DepthAnything/Video-Depth-Anything)) — DA-V2 з часовою головою; має **experimental streaming mode без тренування**: кешуються hidden states temporal attention, на інференсі подається один кадр. Поки depth у нас лише на bootstrap (EFFICIENCY #4 — правильне рішення), це неактуально; стане актуальним, якщо TanDepth-гілка (RELATED_WORK шорт-лист #3) виведе depth у steady state — тоді часова консистентність глибини безкоштовно стабілізує масштаб між кадрами.

## 6. Векторний пошук: не наш вузький хвіст (перевірено логікою, не заміром)

Свіжа хвиля бінарного/ротаційного квантування (RaBitQ SIGMOD'24/25, [Qdrant BQ «40× faster»](https://qdrant.tech/articles/binary-quantization/), [Weaviate RQ](https://weaviate.io/blog/8-bit-rotational-quantization)) дає великі прискорення ANN-пошуку, але на БД масштабу однієї місії (тисячі–десятки тисяч 1024-d векторів) brute-force у LanceDB і так коштує одиниці мс — виграш зникає в шумі на тлі одного forward ViT-L. Актуалізується лише якщо БД виросте до мільйонів векторів (багатомісійна/сателітна база). Зафіксовано, щоб не витрачати на це час зараз.

## 7. Нові статті в мапу related work

- **PiLoT v2** ([arXiv:2606.31098](https://arxiv.org/pdf/2606.31098)) — real-time UAV geo-localization: замість дорогого 3D-рендерингу — легкий кроп TDOM+DSM, >25 FPS на Jetson Orin (v1: [arXiv:2603.20778](https://arxiv.org/html/2603.20778v3)). Новий мешканець «рівня 0» поряд з OrthoTrack; вартий читання цілком саме через фокус на онбордну швидкість.
- **Image Matching for UAV Geolocation: Classical and Deep Learning Approaches** ([J. Imaging 11(11):409, лист. 2025](https://www.mdpi.com/2313-433X/11/11/409)) — свіжий огляд саме нашої матчинг-ланки; додати до списку оглядів.
- **Real-Time Elevation and Orientation-Aware Visual Localization** ([Drones 10(6):445, 2026](https://www.mdpi.com/2504-446X/10/6/445)) — GNSS-denied навігація з урахуванням висоти/орієнтації в реальному часі; перетин із нашою віссю scale-invariance.
- **Online Video Depth Anything** ([arXiv:2510.09182](https://arxiv.org/html/2510.09182v1)) — онлайн-варіант VDA з низькою памʼяттю; супутник до §5.

## Шорт-лист дій (поверх існуючих roadmap-ів, у порядку ефект/зусилля)

| # | Що | Вісь | Зусилля | Залежності |
|---|---|---|---|---|
| 1 | Умова spatial collapse min(σx,σy) у guards (OrthoTrack §3.4) | якість | години | ні |
| 2 | Подвійний адаптивний reprojection-поріг у propagation guards | якість | дні | ні |
| 3 | A4 = тільки FP16, INT8 викреслити | швидкість | −1–2 дні | ні |
| 4 | Recovery: каскад «4 кути @1.0 → піраміда» (підтверджено абляцією OrthoTrack) | швидкість | дні | ні |
| 5 | EffoVPR-style re-rank патч-токенами перед LightGlue | швидкість+якість | ~тиждень | перебудова БД (патч-дескриптори рефів) |
| 6 | EfficientLoFTR як fallback-матчер на ріллі | якість | ~тиждень | ні (окрема модель) |
| 7 | LighterGlue у пакеті з міграцією на XFeat (EFFICIENCY #9) | швидкість | у складі #9 | перебудова БД |

Пункти 1–2 — найдешевший спосіб добрати якість саме в відомому режимі відмови (рілля/низька текстура); 3–4 — чиста економія часу; 5–7 — більші шматки під наступну ітерацію.

---

**Джерела:** [EffoVPR](https://arxiv.org/abs/2405.18065) · [OrthoTrack](https://arxiv.org/abs/2606.25245) · [PiLoT v2](https://arxiv.org/pdf/2606.31098) · [XFeat](https://arxiv.org/abs/2404.19174) / [LighterGlue ONNX](https://github.com/noahzhy/xfeat_lightglue_onnx) · [EfficientLoFTR](https://huggingface.co/docs/transformers/model_doc/efficientloftr) · [MapGlue](https://arxiv.org/abs/2503.16185) · [Video Depth Anything](https://videodepthanything.github.io/) · [Online VDA](https://arxiv.org/html/2510.09182v1) · [TRT DINOv2 практика](https://medium.com/@testth02/accelerating-vision-ai-inference-with-tensorrt-yolov8-and-dinov2-optimization-in-practice-287acd4c73e1) · [INT8-не-швидше issue](https://github.com/NVIDIA/TensorRT/actions/runs/12362619431/job/34502302378) · [Qdrant BQ](https://qdrant.tech/articles/binary-quantization/) · [J.Imaging огляд матчингу](https://www.mdpi.com/2313-433X/11/11/409) · [Drones 2026 elevation-aware](https://www.mdpi.com/2504-446X/10/6/445)
