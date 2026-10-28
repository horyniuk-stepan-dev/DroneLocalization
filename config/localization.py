"""Localization, tracking and homography configuration."""

from typing import Literal

from pydantic import BaseModel, Field


class LayerSearchConfig(BaseModel):
    enabled: bool = False
    # Search work stays bounded when a project contains many reference sources.
    # One slot is reserved for a rotating recovery probe outside the preferred
    # active/nearby layers, so a stale spatial filter cannot hide a new layer.
    max_sources_per_frame: int = Field(default=8, ge=2, le=256)
    candidates_per_source: int = Field(default=4, ge=1, le=32)
    descriptor_batch_size: int = Field(default=4, ge=1, le=32)
    max_verifications: int = Field(default=32, ge=1, le=512)
    budget_ms: float = Field(default=2000.0, gt=0)
    confirmations: int = Field(default=2, ge=1, le=20)
    switch_margin: float = Field(default=0.15, ge=0, le=2)
    agreement_m: float = Field(default=30.0, gt=0)
    lost_after_s: float = Field(default=3.0, gt=0)
    min_spread: float = Field(default=0.015, ge=0, lt=1)
    max_rmse_px: float = Field(default=4.0, gt=0)
    min_inlier_ratio: float = Field(default=0.2, ge=0, le=1)
    max_center_extrapolation: float = Field(default=0.1, ge=0, le=1)
    scale_drift_per_s: float = Field(default=0.1, gt=0)
    require_schema: bool = False
    # Handoff motion gate: after a gap in the active layer's fixes (LOST, or it
    # did not verify this frame) accept one observation within
    # agreement_m + growth * gap of the constant-velocity track prediction,
    # instead of waiting for `confirmations` fresh observations.
    motion_gate: bool = False
    motion_gate_growth_mps: float = Field(default=30.0, gt=0)
    motion_gate_max_gap_s: float = Field(default=10.0, gt=0)
    # Verify each retrieved frame once before retrying any frame at another scale.
    diverse_candidates: bool = False
    # Stop verifying once the active layer yields a strong, near-native-scale fix
    # (1/ratio..ratio) that agrees with the confirmed track.
    early_stop: bool = False
    early_stop_min_quality: float = Field(default=150.0, ge=0)
    early_stop_max_scale_ratio: float = Field(default=1.5, ge=1)


class AltitudeHintConfig(BaseModel):
    """Altitude-guided layer scales (src/localization/altitude_hint.py). Default off.

    The query's ground pixel size comes from telemetry (flight_data: height above
    ground, else above take-off) or, without telemetry, from the last fix's scale;
    the expected query/layer scale of the ``max_layers`` layers nearest to it is
    tried before the rest of the scale pyramid. Neither known -> unchanged search.
    """

    enabled: bool = False
    max_layers: int = Field(default=2, ge=1, le=8)
    # Weight of each telemetry + visual fix in refining the camera footprint factor
    # (ground width per metre of altitude, initially sensor width / focal length).
    telemetry_scale_ema: float = Field(default=0.3, gt=0, le=1)


class ConfidenceConfig(BaseModel):
    inlier_weight: float = 0.7
    stability_weight: float = 0.3
    rmse_norm_m: float = 10.0
    disagreement_norm_m: float = 5.0
    confidence_max_inliers: int = 80


class LocalizationConfig(BaseModel):
    layer_search: LayerSearchConfig = Field(default_factory=LayerSearchConfig)
    altitude_hint: AltitudeHintConfig = Field(default_factory=AltitudeHintConfig)
    min_matches: int = 12
    min_inliers_accept: int = 10
    # Absolute post-fit reprojection gate. Adaptive RANSAC may refine its own
    # threshold, but it cannot redefine what is accurate enough to emit GPS.
    max_geometric_rmse_px: float = Field(default=4.0, gt=0)
    geometric_min_inlier_ratio: float = Field(default=0.2, ge=0, le=1)
    # Maximum distance from the image centre to the inlier convex hull,
    # expressed as a fraction of the image diagonal.  Prevents a correct local
    # patch at one edge from authorizing an unsupported GPS projection.
    geometric_max_center_extrapolation: float = Field(default=0.1, ge=0, le=1)
    geometric_min_reference_eigenvalue: float = Field(default=1e-4, ge=0, le=1)
    # 0.75 = рекомендація Lowe's ratio test; 0.85 пропускало забагато хибних
    # збігів (конфіг перекривав фікс "БАГ 4" у matcher.py)
    ratio_threshold: float = 0.75
    ransac_threshold: float = 3.0
    retrieval_top_k: int = 12
    early_stop_inliers: int = 40
    retrieval_only_min_score: float = 0.90
    auto_rotation: bool = True
    # A3: темпоральний prior кута — пробуємо лише останній вдалий кут
    # (1 DINO-forward замість 4). Якщо retrieval-score prior-кута нижчий за цей
    # поріг — повний батчований скан 4 кутів.
    rotation_rescan_min_score: float = 0.70
    # ── Query rotation (src/localization/query_rotation.py) ──
    # "quarter"    — legacy: only 0/90/180/270 via np.rot90 (current behaviour).
    # "continuous" — any angle: bootstrap scans every rotation_scan_step_deg,
    #                steady state applies the angle measured on the previous fix
    #                (or the telemetry yaw), so ALIKED/LightGlue see ~0° residual.
    #                The view keeps the reference frame size, so DINO sees the
    #                same aspect as the database at every angle.
    rotation_mode: Literal["quarter", "continuous"] = "quarter"
    # Bootstrap / recovery scan step (45 → 8 angles, 30 → 12). Residual after
    # the scan is at most step/2; the refine pass below removes it.
    rotation_scan_step_deg: float = Field(default=45.0, gt=0, le=180)
    # After a verified match whose residual rotation exceeds this, re-extract
    # the query at the corrected angle and re-verify that candidate once
    # (more inliers, better geometry). 0 disables the refine pass.
    rotation_refine_min_deg: float = Field(default=12.0, ge=0)
    # Erosion of the valid area of a rotated view: keypoints on the artificial
    # border of the rotated frame are dropped.
    rotation_border_erode_px: int = Field(default=8, ge=0)
    enable_lightglue_fallback: bool = True
    # ── RESEARCH 2.2: аварійний SIFT+LightGlue фолбек ──
    # Одноразовий перезапуск матчингу через SIFT, коли ALIKED дав
    # < min_matches inliers (in-plane rotation / екстремальна похилість).
    # Вимагає БД, збудованої з database.store_sift_features=True.
    sift_fallback: bool = False
    sift_fallback_max_candidates: int = 3
    fallback_extractor: str = "aliked"
    confidence: ConfidenceConfig = ConfidenceConfig()
    use_patchify: bool = False  # Мультимасштабний retrieval через патч-дескриптори DINOv2 (14× DINOv2 forward passes — повільно на слабких GPU)
    patchify_grids: list[list[int]] = [[1, 1], [2, 2], [3, 3]]  # 1+4+9 = 14 патчів
    patchify_batch_size: int = (
        1  # Кількість патчів за один DINOv2 інференс (1 = послідовно, 4-7 = батч)
    )
    patchify_merge_weight: float = 0.4
    # False (old behaviour): a patchify frame score sums every reference-patch hit, so
    # it can exceed 1 and outrank cosine-scored standard candidates in the merge.
    # True: best hit per (query patch, frame); score = coverage x mean <= 1.
    patchify_unique_coverage: bool = False
    # Legacy multi-layer retrieval: 0 = the layer with the best top score is verified
    # (old behaviour). N > 0: candidates of all layers are ranked together and a layer
    # of higher priority (project.json `priority`, main = 0) wins while its best
    # candidate is within the first N + 1 ranks (src/database/source_priority.py).
    source_priority_rank_margin: int = Field(default=0, ge=0, le=100)
    # Multi-layer + patchify: 0 = old order (the chosen layer's merged candidates).
    # N > 0: candidates found only by patchify are verified after the main layer's
    # standard candidates unless they rank more than N places above the main layer's best
    # (patchify 1st, main 2nd -> main first; main 10th -> patchify first with N = 3).
    patchify_main_rank_margin: int = Field(default=0, ge=0, le=100)
    # Multi-layer: one verification list for all layers (default False = old path).
    # True: the standard candidates of every layer and, with use_patchify, every layer's
    # patchify-only candidates are ranked together; the first joint_top_k are verified in
    # that order, each on its own layer at its own (angle, scale). Place = rank by score
    # within its kind (the standard and patchify scores are not on one scale) + N places
    # per priority step of its layer (N = source_priority_rank_margin; project.json
    # `priority` for standard candidates, `patchify_priority` (default: priority) for
    # patchify ones) + joint_patchify_rank_penalty for patchify candidates.
    # patchify_main_rank_margin is not used while this is on
    # (src/database/source_priority.py::joint_order).
    joint_candidate_ranking: bool = False
    joint_top_k: int = Field(default=5, ge=1, le=50)
    joint_patchify_rank_penalty: int = Field(default=2, ge=0, le=100)
    # Effective layer priority = project priority step + layer_fineness_weight x log2(layer
    # ground pixel size / the finest layer's); pixel size measured from each layer's
    # calibration (a finer layer has the more accurate map). 0 = project priority only (old
    # behaviour). Used by source_priority_rank_margin and for the main layer of
    # patchify_main_rank_margin (src/database/source_priority.py::effective_steps).
    layer_fineness_weight: float = Field(default=0.0, ge=0, le=10)
    # Patchify only where no real layer is close in scale: the query's ground pixel size is
    # estimated from the scale prior and the active layer; with a layer within
    # patchify_gap_ratio of it patchify is skipped for the frame, otherwise it runs on the
    # layers whose crops come closer than any real layer. Unknown scale -> every layer.
    patchify_adaptive: bool = False
    patchify_gap_ratio: float = Field(default=1.5, ge=1.0, le=4.0)
    # Lazy patchify (not with joint_candidate_ranking): the patch search runs only after the
    # chosen layer's standard candidates fail verification, on the same view (the one the
    # scale / rotation priors chose); then the main layer (patchify_main_rank_margin) and the
    # patchify-only candidates as before. Standard candidates keep their retrieval order.
    # False = patch search before every verification (old behaviour).
    patchify_lazy: bool = False
    # ── Scale-invariance (ScaleManager) ──
    # GSD-ratio pyramid for altitude-invariant localization (r = query_alt / db_alt).
    # Scanned when no temporal prior exists (bootstrap / out-of-coverage).
    scale_pyramid: list[float] = [0.5, 0.7, 1.0, 1.4, 2.0]
    # Minimum retrieval score on the prior-scale level to accept it;
    # below this → full pyramid rescan (analogous to rotation_rescan_min_score).
    scale_rescan_min_score: float = 0.65
    # EMA smoothing factor for the temporal scale prior (higher = faster tracking).
    scale_prior_ema: float = 0.7
    # Use DepthAnythingV2 depth_scales (if available in DB) to reorder the pyramid.
    scale_use_depth_hint: bool = True
    # How often (in keyframes) to recompute the depth-based scale hint.
    depth_hint_every_n: int = 30

    # ── ADDENDUM 1.1: просторовий розкид інлаєрів → confidence. Дефолт off. ──
    # Інлаєрів може бути досить, але всі в одному куті кадру → гомографія
    # ill-conditioned (OrthoTrack §3.4, «spatial collapse»). Не відкидаємо кадр
    # (на межі покриття скупчення легітимне), а множимо confidence — далі він
    # їде в Kalman R (B2), тож слабкий фікс отримує меншу вагу, а не бан.
    spread_confidence_enabled: bool = False
    # Розкид, вище якого штрафу нема. 0.15 ≈ половина рівномірного покриття
    # (σ/L = 1/√12 ≈ 0.289 для точок, розкиданих по всьому кадру).
    spread_ref: float = 0.15
    # Нижня межа множника — вироджена хмара не обнуляє confidence.
    spread_floor: float = 0.35
    # Як часто (у keyframe-ах) логувати статистику розкиду (spread stats).
    spread_log_every: int = 50

    # Після стількох поспіль невдалих keyframe-ів локалізатор повертає
    # out_of_coverage і скидає пріори кута/масштабу (наступний кадр — повний скан).
    # Читався з 2026-07, але не був оголошений — user_config не міг його змінити.
    max_consecutive_failures: int = 10

    # ── ADDENDUM 2.1: каскадний recovery замість повного добутку. Дефолт off. ──
    # off = ПОТОЧНА поведінка: 4 кути × 5 масштабів = 20 ViT-forward одним батчем.
    # on: етап 1 — 4 кути на ОДНОМУ масштабі (prior → hint-найближчий → 1.0);
    # етап 2 (лише якщо скор етапу 1 < rotation_rescan_min_score) — РЕШТА
    # комбінацій. Найгірший випадок лишається 20, типовий стає 4.
    recovery_cascade: bool = False

    # ── ADDENDUM §1: MNN-передфільтр кандидатів перед LightGlue. Дефолт off. ──
    # off = ПОТОЧНА поведінка: LightGlue послідовно по всіх top-K з early-stop.
    # on: спершу дешевий mutual-NN скоринг дескрипторів КОЖНОГО кандидата
    # (один матмул на кандидата, ~мс), потім повний LightGlue лише на
    # prefilter_keep найкращих. Найбільший ефект на слабких GPU, де LightGlue
    # дорогий, а поганий кадр інакше коштує top_k повних прогонів.
    candidate_prefilter: bool = False
    # Скільки найкращих (за MNN-парами) кандидатів іде в повний LightGlue.
    prefilter_keep: int = 2
    # When no candidate has a single MNN pair (Lowe ratio 0.75), the prefilter
    # keeps the whole list and LightGlue runs on all top-K. N > 0 = LightGlue
    # only the first N by global score instead. 0 = whole list (old behaviour).
    # Measured 2026-10-10 on 6 routes: that case ran 63 % of all LightGlue
    # calls; 3 of its 163 calls found a frame, at global positions 0, 3, 0.
    prefilter_fallback_keep: int = 0

    # ── PIPELINE_OPTIMIZATION_PLAN §A3: довга сторона кадру для локального
    # екстрактора. Раніше жила ЛИШЕ як дефолт у get_cfg(..., 1600) і не була
    # у pydantic — тобто ключ у user_config.json мовчки ігнорувався
    # (APP_CONFIG = AppConfig(**data).model_dump() відкидає невідомі ключі).
    # Менше значення = менше часу ALIKED і менший пік VRAM, ціною дрібних фіч.
    max_local_edge: int = 1600

    # ── PIPELINE_OPTIMIZATION_PLAN §A1: темпоральний prior кандидатів ────────
    # off = ПОТОЧНА поведінка: кожен keyframe рахує глобальний дескриптор
    # (на GTX 1650 це 470 мс із 945, тобто половина keyframe-а).
    # on = у steady state кандидати беруться з околу останнього збігу, а
    # DINOv3 не запускається взагалі; хибна гіпотеза відсіюється дешевим MNN.
    temporal_candidate_prior: bool = False
    # Півширина вікна сусідів за індексом кадру БД: id-w … id+w.
    temporal_prior_window: int = 2
    # Скільки найкращих за MNN сусідів іде в повний LightGlue.
    temporal_prior_keep: int = 1
    # Мінімум MNN-пар, щоб гіпотеза взагалі дійшла до LightGlue. Це і є
    # «дешева проба»: одиниці мс на кандидата замість ~128 мс LightGlue, тому
    # промах не ламає бюджет worst case.
    temporal_prior_min_mnn: int = 20
    # Скільки інлаєрів потрібно, щоб ПРИЙНЯТИ результат темпорального шляху.
    # Суворіше за min_matches навмисно: помилка тут тиха і самопідтверджувана.
    temporal_prior_accept_inliers: int = 25
    # Кожен N-й keyframe примусово йде повним шляхом — аудит проти дрейфу.
    # 0 = аудит вимкнено (не рекомендується).
    temporal_prior_audit_every: int = 10
    # Temporal prior при наявній підказці курсу (2026-10-28). None = стара
    # поведінка: будь-яка підказка вимикає prior. Число = допуск у градусах:
    # якщо курс з телеметрії збігається з кутом останнього фіксу в межах
    # допуску, prior працює (на wayback-маршруті з телеметрією без нього
    # успішність keyframe падала 0.87 → 0.15). Рекомендовано 20.
    temporal_prior_yaw_hint_tol_deg: float | None = None
    # Query keypoint budget for the temporal-prior check only (MNN probe, its
    # LightGlue verification and the rotation refine of a temporal-prior hit):
    # the strongest N of the extracted keypoints. Retrieval / rescan frames keep
    # the full models.aliked.max_keypoints set. 0 = all keypoints (behaviour
    # before 2026-10-10). Measured 2026-10-10 with a global 2048 cap: GTX 1650
    # temporal-prior frames 348 -> 241 ms median, same fixes on own imagery.
    temporal_prior_max_keypoints: int = 0
    # temporal_prior_max_keypoints applies only when the previous frame's fix had at least
    # this many inliers (0 = always). After a weak or failed fix every keypoint is used:
    # 2026-10-17 a plain 2048 cap lost the wb2019 recovery on the RTX 5070 Ti (161 -> 91
    # fixes; on the GTX 1650 the same 91 came from models.aliked.max_keypoints=2048, the plain
    # prior cap was not run to the end there); its borderline spatial-prior matches have
    # 12-30 inliers, healthy fixes have hundreds (median 772 on the route3min flights).
    temporal_prior_kp_min_inliers: int = Field(default=0, ge=0)
    # Spatial prior (src/localization/spatial_prior.py): when the index-neighbour
    # temporal prior finds nothing, probe the map frames nearest to the
    # constant-velocity prediction from the last two fixes, with the last
    # confirmed angle and scale (the in-frame retry clears those priors, which
    # otherwise turns the temporal prior off for a whole loss), and LightGlue the
    # best `spatial_prior_verify` of them without the temporal_prior_min_mnn
    # gate. Acceptance stays temporal_prior_accept_inliers. Once per frame, only
    # within spatial_prior_max_age_s of the last fix, active layer only.
    # Diagnosed 2026-10-10 on wb2019 (other-satellite imagery): the right frame
    # was accepted by LightGlue 22 of 22 times it got there, but got there in
    # 22 of 176 slots; at each loss start it was the nearest frame of its layer
    # to the prediction. False = old behaviour.
    spatial_prior: bool = False
    spatial_prior_max_age_s: float = 30.0
    spatial_prior_nearest_k: int = 4
    spatial_prior_radius_m: float = 1500.0
    spatial_prior_verify: int = 2


class TrackingConfig(BaseModel):
    kalman_process_noise: float = 2.0
    kalman_measurement_noise: float = 5.0
    # A high-confidence homography is an absolute position observation.  The
    # constant-velocity KF may lag it badly at a sharp turn, so cap that lag
    # and re-anchor the state.  Applied only when
    # outlier_trust_strong_evidence is enabled and its inlier threshold passes.
    trusted_fix_max_filter_offset_m: float = 5.0
    outlier_window: int = 10
    # ── Пороги аутлаєр-детектора. ЧИТАЙ ПЕРЕД ЗМІНОЮ ────────────────────────
    # Виміряно на місії newzap (2026-07-31, 35 keyframe-локалізацій із 2048
    # інлаєрами, широта 47.83°, ground_scale_correction=ON):
    #     крейсерська швидкість  47.0 м/с (169 км/год), медіана 49.6
    #     МАКСИМУМ платформи     83.0 м/с (299 км/год)
    #
    # РЕКОМЕНДОВАНІ значення:
    #     max_speed_mps         = 180.0  (2.17x над виміряним максимумом)
    #     outlier_threshold_std = 8.0
    #
    # Чому threshold_std саме 8, а не класичні 4: вікно детектора змішує
    # keyframe-позиції (dt≈1.0 с) з OF-позиціями (dt=0.1 с) і містить
    # майже-дублі (заміряно: keyframe і OF-семпл за 14 см один від одного).
    # Через це mean_speed у вікні читається ~26 м/с при реальних 47 —
    # база занижена в 1.8x, а z завищений приблизно в 1.6x. 4.0 * 1.64 ≈ 6.6,
    # округлено до 8.0. Полагодиш вікно — повертай 4.0.
    #
    # ЗАСТЕРЕЖЕННЯ щодо значень ВИЩЕ рекомендованих (зараз у user_config.json
    # стоїть 100.0 / 6000.0): це не тюнінг, а вимкнення гейта. 6000 м/с =
    # 21600 км/год ≈ 17 Махів, тобто 72x над фізичною межею платформи. Такі
    # пороги підігнані під конкретні відео так, щоб у лозі не було WARNING.
    # Наслідок: OF-вимірювання зі зривом LK (flow_quality 0.017-0.035) ідуть
    # прямо в GPS-вихід, і про це вже НЕ буде жодного сигналу в лозі.
    # Якщо гейт треба вимкнути — чесніше зробити це через of_outlier_gate=False.
    #
    # Звідки 6000: місія top (FlightSimulator/flight.mp4, 571 кадр, 77 анкерів)
    # дала максимум 5301 м/с, медіану speed-спрацювань 585 м/с і 38 спрацювань
    # на ~50 keyframe-ів. Це НЕ поодинокі викиди — на цьому матеріалі LK не
    # тримає лок майже ніколи. Перш ніж піднімати поріг ще вище, варто
    # перевірити OF, а не гейт.
    #
    # Історія: 150.0 і 1000.0 фактично вимикали обидві гілки фільтра.
    outlier_threshold_std: float = 4.0
    max_speed_mps: float = 120.0
    max_consecutive_outliers: int = 3
    process_fps: float = 1.0
    keyframe_interval: int = 30

    # ── RESEARCH 3.1: ковзний віконний back-end smoother (дефолт: вимкнено) ──
    # Вікно keyframe-фіксів + OF-одометрії, Huber-IRLS вага замість бінарного
    # Z-score-відкидання; корекція KF зсувом (TrajectoryFilter.shift).
    smoother_enabled: bool = False
    smoother_window: int = 60
    smoother_huber_k: float = 1.2
    smoother_fix_sigma_base_m: float = 5.0
    smoother_odom_sigma_base_m: float = 3.0
    # Кламп оцінки дрейфу ПЕРЕД gain і max_step_m. Оскільки далі йде
    # step = corr * gain, обмежений max_step_m, для будь-якого drift >= 12 м
    # (12 * 0.25 = 3.0 = max_step_m) фактичний крок однаковий незалежно від
    # цього ліміту — змінюється лише те, чи буде WARNING у лозі. Тобто
    # підняття цього значення (user_config: 800.0) НЕ впливає на траєкторію,
    # на відміну від max_speed_mps / outlier_threshold_std вище.
    smoother_max_correction_m: float = 50.0
    smoother_entry_prior_sigma_m: float = 15.0
    smoother_irls_iterations: int = 4
    # Fixed-lag servo (v2 після живого прогону): корекція по вузлу з лагом,
    # deadband проти смикання на шумі, гейн + ліміт кроку проти телепортів.
    smoother_correction_lag: int = 10
    smoother_deadband_m: float = 2.0
    smoother_gain: float = 0.25
    smoother_max_step_m: float = 3.0
    # LOCALIZER_IMPROVEMENT_PLAN 1.2: Huber на OF-ребрах одометрії (0 = старі
    # квадратичні ребра). Зрив LK на повторюваній текстурі інакше тягне вікно
    # з повною вагою. Типове значення для ввімкнення — те саме, що для фіксів (1.2).
    smoother_odom_huber_k: float = 0.0

    # ── ADDENDUM 1.2: forward-backward перевірка optical flow. Дефолт off. ──
    # Зараз LK фільтрується лише за status==1; треки, що «сповзли», доживають
    # до estimateAffinePartial2D. RANSAC там їх відкидає, тож на трансформацію
    # вплив малий — але вони роздувають знаменник inlier_ratio і ЗАНИЖУЮТЬ
    # flow_quality, який іде в Kalman R. Це фікс чесності метрики.
    of_fb_check: bool = False
    # Вікно і рівні піраміди LK (2026-10-28). LK трекає ВІД keyframe, тож зсув
    # росте до кінця інтервалу; 15 px / 2 рівні покривають ~60 px, а на 500 м
    # при 50 м/с сцена за 25 кадрів зсувається на ~160 px — трек губиться
    # (route3min: похибка 37–74 px на кадрах 15–25 проти ~1 px при 21 / 4).
    # 15 / 2 = стара поведінка.
    of_lk_win_px: int = 15
    of_lk_max_level: int = 2
    # Максимальний вік якоря OF, с (0 = без ліміту, стара поведінка). Якір
    # оновлюється лише на ПРИЙНЯТОМУ keyframe; після провалу LK трекає від
    # кадру 1–2 с тому, далеко поза зоною захоплення (route3min: сирий OF
    # 235–435 м убік). Застарілий якір скидається, наступний кадр — спроба keyframe.
    of_max_anchor_age_s: float = 0.0
    # Мінімальна flow_quality OF-виміру (0 = без порогу, стара поведінка).
    # Загублений LK-трек дає 0.026–0.042 (route3min 2026-10-28: усі 11 OF-вимірів
    # з похибкою > 20 м), нормальний — ≥ 0.075 на 0.5-му перцентилі.
    # Рекомендовано 0.05: відсікає 11/11 поганих і 0.15% добрих.
    of_min_flow_quality: float = 0.0
    # Базовий множник шуму OF-виміру для KF (1.5 = стара поведінка: «OF шумніший
    # за keyframe»). Після глибшої піраміди LK і порогу якості OF такий самий
    # точний, як фікс; офлайн на 6 маршрутах: q 8 / 1.5 → p95 5.9 м, max 50 м,
    # 46 стрибків; q 128 / 0.1 → 4.1 / 28 / 5 (сирий вихід 4.0 / 15 / 3).
    of_noise_scale: float = 1.5
    # LOCALIZER_IMPROVEMENT_PLAN 1.3: декореляція OF. Усі OF-позиції
    # проєктуються через ту саму гомографію keyframe, тож її похибка спільна;
    # KF рахував їх незалежними. R OF-виміру множиться на (1 + g·(n-1)), n —
    # номер OF-семплу після keyframe. 0.0 = стара поведінка; кандидат — 1.0.
    of_noise_growth_per_step: float = 0.0
    # Інтервал прогнозу KF (2026-10-28). Воркер дає keyframe-у dt від ПОПЕРЕДНЬОГО
    # keyframe (~0.9 с), хоча KF востаннє оновлював OF ~0.17 с тому: модель
    # постійної швидкості щоразу прогнозувала на 0.9 с і між keyframe-ами
    # відставала (сирий OF: крок 0.99 від справжнього, вихід KF: 0.62–0.71,
    # пилка з «стрибком» на кожному keyframe). True = dt для KF з міток часу
    # кадрів від останнього оновлення фільтра; детектор аутлаєрів і smoother
    # лишаються на dt воркера. False = стара поведінка.
    kalman_dt_from_timestamps: bool = False
    # Швидкість при re-anchor до сильного фіксу (2026-10-28). Стара поведінка:
    # (фікс − останній СИРИЙ вимір) / dt, а сирий вимір зазвичай OF; загублений
    # OF-трек перетворює точний фікс на швидкість у сотні м/с. True = швидкість
    # між двома останніми сильними фіксами (≤ 5 с), інакше власна швидкість KF.
    kalman_reanchor_velocity_from_fixes: bool = False
    # Перезапуск KF після довгої паузи без вимірів, с (0 = стара поведінка).
    # Прогноз обмежено 5 с, тож після довшої паузи вся розбіжність ішла у
    # швидкість (wayback 2026-10-28: 15 с без фіксу → 214 м/с замість 45,
    # вихід на 56 м убік при сирих вимірах з похибкою 5 м). Рекомендовано 3.0.
    kalman_max_gap_s: float = 0.0
    of_fb_max_px: float = 2.0

    # ── Аудит §1.2: гейт аутлаєрів на OF-шляху. Дефолт off = стара поведінка. ──
    # localize_optical_flow кликав outlier_detector.add_position, але НІКОЛИ
    # is_outlier — на відміну від keyframe-шляху. При keyframe_interval=30 це
    # означає, що 29 із 30 виданих позицій не проходили жодної перевірки.
    # Вмикати варто РАЗОМ із поверненням outlier_threshold_std і max_speed_mps
    # до фізичних значень — інакше гейт формально є, а фактично не спрацьовує.
    of_outlier_gate: bool = False

    # ── Локальна швидкість на OF-шляху (діагностика 2026-07-31) ──────────────
    # LK трекає завжди ВІД KEYFRAME (tracking_worker.py:486 — prev_gray навмисно
    # не оновлюється, щоб не накопичувати drift), тож зсув росте лінійно з часом
    # від keyframe. А dt для OF рахується від останньої УСПІШНОЇ локалізації
    # (tracking_worker.py:243) і дорівнює одному кроку OF. Бази різні, тому
    # швидкість у гейті завищена рівно в N разів, де N — номер OF-кадру після
    # keyframe. Через це доводилось піднімати max_speed_mps до абсурдних значень.
    #
    # True = миттєва швидкість рахується від ПОПЕРЕДНЬОЇ сирої OF-позиції
    # (локально, кадр відносно кадру). Тоді max_speed_mps знову означає
    # фізичну швидкість платформи і його можна повернути до ~2x над реальним
    # максимумом місії. False = стара поведінка побітово.
    of_local_speed: bool = False

    # ── Z-score гілка аутлаєр-детектора ──────────────────────────────────────
    # True = стара поведінка. False лишає ЛИШЕ фізичний max_speed_mps.
    # Замір на місії top 2026-07-31 (of_stride=1, of_local_speed=ON, 70
    # спрацювань): z-гілка дала 47 із них і 100% спрацювань на правильних
    # вимірах, при цьому НЕ спіймала жодного справжнього зриву — усі 22
    # викиди >12 м зловив max_speed. Гілки не перетинаються, тож z-score тут
    # додає лише шум. Вимикати варто разом з of_local_speed=True: без нього
    # швидкості накопичуються і фізичний гейт сам стає ненадійним.
    outlier_zscore_enabled: bool = True

    # ── Mahalanobis-гейт замість Z-score (флаг, дефолт off) ─────────────────
    # Z-score міряє швидкість проти вікна ПРИЙНЯТИХ швидкостей — тобто поріг
    # формується тим самим потоком, який гейт фільтрує (самопідтримна петля,
    # див. коментар вище). Mahalanobis міряє інновацію проти коваріації самого
    # KF: d² = yᵀ·S⁻¹·y, S = H·P_pred·Hᵀ + R. Ця нормалізація приходить з
    # моделі, а не з відфільтрованої вибірки, тому петлі не утворює і сама
    # послаблюється після скидання фільтра (P велика → гейт м'якший).
    # Гілки незалежні: вмикаючи цю, Z-score має сенс вимикати.
    outlier_mahalanobis_enabled: bool = False
    # χ²(2 ступені свободи, p=0.999) = 13.816 → ~0.1% хибних відсіювань, якщо
    # модель шуму адекватна. p=0.99 → 9.21, p=0.95 → 5.99 (жорсткіше).
    outlier_chi2_threshold: float = 13.816

    # ── LOCALIZER_IMPROVEMENT_PLAN 1.1: повторне захоплення замість стрибка ──
    # Стара поведінка: max_consecutive_outliers-те відкинуте вимірювання
    # приймається БЕЗУМОВНО — пряме джерело стрибка, якщо відкидання були
    # випадковими хибними фіксами. True = серію приймаємо лише тоді, коли два
    # поспіль відкинуті кандидати мають ОДНАКОВЕ зміщення від прогнозу KF
    # (справжня релокація: фільтр помиляється на сталий вектор), з допуском
    # outlier_reacquire_tol_m. outlier_reacquire_max_wait — скільки ще
    # відкидань чекати узгодженості, перш ніж прийняти примусово (0 = ніколи).
    outlier_reacquire_consistent: bool = False
    outlier_reacquire_tol_m: float = 15.0
    outlier_reacquire_max_wait: int = 10

    # ── Обхід кінематичного гейта за силою НЕЗАЛЕЖНИХ доказів ────────────────
    # Спостереження з живого прогону: гейт відкидав keyframe-фікси з 1191–1881
    # інлаєрами, а на OF-шляху — фікси з flow_quality 0.89–1.0. І те, й інше —
    # незалежне від кінематики свідчення, що вимірювання добре: інлаєри дає
    # RANSAC по геометрії, flow_quality — узгодженість самого потоку. Кінематика
    # ж спирається на ПРИПУЩЕННЯ про рух платформи, яке на нетиповому матеріалі
    # (відео з онлайн-карти, різкий монтаж, скрол) просто не виконується.
    #
    # Коли докази сильні — довіряємо вимірюванню, а не пріору. Слабкі докази
    # (flow_quality 0.02 = зрив LK на хмарі чи воді) гейт відкидає, як і раніше.
    # Дефолт False = поведінка без обходу.
    outlier_trust_strong_evidence: bool = False
    # Керуємось порогами, що вже мають сенс у системі: confidence насичується
    # на confidence_max_inliers (80), early_stop_inliers = 40. 100 — свідомо
    # вище обох, тобто «геометрію перевірено з великим запасом».
    outlier_trust_min_inliers: int = 100
    # 0.5 чисто розділяє два спостережені режими: справжній зрив LK давав
    # 0.017–0.035, реальний швидкий рух — 0.625–1.0.
    outlier_trust_min_flow_quality: float = 0.5

    # ── PIPELINE_OPTIMIZATION_PLAN §B1: рахувати OF не на кожному кадрі ──────
    # OF-кадри НЕЗАЛЕЖНІ один від одного: кожен трекається від keyframe-а
    # (tracking_worker навмисно не оновлює prev_gray/prev_pts на OF-кадрах),
    # тож пропуск не накопичує помилку — падає лише частота видачі позиції.
    # 1 = ПОТОЧНА поведінка (кожен кадр). 3 → 30 Гц стає 10 Гц.
    of_stride: int = 1
    # §B2: рахувати LK на половинній роздільності (≈4× дешевше, грубіший зсув).
    of_half_res: bool = False

    # ── Етап 6: корекція проєкційних метрів у справжні наземні ───────────────
    # WebMercator (EPSG:3857) розтягує відстані у 1/cos(lat) — ×1.49 на 47.8°.
    # Без корекції max_speed_mps і всі метрові пороги гейтять на cos(lat) від
    # заявленого значення, і залежать від широти місії. Заміряно на живому
    # прогоні 2026-07-31: 28 із 71 speed-відсіювань (39%) мали справжню
    # швидкість нижче 120 м/с — тобто були хибними суто через одиниці.
    # False = стара поведінка побітово (множник 1.0).
    ground_scale_correction: bool = False


class HomographyConfig(BaseModel):
    backend: str = "opencv"  # "poselib" | "opencv"
    # ransac_threshold REMOVED (2026-10): never read. The thresholds in use are
    # localization.ransac_threshold (localization + propagation) and
    # database.inter_frame_ransac_thresh (database build).
    max_iters: int = 2000
    confidence: float = 0.99
    use_mad_ransac: bool = True  # Адаптивне уточнення порогу через MAD
    mad_k_factor: float = 2.5
