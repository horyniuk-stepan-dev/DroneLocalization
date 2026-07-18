"""Graph-optimization and projection configuration."""

from pydantic import BaseModel


class ProjectionConfig(BaseModel):
    # UTM: справжні метри. WEB_MERCATOR розтягує відстані у 1/cos(lat)
    # (~1.5x на широтах України), через що RMSE/пороги/GSD були неузгоджені
    # з реальними метрами. Існуючі калібрування зберігають свою проєкцію
    # з файлу — зміна впливає лише на нові.
    default_mode: str = "UTM"
    strict_projection: bool = True
    fallback_to_webmercator: bool = True
    anchor_rmse_threshold_m: float = 3.0
    anchor_max_error_m: float = 5.0
    propagation_disagreement_threshold_m: float = 2.0
    localizer_sample_points: int = 9
    localizer_expected_spread_m: float = 150.0


class PropagationConfig(BaseModel):
    """Параметри графової пропагації калібрування (воркер)."""

    # Скільки слотів можна "перестрибнути" при temporal-ребрах (лог-повідомлення).
    max_skip_frames: int = 3

    # Ротаційна робастність temporal-ребер (Етап 5). off = ребро просто падає.
    # При провалі матчу — повтор із поворотом query на кут ланцюга frame_poses,
    # далі перебір k·90°. Для місій БЕЗ heading-hold (дуги/протилежні ноги).
    rotation_retry: bool = False

    # ── Мости через розриви temporal-ланцюга (Етап 8, сесія 2026-07-12). ──
    # off = ПОТОЧНА поведінка (розрив → «острів»/«апендикс», інтерпольований наосліп).
    # on: матч(last, i) упав або відсіяний гейтом → пробуємо i проти глибших
    # сусідів (до max_skip_frames). Розрив 21→22 у lasttest robив кадри 1–21
    # апендиксом без другого якоря → відліт траєкторії на кілометри.
    skip_bridges: bool = False

    # LightGlue віддав < min_matches → повторний матч mutual-NN (L2, детермінований).
    # На повторюваній ріллі LightGlue місцями «сліпне» (12–28 матчів) там, де
    # MNN по тих самих дескрипторах дає 100–800 пар (перевірено офлайн на lasttest).
    mnn_fallback: bool = False


class GraphOptimizationConfig(BaseModel):
    """Конфігурація графової оптимізації пропагації координат."""

    # Просторові ребра (loop closure detection)
    loop_closure_top_k: int = 5
    loop_closure_min_similarity: float = 0.75
    loop_closure_min_frame_gap: int = 3
    loop_closure_min_inliers: int = 15

    # Авто min_frame_gap із геометрії місії (Етап 2.1). off = явна константа вище.
    # gap_min = ceil(overlap_factor · frame_diag_px / median_disp_px) з temporal-ребер.
    loop_closure_auto_min_gap: bool = False
    loop_closure_overlap_factor: float = 1.0

    # Дистанційний префільтр кандидатів (Етап 2.2). off = усі пари матчаться.
    # Якщо прикидка |Δцентрів| (BFS temporal від якорів) > margin×діагональ_кадру_м
    # → LightGlue не запускається (відсікає аліасинг полів ДО матчингу, пришвидшує).
    loop_closure_dist_prefilter: bool = False
    loop_closure_dist_margin: float = 2.0

    # Odometry-consistency (PCM-lite, Етап 2.3). off = лише «сусідський» cluster-гейт.
    # spatial-ребро, несумісне з temporal-ланцюгом (передбачення центру j через ребро
    # vs через ланцюг, допуск росте з довжиною), отримує вагу ×factor (не викидання).
    # Ловить аліасні МІЖ СОБОЮ узгоджені ребра (паралельні ряди), які cluster пропускає.
    loop_closure_odometry_check: bool = False
    odometry_consistency_margin: float = 1.5
    odometry_drift_frac: float = 0.25
    odometry_inconsistency_factor: float = 0.3

    # Вагові коефіцієнти ребер
    temporal_edge_base_weight: float = 1.0
    spatial_edge_base_weight: float = 2.0
    # (мертвий ключ anchor_weight видалено у Етапі 6 — жорсткі fix_node не мають
    #  ваги, а м'які якорі Етапу 1.1 несуть власні anchor_base_w/anchor_sigma_floor_m)

    # ── М'які якорі (Етап 1.1). Дефолт off = fix_node (жорсткий, поточна поведінка). ──
    # Якір стає унарним фактором w_a·(state−anchor), w_a = anchor_base_w/max(σ, floor).
    # σ береться з rmse_m якоря. GT-якорі симулятора (σ≈0) → floor → практично
    # жорсткі (сим-бенчмарк не змінюється); реальні якорі (RMSE 5–10 м) стають
    # м'якими й можуть бути «переголосовані» узгодженим ланцюгом temporal-ребер.
    soft_anchors: bool = False
    anchor_base_w: float = 200.0
    anchor_sigma_floor_m: float = 0.05

    # LOO-валідація якорів (Етап 1.2, read-only): ланцюг сусідніх якорів
    # передбачає стан якоря; розбіжність > поріг → warning у звіті пропагації.
    anchor_loo_threshold_m: float = 5.0

    # Levenberg-Marquardt оптимізатор
    max_iterations: int = 50
    convergence_tolerance: float = 1e-6

    # ПРИМІТКА: robust loss (soft_l1) прибрано свідомо: spatial-ребра
    # (loop closures) мають природно великі резидуали, і robust loss
    # пригнічував саме їх — якорі тримали свої кадри, а решта "пливла".
    # Чистий L2 дає loop closures повний вплив і стягує граф у форму.

    # BFS ініціалізація початкового наближення
    use_bfs_initialization: bool = True

    # Діагностика
    export_geojson: bool = True

    # ── Швидкість (Етап 4). Дефолти = ПОТОЧНА поведінка (2-point FD, BFS). ──
    # Аналітичний якобіан звірено з FD (‖J_a−J_fd‖<1e-5); вмикати після бенчмарка.
    use_analytic_jacobian: bool = False
    warm_start: bool = False  # тепла ініціалізація з попереднього розв'язку замість BFS

    # ── Гейтинг ребер ДО оптимізації (Етап 2). Майстер-прапорець off = ПОТОЧНА ──
    # поведінка (жодне ребро не відсіюється). Дефолти гейтів м'які.
    edge_gate_enabled: bool = False
    edge_gate_max_rotation_deg: float = 40.0  # |dtheta| spatial-ребра
    edge_gate_max_scale_ratio: float = 1.6  # висота не стрибає вдвічі: |log_ds|≤log(1.6)
    edge_gate_min_inlier_ratio: float = 0.25
    edge_gate_mutual_check: bool = True  # взаємність retrieval (j теж бачить i у top-k)
    edge_gate_cluster_consistency: bool = True  # самотнє loop closure без підтримки → вага ×0.5

    # ── Two-stage prune L2→prune→L2 (Етап 3). Дефолт off. ──
    two_stage_prune: bool = False
    prune_mad_k: float = 5.0  # поріг = median + k·1.4826·MAD (у класі spatial)
    prune_max_spatial_frac: float = 0.2  # не більше 20% spatial за прохід

    # ── GNC-переваження spatial (Етап 3): плавна еволюція prune. Дефолт off. ──
    # Раунди L2 із Geman-McClure-вагами w'=w·σ²/(σ²+r²), σ=µ·(median+k·MAD) свого
    # класу, µ від опуклого до 1. Чиста сцена (без викидів > поріг) → no-op.
    # Взаємно виключно з two_stage_prune (gnc має пріоритет, якщо ввімкнено).
    gnc_spatial: bool = False
    gnc_rounds: int = 5
    gnc_mad_k: float = 3.0

    # ── Кінематичний prior (Етап 7.1). Дефолт 0.0 = вимкнено (поточна поведінка).
    # Слабкі фактори другої різниці центрів сусідніх вузлів: центр b тягнеться до
    # α-зваженої (за гепами слотів) лінійної інтерполяції сусідів a, c. Гасить
    # провисання середин між якорями там, де нема loop closures; вага ефективно
    # ~1/середній геп — на розривах keyframe selection prior слабшає сам.
    kinematic_prior_weight: float = 0.0

    # PCHIP-заповнення пропущених кадрів (Етап 4). off = посегментна лінійна.
    # Shape-preserving 5-DoF інтерполяція над УСІМА валідними кадрами (центр-базова)
    # прибирає «сходинки» на дугах розворотів для реальних даних.
    pchip_gap_fill: bool = False

    # Log-scale інтерполяція масштабу (RESEARCH_INTEGRATION_PLAN 1.3). off = ПОТОЧНА
    # поведінка (лінійна по sx, sy). on = інтерполяція log(sx), log(sy) — геодезично
    # коректна для масштабу; діє і на PCHIP, і на лінійне заповнення, і на
    # інтерполяцію між якорями MultiAnchorCalibration.
    log_scale_interp: bool = False

    # Вага temporal-ребра × якість афінного фіту H (Етап 6.2). off = без корекції.
    # Кадри з нахилом/рельєфом (великий залишок 5-точкового фіту) → менша довіра:
    # w *= 1/(1 + k·residual_px).
    temporal_weight_use_fit_quality: bool = False
    temporal_fit_quality_k: float = 0.05

    # ── Вага spatial × DINOv2-подібність (Етап 4.3). Дефолт off. w *= 0.5+0.5·sim ──
    spatial_weight_use_similarity: bool = False

    # ── Санітарний гейт temporal-ребер (Етап 8.1, сесія 2026-07-12). Дефолт off. ──
    # Ловить дегенеративні трансформації від хибного RANSAC-консенсусу на
    # повторюваній ріллі (масштаб 0.8, дикий зсув). Межі свідомо м'які —
    # нормальний рух не чіпають. Відсіяне ребро = розрив → кандидат на skip-міст.
    temporal_edge_gate: bool = False
    temporal_gate_max_rotation_deg: float = 30.0
    temporal_gate_max_scale_ratio: float = 1.4
    temporal_gate_max_shift_frac: float = 1.2  # × діагональ кадру × gap

    # ── Звірка проміжків між якорями (Етап 8.2, сесія 2026-07-12). Дефолт off. ──
    # Консистентний аліасинг (усі ребра проміжку брешуть однаково) невидимий
    # для резидуалів — єдина незалежна опора це якорі. Компонуємо temporal-ланцюг
    # якір→якір і порівнюємо з дельтою якорів: неузгоджений проміжок глушиться
    # (вага ×downweight) ДО оптимізації, а ПІСЛЯ неї його кадри, що відхиляються
    # від прямої якір→якір понад поріг, перезаповнюються штатною інтерполяцією.
    anchor_gap_check: bool = False
    anchor_gap_max_dev_m: float = 150.0
    anchor_gap_downweight: float = 0.05
