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


class GraphOptimizationConfig(BaseModel):
    """Конфігурація графової оптимізації пропагації координат."""

    # Просторові ребра (loop closure detection)
    loop_closure_top_k: int = 5
    loop_closure_min_similarity: float = 0.75
    loop_closure_min_frame_gap: int = 3
    loop_closure_min_inliers: int = 15

    # Вагові коефіцієнти ребер
    temporal_edge_base_weight: float = 1.0
    spatial_edge_base_weight: float = 2.0
    anchor_weight: float = 1e6

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
    edge_gate_max_rotation_deg: float = 40.0     # |dtheta| spatial-ребра
    edge_gate_max_scale_ratio: float = 1.6       # висота не стрибає вдвічі: |log_ds|≤log(1.6)
    edge_gate_min_inlier_ratio: float = 0.25
    edge_gate_mutual_check: bool = True          # взаємність retrieval (j теж бачить i у top-k)
    edge_gate_cluster_consistency: bool = True   # самотнє loop closure без підтримки → вага ×0.5

    # ── Two-stage prune L2→prune→L2 (Етап 3). Дефолт off. ──
    two_stage_prune: bool = False
    prune_mad_k: float = 5.0                     # поріг = median + k·1.4826·MAD (у класі spatial)
    prune_max_spatial_frac: float = 0.2          # не більше 20% spatial за прохід

    # ── Вага spatial × DINOv2-подібність (Етап 4.3). Дефолт off. w *= 0.5+0.5·sim ──
    spatial_weight_use_similarity: bool = False
