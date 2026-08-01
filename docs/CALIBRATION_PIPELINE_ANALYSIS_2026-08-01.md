# Calibration Pipeline Analysis — 2026-08-01

Full audit of the database calibration pipeline: anchor creation → multi-anchor
calibration → graph propagation → HDF5. Covers `MultiAnchorCalibration`,
`MultiCalibrationManager`, `CalibrationMixin`, `CalibrationDialog` (index
mapping), `CalibrationPropagationWorker`, `PropagationPipeline`,
`PoseGraphOptimizer` + mixins (`model_5dof`, `diagnostics`, `pruning`,
`vo_guards`), `affine_utils`, `point_spread`, `GeometryTransforms`,
`CoordinateConverter`.

**Verdict:** the mathematical core is correct (analytic Jacobian and sign
algebra re-derived by hand — they match), but two adjacent mechanisms silently
break when `soft_anchors` is enabled, and several reporting/consistency issues
remain from the previous pass.

## 1. Data flow (verified against code)

1. **Anchors** — `MultiAnchorCalibration` (v2.3): list of `AnchorCalibration`
   (frame_id → pixel→metric affine + QA), atomic JSON writes, transparent
   at-rest decryption, v1.0/v2.0 migration. Between anchors: center-based
   5-DoF PCHIP (`affine_utils.build_5dof_pchip`), shared with propagation
   gap-fill; det sign stored separately (Y-flip fix). Fallback: linear.
2. **`MultiCalibrationManager`** — orchestration only:
   dict[source_id → calibration], per-source load/save.
3. **Anchor fitting** (`CalibrationMixin.on_anchor_added`): deterministic
   6-DoF LSQ over all points (`estimate_affine_lsq`), hard reject on det > 0,
   per-point leave-one-out check (5–12 pts), QA thresholds with user
   confirmation. Strong defense line against bad anchors.
4. **`PropagationPipeline`** (Qt-free; worker is a thin QThread adapter):
   prefetch features → temporal edges (skip-bridges, rotation-retry,
   MNN-fallback, sanity gate) → loop closures (DINOv2 retrieval → LightGlue;
   gates: mutual retrieval, distance prefilter, physical bounds,
   cluster/odometry consistency) → anchor fixing at Local Origin (soft/hard)
   → LM/TRF optimization → anchor-gap check (stage 8.2) → gap fill (PCHIP or
   segment-linear) → HDF5 `calibration/` v3.0 + `frame_gps`.

Error handling is strong: out-of-range anchors abort with an explanation,
prefetch distinguishes empty slots from a corrupted DB (1% threshold), anchors
on non-keyframe slots snap to the nearest keyframe, HDF5 write is under the DB
lock with reload in `finally`.

## 2. Findings

### A. Correctness / consistency

**A1. Prune + soft_anchors: disconnect guard silently no-ops**
(`pruning.py:16-30, 65-70`). `_anchor_reachable` seeds only from
`_fixed_nodes`. With `soft_anchors=True` anchors live in `_anchor_priors`,
`_fixed_nodes` is empty → both reachability sets are empty → `trial ==
base_reach` is always true → the "never disconnect a node from all anchors"
guarantee vanishes and `two_stage_prune` may cut off a graph segment.
Fix: seed from `_fixed_nodes | _anchor_priors` (as `initialize_from_bfs:188`
already does).

**A2. Anchor stress + soft_anchors: report goes empty**
(`diagnostics.py:102`). `compute_anchor_stress` iterates only `_fixed_nodes`
— with soft anchors the "anchor #N stress" diagnostics disappear entirely.
The LOO check handles both kinds (`:123-125`); stress does not.

**A3. `get_metric_position_with_depth` computes and discards the correction**
(`multi_anchor_calibration.py:299-333`). `correction` is computed, clipped,
logged — and `mx, my` is returned unchanged. The method name promises a
depth-corrected position it does not deliver. Either apply it or rename.
`set_gsd_calculator` stores `_gsd` that is never read in this class.

**A4. `frame_disagreement` is not disagreement**
(`propagation_pipeline.py:1107-1122`). It is the std of *neighbor* frame tx
values taken directly, not predictions propagated through edge transforms —
i.e. the natural spread of neighboring positions during motion. It is surfaced
to the user as "drift, m" with a red/green verdict
(`calibration_mixin.py:449-487`) — a placebo metric coloring the summary.

**A5. Edge extrapolation is a frozen constant.** Frames before the first /
after the last valid frame get a copy of the boundary affine with
`frame_valid=True` (`propagation_pipeline.py:1331-1387`). A moving drone gets
a frozen position indistinguishable from a measured one. A
`frame_extrapolated` flag would remove the ambiguity.

**A6. Two anchors snapping to one node silently drops the second**
(`propagation_pipeline.py:341-346`) — warning in the log only, unlike
out-of-range anchors which use `_report_error`. Same failure class: the user
loses an anchor without seeing it in the UI.

**A7. `save_all` never cleans an orphaned calibration**
(`multi_calibration_manager.py:74`). `if not cal.is_calibrated: continue` —
after the user deletes all anchors, the stale calibration.json remains on disk
and is picked up by the next `load_all`.

### B. Units and calibration semantics

**B1. "Meter" thresholds are projection meters, not ground meters.** In
WEB_MERCATOR at ~48° latitude all metric values are inflated by
1/cos(lat) ≈ 1.5×: `anchor_gap_max_dev_m=150`, `anchor_rmse_threshold_m=3.0`,
odometry margins, LOO thresholds, `dev_m` in logs. `ground_scale_factor`
exists (`coordinates.py:54`) but is never applied in the calibration pipeline.
Consistent on sim data (all-Mercator); on real flights the thresholds are
effectively ~50% looser and reported figures inflated. Recompute per-latitude
via cos(lat).

**B2. Isotropy regularizer is hardcoded** (`optimizer.py:633, 761`).
`w_reg = 200·cx ≈ 192 000` forces `log_sx == log_sy` — the model is
effectively similarity, not true 5-DoF. Not a bug (it also makes the
`_predict_inverse` anisotropy approximation negligible), but it is the only
heavyweight coefficient in the optimizer without a config flag, and it
interacts with `anchor_base_w=200` and edge weights.

### C. Minor

- Stale module docstring `calibration_mixin.py:1-17` describes the old
  partial/full selection logic; the code has long used `estimate_affine_lsq`.
- `on_verify_propagation:574` debug log labels RMSE as "m" though it is px
  (the UI label was already fixed).
- `downweight_gap_edges` mutates `edge.weight` permanently; GNC restores base
  weights, but diagnostics reports see the muted temporal weights.
- `_cluster_consistency_factors` is O(n²) over loop-closure specs.
- 5-DoF decomposition drops shear; anchor matrices from real tilted-camera
  fits reconstruct inexactly between anchors. Exact anchor hits return the
  raw matrix, and the isotropy regularizer bounds the effect; measure on real
  calibration.json to quantify.

## 3. Verified clean

- **Analytic Jacobian** (`optimizer.py:665-798`): re-differentiated
  `edge_residual` by hand — all eight blocks correct, including the `−res0`
  self-terms from the 1/sx weighting; FD test exists.
- **Sign algebra in `_predict_inverse`**: the diag(1,−1) conjugation cancels;
  mirroring is handled exactly. The inverse predict is exact only for
  sx = sy — negligible given the isotropy regularizer.
- **Anchor chain**: deterministic LSQ + det>0 hard reject + per-point LOO +
  QA thresholds with confirmation.
- **`check_anchor_gaps` / `downweight_gap_edges` / `select_gap_fallback_frames`**:
  coordinate systems consistent (local throughout).
- **MAD mask refinement** only shrinks the RANSAC mask.
- **`_detect_index_mode`** in the dialog closes the DB-slot bug; propagation
  duplicates the guard on its side.
- **`point_spread`** None-vs-0.0 semantics well thought out.

## 4. Priority

If `soft_anchors` is to be enabled, fix A1–A2 first (both are one-line
set-union extensions). B1 matters for interpreting real missions. The rest is
hygiene.

## 5. Audit scope / risk

`calibration_dialog` was read selectively (index mapping, video loading); the
point/anchor UI code was not audited. `matcher` and `database_loader` are
outside this pass. The 1.5× figure at 48° follows from the Mercator formula.
