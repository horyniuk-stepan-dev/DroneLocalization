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

**Status: all findings below were FIXED on 2026-08-01** (same session).
Behaviour-changing fixes ship flag-gated with defaults = previous behaviour;
`tests/test_calibration_audit_2026_08.py` locks them. See §6.

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

**A3. `get_metric_position_with_depth` is dead code, not just wrong code**
(`multi_anchor_calibration.py:299-333`). `correction` is computed, clipped,
logged — and `mx, my` is returned unchanged, so the name promises a
depth-corrected position it does not deliver. Re-checking call sites raised the
severity in the other direction: it has **zero callers** anywhere in `src/`,
`tests/` or `main.py`, as does `set_reference_depth_scale`. Only
`set_gsd_calculator` is called (`localizer.py:281`), and the `_gsd` it stores is
never read. So the fix is deletion, not repair.

**A4. `frame_disagreement` is not disagreement — and it reaches live
localization** (`propagation_pipeline.py:1107-1122`). It is the std of
*neighbour* frame `tx` values taken directly, not predictions propagated
through edge transforms. `tx = M[0,2]` is the metric position of pixel (0,0),
not of the centre, so the value mixes translation with the neighbour's
*rotation*.

This is not merely cosmetic: `ResultBuilder.compute_confidence:61` reads it as
`stability_score = 1 − (rmse/10·0.5 + disagreement/5·0.5)`, stability is 30% of
the final confidence, and confidence drives R in the Kalman filter. It is also
shown to the user as "drift, m" with a red/green verdict
(`calibration_mixin.py:449-487`).

Measured on a synthetic graph in which *every edge agrees perfectly* (so the
true disagreement is zero by construction): the legacy metric reports a mean of
51 m across well-connected frames and saturates `disagreement_norm_m = 5.0` on
6 of 8 frames, while frames with a single edge report exactly 0. A
worse-connected frame therefore looks more stable than a better-connected one.
The replacement metric reports ~1e-14 m on the same graph and saturates
nothing.

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
Consistent on sim data (all-Mercator). On real flights the constants are
therefore *stricter* than their names suggest, not looser — the earlier draft
of this note had the direction backwards. At 48° (cos = 0.669) the
`anchor_gap_max_dev_m = 150` gate fires at ≈100 ground metres; expressing 150
ground metres needs ≈224 projection metres. Reported `dev_m` figures are
correspondingly inflated.

Scope refinement after re-checking: the *self-consistent* comparisons need no
correction, because both sides are in the same units — `odometry_consistency_factors`,
`_prelim_dist_threshold`, `estimate_min_loop_gap`. Only a hardcoded metre
constant meeting a measured Mercator distance is affected: `anchor_gap_max_dev_m`,
`anchor_loo_threshold_m`, `anchor_rmse_threshold_m` / `anchor_max_error_m`,
`anchor_sigma_floor_m`, `disagreement_norm_m` / `rmse_norm_m`.

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


## 6. Fixes applied (2026-08-01)

Behaviour-changing items are flag-gated with defaults equal to the previous
behaviour, per the repo convention (pydantic defaults OFF; enablement belongs in
`user_config.json`, which was deliberately **not** touched).

| # | Fix | Files | Flag |
|---|-----|-------|------|
| A1 | `_anchor_reachable` seeds from `anchor_states()` (hard + soft), restoring the prune disconnect-guard | `pose_graph/pruning.py` | none (bug fix) |
| A2 | `compute_anchor_stress`, `diagnostics_report.num_anchors`, GeoJSON anchor marking and the optimizer log all use `anchor_states()` | `pose_graph/diagnostics.py`, `optimizer.py` | none |
| A4 | Disagreement = mean spread of edge-predicted centres for the frame, in metres | `propagation_pipeline.py` | `graph_optimization.true_disagreement` |
| A6 | Two anchors snapping to one node → `_report_error` + abort, as for out-of-range anchors | `propagation_pipeline.py` | none |
| A7 | `save_all` overwrites an existing calibration file even when all anchors were deleted | `multi_calibration_manager.py` | none |
| B1 | `anchor_gap_max_dev_m` and `anchor_loo_threshold_m` divided by cos(lat) so the constants read as ground metres | `propagation_pipeline.py` | `graph_optimization.ground_scale_thresholds` |
| B2 | Isotropy regularizer weight is a constructor argument plumbed from config | `pose_graph/optimizer.py`, `propagation_pipeline.py` | `graph_optimization.isotropy_weight` (default 200.0) |
| A3 | `get_metric_position_with_depth` and `set_reference_depth_scale` deleted; `set_gsd_calculator` documented as informational | `multi_anchor_calibration.py` | none |
| C | Stale module docstring rewritten; propagation RMSE debug log relabelled px | `gui/mixins/calibration_mixin.py` | none |

**Not fixed, deliberately:** A5 (`frame_extrapolated` flag) adds an HDF5
dataset and a consumer contract — worth doing, but it is a schema change that
deserves its own change and its own benchmark, not a rider on an audit fix.
The remaining §2C minor items (O(n²) cluster consistency, permanent
`downweight_gap_edges` mutation, shear in the 5-DoF decomposition) are
unchanged.

### Verification performed

- `tests/test_calibration_audit_2026_08.py` — 5 pure tests pass in-sandbox,
  4 more skip here and run where `h5py`/`faiss` are installed (Windows).
  The disagreement tests use a graph whose edges agree exactly, so the correct
  answer is zero and the legacy defect is visible as a failed sanity check.
- Existing runnable suites (`test_affine_utils`, `test_geometry_utils`,
  `test_pose_graph_optimizer`, `test_config_sync`, `test_coordinates*`,
  `test_projections`, `test_multi_anchor_calibration`): 59 passed, 4 skipped.
- Real imports of every touched module; zero null bytes; CRLF preserved on the
  one CRLF file (`calibration_mixin.py`).
- `ruff check` clean. `ruff format` was **not** run wholesale: the repo is not
  format-clean, so only the lines introduced here were formatted, to avoid a
  large diff in unreviewed code.

### Still required (cannot run in this environment)

1. A mission benchmark on Windows before enabling `true_disagreement` — the
   metric drops to near zero on healthy graphs, so confidence rises and Kalman
   R falls. This is a real behaviour change even though the flag defaults off.
2. A mission benchmark before enabling `ground_scale_thresholds` on
   WEB_MERCATOR projects; UTM projects are unaffected (factor 1.0).
3. `soft_anchors` may now be enabled with its guards intact — but that
   enablement is a `user_config.json` edit for the user to confirm.
