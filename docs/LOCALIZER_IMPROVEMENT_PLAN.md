# DroneLocalization — accuracy & performance improvement plan

Date: 2026-10-25. Basis: read-only code audit of the same day (report "Пайплайни AVL з UE5 симулятором").
Companion plan: `DroneSimUE/Docs/SIMULATOR_IMPROVEMENT_PLAN.md` (the simulator side of the benchmark contract).

## Status (2026-10-27)

| Item | State | Where |
|---|---|---|
| 0.2 scorecard | **Done (code)**: `scripts/replay_tracking_ground_truth.py` replays the FULL live loop (keyframes + keyframe-anchored OF, worker dt rules, of_stride), reports p50/p95/max, jumps (step error > 10 m), per-path latency, per-stage timings; `--set key=value` A/B without touching user_config. LK extracted from the worker into `src/tracking/keyframe_flow.py` (bit-exact, tested) so replay and worker run the same code. | Windows run pending |
| 0.3 depth sign | **Resolved, plan premise was wrong**: DA-V2 outputs disparity (vendored `app.py`), so `1/median` grows with distance and `q/db` = query_alt/db_alt — the direction is CORRECT; only the docstring was wrong (fixed). Real limitation: affine-invariant output → `scripts/check_depth_scale_vs_altitude.py` measures whether the ratio carries altitude at all. Keep `scale_use_depth_hint=false` until it does. | Windows run pending |
| 0.3 docs | CLAUDE.md: v2.4, FlightSimulator = RotorPy + ortho warp, DroneSimUE added. | Done |
| 1.1 re-acquisition gate | **Done (flag)** `tracking.outlier_reacquire_consistent` (+ `_tol_m`, `_max_wait`); KF `predict_position()`. | 11 tests |
| 1.2 robust OF edges | **Done (flag)** `tracking.smoother_odom_huber_k`. Measured: one collapsed 60 m edge → max error 24.6 m quadratic vs 9.6 m Huber at 4 IRLS iterations, 4.5 m at 8 (neighbour edges recover full weight). Enable together with `smoother_irls_iterations: 8` (0.56 ms / 60-node solve). | 5 tests |
| 1.3 OF decorrelation | **Done (flag, option a)** `tracking.of_noise_growth_per_step`. | Needs Windows replay |
| 1.4, 1.5 | Open (1.5 waits for simulator S1.1). | |
| **Sawtooth root cause (2026-10-28)** | Jumps were NOT outliers: (1) LK tracked from the keyframe with a 15 px / 2-level pyramid (~60 px capture) while the scene moves up to ~160 px per interval → track lost (37–74 px error late in the interval); (2) the KF predicted over the worker's keyframe dt (~0.9 s) instead of the time since its last update (~0.17 s). Fix: `tracking.of_lk_win_px=21`, `of_lk_max_level=4`, `kalman_dt_from_timestamps=true`, `kalman_process_noise=8`. route3min full-loop replay: p50 6.1→2.0 m, p95 25.6→13.1 m, max 57→38 m, jumps >10 m 178→11, step p95 19→2.7 m. **Enabled in user_config.json 2026-10-28** (backup `.bak-20261028-pre-lk-kfdt`). Phase-1 flags (reacq/odom/ofdecor/maha) measured ~no effect or trade-offs → left off. | Done |
| **Route matrix (2026-10-28)** | Six route3min variants (base, heading, wayback 2019, winter, yawsweep, z17), full-loop replay. Found and fixed (all enabled in user_config): OF quality floor `of_min_flow_quality=0.05` (lost LK tracks, max 178→48 m); `kalman_max_gap_s=3` (15 s gap poisoned KF velocity, max 56→21 m); KF trusts OF (`kalman_process_noise=128`, `of_noise_scale=0.1`) — final vision-only: p95 3.4–4.8 m, max 6.7–15.6 m, 0–3 jumps per route. Telemetry heading hurt (disabled temporal + spatial prior) → `use_heading=false`, `temporal_prior_yaw_hint_tol_deg=20`; altitude-only telemetry helps (wayback coverage 0.90→0.97). Layer maps re-propagated with current config + `temporal_neighbors=8`: layer 3 p95 10→2.5 m. Smoother now has no measurable effect. | Done |
| **OF chain through failed keyframes (2026-10-28)** | `of_chain_on_keyframe_fail=true`, `keyframe_retry_every=5`, `of_chain_max_s=30` (enabled). Every position now carries `fix_source` (keyframe / optical_flow / dead_reckoning) + `since_fix_s` in GUI status bar, broker messages, CSV/GeoJSON export and replay rows; dead-reckoning confidence halved. wayback coverage 0.90→1.00, dead-reckoning error p50 9.9 / max 15.5 m over ~22 s; confirmed positions unchanged (p95 3.3–4.7 m). Unverified guess: first fix after a chain raises some route maxima by 3–5 m. | Done |
| Open | heading/yawsweep: 1–3 jumps, max 12–16 m (fast yaw of a tilted camera: KF tracks the image-centre ground point); wayback: 13% keyframes fail (retrieval/matching across imagery date → VLAD layer / GIM); GTX 1650 perf; corridor scaling. | |
| Next | 2.1 oracle replay (`tracking_oracle_ab.ps1`): keyframe raw p95 13 m is now the floor; telemetry A/B (`-Only tele,tele_alt`). | Windows run pending |
| A/B batch | `scratch/testtopboch_analysis/tracking_phase1_ab.ps1` (base / reacq / odom / ofdecor / maha / all on testtopboch_max_dino336 × route3min) + `tracking_table.py`. | Windows run pending |
| 2.1 robust `least_squares` loss | **Struck**: already tried as RESEARCH_INTEGRATION_PLAN 1.2 and cancelled — a robust scipy loss suppressed loop closures (see config/graph.py). Keep the calibration-error measurement; use real GNC/PCM instead. | |

Finding while doing 0.2: the existing replay scripts call the localizer only on GT slots (keyframes), so the OF path, smoother odometry edges and OF noise were never exercised by any benchmark — 1.2/1.3 could not have been measured before.

## Targets (from the user's stated requirements)

| Target | Value | Measured by |
|---|---|---|
| Absolute error | p95 < 10 m per keyframe fix | `validate_vs_ground_truth.py`, replay scripts |
| No jumps | jump rate = 0: a jump is a fix whose step error (|Δest − ΔGT| between consecutive fixes) > 10 m | replay "step error" output |
| Coverage | confirmed-fix share ≥ current baseline, false confirmations ≤ baseline (bootstrap CI) | `compare_replays.py` |
| Keyframe latency, GTX 1650 | p95 ≤ 1000 ms (now: 945 ms steady, 2385 ms worst) | `scripts/benchmark_hardware.py --compare` |
| Optical-flow (OF) step latency, GTX 1650 | ≤ 33 ms at 30 fps (now: 79 ms) | same |
| Map scale | corridor up to ~1000 km × 10–70 km, multi-layer (e.g. 500/1000/2000 m) | phase 4 sizing |
| Tilt | must work with a tilted camera, not only nadir | DroneSimUE gimbal sweep |

Every item below is an untested proposal. Every behaviour change ships behind a flag that is off by default. You enable it with a `user_config.json` snippet that the user confirms; pydantic defaults stay unchanged. Benchmarks and GUI runs happen on Windows; the user runs the given command and reports the output.

## Phase 0 — Honest measurement first (blocking)

You cannot reach "p95 < 10 m" while every benchmark is optimistic. Today, query and reference come from the same Esri orthophoto (FlightSimulator, and DroneSimUE with imagery=1).

- **0.1 Benchmark matrix.** Freeze one evaluation set:
  - (a) existing FlightSimulator runs (regression only);
  - (b) DroneSimUE runs through the new contract (simulator plan S0). These sweep reference epoch vs query epoch (same / different Wayback release / Bing), time of day, gimbal −90/−75/−60°, altitude ladder 500/1000/2000 m, and terrain-following vs constant-baro altitude;
  - (c) one external real dataset (UAV-VisLoc or AnyVisLoc subset) once (b) works.

  Store it as `benchmarks/eval_set_v1.json` (paths and sha256) so every later A/B uses identical inputs.
- **0.2 One scorecard script.** Wrap the replay, validate and compare scripts into one command. It should output all target rows above plus a per-stage latency breakdown (normalize / rotation / DINO+VLAD / FAISS / prefilter / ALIKED / LightGlue / MAGSAC / OF / KF+smoother). Without a stage breakdown, the performance work in phase 3 is guesswork.
- **0.3 Close the open suspicions.**
  - Depth sign: resolved 2026-10-27, see Status — direction correct, informativeness unproven.
  - Docs: CLAUDE.md says v2.3, but the code and files are at v2.4. The CLAUDE.md simulator line is also stale.

Exit criterion: the scorecard runs end-to-end on the 0.1 set and gives a baseline row with CIs.

## Phase 1 — Temporal back-end correctness (largest cheap accuracy gain, aimed at "no jumps")

The audit found the filter and smoother are probabilistically inconsistent. That is the most likely source of jumps.

- **1.1 Mahalanobis gate.** It already exists. A/B `tracking.outlier_mahalanobis_enabled=true` (χ²₂, p=0.999 → 13.816).
  - Re-examine "forced acceptance after 3 consecutive rejections". It is a direct jump mechanism. Replace it with a re-acquisition mode: require 2 mutually consistent strong fixes before re-anchoring.
- **1.2 Robust odometry factors in the smoother.** Apply Huber or DCS to the OF edges as well, not just to the fixes (`src/tracking/smoother.py`).
  - Gate: jump rate, p95. A collapsed OF track on repetitive farmland must not drag the window.
- **1.3 Decorrelate OF measurements.** Today OF positions are projected through the last keyframe's homography and enter the KF as independent absolute measurements, so the keyframe's error is counted N times. Options, cheapest first:
  - (a) inflate R with the number of OF steps since the keyframe;
  - (b) feed OF only as a relative factor between smoother nodes, with keyframe fixes the only absolute measurements.
  - Choose by scorecard.
- **1.4 Fixed-lag servo audit.** The `smoother_max_correction_m` cap is 90 m and the step clip is 3 m. Verify that a large, legitimate correction is not delivered as a visible "slide" that counts as a jump. Measure the step-error distribution during servo activity.
- **1.5 Telemetry altitude for layer choice.** This is what the user stated telemetry is for.
  - Set `flight_data.source=csv` (DroneSimUE "generic" preset) or `mavlink` live, with `altitude_hint.enabled=true`.
  - Wait for simulator fix S1.1 (geoid) first.
  - Gate: time-to-first-fix and wrong-layer rate on the altitude ladder.

## Phase 2 — Front-end and calibration accuracy

- **2.1 Reference-calibration error budget.** Fix error ≈ homography error + the propagated calibration error of the reference keyframe.
  - First measure the second term alone: compare `get_frame_transform` against simulator ground truth on the DB flight itself. If it already eats a large part of the 10 m, matcher work cannot fix it.
  - Measures for the graph (`src/geometry/pose_graph/`):
    - ~~robust `least_squares` loss~~ — already tried and cancelled (suppressed loop closures);
    - a real GNC (Yang et al. 2020) or PCM as an option next to `gnc_spatial`;
    - A/B the stage 7.1 `kinematic_prior_weight` > 0.
- **2.2 Tilt-aware terrain prior.** `terrain_scale.py` assumes nadir and constant baro altitude. When telemetry gives camera pitch/roll, replace the node prior log GSD = log((H−h)/f) with the slant-range form along the optical axis, and evaluate on the DroneSimUE gimbal sweep. This is the user's "must also work under tilt" requirement.
- **2.3 Rotation.**
  - When telemetry yaw is available, use it as the primary prior and skip the 4-way scan (already supported via the yaw hint).
  - Without yaw, A/B `rotation_scan_step_deg` 90 vs 45 (accuracy vs cost), then Steerers (arXiv 2312.02152) as a rotation-equivariant option.
  - Test bed: DroneSimUE FPV/Manual runs with large yaw rates.
- **2.4 Matcher weights.** GIM-trained LightGlue weights (arXiv 2402.11095) as a drop-in A/B, especially on farmland.
- **2.5 Retrieval ablations.** Run each on the 0.1 set, with a new VLAD vocabulary per setting:
  - `models.vlad.layer` (intermediate DINOv3 layer, AnyLoc);
  - `models.vlad.low_norm_fraction=0.1` (SALAD-style dustbin);
  - adaptive patchify per the user's requirement (keep patchify, make it depend on layer altitude spacing).
  - Report Recall@K and PDM@K (`retrieval_metrics.py`).
- **2.6 Rescue matcher (Stozhar tier only).** RoMa v2 (arXiv 2511.15706) or MASt3R, invoked only after `recovery_cascade` fails.
  - Gate: fixes recovered vs added false confirmations.
  - Never on the GTX 1650 tier unless its latency fits.

## Phase 3 — Performance (GTX 1650 is the gate, Stozhar is where you profile first)

Do this only after phase 0.2 gives a stage breakdown. Rank by measured share of time, not by intuition.

- **3.1 Worst-case keyframe (2385 ms).** Likely the rotation scan × scale pyramid fan-out (inference).
  - Batch all rotation/scale variants into one ALIKED forward pass.
  - Cap total LightGlue pairs per keyframe.
  - Stop early on the first candidate passing ≥40 inliers (already exists for MAGSAC; extend it to the candidate loop).
- **3.2 Steady keyframe (945 ms).**
  - Raise the A1 temporal-prior hit rate (it skips DINO entirely).
  - Use 2048 keypoints on the 1650 tier, since the DB stores 2048 anyway.
  - A/B XFeat on the 1650 tier (speed vs accuracy row in the scorecard).
  - Keep fp32 on TU117 (fp16 is 4× slower there).
- **3.3 Optical flow (79 ms → ≤33 ms).**
  - Run LK on a downscaled grey frame with a fixed small point budget.
  - Reuse the pyramid between frames.
  - Move decoding off the main thread (decord or a hardware decode path for the query too, not only for the DB).
  - If OF still misses 33 ms, run OF at 15 Hz with KF prediction in between and measure the accuracy cost.
- **3.4 Stozhar tier.** TensorRT or `torch.compile` for DINOv3/ALIKED/LightGlue (YOLO is already TRT). Check SM 12.0 support for each export before investing time (inference: support may lag).
- **3.5 Pipelining.** Overlap the next frame's decode and normalize with the current keyframe's matching (producer/consumer). Latency stays the same, throughput rises; this matters for live RTSP from DroneSimUE.

## Phase 4 — Scaling to the target corridor (strategic, decide early)

These are rough estimates. Assumptions: 1000 m layer, fx = 1920 px → 0.52 m/px, footprint ≈ 0.25 km²; keyframe overlap 0.5 and 30% sidelap → ≈ 0.0875 km² of new area per keyframe.

| Corridor | Layer | Keyframes | Local features (2048 × 128-D, fp32) | Global + patchify (15 × 256-D) |
|---|---|---|---|---|
| 1000 × 10 km | 1000 m | ~114 k | ~120 GB | ~1.8 GB |
| 1000 × 70 km | 1000 m | ~800 k | ~840 GB | ~12 GB |
| 1000 × 70 km | 500 m | ~3.2 M | ~3.4 TB | ~49 GB |

What this implies, by inference:
- Local-feature storage, not retrieval, is the first wall. Store descriptors as fp16 (half the size) or int8/PQ-compressed, and memory-map HDF5 by geo-tile.
- Exact `IndexFlatIP` over ~12 M vectors does not fit in 4 GB VRAM. Make the existing `SpatialIndex` geo-sharding mandatory when a prior exists. Use an IVF index (or the LanceDB IVF-PQ that is already built) only for the cold-start / lost case.
- Building the DB from a flown reference video does not scale to 70,000 km². Decide whether DB layers are generated by DroneSimUE flights over the corridor (simulator plan S3.3) or built directly from orthophoto tiles. The second option makes the cross-view literature (University-1652, Game4Loc) relevant again. This is a user decision; this plan does not make it.

## Order and dependencies

0.1–0.3 → 1.1–1.4 (cheap, high value) → 2.1 (budget) → 1.5 / 2.2 (need simulator S1) → 2.3–2.5 → 3.x (needs the 0.2 breakdown) → 2.6 → phase 4 decision.

## Risks

- All phase 1–3 gains are hypotheses. The specific risk is that phase 2.1 reveals the reference-calibration error already dominates the 10 m budget; then propagation, not the per-frame front end, becomes the main work.
- Synthetic-only validation can overfit to Cesium imagery. Keep a real dataset (0.1c) in every final gate.
- Paper IDs marked [mem] in the source report must be verified before citation.
