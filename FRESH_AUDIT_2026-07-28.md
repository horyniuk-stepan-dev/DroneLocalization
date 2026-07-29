# Fresh Audit — DroneLocalization (2026-07-28)

Independent read of the code, config and tests. No file under `docs/`, `.agents/`
or `KnowledgeBase/` was opened; every claim below is derived from source, from
`user_config.json`, or from a computation run in this session. Line references are
to the working tree at the time of the audit.

---

## Verdict

The architecture is sound and the modular split is real, not cosmetic. Three
things are wrong enough to change behaviour in the field, and they are all cheap
to fix:

1. **`homography.use_mad_ransac: true` does nothing under `backend: "poselib"`,
   and when it *does* run it can inflate the inlier count instead of tightening
   it.** Both halves are verified below.
2. **The global descriptor is fed a full-resolution frame that is converted to
   float32 on the CPU and shipped over PCIe, only to be resized to 224×224 on the
   GPU.** ~4× more host→device traffic than necessary, per rotation, per scale.
3. **The realtime worker decodes, detects, and localizes on one thread.** At a
   ~1 s keyframe the video display and the whole pipeline stall together.

Everything else is ranked below by cost of being wrong, not by difficulty.

---

## 1. Correctness — highest cost of error

### 1.1 MAD-RANSAC: inert on the configured path, and inverted on the fallback path

`src/geometry/transformations.py:174-230`.

```python
if backend == "poselib" and _POSELIB_AVAILABLE:
    H, mask = _estimate_homography_poselib(...)
    if is_matrix_valid(H, is_homography=True):
        return H, mask          # ← returns here
...
if use_mad_ransac and H is not None:   # ← only reachable via the OpenCV branch
    mad_threshold = compute_mad_threshold(src_pts, dst_pts, H, k=mad_k_factor)
    errors = ...
    mask = (errors < mad_threshold)
```

`user_config.json` sets `homography.backend: "poselib"` **and**
`use_mad_ransac: true`. Whenever PoseLib returns a matrix that passes
`is_matrix_valid`, the function returns before the MAD block. The flag is
therefore inert on the normal path — a knob that reads as ON and is OFF.

The second half is worse. `compute_mad_threshold`
(`transformations.py:116-147`) derives the threshold from the error distribution
of **all** correspondences, outliers included:

```
threshold = median(errors) + k · 1.4826 · MAD(errors)
```

The mask is then recomputed with that threshold — which *widens* the inlier set
when outliers dominate. Simulated in this session (k = 2.5, the configured
value):

| true inliers | outliers | outlier error | MAD threshold | reported "inliers" |
|---:|---:|---|---:|---:|
| 20 | 40 | 50–500 px | 661 px | **60** |
| 10 | 50 | 20–200 px | 280 px | **60** |
| 30 | 30 | 50–500 px | 142 px | 32 |
| 45 | 15 | 50–500 px | 3.0 px | 45 ✅ |

Above ~50 % outliers the count saturates at "all matches are inliers". That
number then drives, in order: candidate selection in
`GeometricVerifier.verify` (`inliers > best_inliers`,
`geometric_verifier.py:112`), the `early_stop_inliers: 40` break
(`geometric_verifier.py:124`), `_compute_confidence` → the Kalman
`noise_scale = 1/confidence` (`localizer.py:757`), `inlier_spread`, and the FOV
polygon. A wrong candidate frame with an inflated count can outrank a correct
one with an honest count — and the position is then read off the wrong database
frame.

Today the exposure is bounded: the MAD block only runs when PoseLib is missing
or returned an invalid matrix — i.e. precisely on the hardest frames.

**Fix:** either delete the MAD recomputation, or compute the threshold over the
RANSAC inliers only and use it strictly to *shrink* the mask
(`mask_new = mask_ransac & (errors < thr)`). Also decide what the flag should
mean under `poselib` and make the code match the config.

### 1.2 The outlier gates are weak, and the OF path has none

> **Correction (2026-07-29).** The first edition of this section claimed the
> Z-score gate "will essentially never fire". Writing
> `tests/test_outlier_detector.py` disproved that, and the claim is replaced
> below with the derived condition. The second half of the section — that the
> optical-flow path never calls `is_outlier` — was and remains structural fact.

`user_config.json`: `tracking.outlier_threshold_std: 80.0`,
`max_speed_mps: 350.0`.

`OutlierDetector.is_outlier` tests `|v − mean| / max(std_v, 1.0) > threshold_std`
(the `1.0` is a floor in the code). So the **effective** Z-gate, expressed in
m/s, is `threshold_std · max(std_v, 1.0)` — it scales with the observed speed
variance:

| speed variance of recent history | effective Z-gate | which gate binds |
|---|---:|---|
| `std_v → 0` (floored at 1.0) | 80 m/s | Z-gate — still useful |
| `std_v = 2 m/s` | 160 m/s | Z-gate |
| `std_v = 4.375 m/s` | 350 m/s | **crossover** |
| `std_v = 10 m/s` | 800 m/s | only the 350 m/s cap |

The crossover is `350 / 80 = 4.375 m/s`. Above it the Z-gate threshold exceeds
the speed cap and becomes dead code, leaving only `max_speed_mps: 350.0`
(1 260 km/h). Verified empirically: with 10 m/s speed noise in the window, a
340 m/s jump passes both gates untouched.

So the honest statement is not "the Z-gate never fires" but **"the Z-gate dies
as soon as the trajectory acquires ordinary speed noise"** — which, for a drone
in wind or on manoeuvres, is most of the time. Both branches are pinned by
`TestZScoreDependsOnHistoryVariance` in `tests/test_outlier_detector.py`.

Independently, the optical-flow path has **no gate at all**:
`localize_optical_flow` (`localizer.py:940-943`) calls
`trajectory_filter.update` and then `outlier_detector.add_position` — it never
calls `is_outlier`. With `keyframe_interval: 30`, 29 of every 30 emitted
positions are ungated OF fixes. A flow lock onto a moving cloud or a water
surface reaches the GPS output directly.

The smoother (Huber IRLS, `tracking/smoother.py`) is now the only robustness
layer, and it corrects with `smoother_correction_lag: 10` and
`smoother_max_step_m: 3.0` — it arrives late and moves slowly by design. That
is a deliberate trade, but it means there is currently **no fast rejection path**
for a single catastrophic fix. Worth an explicit decision rather than an
emergent one.

`src/tracking/outlier_detector.py` (100 lines) is also not referenced by any
test.

### 1.3 `--rest-port` argparse default silently overrides the config

`main.py:282` — `default=8080`; `main.py:326` — `APP_SETTINGS.network_api.rest_port = args.rest_port`,
executed unconditionally in headless mode.

`user_config.json` sets `network_api.rest_port: 8081`. Headless runs therefore
bind **8080**, not 8081, whether or not the user passed the flag. `--ws-port`
has the same structure; its default (8765) happens to match the config, so the
bug is invisible there.

**Fix:** `default=None` on both, assign only when `is not None`.

### 1.4 `UnboundLocalError` waiting in the XFeat path

`src/models/wrappers/feature_extractor.py:263-302`. `aliked_out` is assigned only
in the non-XFeat branch (line 274), but line 299 references
`len(aliked_out['keypoints'][0])` inside the "YOLO mask killed every keypoint"
warning — a branch reachable from *both* extractors.

Latent today (`models.local_extractor: "aliked"`), live the moment anyone
switches to XFeat. Use `len(keypoints)`, which is in scope in both branches.

### 1.5 Hard-coded descriptor width in the batch path

`feature_extractor.py:446` — `desc = np.empty((0, 128), dtype=np.float32)` when a
mask removes all keypoints. ALIKED is 128-d; XFeat is 64-d, RDD 256-d. Use
`descriptors.shape[1]`.

### 1.6 The model pin list assumes ALIKED

`src/workers/tracking_worker.py:85` — `model_manager.pin(["aliked", "lightglue_aliked", "dinov2"])`.

The registry names are `aliked`, `rdd`, `xfeat`, `superpoint`, `dinov2`, `yolo`,
`lightglue_<features>` (`model_manager.py:217-716`). With
`models.local_extractor` set to `rdd` or `xfeat`, the pin list protects models
that were never loaded while the ones actually in use stay evictable — and
`_ensure_vram_available` (`model_manager.py:174-187`) will unload them under
pressure. On a 4 GB card that is exactly when it matters.

**Fix:** build the list from `models.local_extractor`, the same way
`prewarm()` already does at `model_manager.py:196-201`.

### 1.7 Silent swallow in feature prefetch

`src/workers/propagation_pipeline.py:502-508`:

```python
try:
    features[i] = self.database.get_local_features(i)
except Exception:
    pass
```

No log, no counter. A corrupt HDF5 range, a permissions error and "this slot has
no keyframe" are indistinguishable — propagation just runs on a smaller graph and
reports success. This is the one place in the pipeline where a silent data loss
turns into a plausible-looking but wrong calibration. At minimum: count the
failures, log the first few, and refuse to proceed past a threshold.

There are **21 `except`-handlers whose entire body is `pass`** in `src/` (AST
count, this session). Several are defensible — `logging_utils` (3),
`atomic_io`, `debug_renderers`, `ws_server`, `coordinates_broker`. `result_builder.py:207`,
`multi_database_manager.py:148` and `propagation_pipeline.py:920,1116` deserve
the same review.

---

## 2. Performance — the hot path

Measured structurally, not by profiling (no GPU in this environment). Sizes are
computed from the configured resolutions.

### 2.1 Full-resolution frames are shipped to the GPU for a 224×224 descriptor

`feature_extractor.py:143-150`:

```python
dino_tensor = torch.from_numpy(image).float().div_(255.0)     # CPU, full-res, float32
dino_tensor = dino_tensor.permute(2,0,1).unsqueeze(0).to(self.device)
dino_input  = self.dinov2_transform(dino_tensor)              # → Resize(224,224) on GPU
```

Active backend is `dinov3`, `input_size: 224`. For a 1920×1080 frame the CPU
allocates and writes ~25 MB of float32, then transfers 25 MB over PCIe, to
produce a 224×224×3 tensor (~0.6 MB). At 4K it is ~100 MB per call. The batched
variant (`extract_global_descriptors_multi:186-191`) does the same per image, so
a full recovery scan (4 angles × 5 scales = 20 variants) moves ~500 MB at 1080p.

**Zero-risk fix (no numerical change):** upload `uint8`, convert on device.
`torch.from_numpy(image).to(device).permute(2,0,1).unsqueeze(0).float().div_(255.0)`
— 4× less PCIe traffic and no CPU float conversion, bit-identical output, so
existing databases stay comparable.

**Larger fix (needs care):** resize on the CPU with `cv2.INTER_AREA` before
upload — ~40× less traffic, but it changes the resampling filter and therefore
the descriptor values. That must be applied to `extract_features_batch` (the DB
build path) at the same time, or query and database descriptors stop being
comparable. Verify with cosine similarity between old and new descriptors on a
sample before committing.

Same pattern, same fix, at `extract_patch_tokens:226-230` (debug view only).

### 2.2 Every keyframe rotates and rescales the frame twice

`RotationSelector._prepare_frame` (`rotation_selector.py:188-209`) materialises
the rotated (and optionally scale-normalised) frame, caches it in a per-stage
`rot_cache`, computes the global descriptor — and then throws the cache away.
`Localizer._prepare_and_extract` (`localizer.py:966-1005`) immediately redoes
`np.rot90(...).copy()` and `_scale_manager.normalize(...)` for the winning
(angle, scale) pair before running ALIKED.

`np.rot90` for k=1,3 is a transposed view; the `.copy()` materialises it —
~6 MB at 1080p, ~25 MB at 4K, plus a `cv2.resize` when scale ≠ 1. Returning the
winning frame from `RotationSelector` (it already holds it) removes one full
copy and one resize per keyframe.

Note the existing `_feat_cache` in `localize_frame:384` already solves exactly
this problem for the temporal-prior path — the same idea just needs to reach
across the selector boundary.

### 2.3 The realtime loop is single-threaded end to end

`RealtimeTrackingWorker.run` (`tracking_worker.py:172-521`) does, in sequence, on
one thread: `video_src.read()` → `cvtColor` → YOLO `detect_and_mask` →
`localize_frame` → signal emits → sleep.

`frame_ready.emit(frame)` is at the *top* of the loop (line 191), so while a
keyframe is being localized no frame is read and none is displayed. With
`keyframe_interval: 30`, that is 29 cheap OF frames followed by one long stall.
The GUI video visibly freezes for the duration of every keyframe.

`VideoDecodeWorker` exists (`src/workers/video_decode_worker.py`, 169 lines) but
is used **only** by `calibration_dialog.py:84` — the realtime path never touches
it. A decode thread feeding a bounded queue (drop-oldest for live sources) would
decouple display smoothness from localization latency and is the single largest
perceived-performance win available.

Secondary: `fps_updated.emit(1.0 / process_duration)` (line 511) reports the
reciprocal of *this frame's* processing time, not a frame rate. On keyframes it
reads ~1; on OF frames ~600. It is not a usable metric as displayed.

### 2.4 CUDA stream usage in the batch path is unsynchronised

`extract_features_batch:367-417`. `dino_batch` and `local_batch` are created and
copied on the **default** stream with `non_blocking=True` from pinned memory
(line 336/345 use `pin_memory=True`, so the copy is genuinely async). Work is
then enqueued on `self.stream_global` / `self.stream_local` without a
`wait_stream` on the default stream and without `record_stream` on the tensors.

The trailing `torch.cuda.synchronize()` makes the *result* correct, but nothing
orders the consumer kernels after the producer copy. This is a real race, of the
kind that shows up as a rare NaN or a garbage descriptor rather than a crash.

Also: `torch.tensor(img, pin_memory=True)` allocates pinned host memory per
image per batch. `cudaHostAlloc` synchronizes; doing it in a loop is a known
anti-pattern. A reusable pinned staging buffer removes it.

### 2.5 Pose-graph optimization runs on numerical derivatives

`user_config.json` sets `graph_optimization.use_analytic_jacobian: false`, so
`optimizer.py:521-524` takes the `jac="2-point"` branch. The analytic Jacobian
is implemented (`_jacobian_vec`, `optimizer.py:665-798`, returns a proper CSR
sparse matrix) and is verified against finite differences by
`tests/test_pose_graph_jacobian.py` — which passes in this session's run.

Its own docstring claims 3–10×. The budget is `max_nfev = max_iterations *
n_vars` (line 533), which for a 1 000-frame graph is 250 000 residual
evaluations. This is the cheapest available speedup on the propagation path and
it is already written and tested.

### 2.6 Propagation loads every keyframe's features into RAM

`_prefetch_features` (`propagation_pipeline.py:499-515`) builds a dict over all
frames. At 2 048 keypoints × 128-d float32 that is ~1 MB per frame before
keypoint arrays — ~1 GB at 1 000 keyframes, ~3 GB at 3 000. There is no cap, no
spill, and no estimate logged before allocation. On the target machine this is a
hard ceiling on mission length. Worth at least a pre-flight estimate and a
warning.

---

## 3. Architecture and tech debt

### 3.1 Layering is clean; the cycles are shallow

Package dependency edges point the right way overall: `gui → workers → localization → geometry → utils`, with `config` and `utils.logging_utils` as
the two universal leaves (fan-in 26 and 67).

Five two-node cycles exist, each a single import:

| cycle | edge |
|---|---|
| `database ↔ localization` | `geo_aware_retriever.py:19` (inside a guarded/lazy import) |
| `core ↔ security` | `project_scan.py:89` (function-local import of `ProjectSettings`) |
| `core ↔ database`, `core ↔ calibration`, `utils ↔ video` | one import each |

None is load-bearing. `utils → video` (`fault_injection.py:35`) is the only one
that looks genuinely misplaced — a test/fault-injection helper importing a domain
module pulls `src.video` into anything that touches `src.utils`. Moving
`fault_injection` out of `utils` would clear it.

### 3.2 Four functions carry a disproportionate share of the complexity

Measured over 812 functions in `src/`, `config/`, `main.py`:

| function | lines | cyclomatic |
|---|---:|---:|
| `database_builder.build_from_video:74` | 532 | 89 |
| `tracking_worker.run:75` | 449 | 77 |
| `propagation_pipeline._detect_loop_closures:691` | 210 | 57 |
| `localizer.localize_frame:337` | 483 | 55 |

27 functions exceed 100 lines; 17 exceed cyclomatic 20. The distribution matters
more than the totals: **these four alone hold ~1 670 lines and are the four
least testable places in the codebase.** `build_from_video` contains nested
`prefetch_frames`, `_flush_mask_batch` and `_process_single_frame` closures —
each of which is a natural free function with a testable signature.

`localize_frame` is the one to split first, because it is on the accuracy path.
Its shape is already visible in the code: rotation/scale selection → feature
extraction → candidate expansion → verification → SIFT rescue → fallback ladder →
coordinate transform → filtering → FOV. Steps 4–9 are pure given their inputs.

### 3.3 The refactor bought testability that was never cashed in

`GeometricVerifier` and `CandidateRetriever` were extracted from `Localizer` as
explicitly stateless collaborators (their own docstrings say so). Both are
trivially unit-testable. Neither is named in any test file.

Coverage by module reference across all 69 test files:

- **48 of 95 modules (10 928 of 23 561 LOC, 46 %) are not named in any test.**
- Excluding GUI (5 008 LOC of that), **5 920 LOC of non-GUI logic is untested**.
- The largest gaps: `propagation_pipeline` (1 317), `tracking_worker` (610),
  `multi_database_manager` (386), `geometric_verifier` (259),
  `patchify` (226), `geo_aware_retriever` (186), `pose_graph/pruning` (158),
  `outlier_detector` (100), `candidate_retriever` (87), `object_projector` (85).

The pose-graph subsystem, by contrast, has ~20 dedicated test files and a
Jacobian verified against finite differences. The testing effort is real — it is
just concentrated on one subsystem while the localization path that produces the
actual output is comparatively bare.

**Cheapest high-value tests, in order:**
1. `GeometricVerifier.verify` — candidate ranking and early-stop, with a synthetic
   database. Would have caught 1.1 directly.
2. `OutlierDetector` — the safety gate, 100 lines, pure.
3. `CandidateRetriever.merge` — pure function, weighted merge arithmetic.
4. `estimate_homography` with `use_mad_ransac=True` on a majority-outlier set —
   locks 1.1 shut.

### 3.4 Configuration surface

`user_config.json` currently exposes **383 keys** across 15 top-level sections,
of which `graph_optimization` alone has 53. Most are flag-gated behaviour switches whose
Pydantic default is "old behaviour". That discipline is right — but the
population is now large enough that:

- Flags whose enablement is silently conditional (§1.1) are hard to notice.
- There is no single place that reports the *effective* configuration after
  `hw_profile.auto_tune` has applied overrides (`main.py:257-267`). The overrides
  are logged, but the resolved end state is not.

A `--print-config` that dumps the post-auto-tune `APP_CONFIG` would make every
"is this flag actually on?" question a one-line check instead of a code read.

### 3.5 Lint configuration hides a bug-finding rule

`pyproject.toml` `[tool.ruff.lint].ignore` includes `F841` (assigned-but-unused
local). Currently 8 real instances. `F841` is one of the few pyflakes rules that
routinely catches a *typo* rather than a style preference — an assignment to a
variable nobody reads is often a branch that was meant to do something.

Also `target-version = "py311"` while `requires-python = ">=3.10,<3.12"`: ruff
will accept 3.11-only syntax that the declared floor does not support.

Current lint state is otherwise clean: 29 findings, 28 auto-fixable, all
cosmetic (whitespace, unused imports, import order).

### 3.6 Repository weight

`third_party/Depth-Anything-V2` accounts for 49 MB of the tracked tree, of which
~36 MB is `assets/` (example images and videos) that no code path reads. The
`.git` directory is 70 MB. Large binaries (`models.zip` 3.6 GB, `dist/` 11 GB,
`build/`, `yolo11n-seg.pt`) are correctly untracked — `.gitignore` is doing its
job. A sparse vendoring of Depth-Anything-V2 would halve the clone size, but this
is cosmetic next to everything above.

---

## 4. Better ways to solve the tasks already being solved

These are alternatives to *existing* mechanisms, not new features. Each names the
condition under which the current approach is still the right one.

### 4.1 Robust loss instead of the prune/GNC scaffolding

`optimizer.py:528` calls `scipy.optimize.least_squares` with **no `loss=`
argument** — verified by grep across `src/`: the string `loss=` does not appear.
That is a pure L2 objective, in which one bad loop closure pulls the entire
graph.

The code compensates with three separate mechanisms: `edge_gate_*` (8 keys),
`two_stage_prune` + `prune_mad_k`, and `gnc_spatial` + `gnc_rounds` +
`gnc_mad_k`. That is a lot of machinery to reproduce what
`loss="soft_l1"` or `loss="huber"` with a tuned `f_scale` gives natively, inside
the same TRF solver, at no implementation cost.

Concretely: `loss="huber", f_scale=σ_edge` down-weights a residual as it grows
instead of binarily deleting it — which is what GNC is approximating with an
outer loop. It composes with the analytic Jacobian (scipy applies the loss
derivative itself).

**When the current approach still wins:** if the bad edges are *grossly* wrong
(hundreds of metres), a robust loss still lets them bias the solution slightly,
while pruning removes them outright. The right shape is probably
`loss="soft_l1"` as the default *plus* the existing gates for gross rejection —
and then `gnc_spatial` can be retired. Worth an A/B on one recorded mission
before committing.

### 4.2 Two-stage retrieval instead of a wider `top_k`

Current: `retrieval_top_k: 12` candidates → optional MNN prefilter keeps 2 →
LightGlue on those. The prefilter (`geometric_verifier.py:238-259`) is a good
design; the ordering is what can be improved.

DINOv3 CLS retrieval on farmland and forest is known to be weakly
discriminative — that is why `patchify`, `vlad` and `cesp` all exist in this
codebase as alternative aggregations, all currently disabled. Rather than adding
a fourth aggregation, the cheaper lever is **geometric re-ranking of the existing
top-k**: the preliminary centres already computed for loop closure
(`propagation_pipeline.py:281`) give an approximate world position for every
database frame. At localization time the last accepted fix bounds where the drone
can be. Candidates outside that radius can be dropped before any descriptor
comparison.

`geo_aware_retriever.py` already does something in this direction for the
multi-database case. Extending it to single-database mode would make the
temporal-candidate-prior (`localizer.py:1035-1088`) a special case of a general
mechanism rather than a parallel code path with its own accept thresholds
(`temporal_prior_min_mnn`, `temporal_prior_accept_inliers`,
`temporal_prior_audit_every`).

**When the current approach still wins:** at bootstrap and after out-of-coverage,
when there is no prior position. The pyramid scan is the right answer there and
should stay.

### 4.3 Scale: the pyramid is a search over something already measurable

`ScaleManager` scans up to 5 discrete scale levels × 4 rotations = 20 forward
passes at bootstrap (`rotation_selector.py:170`), then tracks an EMA prior
extracted from the homography. The `recovery_cascade` flag already halves the
common case.

But the scale ratio is `GSD_query / GSD_db`, and `GSDCalculator`
(`geometry/gsd_calculator.py`) already computes GSD from altitude, focal length
and sensor width — quantities the project settings carry
(`localizer.py:241-256`). When those are present and trustworthy, the pyramid is
searching for a number that is already known to within a few percent.

**Proposal:** when project settings supply altitude and optics, seed
`ScaleManager._prior` from GSD directly instead of `None`, and keep the pyramid
strictly as the fallback for the case where they are absent or the measured
homography disagrees with them by more than a factor. This turns a 20-forward
bootstrap into a 4-forward one (rotation only) on the common path.

**When the current approach still wins:** if reported altitude is unreliable
(barometric drift, terrain relief), the measured scale is the only truth and the
pyramid must run. Gate the seeding on agreement, not on presence.

### 4.4 Confidence should be a variance, not a score

`_compute_confidence` returns a 0–1 blend of inlier count, RMSE and spread
(`result_builder.py`), which is then inverted into a Kalman noise scale:
`noise_scale = 1.0 / max(confidence, 0.25)` (`localizer.py:758`) and
`1.5 / max(of_conf, 0.25)` (`localizer.py:941`).

That is a heuristic standing in for a covariance. The homography estimate
already carries the information needed to do it properly: the inlier
reprojection residuals give a pixel-space σ, and propagating it through
`affine_ref` gives a metric-space covariance for the centre — an actual
2×2 matrix instead of a scalar fudge. `TrajectoryFilter` takes a measurement
noise, so it can consume it.

This matters because the same scalar currently mediates the interaction between
KF, smoother and outlier detector — three components whose disagreement is
exactly what §1.2 is about.

**When the current approach still wins:** if you are not going to validate the
propagated covariance against ground truth, a tuned heuristic is more honest than
an unvalidated covariance that *looks* principled. Only do this with the
`FlightSimulator` ground-truth pairs in hand.

### 4.5 Rotation: four 90° hypotheses is a workaround for a rotation-variant descriptor

The pipeline runs DINOv3 on 4 rotations because neither the descriptor nor ALIKED
is rotation-invariant. `sift_fallback` exists for the same reason
(`localizer.py:1142-1205`) and its own comment says so.

The cheaper structural answer, if a yaw source exists at all (telemetry, IMU,
`yaw_hint_deg` is already plumbed through `localize_frame:342`), is to make the
hint the primary path rather than an optimization. `RotationSelector` already
self-heals from a wrong hint via `rotation_rescan_min_score`. The change is not
in the code — it is in whether the recording side is required to provide yaw.

Worth raising with `FlightSimulator`, since the calibration format is versioned
and shared: adding a yaw field costs one version bump and removes 3 of 4 forward
passes from the common path.

---

## 5. What is working — do not rebuild it

Stated so effort does not go here:

- **The pose-graph subsystem.** Analytic Jacobian verified against finite
  differences, sparse CSR assembly, BFS initialization, soft anchors,
  leave-one-out anchor checks, ~20 dedicated test files. 105 tests pass in this
  session across config, geometry, coordinates, scale and pose-graph suites.
- **The flag-gating discipline.** Pydantic defaults = old behaviour, enablement
  in `user_config.json`. It is why §1.1 is a config question and not a rollback.
- **The debug-view backpressure design** (`tracking_worker.py:565-580`) —
  drop-instead-of-queue with a staleness self-heal is the correct pattern and is
  implemented correctly.
- **`crop_to_affine`** (`scale_manager.py:51-91`) derives per-axis scales from
  actual crop/output sizes rather than accumulating `resize_scale` rounding. That
  is the subtle version, and it is the right one.
- **The OF-seam fix** (`localizer.py:744-748`) — committing `_last_state` only
  after the outlier gate. The comment explains a real bug that was really fixed.
- **Repository hygiene.** No model weights, no build output, no `dist/` in git.

---

## 6. Suggested order

Ranked by (cost of being wrong) × (cheapness of fix):

| # | Item | § | Effort |
|---|---|---|---|
| 1 | Decide what `use_mad_ransac` means; fix or delete | 1.1 | ~1 h |
| 2 | `--rest-port` / `--ws-port` defaults → `None` | 1.3 | 10 min |
| 3 | Build the pin list from `local_extractor` | 1.6 | 15 min |
| 4 | `aliked_out` → `keypoints`; descriptor width from shape | 1.4, 1.5 | 15 min |
| 5 | uint8 upload in all three DINO entry points | 2.1 | ~1 h |
| 6 | Enable `use_analytic_jacobian` (already tested) | 2.5 | 5 min + benchmark |
| 7 | Log and cap prefetch failures | 1.7 | 30 min |
| 8 | Unit tests for `GeometricVerifier` and `OutlierDetector` | 3.3 | ~3 h |
| 9 | Return the winning frame from `RotationSelector` | 2.2 | ~1 h |
| 10 | Decide the outlier-gate policy for the OF path | 1.2 | design first |
| 11 | Decode thread for the realtime path | 2.3 | ~1 day |
| 12 | A/B `loss="soft_l1"` against prune/GNC | 4.1 | ~1 day |

Items 1–7 are independent of each other and independently verifiable.

---

## 7. Risks in this audit

**What would change these conclusions:**

- **§2.1 and §2.2 are structural estimates, not measurements.** I have no GPU
  here. The byte counts are arithmetic from the configured resolutions and are
  solid; the *wall-clock* share is not. Item 5 should be benchmarked on the
  GTX 1650 before and after, not assumed.
- **§1.1's field impact depends on how often PoseLib returns an invalid
  matrix** — which I cannot measure from source. If that is ~0 %, the bug is
  purely "a config flag that lies"; if it is a few percent of hard frames, it is
  a source of wrong fixes. A counter on the fallback branch would settle it in
  one mission.
- **§1.2 may be a deliberate trade** that the thresholds encode intentionally.
  I read the configured values, not the reasoning behind them. The claim I stand
  behind unconditionally is the narrower one: **the OF path calls
  `add_position` without ever calling `is_outlier`** — that is structural, not a
  tuning choice.
- **The test-coverage figure (46 %) is a module-name reference count, not line
  coverage.** It overstates coverage where a module is merely mocked and
  understates it where a module is exercised through a package import. The
  direction is reliable; the exact number is not. `pytest --cov` on the
  Windows side would give the real one.

**The strongest objection to this whole audit:** several findings — the pin list,
the XFeat `UnboundLocalError`, the hard-coded 128 — are conditional on
configuration paths nobody currently uses. They are cheap to fix and they are
real, but they are not why anything is currently going wrong. §1.1, §1.2, §2.1
and §2.3 are the ones that affect the system as it is configured today.
