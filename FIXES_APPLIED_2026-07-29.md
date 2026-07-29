# Fixes Applied — 2026-07-29

Companion to `FRESH_AUDIT_2026-07-28.md`. Everything below is **code already
changed** in the working tree. Nothing here needed a database rebuild.

Not touched, by design: the proposals in `BETTER_APPROACHES_2026-07-28.md` (they
need a rebuild and an A/B), and `user_config.json` values (see §Decisions).

**Test state:** 444 passed in the sandbox. The 2 failures and 13 collection
errors are all `ModuleNotFoundError` for torch / h5py / filterpy / supervision —
the sandbox limits recorded in `CLAUDE.md`, not regressions. **This means the
localization hot path is not covered by the sandbox run — see §Verification.**

---

## 1. Correctness

### 1.1 MAD-RANSAC — `src/geometry/transformations.py`

Two defects, one fix.

*Was inert on the configured path.* Under `homography.backend: "poselib"` the
function returned at line 180, before the MAD block. The flag read ON and
behaved OFF. It now runs on both backends.

*Widened the inlier mask instead of narrowing it.* The threshold came from the
error distribution of **all** correspondences, outliers included, and the mask
was rebuilt from scratch. Measured on synthetic data (k = 2.5):

| true inliers | outliers | old threshold | old "inliers" | new |
|---:|---:|---:|---:|---:|
| 20 | 40 | 975.6 px | **60** | 20 |
| 10 | 50 | 280 px | **60** | ≤10 |

New behaviour:

- `compute_mad_threshold` takes an `inlier_mask` and computes the threshold over
  the RANSAC inliers only.
- New `_refine_mask_mad` intersects: `mask & (errors < thr)`. **The mask can only
  shrink** — a MAD threshold can no longer re-admit a point RANSAC rejected.
- Refinement is skipped if it would leave fewer than 4 points (`_MIN_HOMOGRAPHY_PTS`),
  so it cannot produce a degenerate inlier set.

Locked by `tests/test_mad_ransac.py` (12 tests), including a parametrised
"mask never grows" property across five inlier/outlier ratios.

**Scope note:** `use_mad_ransac` is also read by `database_builder.py:639`,
`keyframe_selector.py:84` and `propagation_pipeline.py:1186`. A rebuilt database
and a re-run propagation will now produce slightly different — more honest —
inlier counts than before.

### 1.2 Optical-flow outlier gate — `src/localization/localizer.py`, `config/localization.py`

`localize_optical_flow` called `outlier_detector.add_position` but never
`is_outlier`. At `keyframe_interval: 30` that is 29 of 30 emitted positions with
no check at all.

Added the gate behind **`tracking.of_outlier_gate`, default `false`** — bit-for-bit
current behaviour until switched on. A rejected OF fix still enters the smoother
window as odometry, mirroring how rejected keyframes are handled.

**Correction to the audit.** Writing `tests/test_outlier_detector.py` disproved
the audit's claim that the Z-gate "will essentially never fire". The effective
gate in m/s is `threshold_std · max(std_v, 1.0)`. With `threshold_std: 80` it
catches jumps > 80 m/s on a perfectly smooth track, but crosses the
`max_speed_mps: 350` cap once speed variance exceeds **350/80 = 4.375 m/s** —
ordinary for a drone in wind. Verified: with 10 m/s speed noise, a 340 m/s jump
passes both gates. `FRESH_AUDIT_2026-07-28.md` §1.2 has been rewritten with this
derivation.

### 1.3 CLI ports overrode the config — `main.py`

`--ws-port`/`--rest-port` had numeric argparse defaults that were assigned
unconditionally, so headless always bound 8080 even though
`network_api.rest_port` is 8081. Both are now `default=None`, assigned only when
explicitly passed, and the resolved ports are logged at startup. The supervisor
passes them to the child only when set, so the child reads the config the same
way a standalone run would.

### 1.4 `UnboundLocalError` in the XFeat path — `feature_extractor.py`

The "YOLO mask removed every keypoint" warning referenced `aliked_out`, bound
only in the ALIKED/RDD branch. Uses `keypoints`, in scope in both.

### 1.5 Hard-coded descriptor width — `feature_extractor.py`

`np.empty((0, 128))` replaced with the actual `desc.shape[1]` (ALIKED 128,
XFeat 64, RDD 256).

### 1.6 Model pin list assumed ALIKED — `tracking_worker.py`

`pin(["aliked", "lightglue_aliked", "dinov2"])` protected models that are never
loaded when `models.local_extractor` is `rdd` or `xfeat`, leaving the ones
actually in use evictable under VRAM pressure. New `_models_to_pin()` derives the
list and mirrors `ModelManager.load_local_extractor()` exactly — including that
anything other than `rdd`/`xfeat` silently loads ALIKED, so `"superpoint"` maps
to the ALIKED set and warns.

### 1.7 Silent swallow in feature prefetch — `propagation_pipeline.py`

`except Exception: pass` with no logging made a corrupt HDF5 indistinguishable
from an empty keyframe slot; propagation built a smaller graph and reported
success. Now:

- `ValueError`/`KeyError` → empty slot, counted, no noise;
- anything else → counted, first 5 logged with type and message;
- above `_PREFETCH_MAX_ERROR_FRAC` (1 %) → `_report_error` and propagation stops
  rather than emitting a wrong calibration;
- the caller exits early instead of burning matching and loop closure on an
  empty graph.

---

## 2. Performance

### 2.1 uint8 upload — `feature_extractor.py`

New `_upload_chw()`: the frame goes to the device as **uint8**, and
`.float().div_(255.0)` happens there. Was: float32 conversion on the CPU, then a
4× larger PCIe transfer, only for the GPU to resize to 224×224.

Applied at all four entry points — `extract_global_descriptor`,
`extract_global_descriptors_multi`, `extract_patch_tokens`,
`extract_local_features` — plus the batch path in `extract_features_batch`.

**Numerically identical**, so existing databases stay comparable: uint8→float32
is exact (all 256 values representable) and division by 255.0 is a single
correctly-rounded IEEE-754 operation. Verified over all 256 values in numpy;
the CPU-vs-CUDA half is an IEEE-754 argument, not a measurement — see
§Verification.

### 2.1b CPU resize before upload — flag-gated (added after the rebuild constraint was lifted)

§2.1 above was deliberately the conservative half, because changing descriptor
values would have invalidated existing databases. With old projects slated for
rebuild, the full form is now available behind
**`models.performance.dino_cpu_resize`, default `false`**.

When on, the frame is resized to `(S, S)` on the CPU as uint8 *before* upload
(`_cpu_resize_dino`), and only `Normalize` runs on the device. Transfer per
DINO forward, with `dinov3.input_size: 224`:

| source | original (float32) | §2.1 (uint8) | §2.1b (CPU resize) |
|---|---:|---:|---:|
| 1080p | 24.88 MB | 6.22 MB | **0.151 MB** (165×) |
| 4K | 99.53 MB | 24.88 MB | **0.151 MB** (661×) |

Per keyframe during a full recovery scan (4 angles × 5 scales = 20 forwards) at
1080p: 498 MB → 124 MB → **3.0 MB**.

Design notes:

- **One preprocessing path.** `_dino_input()` is the single entry used by both
  online localization and the database build, so query and database descriptors
  cannot diverge by preprocessing. `extract_features_batch` (the build path)
  applies the same branch.
- **`extract_local_features` is untouched** — it legitimately needs full
  resolution up to `max_local_edge`.
- **Filter choice mirrors `ResolutionNormalizer`:** `INTER_AREA` downscaling,
  `INTER_CUBIC` upscaling.
- **Aspect ratio is deliberately not preserved** — the target is square,
  exactly like the `T.Resize((S, S))` it replaces. Preserving it would change
  input geometry relative to the GPU path.

**Schema impact — the important part.** `cv2.INTER_AREA` and torchvision's
`Resize(antialias=True)` are different filters, so descriptor values shift. The
flag is therefore in `schema_fingerprint.SCHEMA_FIELDS`: a database built with
one setting is *detected as incompatible* with the other instead of silently
corrupting matches. It is deliberately **not** in
`hardware_profile.TUNABLE_KEYS` — auto-tune must never make databases
machine-dependent. Both properties are asserted in
`tests/test_dino_cpu_resize.py` (15 tests).

Turning it on requires rebuilding the database. Turning it back off requires
rebuilding again.

### 2.2 Frame rotated and rescaled twice per keyframe — `rotation_selector.py`, `localizer.py`

`RotationSelector` built the rotated (and GSD-normalised) frame to compute the
global descriptor, then discarded it; `Localizer._prepare_and_extract` immediately
redid `np.rot90().copy()` and `_scale_manager.normalize()` for the same
(angle, scale).

`RotationResult` now carries `frame` and `crop_info` for the winning pair, and
`_prepare_and_extract` takes an optional `prepared=` to skip the redundant work.
The mask is still rotated and normalised locally — the selector never sees it.

`_prepare_frame` now returns `(frame, crop_info)`; `tests/test_rotation_cascade.py`
was updated for the new contract (it caught the change, which is what it is for).

### 2.4 CUDA stream race — `feature_extractor.py`

`dino_input` and `local_batch` were produced on the default stream and consumed
on side streams with no ordering — only a trailing `torch.cuda.synchronize()`.
Added `wait_stream` on both side streams and `record_stream` on both tensors, so
the caching allocator cannot recycle a block still being read.

Also removed the per-image `torch.tensor(..., pin_memory=True)` loop (a
synchronising `cudaHostAlloc` per frame) in favour of one `np.stack`.

---

## 3. Tests added

| file | tests | what it locks |
|---|---:|---|
| `tests/test_mad_ransac.py` | 12 | §1.1 — mask never grows; threshold tight on inliers; degenerate guard; end-to-end through `estimate_homography` |
| `tests/test_outlier_detector.py` | 17 | warm-up, speed gate, Z-gate variance dependence, consecutive-reset escape hatch, `reset_consecutive=False` |
| `tests/test_geometric_verifier.py` | 14 | candidate selection, early-stop, prefilter, MNN edge cases, and the §1.1 regression at the level where it caused wrong fixes |
| `tests/test_dino_cpu_resize.py` | 15 | §2.1b — flag changes the schema fingerprint, is absent from `TUNABLE_KEYS`, square output, uint8 preserved, filter choice, transfer volume |

`GeometricVerifier` and `CandidateRetriever` were extracted from `Localizer`
specifically to be testable; this closes the first of those gaps.

Also ran `ruff check --fix` repo-wide: 48 cosmetic fixes (whitespace, unused
imports, import order). **See §Scope creep below** — this touched files outside
the fix set.

---

## Decisions left to you

`user_config.json` was **not** modified except to add the new flag at its safe
(off) default. These are proposals, not changes:

| key | now | proposed | why |
|---|---|---|---|
| `models.performance.dino_cpu_resize` | `false` (added) | `true` | Enables §2.1b. **Requires a database rebuild** — the schema fingerprint changes, so a database built with the other setting is rejected on open. |
| `tracking.of_outlier_gate` | `false` (added) | `true` | Enables §1.2. Only meaningful with the two below. |
| `tracking.outlier_threshold_std` | `80.0` | `4.0` | At 80 the gate dies above 4.375 m/s speed noise (§1.2). |
| `tracking.max_speed_mps` | `350.0` | `120.0` | 350 m/s = 1 260 km/h. |
| `graph_optimization.use_analytic_jacobian` | `false` | `true` | Analytic Jacobian is implemented, returns sparse CSR, and is verified against finite differences by `test_pose_graph_jacobian.py`. Its docstring claims 3–10×. |

The outlier trio interacts with the smoother, which was tuned against the current
values — change them together and re-run one mission, not piecemeal.

---

## Verification still required on Windows

The sandbox has no torch, so **nothing on the localization hot path was executed**
— only imported and unit-tested around the edges.

1. `pytest tests/test_localization_characterization.py` — the golden test for
   `localize_frame`. It is the one that would catch a regression from §2.2
   (frame reuse) or §1.1 (inlier counts). **Run this first.**
2. `pytest tests/` in full.
3. One tracked mission, comparing against `ground_truth.json` from a v2.3
   `FlightSimulator` recording — not against a previous run of this pipeline.
4. Benchmark the keyframe time before/after §2.1. The audit's claim that the
   470 ms global descriptor is dominated by transfer overhead is arithmetic, not
   a measurement. **This is the measurement that decides whether §2.1b is worth
   a rebuild** — if the 470 ms turns out to be mostly ViT compute rather than
   transfer, `dino_cpu_resize` buys far less than the byte counts suggest and
   should stay off.
5. If enabling §2.1b: rebuild one database, confirm the old one is *rejected* on
   open with a fingerprint mismatch (that rejection is the safety mechanism
   working), then re-measure Recall@1 — the resampling filter changed, so
   retrieval quality could move in either direction.
6. Confirm §2.4 changed nothing observable — a stream race is silent when it is
   not biting.

---

## Scope creep to review

`ruff check --fix` was run across `src`, `config`, `main.py` and `tests`, which
applied cosmetic fixes to seven files outside the fix set:

```
src/gui/dialogs/config_dialog.py     +36 -27   (import sorting, unused import)
src/utils/hardware_profile.py        +1  -1    (UP037 quoted annotation)
tests/test_object_tracker.py         +10 -9
tests/test_rdd.py                    +5  -4
tests/test_hardware_profile.py       +1  -4
tests/test_scale_manager.py          +0  -1
tests/test_config_defaults.py        +1  -0
```

These are valid fixes for findings the audit listed, but they are unrelated to
the bugs above and were not asked for. To drop them and keep the diff minimal,
on Windows:

```
git checkout -- src/gui/dialogs/config_dialog.py src/utils/hardware_profile.py tests/test_object_tracker.py tests/test_rdd.py tests/test_hardware_profile.py tests/test_scale_manager.py tests/test_config_defaults.py
```

Your parallel work on the supervised-passphrase pipe (`src/security/at_rest.py`,
`tests/test_at_rest.py`, `docs/THREAT_MODEL_AND_SECURITY.md`, and the
`_run_supervised` body in `main.py`) was **not** modified by ruff — checked
specifically for import reordering, none found. My `main.py` change sits beside
yours in `_run_supervised`; both are present and consistent, but that function is
worth a read before committing.

`ruff format` was deliberately **not** run: the sandbox has ruff 0.16.0 while
`.pre-commit-config.yaml` pins 0.1.9, and reformatting 10 files with a different
version would produce a large diff inconsistent with your own tooling.
