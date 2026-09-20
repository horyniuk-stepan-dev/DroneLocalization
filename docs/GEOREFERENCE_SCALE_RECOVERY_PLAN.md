# Georeferencing repair and scale-layer recovery plan

Date: 2026-09-19

Status: implementation completed through the correct-map, single-layer scale/tilt replay gate;
cross-layer mission replay and held-out-area validation remain open.

Projects: `DroneLocalization` and the sibling `FlightSimulator`.

## 0. Implementation record (2026-09-20)

The implementation followed the dependency order in this document.  The original
mission source was not promoted or overwritten: controlled rebuilds live under
`D:/My Projects/TEST/test_astra_changes/sources/`.

Implemented contracts:

- Homography direction, gauge normalization, exact anchor retention, source-file
  identity, v2.4 calibration metadata round-trip and explicit chain-break state.
- Anchor-connected graph support and per-frame geographic provenance.  Availability
  of a matrix no longer implies supported geography; localizer candidates are gated
  by map status.
- Transaction-like HDF5 calibration writes: a pending generation is written fully
  and swapped only after success; cancellation or exception preserves the active
  generation.
- A validated versioned scale-layer metadata contract, descriptor-schema checks,
  source-local layer beliefs and bounded handoff/rollback logic.
- Absolute geometric acceptance checks (RMSE, inlier ratio, spatial support,
  reference rank and projective-pole rejection).  Adaptive MAD refinement can no
  longer relax beyond the configured RANSAC threshold.
- A trusted-fix temporal-filter contract.  When independently strong image geometry
  disagrees with the constant-velocity Kalman output by more than 5 projected metres,
  the filter re-anchors and refreshes velocity.  Stale smoother state is reset and a
  delayed correction cannot violate the same bound.  This was added only after the
  correct-map independent replay isolated temporal filtering as the error source;
  it is not used to hide reference-map errors.

Controlled reference build:

- `sources/main_full/database.h5`: all 43 sampled reference slots retained, all 11
  exact anchors preserved, 43/43 frames classified `SUPPORTED`.
- `reports/main_full_ground_truth.csv`: true surface-centre median 0.18 m, p95
  0.64 m, maximum 0.69 m; affine p95 0.56 m; angle p95 0.12 degrees; scale p95
  0.26 percent.

Independent query evidence (query GT and altitude are evaluator-only and never enter
`Localizer.localize_frame`):

| Replay | Height / attitude | Confirmed | Raw visual error | Final output error | Warm latency |
| --- | --- | ---: | --- | --- | --- |
| `ascent` | AGL 484-983 m, nadir | 60/60 | median 1.01 m, p95 1.47 m, max 1.53 m | median 1.04 m, p95 2.21 m, max 4.60 m | median 46.1 ms, p95 76.9 ms |
| `combined_scale_tilt` | AGL 500-1000 m, pitch to 20 degrees, roll to 10 degrees | 80/80 | median 0.81 m, p95 1.44 m, max 1.49 m | median 0.79 m, p95 3.52 m, max 4.20 m | median 59.3 ms, p95 80.8 ms |

The first model-load sample is intentionally included in report maxima (about
1.4-2.2 s), but excluded from the warm-latency wording above.  The 1000 m layer has
therefore demonstrated a measured 0.5-1.0 query/reference height-ratio interval,
including tilt.  That interval is sufficient to overlap a neighbouring layer centred
near 500 m, but an actual two-layer switch has not yet been exercised end to end.

Work was paused at the user's request while generating that physical 500 m layer.
No incomplete run is presented as valid: the two partial simulator outputs are named
`reference_500m_failed_cp1252` and `reference_500m_interrupted`.  The first exposed a
Windows CP1252 console crash during calibration finalisation; the simulator entry
point now configures UTF-8 output before printing diagnostics.  A clean 500 m layer,
its database and the cross-layer replay still need to be generated after resumption.

Primary reports:

- `D:/My Projects/TEST/test_astra_changes/reports/ascent_main_full_reanchored.json`
- `D:/My Projects/TEST/test_astra_changes/reports/combined_scale_tilt_main_full_reanchored.json`
- The no-smoother ablation is
  `combined_scale_tilt_main_full_no_smoother.json`; its unchanged 96.30 m maximum
  proved that the Kalman motion model, rather than the smoother, caused the turn lag.

## 1. Decision and scope

Repair reference-map correctness before evaluating unknown-scale localization. A correct image match to an incorrectly georeferenced database frame can produce a confidently wrong location. More scale hypotheses or another feature model cannot correct that mapping.

Preserve the existing layer-search implementation and its targeted fixes. Complete the missing contracts and evaluate it on separate query flights after the reference maps pass validation. Do not rewrite already implemented scale management, PCHIP, pose-graph optimization, masking, or handoff logic.

Runtime inputs remain camera images and timestamps. Reference imagery and its offline calibration are available. Query altitude, GNSS, IMU and query ground truth are not algorithm inputs. The required result under tilt is initially the ground point observed by the image principal point, not the camera's vertical ground projection.

The previous [scale-layer plan](SCALE_LAYER_HANDOFF_IMPLEMENTATION_PLAN.md) remains the architectural reference. This plan revises the implementation order based on the actual failed GT run and additional reproductions. The sibling simulator plan is `D:/My Projects/FlightSimulator/docs/SIMULATOR_VALIDATION_IMPLEMENTATION_PLAN.md`.

## 2. Evidence and corrections to the earlier diagnosis

### 2.1 Baseline artifacts

| Artifact | Location |
| --- | --- |
| Database | `D:/My Projects/TEST/test_astra_changes/sources/main/database.h5` |
| Reference video and original calibration | `D:/My Projects/FlightSimulator/output/reference_1000m/` |
| Slot GT | Same directory, `ground_truth.json` |
| Frame GT for future replay evaluation | Same directory, `video.frames.jsonl` |
| Per-slot results | `D:/My Projects/TEST/test_astra_changes/reports/ground_truth_per_slot.csv` |
| Detailed report | Same reports directory, `ground_truth_detailed.json` |
| Original run log | `C:/Users/horyn/.codex/attachments/583e22d5-a73f-4d04-a1de-d61a55049989/Pasted text.txt` |

The database contains 43 slots, 39 stored keyframes, 11 input anchors, 32 temporal edges and zero spatial edges. Slots 9, 17, 19 and 28 have no stored local features. All 43 slots nevertheless have `frame_valid=1` after propagation. This is availability of transforms, not evidence of correct georeferencing.

The direct-center validator reports median 0.36 m, p95 146.47 m and maximum 172.78 m. Twelve of 43 slots exceed 10 m; seven exceed 100 m. The detailed validator reports median 0.335 m, p95 146.501 m and maximum 172.645 m in its ground-distance summary.

These validators do not use exactly the same target: `validate_vs_ground_truth.py` compares against the directly projected `center_mercator`; `validate_vs_telemetry.py` loads GT affines and compares affine-derived centres. Their Mercator scale factors also differ slightly. Do not mix their values into one statistic. Exact surface centre, affine approximation and camera position must be separate metrics, especially under tilt.

The previous session passed 33 focused tests for the added localizer/database/geometry changes, and the five simulator verification scripts passed when executed directly in the simulator environment. These tests do not establish end-to-end correctness of adaptive selection, imported anchors or layer transitions.

### 2.2 Findings

| ID | Finding and evidence | Consequence |
| --- | --- | --- |
| F1 | `PropagationPipeline` attaches the unchanged affine of missing anchor slot 19 to keyframe 18. The log records this explicitly. | Frame 18 receives calibration belonging to a different image. Its error is approximately 150 m; nearby optimized/interpolated frames are also wrong. |
| F2 | Stage 8 flags broken/inconsistent anchor gaps, then selects fallback frames by deviation from a time-parametrized straight line between anchor centres. Slots 11, 12, 20, 22, 23, 33 and 34 were replaced. | Valid curved motion can fail a straight-line assumption. Missing trajectory evidence is replaced by interpolation and then marked valid. |
| F3 | `frame_origin` records production method, but there is no effective geographic-trust gate for all consumers. New layer search rejects unknown/extrapolated origins but accepts interpolated origins. | The log shows approximately 109 m map error at slot 11 alongside 2048 inliers and confidence 0.99 during self-replay. Strong image geometry does not prove accurate geography. |
| F4 | Simulator keyframe prediction uses analytic transforms; the builder uses image-estimated transforms and match-failure decisions. They disagree on slot 19. | Matching configuration values cannot guarantee identical retained frames. Predicted indices must not determine anchor identity. |
| F5 | `compute_inter_frame_homography(fa, fb)` documents `fb -> fa`, but passes points from `fa` as source and `fb` as destination. | Its return direction conflicts with the accumulation and overlap contract in the builder. A numerical reproduction is described below. |
| F6 | `overlap_fraction` tests the raw determinant and positive homogeneous denominators without canonicalizing the matrix scale/sign. Accumulated poses are not renormalized. | Equivalent homographies can cause different keyframe decisions. Very small stored matrix coefficients are a numerical-risk signal, not by themselves proof of wrong projected coordinates. |
| F7 | Import reads v2.4, then save emits v2.3 and omits `keyframe_selection` and `generator`. | Compatibility and recording provenance are lost on round-trip. |
| F8 | Layer search is disabled in the current configuration; this project has one source with `scale_layer=null`, and the replay uses the reference video itself. | This run does not measure scale handoff, independent-flight retrieval or real-time performance of the new path. |

Corrections to the earlier chat analysis:

- The configured gap filler is PCHIP, not necessarily straight-line interpolation. The straight-line assumption occurs in the fallback selector; PCHIP subsequently interpolates available transforms. Replacing PCHIP with another smooth curve is not a demonstrated fix.
- Homographies are defined up to a nonzero scalar. Tiny matrix entries and determinants alone do not establish degeneracy or an incorrect mapping. The demonstrated defect is that downstream decisions depend on that arbitrary scalar, combined with unnormalized long products and the direction mismatch.
- The observed errors identify failure locations. Exact attribution of each neighbour's error to snapping, edge downweighting or interpolation requires controlled reruns; it has not yet been isolated by ablation.

### 2.3 New bounded reproductions

A grid of 30 points was translated by `(10, 5)` pixels and passed through the existing inter-frame helper. Applying the returned H from previous to current reproduced the translation with maximum coordinate error below `6e-13` pixels. Applying it in the documented current-to-previous direction gave a maximum coordinate error of 20 pixels. Existing helper tests check shape/dtype, not this directional contract.

For a 1280 x 720 image, the existing overlap implementation returned:

| Matrix | Actual projective mapping | Returned overlap |
| --- | --- | --- |
| `I` | Identity | 1.0 |
| `-I` | Identity | 0.0 |
| `1e-5 * I` | Identity | 0.0 |

These are deterministic reproductions independent of simulator accuracy or neural feature quality.

## 3. Geometry contracts that implementation must enforce

### 3.1 Homography direction and gauge

Use explicit names: `H_current_to_previous`, `P_current_to_origin`, and `H_current_to_last_keyframe`. A homogeneous point is transformed by `p_out ~ H @ p_in`, followed by division by the third component.

If the matcher estimates previous-to-current, invert it once at the builder boundary, or estimate using reversed source/destination points. Choose one convention and update its callers and tests together. Do not flip propagation edges elsewhere without separately establishing their conventions.

For the current-to-previous convention:

```text
P_0 = identity
P_i = normalize(P_(i-1) @ H_i_to_(i-1))
H_i_to_k = normalize(inverse(P_k) @ P_i)
```

After a failed adjacent match, do not pretend the reused pose is a newly measured pose. Mark the chain broken, retain the new frame, and start a local segment or re-establish a checked bridge. Store that quality/segment information with the transform.

Normalize safely before scale-sensitive decisions and after composition. Normalization may use `H[2,2]` when numerically safe, with a stable alternative gauge otherwise. Do not classify every valid H with small `H[2,2]` as degenerate. Check finite projections, a pole crossing the relevant support, rank and conditioning in appropriate normalized coordinates. A large raw condition number caused by pixel translation alone is not a universal rejection criterion.

Prefer accumulating motion relative to the last retained keyframe for overlap selection, resetting that accumulator after retention. This limits long-chain numerical growth. If full origin poses are still required, maintain them separately with explicit chain validity.

### 3.2 Anchor identity

An anchor belongs to an exact source image: `(source_id, video_frame_index, timestamp, dimensions, preprocessing convention)`. A DB slot is an indexing convenience, not interchangeable image identity.

If transferring an anchor from image a to image b is unavoidable, the required relationship is:

```text
G_b = G_a @ H_b_to_a
```

Here G maps image pixels to map coordinates, and both matrices must use compatible pixel conventions. The product can be projective even if G_a was affine; fitting it back to an affine requires a measured residual over supported pixels. Transfer is permitted only through independently verified visual geometry, with propagated uncertainty and recorded origin. A nearest-slot offset alone cannot justify it.

The initial repair should recover/store the exact anchor image or fail with a clear diagnostic. Automatic anchor transfer is a later optional feature, not required for the first safe implementation.

### 3.3 Geographic trust versus transform availability

Retain provenance and add an independent geographic status. Proposed fields:

```text
frame_origin: direct_anchor / optimized / interpolated / extrapolated / unknown
georef_status: supported / provisional / invalid / unknown
georef_reason
support_anchor_ids and graph_component_id
support_distance_seconds
georef_uncertainty_m: value or explicitly unknown
```

An anchor is not automatically accurate, an optimized node is not automatically supported by a valid anchor, and an interpolated transform is not automatically invalid. Status follows evidence: anchor identity, anchor QA and units, graph connectivity, accepted constraints and gap policy. Keep heuristic quality separate from statistically calibrated covariance.

Unknown/invalid geography cannot produce a confirmed geographic fix. Such frames may still support visual tracking or candidate retrieval. Provisional interpolation may be displayed as an estimate but must not reset absolute-fix age, promote a scale prior to trusted, or act as an absolute smoother constraint.

### 3.4 Output and scale semantics

For a planar approximation with matching effective intrinsics, `r = GSD_query / GSD_reference` is approximately a height ratio. Under tilt or relief it is a local image-scale hypothesis, not an altitude measurement.

Preserve the complete image transform chain:

```text
H_original = inverse(T_reference) @ H_matcher @ T_query
ground_observation = G_reference(H_original(principal_point_query))
```

T includes actual crop, resize, padding and rotation with integer output dimensions. If calibration does not provide the principal point, record that image centre is an approximation. Compare the output with the simulator's directly projected ground observation, not camera GNSS under tilt.

## 4. Implementation sequence

### P0. Freeze the failing case and improve the evaluator

Complexity: low to medium. Dependencies: none.

- Preserve the existing mission and dataset. Write experimental outputs to new directories; never overwrite the only failing example.
- Record code revision plus dirty diff, configuration snapshot, model/schema identity, video/calibration hashes, frame indexing and hardware. The commit alone cannot identify the current uncommitted implementation.
- Add identity checks before comparing DB and GT: source/run identity when available, dimensions, FPS, frame mapping, timestamps, duplicates and missing rows. Index equality alone is insufficient.
- Report exact surface-centre error and affine-approximation error separately. Use geodesic ground distances for final geographic metrics; label projected/pixel residuals explicitly.
- Report stored-keyframe coverage, transform availability and confirmed-geography coverage separately. Include unsupported and failed frames in denominators, with reasons.
- Preserve existing slot-based validators and extend them rather than building another incompatible evaluator. Add a frame-based evaluator for query replay.

Deliverable: reproducible baseline manifest plus JSON/CSV reports. Gate: the current bad result is reproduced; missing or mismatched GT produces an error rather than an apparently clean score.

### P1. Correct selection geometry and preserve calibration metadata

Complexity: medium. Dependencies: P0.

Primary files: `src/database/keyframe_selector.py`, `src/database/frame_processor.py`, `src/calibration/multi_anchor_calibration.py`; simulator predictor only where it shares the same geometric primitive.

- Fix the documented/actual H direction and establish the accumulator convention.
- Make overlap and motion decisions invariant to nonzero scalar multiplication of H. Audit consumers of raw affine blocks for the same mistake.
- Normalize compositions and preserve chain-break state on match failure.
- Preserve supported v2.4 metadata on load/save; explicitly validate supported versions and required identity fields. Round-trip tests must include `generator` and `keyframe_selection`.
- Specify gap limits in seconds or video-frame units with unambiguous conversion. Current `keyframe_max_gap_frames` increments per processed slot, not per source video frame.

Gate: translation plus rotation plus scale in a noncommuting sequence maps known points correctly; decisions match for `H`, `-H` and scaled H; long sequences remain finite; failed-match segments are not silently connected. Rebuild a fresh experimental DB because keyframe selection may change.

### P2. Preserve exact anchors and repair the simulator-to-database contract

Complexity: medium to high. Dependencies: P1.

- Remove unchanged-affine nearest-keyframe snapping from propagation.
- When calibration is known before DB construction, pass required anchor image IDs to the builder and force their retention independently of overlap.
- When calibration is imported later, recover features for the exact missing images from the verified source video, or return an actionable rebuild requirement. A virtual anchor node may store its direct transform without features, but cannot constrain neighbouring nodes without valid connecting measurements.
- Keep simulator keyframe prediction as a preview only. Either retain its chosen anchors explicitly, or export simulator reference calibration after reading the actual retained frame mapping.
- In the latter workflow, retrieve the GT transform for each exact selected reference image. Reject missing GT, ambiguous remapping and slot/frame-step mismatches. If changing frame_step, remap by video-frame identity using frame data/re-export, never relabel old slots.
- Select turn anchors before optional compression; do not silently discard them through collision resolution. Regenerate reference calibration as needed while preserving the existing video and original artifacts.

Gate: anchor 19 stays associated with video frame 570, or the operation fails clearly. No anchor acquires another image's affine. Forced anchors survive the actual builder, not merely the predictor.

### P3. Repair broken-graph handling and propagate trust to every consumer

Complexity: high. Dependencies: P2.

Primary files: `src/workers/propagation_pipeline.py`, `src/geometry/pose_graph/vo_guards.py`, `src/geometry/calibration_provenance.py`, `src/database/database_loader.py`, `src/localization/result_builder.py`, `src/localization/localizer.py`, `src/localization/layer_search.py`, tracking and GUI consumers.

- Treat broken connectivity and inconsistent measured constraints as different conditions. Inspect connected components and anchor support rather than only a greedy temporal route.
- Replace straight-line deviation as a correctness gate with checks grounded in accepted image correspondences, endpoint consistency, component support and residuals. Motion models can provide labelled predictions, not proof that curved motion is wrong.
- Attempt bounded recovery with existing rotation retry/skip bridges, direct matches to nearby supported references, or intermediate video frames around failed pairs. Log which recovery supplied each constraint.
- For unresolved gaps, retain optional PCHIP display estimates as provisional. Permit trusted interpolation only after a defined short-gap/support policy has passed independent validation; distance in slots alone is insufficient.
- Populate geographic status and reasons for every slot. Avoid zero residual/zero disagreement defaults being interpreted as perfect evidence when no constraints exist.
- Apply the same map-trust gate to legacy localization, layer search, optical-flow anchors, object projection, spatial activation and UI success reporting. A high-inlier self-match must not bypass it.
- Save calibration as a validated generation with recoverable commit semantics. Cancellation or write failure must leave the previous complete generation usable; do not delete the active HDF5 group before a replacement is ready.

Gate: curved synthetic trajectories with correct edges remain valid; unsupported components remain unconfirmed; injected bad-map self-matches cannot emit confirmed fixes. Cancellation and write-failure tests verify both emitted status and persistent data.

### P4. Establish a usable reference map without hiding coverage loss

Complexity: medium, with real-data evaluation. Dependencies: P0-P3.

Run controlled variants, changing one factor at a time:

| Variant | Purpose |
| --- | --- |
| Existing DB/calibration | Frozen failure baseline |
| Correct anchor identity, otherwise equivalent settings | Measure the snap contribution |
| Corrected selection with all sampled slots retained | Isolate sparsification/overlap effects |
| Supported graph recovery and explicit unresolved gaps | Measure honest accuracy and coverage |
| Denser temporal samples around turns | Measure recovery from large rotations between observations |

Initially reuse the current video to isolate algorithm changes. At 30 fps and frame_step 30, slots are one second apart; the nominal 150 m/s flight travels approximately 150 m per slot on steady segments. The observed turns can change yaw by about 100 degrees between slots. Use frame_step 15 or 10 as experiments, not a guarantee; align calibration and GT to the new indices. Merely saving every existing one-second slot cannot recover missing intermediate evidence.

Maintain two evaluation tracks: (A) references calibrated directly from exact simulator reference GT to isolate live localization, and (B) sparse anchors plus production propagation to evaluate map construction. Track A must not be claimed as evidence that propagation works. Query GT is evaluator-only in both tracks.

Provisional engineering targets for the clean nadir baseline: supported-map p95 below 3 m and no accepted map error above 10 m. Also require coverage recovery: for this fully covered 43-slot case, aim for all 39 existing keyframes to have supported geography after repair. Rejecting every difficult slot is an intermediate containment result, not completion. Freeze final numeric requirements before held-out evaluation.

### P5. Complete scale search and layer routing

Complexity: high. Dependencies: P4.

- Preserve the source-scoped candidates, geometric comparison, fresh confirmations, rollback and source-local beliefs already implemented.
- Replace unvalidated `scale_layer` dictionaries with a versioned validated contract: coverage, GSD convention/range or unknown, map quality, full descriptor identity and neighbours.
- Verify registration between geographically overlapping sources before enabling handoff. A successful match to two maps that disagree must remain a diagnostic outcome.
- Search the active source and neighbours during tracking; periodically probe alternatives and broaden scope after loss. Existing source activation/spatial filters must not permanently exclude a layer needed for recovery.
- Add source-aware regional retrieval through existing patch infrastructure where possible. For lower queries, retrieve reference subregions; for higher queries, use several query regions. Deduplicate `(source, frame, region, transform)` hypotheses and retain per-source quotas through geometric verification.
- Gate reference geographic quality before choosing a challenger. Measure support in both images and at the requested output point; do not confirm an unsupported centre from an unrelated corner patch.
- Keep beliefs source-local. If compatible GSDs are known, a proposal can be converted as `r_B = r_A * GSD_A / GSD_B`; otherwise initialize from geometry in B. Conversion is a search proposal, not proof of altitude or location.
- Batch compatible descriptor hypotheses and cache features. Preserve narrow temporal searches and reserve recovery work explicitly. Do not process the full angle/scale grid on every stable frame.
- Audit rollback beyond source/filter objects: timestamps, failure counters, flow state and pending handoff evidence must obey the accepted-frame contract.

Gate: lower-retrieval-score correct layers survive to geometry; scale-prior failure can recover within the declared budget; partial crops and non-square rotations project correctly; ambiguity, map disagreement and stale observations cannot commit a switch.

### P6. Evaluate independent flights and real-time scheduling

Complexity: high. Dependencies: P5; evaluator development can begin with P0.

Start with two or three valid reference layers and separate queries. The previously suggested `500, 707, 1000, 1414, 2000 m` set is an optional experimental grid, not an established optimum. Choose spacing from measured overlap of successful scale/viewpoint ranges. Higher-altitude FOV must remain within actual map/DEM coverage.

Replay ascent/descent, intermediate-height holds, tilt-only, combined scale/tilt, boundary oscillation, missing-layer, out-of-coverage and reacquisition scenarios. Include a held-out route and then another geographic area. Validate simulator pixel/GT consistency independently; renderer and exporter sharing code is not independent proof.

Record each source timestamp, result status, selected layer/frame/region, H and support, map status, rejection reason, processing duration, frame age, dropped frames and handoff decision. Report raw confirmed observations, predictions and smoothed output separately.

Use latest-frame scheduling with bounded queues and stale-result rejection for live sources. Preserve chronological deterministic replay as a separate benchmark mode. The current 2000 ms soft search budget is not a demonstrated real-time guarantee.

Freeze required accuracy, minimum confirmed-fix rate, maximum fix gap, maximum handoff delay/frame age and allowed false confirmations before held-out runs. Define a false confirmation using a fixed geographic error threshold and include unconfirmed frames in coverage metrics. Report finite-sample counts; zero errors on a short replay is not a universal guarantee.

For a nadir approximation, if both layers satisfy the gates over log-scale width `delta_z`, the available transition time is approximately `delta_z / abs(d(log(r))/dt)`. It must exceed search, confirmation and processing-age delays with margin. Tilt, relief and lost overlap can invalidate this estimate, so measured replay behaviour decides layer spacing.

## 5. Required regression cases

| Case | Required observable result |
| --- | --- |
| Translation and noncommuting transforms | Correct pixel correspondences and composition direction, not just a 3x3 shape |
| H, -H and scaled H | Identical valid-region overlap and retention decisions |
| Missing anchor slot | Exact image recovery/retention or explicit failure; never unchanged-affine snapping |
| Failed match followed by recovery | Broken segment is represented; no invented measurement |
| Curved trajectory with correct edges | No rejection merely for deviating from an anchor chord |
| Disconnected/unanchored component | No confirmed geography, even if optimization returns matrices |
| PCHIP through unsupported gap | Provisional output with reason, not a perfect map-quality score |
| Perfect self-match to bad map | Geographic confirmation rejected despite many inliers |
| v2.4 import/save | Identity, selection and generator fields survive; incompatible indices rejected |
| Interrupted calibration write | Previous committed data remains readable and active |
| Scale/rotation/crop combination | Correct original-coordinate centre and region projection |
| Cross-layer ambiguity or disagreement | No arbitrary confirmed switch |
| Delayed/duplicate timestamps | No extra confirmation and no stale committed result |
| Tilt | Ground-observation metric distinct from camera-position error |
| Query evaluation | Localizer inputs exclude query GT and runtime altitude |

## 6. Completion and handoff checklist

- [x] Baseline and code/config identities archived; validators distinguish their targets and units.
- [x] Selection direction and projective-scale invariance corrected and tested.
- [x] Exact anchor identity preserved across simulator, builder, import and propagation.
- [x] Unsupported transforms cannot generate confirmed geographic fixes in localization paths.
- [x] Curved paths survive evidence-based checks; unresolved gaps remain explicit.
- [x] Reference GT targets met together with meaningful coverage recovery.
- [x] Metadata and persistence survive round-trip, cancellation and write failure.
- [ ] Separate query flights demonstrate scale/tilt recovery and bidirectional handoff. Scale/tilt is proven; physical two-layer handoff remains.
- [ ] Measured processing age, fix gaps and handoff delay meet frozen cross-layer requirements. Single-layer warm latency is measured.
- [x] Final report states achieved operating ranges and unresolved cases without treating hypotheses as guarantees.

Implement P0-P4 first, then complete P5-P6. Do not tune smoother limits, increase confidence or relax geometric gates merely to hide the current map errors. Do not introduce a new depth/scale model until a controlled failure on a correct reference map shows which missing capability it would supply.
