# Scale-Robust Localization and Database Layer Handoff: Implementation Plan

Date: 2026-09-18
Status: Proposed; implementation is not authorized by this document.
Scope of this change: documentation only.

## 1. Objective and operating assumptions

The application must localize a live camera stream when flight height changes, even when the current database does not contain a frame captured at exactly that height. It should continue tracking while a neighboring database layer takes over. The camera is predominantly downward-facing, but tilted views must also be evaluated.

The first implementation must work without runtime altitude, GNSS, IMU, or a rangefinder. Offline ground truth may be used for evaluation, but must not leak into the localization inputs.

The practical objective is to identify a geometrically compatible reference region and scale layer, not to recover absolute flight height from every image. No universal altitude range or confirmed localization frequency has been established yet. These are measured acceptance parameters, not promises.

This plan integrates the earlier pipeline audit and the existing research documents. It does not authorize code edits, configuration changes, model downloads, database rebuilds, or changes to the sibling FlightSimulator project.

## 2. Existing work to preserve

Read these documents before implementing related components:

| Document | Relevant contribution | Limitation |
| --- | --- | --- |
| [SCALE_INVARIANCE.md](SCALE_INVARIANCE.md) | Scale prior, pyramid search, asymmetric crop strategies, Scale-Net, HE-VPR, mosaic option | Expected coverage ranges and synthetic tests do not establish field performance |
| [RELATED_WORK_MAP.md](RELATED_WORK_MAP.md) | Comparison with complete localization systems and their map requirements | The document explicitly labels many numerical results as abstract/search-summary information |
| [RESEARCH_ADDENDUM_2026-07.md](RESEARCH_ADDENDUM_2026-07.md) | OrthoTrack-inspired recovery and correspondence-distribution checks | Full visibility-aware cropping requires geometry unavailable in a simple 2D frame database |
| [DEPTH_FOR_TILT_FEASIBILITY_2026-07-25.md](DEPTH_FOR_TILT_FEASIBILITY_2026-07-25.md) | Risks of monocular depth for aerial views | It is not a direct benchmark establishing that all depth-based tilt methods fail |
| [RESEARCH_INTEGRATION_PLAN.md](RESEARCH_INTEGRATION_PLAN.md) | Implementation history and rejected approaches | Status statements must be reconciled with current code |
| [VLAD_PLAN_2026-08-02.md](VLAD_PLAN_2026-08-02.md) | Documented retrieval A/B experiment | Primarily photometric self-retrieval; not proof of scale/tilt handoff |

Existing scale management, patchify, VLAD, RootSIFT fallback, spatial-spread checks, smoothing, and retrieval metrics must be inspected and extended rather than implemented again. Existing flags and actual database contents determine what is usable.

The earlier global soft_l1 experiment is not a new proposed fix: the integration notes record that it suppressed useful graph closures. Preserve that finding when improving calibration.

## 3. What scale means in this system

### 3.1 Image scale is not automatically altitude

For an approximately horizontal plane, a nadir pinhole camera, and consistent image coordinates:

    GSD approximately equals height_above_plane / focal_length_in_pixels

Define the query-to-reference ratio as:

    r = GSD_query / GSD_reference

Only with matching effective focal lengths and the same ground-plane assumptions does r approximately equal the ratio of heights.

Examples under those assumptions:

- r > 1: the query covers more ground per pixel. It may contain a reference view as a smaller region.
- r < 1: the query covers less ground. It may correspond to only a small part of the reference view.

Different intrinsics, digital zoom, preprocessing, terrain relief, and camera tilt break a direct altitude interpretation. Store and expose this value as an image-scale hypothesis, with its coordinate convention and uncertainty.

Absolute height is not uniquely recoverable from an arbitrary monocular image without additional assumptions or reference information. A georeferenced database supplies useful reference information, but that does not make every frame localizable.

### 3.2 Why resizing alone is insufficient

Resizing changes sampling; it does not reveal ground outside the captured field of view.

A low-altitude query may show one courtyard inside a database frame containing several blocks. Its global descriptor need not resemble the descriptor of the entire database frame. Reference subregions address this retrieval problem.

A high-altitude query may show several blocks while a reference frame shows one courtyard. Query crops may expose the shared region, but a center-only crop can miss it. Regional search or another layer is needed.

Upsampling cannot recreate missing detail. If no layer supplies enough common visible structure, the correct output is loss of localization. Mosaics or new reference coverage are then a separate data requirement.

### 3.3 Tilt changes scale across the image

For tilted views, different parts of the image can have different local scales. A homography can model the projective relation between two views of approximately planar terrain. It cannot fully account for strong terrain relief, tall structures, moving objects, or severe parallax.

A useful diagnostic is the local Jacobian J of the query-to-reference homography. Its singular values describe local directional stretch; sqrt(abs(det(J))) describes local area-based scale. Evaluate this at several supported image locations, not just at the center.

These quantities help detect anisotropy and unstable geometry. They are not direct pitch/roll or altitude measurements. Raw h31/h32 values are also not physical tilt angles: their interpretation depends on image coordinates and normalization.

Avoid rejecting every tilted view because its stretch is anisotropic. Combine local distortion with reprojection errors, spatial support, and the location of the requested output point.

### 3.4 Define the coordinate being localized

Mapping the query principal point through a planar homography estimates the ground point intersected by the optical axis. Under tilt, that point is generally different from the camera's vertical ground projection.

The baseline output must distinguish:

- a confirmed ground observation coordinate;
- a predicted coordinate propagated from a previous observation;
- an unverified retrieval candidate;
- a camera position, only if a separate pose model actually estimates it.

Use a calibrated principal point when available; otherwise identify the image-center approximation explicitly. Do not evaluate a ground observation coordinate against camera GNSS without accounting for this distinction.

## 4. Target architecture and data contracts

The proposed flow is:

    Frame + timestamp
      -> bounded scale/rotation/crop hypotheses
      -> candidates from several compatible layers
      -> local matching
      -> geometric verification in original image coordinates
      -> ambiguity and map-quality checks
      -> layer handoff controller
      -> confirmed observation or explicit unconfirmed state
      -> tracking/smoothing and UI

Retrieval similarity proposes candidates. Geometry confirms image correspondence. Reference-map quality determines whether that correspondence supports trustworthy geographic coordinates. These are different checks.

### 4.1 Layer metadata

Extend the source metadata with a versioned layer description:

| Field | Meaning |
| --- | --- |
| layer_id and source_ids | Logical scale layer and the physical database sources supporting it |
| geographic_coverage | Known valid reference coverage, including gaps |
| descriptor_fingerprint | Model/checkpoint, preprocessing, descriptor mode, dimensions, vocabulary/PCA identity where relevant |
| scale_reference | Metric GSD estimate or explicitly relative scale convention; unknown is permitted |
| scale_quality | Origin, uncertainty, spatial validity, and limitations of the scale estimate |
| neighbor_layer_ids | Layers with overlapping geography and potentially compatible scale ranges |
| map_quality | Anchor support, calibration provenance, and available error diagnostics |
| empirical_operating_envelope | Measured scale/viewpoint conditions and the dataset used to establish them |

One video source is not necessarily one uniform scale layer: a reference flight may itself change height. Keep physical storage and logical scale grouping separate. For the initial version, accept source-level layers only when their scale variation is bounded; otherwise retain per-frame estimates or explicit unknown values.

Legacy databases remain loadable. Missing metadata must not be interpreted as known altitude, zero error, or perfect calibration.

### 4.2 Candidate and observation records

A candidate must retain its source_id, layer_id, frame_id, retrieval rank/score, crop identifier, transform chain, and descriptor fingerprint. Frame IDs alone are not unique across databases.

A verified observation additionally records the full homography, inlier statistics, coverage, reprojection errors, projected ground point, map provenance, timestamp, and uncertainty or explicitly labeled quality estimate.

Keep geometric verification separate from geographic validity. A strong match to a badly calibrated reference is not a strong geographic fix.

A heuristic confidence score is not automatically a calibrated probability or covariance. Use empirical calibration before relying on it for statistical gates.

## 5. Implementation stages

### Stage 0 — Reproducible baseline and acceptance manifest

Tasks:

- Record the code revision, effective configuration, model fingerprints, database IDs, calibration version, and hardware.
- Reuse existing retrieval metrics and replay utilities where possible.
- Log per-frame search scope, scale hypothesis, source, match geometry, output status, rejection reason, and timing.
- Separate database-building data, threshold-tuning data, and held-out evaluation flights.
- Reproduce relevant audit findings against the current code before changing them.

Define an acceptance manifest containing required geographic accuracy, maximum confirmed-fix gap, handoff delay, processing-age budget, minimum confirmed localization frequency, allowed false confirmations, and tested viewpoint conditions.

These numerical requirements remain to be chosen from the intended use and baseline measurements. Freeze them before the held-out evaluation; do not choose them after seeing its results.

Deliverable: a baseline report and replay specification, with unavailable evidence explicitly marked.

### Stage 1 — Repair correctness defects before scale experiments

| Audit item | Intended action | Verification |
| --- | --- | --- |
| Preliminary pose-graph propagation can use the wrong orientation sign | Establish the coordinate orientation before generating preliminary states | Mirrored-anchor regression and existing graph tests |
| Cancellation can continue toward saving/completion | Check cancellation at stage boundaries and before committing output; distinguish canceled from completed | Cancel before optimization and before save |
| Fully masked feature extraction can retain features | Return an empty valid feature set when every point is rejected | Fully masked and partially masked inputs |
| Decoder failure can leave a blocking consumer waiting | Guarantee a terminal event carrying EOF, failure, or cancellation | Inject producer failure and verify bounded shutdown |
| Retrieval-only output can be mistaken for a fix | Audit every consumer; prevent anchor, scale-prior, and confirmed-pose updates | Retrieval-only result through worker, filters, smoother, and UI |
| Interpolation/extrapolation loses calibration provenance | Preserve direct/optimized/interpolated/extrapolated/unknown status and support distance | Interior gap and both sequence boundaries |
| Runtime/database feature compatibility is incomplete | Validate descriptor fingerprints before shared retrieval | Deliberately incompatible model or vocabulary |
| Confidence configuration and QA units can diverge from their use | Reconcile the implemented formula, configured weights, and metric/pixel units | Controlled synthetic quality inputs |

Some protections may already exist in individual call sites. For example, tracking currently excludes retrieval-only results in one optical-flow initialization branch. Do not remove that protection or assume it covers every consumer.

Also resolve object-coordinate signal delivery and effective process_fps scheduling as separate bounded fixes. Preserve timestamps and ensure their tests exercise observable behavior.

Deliverable: trustworthy result semantics and targeted regression coverage. Do not bundle unrelated refactors into this stage.

### Stage 2 — Add layer metadata and compatibility checks

Tasks:

- Add the versioned metadata and candidate identity contract.
- Construct neighbor relationships using geographic overlap and available scale evidence.
- Represent unknown scale honestly and include such layers in discovery/recovery.
- Validate coordinate systems and map registration between overlapping layers.
- Group compatible descriptor spaces; never compare unrelated embedding spaces as if cosine scores were interchangeable.
- Avoid rebuilding local features unless their actual input or model contract changes.

Deliverable: existing databases work through a compatibility adapter, and compatible layers can participate in one search.

### Stage 3 — Preserve candidates across layers until geometry

The present get_best_match path in multi_database_manager.py selects the source with the largest top retrieval score and returns candidates only from that source. This can discard the correct source before local matching.

Replace that decision boundary with a multi-source candidate result:

1. During stable tracking, query the active layer and relevant neighbors.
2. Reserve a configurable candidate quota per searched layer.
3. Deduplicate repeated full-frame/crop/rotation hypotheses while preserving their provenance.
4. Order work within a bounded matching budget.
5. Verify geometry before choosing a layer.
6. During recovery, expand geographic and layer scope.

Illustrative failure: layer A returns similarity 0.91 but inconsistent correspondences; layer B returns 0.87 and valid geometry. The old source-first choice can hide B. The new path retains B long enough to verify it. These scores are examples, not thresholds.

Even compatible embeddings can have different score distributions across datasets. Use within-layer rank and quotas initially; calibrate cross-layer score use on data rather than assuming raw similarity is sufficient.

Deliverable: a regression scenario where the correct lower-similarity source wins after geometry.

### Stage 4 — Extend scale search and geometric verification

#### A. Maintain uncertainty, not one permanent scale guess

Track log-scale because ratios are multiplicative:

    z = log(r)

For example, doubling and halving correspond to equal-magnitude changes with opposite signs. The prior should contain a mean, uncertainty, timestamp, supporting layer, and last verified observation.

Use the existing interpolation/scale mechanisms where applicable. Update them only from verified geometry with adequate map support. Increase uncertainty with elapsed time and failed observations; do not assume constant keyframe spacing.

A useful search order is:

1. Narrow hypotheses around the recent scale prior.
2. A wider local scale set when geometry weakens.
3. Alternative query/reference regions and neighboring layers.
4. Broader recovery across compatible layers.

A failed prior must allow alternative scales in the current search cycle when budget remains. Do not keep retrying only rotation at a stale scale.

#### B. Keep priors consistent across layer changes

A scale ratio relative to layer A cannot be copied unchanged into layer B.

If GSD values are known under compatible conventions:

    r_B = r_A * GSD_A / GSD_B

For example, a query GSD of 0.20 m/pixel gives r_A = 2 for a 0.10 m/pixel layer and r_B = 1 for a 0.20 m/pixel layer.

With only relative layer relationships, use the verified relative conversion and carry its uncertainty. If that relationship is unknown, initialize the new layer's prior from its own successful match. Reset rather than invent a conversion.

Under tilt, a scalar prior is only a search convenience. Preserve distortion diagnostics and do not treat the scalar as a complete geometric description.

#### C. Preserve the entire coordinate transform chain

Let Tq map original query pixels into the image used by the matcher, and Tr map original reference pixels into its matcher image. Each includes the actual crop, resize, padding, and rotation, in the order applied.

If Hm maps matcher-query coordinates to matcher-reference coordinates:

    H_original = inverse(Tr) * Hm * Tq

Only H_original should feed original-frame projection and geometric measurements. Use actual output dimensions, including integer rounding. Carry the same transforms for masks and valid image regions.

Coordinate tests must cover non-square frames, off-center crops, rotations, padding, and the combination of these operations. A visually convincing match can still produce systematically wrong GPS if this chain is incorrect.

#### D. Confirm geometry and preserve ambiguity

Evaluate reprojection error in a declared pixel convention, inlier count/ratio, correspondence spread in both images, degeneracy, supported projection region, and reference-map quality.

Measure support within the visible matched region as well as the full frame. A valid partial-overlap match should not fail simply because it covers a small fraction of a much larger reference. Conversely, extrapolating the query center far outside the inlier support should reduce confidence or prevent a geographic fix.

Use hard rejection for invalid/degenerate geometry and calibrated uncertainty for weaker but usable observations. Retrieval similarity cannot override a failed geometric gate.

If two geometrically plausible candidates imply distant locations and available evidence cannot disambiguate them, retain ambiguity. Do not manufacture a confirmed location by picking the largest scalar score.

Deliverable: stale-prior recovery, correct coordinate remapping, and explicit handling of partial overlap and ambiguous scenes.

### Stage 5 — Layer handoff state machine

| State | Meaning | Search behavior |
| --- | --- | --- |
| SEARCHING | No currently confirmed layer | Broad, budgeted candidate discovery |
| TRACKING | Active layer has recent valid observations | Active layer plus periodic neighbor probes |
| HANDOFF_PENDING | A neighboring layer is a plausible replacement | Compare active and challenger using fresh geometry |
| LOST | The last confirmed observation is too old or inconsistent | Widen scale, region, and layer search; label predictions |

Transitions:

- SEARCHING -> TRACKING: a candidate meets bootstrap geometry and ambiguity requirements.
- TRACKING -> HANDOFF_PENDING: a neighbor repeatedly becomes competitive, or active-layer support degrades.
- HANDOFF_PENDING -> TRACKING on the new layer: the challenger passes confirmation and coordinate-consistency checks.
- HANDOFF_PENDING -> TRACKING on the old layer: the challenger fails while the old layer remains valid.
- Any state -> LOST: confirmed support expires or all relevant hypotheses fail.
- LOST -> TRACKING: reacquisition passes its confirmation policy.

Hysteresis means the evidence needed to switch away from a usable layer is stronger than the evidence needed to keep it. It prevents oscillation at a scale boundary. Confirmation should use fresh observations separated in time, not duplicate hypotheses from the same frame.

Avoid a handoff deadlock: do not require both layers to stay valid indefinitely. If the old layer is lost, the challenger may establish a new track using the stricter reacquisition policy. Hysteresis must not force continued use of an invalid layer.

Compare layer observations at the same timestamp and for the same physical output quantity. If using covariance, transform it into the same metric coordinate frame and account for shared map errors; otherwise use empirically established bounds without calling them probabilistic guarantees.

A persistent offset between layers is a calibration fault. Log and expose it; do not silently correct the map online or hide it through smoothing.

The switch must update source, reference transform, map converter, geometry state, and scale prior coherently. Optical-flow state tied to the old reference must be transformed consistently or reinitialized.

Deliverable: continuous ascent/descent replays with explicit transitions, no stale-source projections, and no unjustified coordinate jumps.

### Stage 6 — Real-time scheduling and result delivery

Use an explicit per-frame budget and a bounded queue. Prefer the newest usable live frame when processing falls behind; preserve capture timestamps and record dropped frames. Replay evaluation should offer both deterministic offline comparison and realistic live scheduling.

Separate fast confirmed tracking from expensive recovery, but prevent starvation: reserve opportunities for neighbor discovery and scale rescans. Cache query descriptors by transform/model identity and reference data by database version.

If recovery runs asynchronously, attach a generation ID and capture timestamp. Reject stale results after a newer observation or incompatible layer switch; they must not overwrite current state.

Report confirmed-fix frequency separately from optical-flow update rate and display FPS. A smooth trajectory between fixes is not evidence of new absolute localization.

Do not run every optional fallback on every frame. Log deadline exhaustion separately from a geometric failure.

Deliverable: bounded processing age and measured p50/p95 latency, recovery delay, queue behavior, and peak memory on both development hardware and the GTX 1650 4 GB support target.

### Stage 7 — Optional research experiments

These experiments follow a working baseline. They do not block stages 0-6.

| Method | Hypothesis to test | Integration gate |
| --- | --- | --- |
| HE-VPR | A learned height-partition proposal reduces the number of searched layers | Usable checkpoints, transfer to this database type, preserved correct-layer recall, acceptable resources |
| Scale-Net | Pairwise scale normalization rescues an already-correct retrieval candidate | More verified fixes at acceptable additional latency |
| Dense matching | Difficult sparse-matching failures can be recovered | Reduced loss time without increased false confirmations or excessive memory |
| VLAD | Aggregation improves candidate recall across real scale/viewpoint changes | Held-out flight improvement through the full production retrieval index |
| Patch-token pooling | Cheaper regional descriptors can replace some crop forwards | Direct A/B comparison against actual crop descriptors |
| Mosaic reference regions | Missing field-of-view overlap prevents otherwise viable matches | Demonstrated coverage problem that simpler layer search cannot resolve |

Use HE-VPR as a proposal mechanism initially: retain neighboring alternatives until geometry confirms the choice. Do not claim to reproduce the paper by adding layer routing to an unmodified DINO model.

Depth, DSM, and full pose estimation become a separate project branch only if tilted/non-planar evaluation demonstrates that the 2D map model is insufficient. Relative query depth alone is not a georeferenced metric surface.

## 6. Likely code ownership

These are existing entry points to inspect, not instructions to place all new logic in them:

| Existing file | Responsibility affected |
| --- | --- |
| src/core/project_video_source.py | Source/layer metadata and compatibility |
| src/database/multi_database_manager.py | Multi-source candidate collection |
| src/database/database_builder.py | Descriptor/crop metadata and optional rebuild scope |
| src/database/video_frame_source.py | Producer lifecycle and terminal events |
| src/localization/localizer.py | Search orchestration and geometry-confirmed selection |
| src/localization/scale_manager.py | Scale hypotheses, uncertainty, remapping |
| src/localization/patchify.py | Reference subregions and transform provenance |
| src/localization/result_builder.py | Result status, quality, and output-coordinate semantics |
| src/workers/tracking_worker.py | State delivery, scheduling, tracking and object outputs |
| src/workers/propagation_pipeline.py | Cancellation and calibration provenance |
| src/geometry/pose_graph/optimizer.py | Preliminary-state orientation and graph regressions |
| src/models/wrappers/feature_extractor.py | Fully masked feature behavior |
| config/localization.py and config/database.py | Validated optional configuration |
| scripts/retrieval_metrics.py and existing tests | Reuse of evaluation and regression infrastructure |

Keep the handoff controller and candidate contracts separately testable rather than growing a monolithic Localizer. Final new filenames should follow the current package organization.

## 7. Evaluation and acceptance

### 7.1 Required scenarios

| Scenario | Expected behavior |
| --- | --- |
| Cold start at an unseen intermediate height | Search appropriate layers and confirm geometry without altitude input |
| Continuous ascent and descent | Transfer across overlapping layer envelopes in both directions |
| Wrong scale prior | Expand search and recover without waiting through repeated identical failures |
| Correct source has lower retrieval similarity | Retain and select it after geometry |
| Partial overlap and tilted views | Accept supported geometry; reject unsupported projection |
| Repeated roads, fields, roofs, or similar blocks | Preserve ambiguity rather than emit a confident distant fix |
| Blur, occlusion, or empty static mask | Degrade explicitly and recover when evidence returns |
| No suitable reference coverage | Report loss; do not substitute a database-frame center as a fix |
| Conflicting layer georeferencing | Detect the conflict and avoid smoothing it into apparent correctness |
| Decoder failure, cancellation, or asynchronous late result | Shut down or discard work without blocking or corrupting current state |

### 7.2 Metrics and denominators

Measure correct confirmed fixes, false confirmations, time without a confirmed fix, longest outage, reacquisition time, handoff delay, wrong-layer decisions, coordinate error, p50/p95 processing latency, processing age, and peak memory.

Report accuracy both among confirmed fixes and across the eligible input timeline. A method that confirms only easy frames must not appear equivalent to one that maintains coverage.

A layer ID is not inherently correct or incorrect at a boundary: several layers may localize the same frame correctly. Evaluate geometric/geographic correctness first, and layer choice against a valid-layer set where available.

Include failed frames in coverage/outage statistics. Report error in pixels separately from map error in meters. Ground truth must match the physical point being estimated.

### 7.3 Determine whether neighboring layers overlap enough

For each geographic region and viewpoint condition, measure the range of query scales where each layer meets the frozen accuracy, confirmation, and latency requirements.

A handoff needs overlap between the usable ranges of neighboring layers. Merely touching at one scale is insufficient if confirmation takes time.

If usable overlap width in log-scale is W, a measured scale-change rate is v = abs(d(log(GSD_query))/dt), and total detection/confirmation time is T, a first planning check is:

    W > v * T + uncertainty_margin

This is a design condition under the measured motion assumptions, not a guarantee. It explains why faster climbs, slower processing, or greater uncertainty require more overlap.

Tilt makes the operating envelope depend on more than one scalar. Report scale coverage separately for viewpoint/terrain groups rather than advertising one universal altitude interval.

If overlap is insufficient, add reference coverage, improve regional matching, or revise the supported operating envelope. Threshold relaxation alone is not evidence that the gap is solved.

### 7.4 Evaluation ladder

1. Targeted regression tests for correctness defects and transform composition.
2. Controlled image pairs with known transformations and adversarial ambiguities.
3. Synthetic scale changes for diagnosis, without claiming field validation.
4. Held-out real flights with actual height and viewpoint changes.
5. Live scheduling and hardware measurements.
6. A/B ablations for every optional research method.

Evaluate on development hardware and the supported minimum GPU. Do not transfer published timing results directly to either platform.

## 8. Rollout and completion order

| Work package | Depends on | Completion evidence |
| --- | --- | --- |
| A. Baseline and acceptance manifest | Available replay inputs | Reproducible baseline, declared output semantics, frozen evaluation criteria |
| B. Correctness fixes | A | Focused regressions and trustworthy result/provenance contracts |
| C. Layer metadata and multi-source candidates | B | Legacy compatibility and correct candidate retention |
| D. Scale recovery and geometry | C | Wrong-prior recovery and exact coordinate remapping |
| E. Handoff controller | D | Stable bidirectional layer transitions and ambiguity handling |
| F. Scheduling and field validation | E | Live latency/coverage/error results against the manifest |
| G. Optional research integrations | F baseline | Isolated, measured improvement over the baseline |

New algorithmic behavior should be opt-in using validated configuration, preserving existing defaults until explicitly enabled for evaluation. Correctness repairs should restore their stated contracts rather than require users to opt into correctness.

Do not silently enable VLAD, patchify, or depth in user_config.json. Database rebuilds must identify which descriptor or metadata changes require them. Preserve the old usable database until replacement validation is complete.

Rollback should allow returning to the previous algorithm/database version without incompatible metadata or descriptor mixing.

The first complete implementation milestone is A-F: reliable result semantics, multi-layer geometric search, scale recovery, controlled handoff, and measured live behavior. G is justified only by residual failure evidence.

### Implementation status (2026-09-18)

Completed in the initial low-risk pass:

- Fully masked frames now return empty local keypoints and descriptors instead of leaking the unfiltered feature set.
- The database video producer always emits its terminal queue item. Decoder exceptions are retained and raised in the database builder, so a truncated stream cannot be reported as a successful build.
- Projected object coordinates are emitted through the existing objects_gps_updated signal, including an empty list that clears stale consumers.
- Retrieval-only fallback results no longer emit anchor_fix. Their broader result-status redesign remains pending.

Implemented in the subsequent core-logic pass:

- Opt-in `localization.layer_search.enabled` search retains candidates independently from every active source. Frame IDs remain scoped by source. Round-robin geometric verification and rotating source order reduce starvation under a bounded workload.
- A homography, inlier ratio, query spread, reference non-collinearity, reprojection error, centre support, valid projection denominator, and usable calibration are required before a candidate enters handoff selection. Retrieval-only fallback now has `success=false`, `status=candidate_only`.
- Scale beliefs are source-local and updated only after downstream acceptance. Their uncertainty grows with age and projective anisotropy. Failure of the initial geometric hypotheses expands to the configured scale pyramid in the same call, subject to the remaining budget.
- Handoff uses fresh-frame confirmations, a quality margin, same-frame geographic agreement in geodesic metres, and recovery after loss. Repeated timestamps cannot increase the confirmation count. Similar-quality distant solutions are ambiguous.
- Source, calibration, trajectory filters, scale manager, and optical-flow state are restored if downstream acceptance fails or raises. Source switches reset metric-space filters rather than mix different map projections. Object projection uses the accepted source calibration.
- Pose-graph orientation is inferred from anchor affine determinants before preliminary traversal and odometry. Conflicting anchor handedness is rejected.
- Propagation records anchor/optimized/interpolated/extrapolated/unknown provenance and distance to optimized support in frame slots. New layer search excludes explicitly unknown or extrapolated slots. Legacy files without provenance retain compatibility; their support is not claimed to be known.
- Cancellation is checked between graph phases, during optimizer progress callbacks, and before entering the HDF5 write section. The GUI receives a separate cancellation signal. Cancellation during an already-started HDF5 write does **not** roll back that write; atomic persistence remains pending.
- Available schema fields are checked against runtime settings before multi-source retrieval. Known mismatches are rejected; `require_schema=true` also rejects missing fields. This does not establish checkpoint or VLAD vocabulary identity.

### Implemented scale and transform logic

The search does not estimate absolute altitude. It estimates an image ratio relative to a particular reference source. Each source keeps its own belief, so an estimate such as `1.4` for source A is never interpreted as a measured `1.4` for source B. Belief hypotheses currently form a shared search proposal pool; they do not transfer a calibrated physical scale between layers.

Let `C` map the rotated, resolution-normalized full query into the crop/resize used by the matcher. Let `H_match` map those matcher pixels into reference pixels. The full-query mapping is:

```text
H_full = H_match @ C
query centre -> H_full -> reference pixels -> reference affine -> metric coordinates -> GPS
```

`C` uses the actual integer crop dimensions and output width/height separately. This avoids applying an approximate scalar resize twice or forgetting the crop translation. The same composition is used for the accepted optical-flow mapping. A centre too far outside the query inlier hull is rejected, because a well-matched corner patch does not establish an accurate centre projection.

At the image centre, the Jacobian `J` of `H_full` gives local pixel scale. The proposal ratio is `sqrt(abs(det(J)))`. The ratio of the two singular values measures directional distortion: under tilt, different directions can have different scale, so uncertainty widens instead of interpreting the estimate as height. The source-local mean is smoothed in log space. This is a search heuristic, not a calibrated uncertainty distribution or a terrain/camera pose estimator.

The returned coordinate is labelled `ground_observation`: it is the mapped image centre. With camera tilt it is not necessarily the drone's ground position. Strong relief, parallax, missing overlap, and repeated texture can still defeat a planar homography.

### Implemented handoff logic

Selection and commitment are separate. `choose()` may propose a new source, but `commit()` runs only after the ordinary localization acceptance path succeeds. While a stronger challenger gathers confirmations, a valid current source can continue providing fixes. Similar-quality candidates that disagree geographically return ambiguity; a much stronger challenger that disagrees with a still-valid current source does not silently move the map.

After the loss timeout, reacquisition does not require the old source to match. It still requires fresh consistent observations. Geographic consistency between confirmations permits the configured maximum travel speed plus the agreement tolerance. Filter reset on a committed switch avoids mixing local metric origins; it also means smoothing restarts at that transition.

The search budget is soft: it is checked between inference/matching operations, which cannot currently be interrupted. One verification may complete beyond the deadline after a slow extraction. Diagnostics report attempted hypotheses, verifications, elapsed time, and budget exhaustion. The default 2000 ms is an evaluation setting, not a demonstrated real-time rate.

### Activation, validation, and remaining work

The new mode defaults to **off**. For an evaluation run, supply this configuration override without changing the stored user's settings:

```json
{"localization": {"layer_search": {"enabled": true}}}
```

It requires the multi-database and calibration managers. It searches the currently active sources; optional `scale_layer` metadata is stored but does not yet drive neighbor routing. Descriptor rebuilds are not required just to exercise the search with compatible databases. Re-running propagation is required to populate provenance for an older database.

Regression validation: 386 passed, 7 skipped across localization, geometry, database, configuration, project serialization, and calibration tests. Subsequent focused tests also passed for zero timestamps, rollback on exceptions, source quota/schema checks, rejection of extrapolated slots, projective tilt, and unsupported centre projection. Ruff passed on changed Python files. Pytest's Loguru shutdown handler still reports a closed capture stream after successful completion; this is a separate existing logging issue.

Still pending before claiming milestone A-F complete:

- Frozen real-flight acceptance manifest, actual ascent/descent and tilt replays, overlap envelope, false-fix rates, and hardware latency measurements.
- `process_fps` scheduling, latest-frame queue policy, asynchronous generation IDs, source-neighbor routing, and source-aware regional/patchified recovery.
- Full checkpoint/vocabulary provenance, local descriptor dimension enforcement, and confidence-unit corrections across all consumers.
- Uniform candidate/predicted/ambiguous/confirmed contracts throughout legacy and optical-flow consumers.
- Transactional calibration persistence, including cancellation/failure during HDF5 writes, and richer interpolation support policies.
- Camera-pose/terrain reasoning if the required output is the drone position under tilt rather than the mapped image centre.

These changes implement the geometric search and handoff core. They do not yet demonstrate continuous real-time localization or guaranteed overlap between the available flight layers.

## 9. Research provenance and limits

The source checks below belong to the documentation review preceding this plan. Recheck checkpoint availability and version-specific dependencies before integration.

- [HE-VPR paper](https://arxiv.org/html/2603.04050v1): height-partition retrieval followed by place retrieval, using trained adapters on a frozen backbone. The [repository README](https://github.com/hmf21/HE-VPR) reviewed for this plan stated that data and pretrained weights would be released after review, despite the paper's release claim. Usable weights are not assumed.
- [OrthoTrack](https://arxiv.org/html/2606.25245v2): useful recovery and geometric-support mechanisms; its full system relies on orthophoto/DSM map information and camera intrinsics. Those prerequisites differ from the current frame database.
- [Scale-Net](https://arxiv.org/abs/2112.10485): pairwise scale normalization before matching. It does not by itself recover a correct candidate omitted by retrieval.
- [TanDepth](https://arxiv.org/html/2409.05142v2): uses DEM and pose-related inputs, including AGL/pitch in its algorithm. It is not a sensor-free source of otherwise unknown height for the current setup.
- [PiLoT v2](https://arxiv.org/html/2606.31098v1): the reviewed version includes pose/sensor assumptions and RTX 5090 timing. The earlier notes' Jetson Orin performance statement was not verified for that version.
- The documented VLAD experiment supplies useful photometric evidence; it does not establish scale/tilt handoff performance. The experiments were not rerun while writing this plan.

No paper's benchmark score is an acceptance result for this project. The final supported operating envelope must come from the complete pipeline on representative held-out data.

## 10. Definition of done

- [ ] Audit findings are reproduced or marked already resolved against the implementation revision.
- [ ] Confirmed, predicted, candidate-only, ambiguous, and lost outputs are distinguishable.
- [ ] Multiple layers survive retrieval until geometric verification.
- [ ] Scale priors recover from failure and convert or reset correctly across layers.
- [ ] Crop/resize/rotation transforms are reversible and tested together.
- [ ] Tilted-view output semantics and planar-model limits are explicit.
- [ ] Handoff does not oscillate, deadlock, or mix source-specific state.
- [ ] Unknown coverage and map disagreement produce explicit outcomes.
- [ ] Live scheduling meets the frozen acceptance manifest on supported hardware.
- [ ] Real ascent/descent and tilted-flight evaluation establishes usable layer overlap.
- [ ] Optional methods are included only when an isolated experiment justifies them.
- [ ] A report lists achieved limits, remaining failure cases, and reproducible settings.

End of plan.
