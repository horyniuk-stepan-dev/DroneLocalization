# Better Approaches — DroneLocalization (2026-07-28)

**Constraints:** no model retraining, no drone telemetry. Everything else is on
the table.

Companion to `FRESH_AUDIT_2026-07-28.md`. That document listed defects; this one
proposes different ways to solve the same tasks. Every line reference was checked
against the working tree in this session.

---

## What the constraints remove

From the previous report, **§4.5 (yaw hint from telemetry/IMU) is dead** — it
required a heading source the constraint forbids. The rest survives, and §4.3
survives in weakened form: `project_manager.settings.altitude_m` is an
operator-entered mission parameter, not telemetry, so seeding scale from GSD is
still allowed — but it is a nominal altitude, not a measured one, so it can only
seed, never override.

Two clarifications on what "no retraining" does *not* exclude:

- **k-means over frozen DINOv3 features** (the VLAD vocabulary) is fitting a
  codebook, not training a model. No gradients, no weight updates. Allowed.
- **PCA / whitening fitted on the database's own descriptors** is likewise a
  linear statistic of frozen features. Allowed.

Both matter, because they are the two highest-leverage items below.

---

## Verdict

The three highest-value moves need no new algorithm at all:

1. **VLAD aggregation is fully implemented in this repo and switched off.**
   `models.vlad.enabled: false`, `vocab_path: null`. The builder script exists
   (`scripts/build_vlad_vocab.py`). It directly targets the farmland/forest
   aliasing that the CLS token handles badly.
2. **`GeoAwareRetriever` + `SpatialIndex` are implemented and unreachable in
   single-database mode** — the default. Geometric candidate filtering is built
   and unused.
3. **The exact query→database rotation is computed every keyframe and thrown
   away** (`scale_manager.py:269`, `_angle`). It is the missing half of a
   continuous rotation prior.

Beyond those, the biggest structural lever is **moving rotation and scale
handling from online to offline** — precompute database descriptors at every
rotation instead of rotating the query four times at runtime.

---

## A. Already in the repository, off or unreachable

### A.1 VLAD over DINOv3 patch tokens (AnyLoc)

`src/models/wrappers/vlad_aggregator.py` (240 lines) implements hard-assignment
VLAD with intra-normalization and PCA-whitening; `scripts/build_vlad_vocab.py`
fits the codebook by k-means over patch tokens from the reference video.
`FeatureExtractor` already routes to it (`feature_extractor.py:33-50, 128-140`)
when `models.vlad.enabled` is true and a vocabulary exists. Neither is set.

**Why it matters here.** The CLS token is a single global summary; on farmland,
forest canopy and water, adjacent fields produce near-identical CLS vectors —
which is the aliasing that forces `retrieval_top_k: 12`,
`candidate_prefilter`, and the whole recovery cascade. VLAD is an orderless
aggregation of *local* cluster residuals, so it discriminates on texture
composition rather than global gist. AnyLoc (arXiv:2308.00688) reports exactly
this on aerial and unstructured domains.

**Honest limit on rotation.** VLAD is invariant to the spatial *arrangement* of
patches, not to the rotation of each patch's content — DINOv3 patch tokens are
themselves rotation-variant. Expect partial, not full, rotation robustness.
Do not budget for dropping the rotation scan because of VLAD alone.

**Cost:** one k-means fit over the reference video (offline, GPU, minutes) plus a
full database rebuild, because the descriptor dimension changes
(`build_vlad_vocab.py` docstring says so explicitly).

**How to validate:** `tests/test_retrieval_metrics.py` already exists. Measure
Recall@1 and Recall@5 of the current CLS database versus a VLAD database on the
same recorded mission. If R@1 does not improve on the homogeneous-terrain
segments, revert — the rebuild cost is not worth a tie.

### A.2 `GeoAwareRetriever` is unreachable in single-database mode

`Localizer.__init__` (`localizer.py:102-108`) selects `LanceDBRetrieval` or
`FastRetrieval` and never `GeoAwareRetriever`. The geo-aware path is only
constructed by `MultiDatabaseManager` (`multi_database_manager.py:83-89`).

Yet `DatabaseLoader` already builds a `SpatialIndex` from `frame_gps` in single
mode (`database_loader.py:289-303`). The data and the index are both there; only
the wiring is missing.

**What this buys.** After propagation, every database frame has a world position.
Given a last accepted fix and a bounded speed, the set of database frames that
could plausibly match is a disc, not the whole map. Restricting the FAISS search
to that disc:

- removes far-away aliases *before* any descriptor comparison — the failure mode
  §1.1 of the audit can turn into a wrong fix;
- makes `retrieval_top_k` meaningful (12 candidates from the neighbourhood beats
  12 from the whole flight);
- generalises `temporal_candidate_prior`, which currently only considers
  database indices `id ± 2` (`localizer.py:1007-1033`). Index adjacency is a
  proxy for spatial adjacency that breaks on repeat passes, at route crossings,
  and when the mission flies the reference route in reverse. Spatial adjacency
  does not.

**Cost:** wire `GeoAwareRetriever` into the single-database branch and call
`update_position` from the existing hook in `tracking_worker.py:499-502`. No
rebuild. Roughly half a day.

**When the current approach still wins:** at bootstrap and after
out-of-coverage, when there is no position prior. `GeoAwareRetriever` already
degrades to a full index in that case (`geo_aware_retriever.py:64-68`) — the
behaviour is correct, it just needs to be reachable.

### A.3 The rotation angle is measured and discarded

`scale_manager.py:254-296` does:

```python
M = homography_to_affine(H, frame_w, frame_h)
_tx, _ty, sx, sy, _angle = decompose_affine_5dof(M)
r_measured = sqrt(|sx| · |sy|)          # ← scale kept
                                         # ← _angle dropped
```

`_angle` is the exact in-plane rotation between the query frame and the matched
database frame, on every successful keyframe. The system currently carries only
`_last_best_angle` — a bucket from {0, 90, 180, 270} (`localizer.py:165, 802`).

**Proposal: a `RotationManager`, structurally identical to `ScaleManager`.**

- EMA prior over the continuous angle, updated from `_angle` each keyframe.
- Propagated between keyframes by the rotation component of `flow_affine`,
  which the tracking worker already computes with
  `cv2.estimateAffinePartial2D` (`tracking_worker.py:424-431`) and currently uses
  only for translating the frame centre — `atan2(S[1,0], S[0,0])` is free.
- De-rotate the query by the **continuous** angle before descriptor and local
  feature extraction, instead of snapping to a 90° bucket.

**Why this is more than an optimization.** Today, if the mission flies a heading
45° off the reference flight, *no* rotation in {0, 90, 180, 270} aligns the
frames. Both DINOv3 retrieval and ALIKED matching degrade at once, and the
system falls back on `sift_fallback` — which exists in the code for precisely
this reason (`localizer.py:1148-1152` names in-plane rotation as the cause).
Continuous de-rotation removes the constraint that the mission heading must
approximately match the reference flight's heading.

**Drift control:** the OF-integrated angle is re-anchored to the measured
`_angle` at every accepted keyframe, so drift is bounded by one keyframe
interval. On failure, fall back to the existing 4-angle scan — the recovery path
already exists and needs no change.

**Cost:** no rebuild. ~1 day, plus the `rot90` fast path becomes a general
`cv2.warpAffine` (slightly more expensive per frame, far cheaper than a wrong
rotation).

---

## B. Rebuild-free improvements

### B.1 Guided matching — warp the query before matching, not after

Currently ALIKED and LightGlue see a query that may differ from the database
frame by rotation, scale and perspective, and LightGlue is asked to solve all of
it. But by the time a candidate is chosen, a *predicted* homography is available:
the last accepted `_last_state["H"]` composed with the OF similarity since then.

Pre-warping the query into the candidate's geometry with one
`cv2.warpPerspective`, matching in that space, and composing the result back is
standard guided matching. It puts LightGlue in its best regime (near-aligned
pairs), which raises the inlier count and *lowers* the false-match rate — the
input to every downstream gate.

**When the current approach still wins:** when the prediction is stale
(long OF coast, after a failure). Gate the warp on prediction age and on
`flow_quality`, which the worker already computes
(`tracking_worker.py:432-435`).

### B.2 Sequence consistency over the existing top-k lists

Full SeqSLAM needs a complete similarity row per query, which `retrieval_top_k:
12` does not give. But a cheap variant does work with what is already returned:
keep the top-k lists of the last N keyframes, and re-score each candidate by
whether it is consistent with a constant-velocity trajectory through the previous
lists (`id ≈ id_prev + v·Δt`).

On repetitive terrain, any single frame is ambiguous while a sequence of five is
close to unique. This is the classic answer to perceptual aliasing and it costs
a few hundred integer operations per keyframe. It also composes with A.2 — one
filters by geometry, the other by temporal consistency.

**When the current approach still wins:** hover and sharp manoeuvres break the
constant-velocity assumption. Weight the sequence term by measured OF speed and
disable it below a threshold.

### B.3 Multi-candidate agreement as a verification signal

`GeometricVerifier.verify` (`geometric_verifier.py:81-128`) keeps only the single
best candidate by inlier count. If the top 2–3 candidates each produce a valid
homography, their *metric* predictions can be compared: genuine matches agree to
within a few metres; a wrong-frame match disagrees grossly.

This is a direct antidote to the §1.1 failure in the audit, where an inflated
inlier count lets a wrong candidate win — an inflated count cannot fake
agreement with an independent candidate. The database already carries
`frame_disagreement` (`database_loader.py:186`) and the confidence config already
has `disagreement_norm_m: 5.0`, so the concept exists in the codebase; it is just
not applied at localization time.

**Cost:** the extra candidates are already matched when `early_stop` does not
fire; the added work is one affine application each. Note it does interact with
`early_stop_inliers: 40` — agreement checking requires *not* stopping at the
first good candidate, so make it a flag and measure the latency cost.

### B.4 Robust loss in the pose graph (carried over, still valid)

`optimizer.py:528` calls `least_squares` with no `loss=` argument — verified by
grep across `src/`, the string does not appear. Pure L2. `loss="soft_l1"` or
`"huber"` with a tuned `f_scale` is a one-parameter change inside the same TRF
solver, and it does natively what `gnc_spatial` + `gnc_rounds` + `gnc_mad_k`
approximate with an outer loop.

Requires re-running propagation, not a database rebuild.

### B.5 Uncertainty instead of a confidence scalar (carried over)

`noise_scale = 1.0 / max(confidence, 0.25)` (`localizer.py:758`) is a heuristic
standing in for a covariance that the inlier residuals already determine. This
becomes more valuable if D.1 below is adopted, since a particle filter consumes
a likelihood, not a score.

---

## C. Requires one database rebuild — batch these together

Several changes alter what a descriptor *means*, so query and database
descriptors stop being comparable unless both are regenerated. **Do them in one
rebuild, not serially.** Each rebuild on a real mission is expensive; three
sequential rebuilds for three independent experiments is the avoidable cost here.

| Change | Why a rebuild |
|---|---|
| C.1 Multi-rotation descriptors | new column in the vector table |
| C.2 GeM pooling instead of CLS | descriptor semantics change |
| C.3 PCA-whitening on database statistics | descriptor space changes |
| A.1 VLAD | descriptor dimension changes |
| Audit §2.1 (CPU resize before upload) | resampling filter changes descriptor values |

Note C.2, C.3 and A.1 are partly alternatives to each other — VLAD already
includes PCA-whitening. The sensible experiment is **two** rebuilt databases
(CLS+GeM+whitening, and VLAD), compared on the same mission, not five.

### C.1 Precompute database descriptors at every rotation

This is the structural change with the best cost profile.

Today the query is rotated up to 4 times (and up to 5 scales, so up to 20
forward passes) and DINOv3 runs on each. Rotation is symmetric: a query at 0°
against a database frame rotated +90° is the same comparison as a query rotated
−90° against the database frame at 0°. So the rotations can be computed **once,
offline, during the database build**, and the online path becomes a single
forward pass with the retrieval returning `(frame_id, rotation)` directly.

Storage, at `descriptor_dim: 1024` float32 (4 KiB per vector):

| database | vectors | size |
|---|---:|---:|
| 30 min flight, now (1/frame) | 1 800 | 7.0 MiB |
| 30 min, 4 rotations | 7 200 | 28.1 MiB |
| 30 min, 4 rot × 5 scales | 36 000 | 140.6 MiB |
| 60 min, 4 rot × 5 scales | 72 000 | 281.2 MiB |

Search cost is negligible: `IndexFlatIP` over 36 000 × 1024 vectors is ~74 MFLOP
per query — sub-millisecond with BLAS.

Schema: `database_builder.py:683-689` defines
`pa.schema([frame_id: int32, vector: list<float32>[dim]])`. Adding a `rot` column
and allowing repeated `frame_id` values is the whole change; `LanceDBRetrieval`
(`matcher.py:113-131`) needs `.select(["frame_id", "rot", "_distance"])`.
`schema_fingerprint.py` already exists to version this.

**Build cost is the real question.** The build does 4× (or 20×) more DINOv3
forwards. Using the figure in the codebase's own comment
(`localizer.py:122-124` — 470 ms per global descriptor on a GTX 1650), a
1 800-frame build goes from ~14 min to ~56 min at 4 rotations. That is
tolerable, and it is a strong second reason to fix audit §2.1 first: if the
470 ms is mostly PCIe and CPU-float overhead — which the arithmetic suggests,
since ViT-L/16 at 224 is ~61 GFLOPs and a GTX 1650 does ~2.9 TFLOPS fp32 — then
fixing the upload path makes the multi-rotation build cheap rather than merely
tolerable.

**Combine with A.3, do not duplicate it.** With a continuous rotation prior the
steady state already costs one forward pass. C.1's value is concentrated in
**bootstrap and recovery**, where there is no prior — exactly the cases where
the system currently pays 4–20 forwards. The two changes are complementary:
A.3 makes the steady state cheap, C.1 makes recovery cheap.

### C.2 GeM pooling instead of the CLS token

`VFM-Loc` (arXiv:2603.13855, training-free, frozen VFM) reports that vanilla
DINOv3 features give 21.56 % R@1 on their cross-view benchmark, and that
hierarchical pooling plus statistical alignment raises it to 73.30 % — with no
training anywhere.

Their pooling half is Generalized Mean over patch tokens:
`d = (1/N · Σ xᵢᵖ)^(1/p)`, `p ≈ 3`. It interpolates between average pooling
(p=1) and max pooling (p→∞) and consistently beats the CLS token for retrieval.
`forward_features` already returns `x_norm_patchtokens` and the code already
consumes them on the CESP and VLAD paths (`feature_extractor.py:157-165`), so
this is a handful of lines.

**Calibration:** VFM-Loc's gain is measured drone→satellite, where the domain gap
is enormous. Here both sides are drone imagery from the same camera, so the gap
is small and **the gain will be far smaller**. I would not predict a number. It
is worth including in the batched rebuild because it is nearly free to
implement, not because 50 points are on offer.

### C.3 PCA-whitening on the database's own descriptors

The other half of VFM-Loc. Fitting PCA on the database descriptors and whitening
both sides decorrelates dimensions and equalises variance, which usually helps
cosine retrieval on features that were never trained for it. Fit once at build
time, store alongside the vectors.

Redundant if VLAD is enabled — `VladAggregator` already applies PCA-whitening
internally (`vlad_aggregator.py:47-51`).

---

## D. Architectural alternatives

These change the shape of the system, not a parameter. Listed because the
question was "better ways", not "cheaper ways" — but they are not first moves.

### D.1 Multi-hypothesis filter instead of single-best + unimodal Kalman

Current chain: retrieval → pick the single best candidate → hard decision →
Kalman (unimodal Gaussian) → smoother.

**A unimodal filter structurally cannot represent "I am at one of these three
identical field corners."** It must commit, and if it commits wrongly the
smoother spends its window dragging the trajectory back. That is the exact
situation on repetitive terrain.

LSVL (Kinnari et al., *Robotics and Autonomous Systems* 2023, arXiv:2212.03581)
solves the closely related problem — UAV localization against a georeferenced map
with no GNSS — with a particle filter over position **and heading**, and reports
convergence from an uninformed start in 23–44 updates. A particle filter here:

- carries competing hypotheses until the sequence disambiguates them (subsuming
  B.2 as a natural consequence rather than a bolted-on heuristic);
- **estimates heading without telemetry**, which is what A.3 is reaching for;
- degrades gracefully out of coverage instead of tripping a
  `max_consecutive_failures` counter;
- consumes a likelihood per candidate, which is what B.5 produces.

**Cost:** significant. It replaces `TrajectoryFilter`, interacts with
`SlidingWindowSmoother`, and needs its own tests. Do not start here. But if
A.2 + A.3 + B.2 all end up implemented, they are approximating a particle filter
with three separate heuristics, and at that point the honest move is to build the
filter.

### D.2 Match against a local orthomosaic instead of a single frame

After propagation, the database is a globally consistent set of georeferenced
affines — i.e. an implicit orthomosaic. `panorama_worker.py` and
`panorama_overlay_worker.py` already compose imagery from it.

Matching the query against a rendered mosaic patch around the predicted position
removes the "which single database frame" question, gives more overlap than any
single frame (important when the mission flies between two reference passes),
and makes the position a direct map lookup.

**When the current approach still wins:** the mosaic inherits every propagation
error, and rendering it costs memory and time. Single-frame matching keeps errors
local. This is only worth it if propagation quality is already validated against
ground truth.

---

## Suggested order

Rebuild-free first, so the rebuild is done once with the winners already known.

**Phase 1 — no rebuild, independently testable**

| # | Item | § | Effort |
|---|---|---|---|
| 1 | Audit §2.1 uint8 upload (prerequisite for everything offline) | — | ~1 h |
| 2 | Wire `GeoAwareRetriever` into single-database mode | A.2 | ~0.5 d |
| 3 | `RotationManager` — continuous angle from `_angle` + OF | A.3 | ~1 d |
| 4 | Sequence consistency over top-k lists | B.2 | ~0.5 d |
| 5 | Multi-candidate agreement check (flag-gated) | B.3 | ~0.5 d |
| 6 | `loss="soft_l1"` A/B in the optimizer | B.4 | ~1 d |
| 7 | Guided matching (pre-warp on prediction) | B.1 | ~1 d |

**Phase 2 — one batched rebuild, two candidate databases**

- Database X: CLS + GeM + PCA-whitening + multi-rotation vectors.
- Database Y: VLAD + multi-rotation vectors.
- Compare both against the current database on the same recorded mission.

**Phase 3 — only if Phase 1 items 2, 3 and 4 all pay off**

- Particle filter (D.1), which subsumes them into one principled mechanism.

---

## How to know whether any of this worked

Each item needs a number, not an impression. The repository already has the
instruments: `tests/test_retrieval_metrics.py`, `scripts/validate_vs_telemetry.py`,
`benchmarks/`, and the `FlightSimulator` ground-truth pairs.

| item | metric | pass condition |
|---|---|---|
| A.1, C.1–C.3 | Recall@1 / @5 on the recorded mission | R@1 improves on homogeneous-terrain segments |
| A.2 | candidates examined per keyframe; wrong-frame rate | fewer candidates, no recall loss |
| A.3 | keyframe success rate at off-heading | improves where heading ≠ reference heading |
| B.2 | wrong-frame matches per mission | drops on repetitive terrain |
| B.3 | wrong fixes surviving all gates | drops; latency cost measured |
| B.4 | anchor RMSE after propagation | ≤ current, with fewer pruned edges |
| C.1 | forwards per keyframe at bootstrap | 4–20 → 1 |

For anything touching accuracy, the comparison must be against
`ground_truth.json` from a v2.3 `FlightSimulator` recording — not against the
previous run of the same pipeline, which shares its biases.

---

## Risks and what I could not check

- **The 470 ms global-descriptor figure is quoted from a code comment**
  (`localizer.py:122-124`), not measured by me. Every timing estimate in section
  C.1 inherits that uncertainty. Measure before committing to the build-time
  budget.
- **No numeric prediction is offered for A.1, C.2 or C.3.** The literature
  numbers are from cross-view benchmarks (drone→satellite) where the domain gap
  dominates. Here both sides are drone imagery from the same camera, so the
  published gains do not transfer and I decline to estimate a substitute. These
  are worth *trying* because they are cheap, not because a specific gain is
  expected.
- **A.3's drift bound is an argument, not a measurement.** OF rotation
  integration over 29 frames at `keyframe_interval: 30` accumulates error I have
  not quantified. If measured Δθ drift exceeds ~10° between keyframes, the
  continuous prior is worse than the 90° bucket and the item should be dropped.
  This is the single assumption most likely to break the proposal.
- **C.1 assumes descriptor comparison is symmetric under rotation** — that
  `desc(rot(db, +90))` matched against `desc(query)` equals
  `desc(db)` matched against `desc(rot(query, −90))`. This holds for the
  comparison, but resampling differs slightly between rotating a database frame
  and rotating a query frame. For k·90 rotations `np.rot90` is exact (no
  interpolation), so the assumption is safe **only for 90° multiples** — it does
  not extend to a continuous-angle precomputation.
- **The strongest objection to this whole list:** A.2, A.3, B.2 and B.3 are four
  separate heuristics that each approximate part of what D.1 does properly. If
  all four get built, the system will have four interacting tuning surfaces
  instead of one filter. That is a real risk of incremental improvement, and it
  is why D.1 is named rather than omitted.

---

## Sources

- AnyLoc: Towards Universal Visual Place Recognition — arXiv:2308.00688 (cited in
  `vlad_aggregator.py:1`)
- VFM-Loc: Training-Free Cross-View Geo-Localization via Aligning Discriminative
  Visual Hierarchies — https://arxiv.org/html/2603.13855v2
- LSVL: Large-scale season-invariant visual localization for UAVs —
  https://arxiv.org/abs/2212.03581
- SeqSLAM (Milford & Wyeth) and Fast-SeqSLAM — sequence-based VPR against
  perceptual aliasing
- OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata —
  https://arxiv.org/html/2509.18350v2
