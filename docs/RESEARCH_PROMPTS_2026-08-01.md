# Research prompts for an external frontier model — DroneLocalization

Date: 2026-08-01. Source of the problem statement: `README_Eng.md`, `docs/architecture.md`,
`docs/RELATED_WORK_MAP.md`, `docs/SCALE_INVARIANCE.md`, `BETTER_APPROACHES_2026-07-28.md`,
`docs/REMAINING_WORK_PLAN.md`.

The prompts are self-contained: the external model has no access to this repository, so each
one restates the constraints it needs. Written in English on purpose (repo convention);
the answer can be requested in any language.

---

## Part 1 — What problems the system actually solves

**Primary problem.** Absolute geo-localization of a UAV from a monocular video stream in
GNSS-denied conditions, with **no telemetry at all** (no IMU, no barometer, no compass,
no measured altitude) and **no model retraining**.

Sub-problems, in the order the pipeline hits them:

1. **Reference map without public geodata.** The map is the operator's own earlier
   fly-over video, turned into a keyframe database (local features in HDF5, global
   descriptors in LanceDB) — not satellite tiles, not a DSM.
2. **Geo-referencing that map from sparse anchors.** A handful of manually placed GPS
   anchors are propagated across the whole database via a 5-DoF pose-graph
   (anisotropic affine, Levenberg–Marquardt). Anchor sparsity, anchor error and
   drift accumulation between anchors are the core error budget.
3. **Appearance change.** Query flight and map flight differ in season, weather, sun
   angle, shadows. Handled by frozen foundation-model descriptors (DINOv2/v3) +
   CLAHE, rather than by trained place-recognition heads.
4. **Dynamic content.** Cars/people must not become anchor geometry → YOLOv11-seg masking.
5. **Scale / altitude mismatch.** DB built at ~100 m, query flown at 50–200 m. Geometry
   survives (homography absorbs scale); *retrieval* and *matching* break at ratio
   r ≳ 1.3–1.5 and r ≳ 1.5–2 respectively. Estimating r without telemetry is an open item.
6. **Rotation / heading ambiguity.** No compass → yaw must come from the imagery itself.
7. **Temporal continuity.** Sparse absolute fixes on keyframes + optical-flow propagation
   between them; Kalman filter + Huber fixed-lag smoother, single hypothesis (no particle
   filter / factor graph).
8. **Failure detection.** Distinguishing "out of coverage" from "confidently wrong" —
   a wrong-but-confident fix is worse than no fix.
9. **Compute budget.** Real-time-ish on a 4 GB GTX 1650 desktop, plus headless mode.

---

## Prompt 1 — Telemetry-free scale/altitude estimation

> I am building a monocular UAV absolute-visual-localization system. Reference map = a
> database of keyframes from a previous fly-over of the same area at ~100 m AGL (local
> features + frozen-DINO global descriptors). Query flights happen at 50–200 m AGL.
> Hard constraints: **no telemetry of any kind** (no IMU, altimeter, barometer, compass),
> **no model retraining or fine-tuning** (frozen off-the-shelf checkpoints only), single
> 4 GB consumer GPU, near-real-time.
>
> Diagnosis I have already made: coordinate geometry does not break with scale ratio r
> (the homography absorbs it); what breaks is (a) global-descriptor retrieval at r ≳ 1.3–1.5
> and (b) local matching (ALIKED/RDD + LightGlue) at r ≳ 1.5–2.
>
> Task: survey and critically compare every practical way to estimate r and/or to make
> those two stages scale-robust *under these constraints*. Cover at minimum: scale
> pyramids over the query, temporal scale priors from the previous frame's decomposed
> affine, monocular metric-depth models (UniDepth v2, MoGe v2, Depth-Anything-V2) as
> hints, pairwise scale-ratio regressors (Scale-Net / SDAIM), scale-covariant classical
> detectors as a fallback, and multi-scale database indexing.
>
> For each option give: expected accuracy of r, compute cost per frame on a 4 GB GPU,
> failure modes, and whether published evidence supports it for **nadir aerial** imagery
> specifically (domain shift of depth models on nadir views is a known problem — verify,
> don't assume). Cite papers with arXiv IDs/venues and state explicitly where a number
> comes from an abstract vs a read full text.
>
> Deliverable: a ranked recommendation of at most 3 options with the decision rule for
> choosing between them at runtime, plus a concrete experiment that would falsify the
> top-ranked one.

## Prompt 2 — Geo-referencing a self-recorded map from sparse GPS anchors

> Setting: a keyframe database built from a UAV fly-over video. It is geo-referenced by a
> small number (2–10) of manually placed GPS anchors, propagated to all other frames by
> optimizing a pose graph of frame-to-frame 5-DoF anisotropic affine transforms
> (Levenberg–Marquardt). No GCPs, no photogrammetric bundle adjustment with a DSM, no
> telemetry.
>
> Questions:
> 1. What does the literature say about the *error model* of this construction — how does
>    residual geo-error grow with distance from the nearest anchor, and what governs it
>    (drift of relative estimates, anchor GPS noise, terrain non-planarity)?
> 2. Optimal anchor placement: given a budget of N anchors on a known flight path, where
>    should they go? Is there a principled criterion (D-optimality / covariance-based)
>    rather than "spread them out"?
> 3. Robust estimation: which loss/outlier scheme is best when one anchor is simply wrong
>    (operator misclick)? Compare switchable constraints, dynamic covariance scaling,
>    max-mixtures, graduated non-convexity for this specific graph topology.
> 4. Is a planar 5-DoF affine the right relative-motion model, or does terrain relief force
>    a homography / partial 3D model, and at what terrain-relief-to-altitude ratio does the
>    planar assumption cost more than the added complexity?
> 5. Uncertainty propagation: how do I get a per-frame covariance out of this graph that is
>    calibrated enough to feed a downstream Kalman filter?
>
> Prefer robust-SLAM and geo-referencing literature with citations; flag where you are
> extrapolating from a related problem rather than reporting a result on this exact setup.

## Prompt 3 — Frozen-VFM retrieval under seasonal and viewpoint change

> Constraint: no training and no fine-tuning; only frozen checkpoints. I currently do
> aerial place recognition by taking DINOv2/DINOv3 CLS (and optionally patch-token) features
> of nadir keyframes and doing nearest-neighbour search in LanceDB.
>
> Survey what can still be done *without gradients* to improve recall@1 in cross-season /
> cross-illumination conditions: patch-token pooling schemes (GeM, VLAD over frozen
> features with a k-means codebook fitted on my own database, SALAD-style assignment
> without its trained head), PCA-whitening fitted on my own descriptors, multi-crop and
> multi-scale aggregation, query expansion / alpha-QE, diffusion-based re-ranking, and
> geometric re-ranking of top-K.
>
> Note: fitting a k-means codebook or PCA on my own frozen features is allowed under my
> constraints (no gradients, no weight updates). Anything requiring backprop is not.
>
> For each technique give expected recall gain with a source, memory cost per frame,
> and query latency. Also tell me which of these are known to *hurt* on nadir aerial
> imagery versus ground-level VPR benchmarks — that distinction matters more to me than
> average-case numbers. End with the single highest-leverage change and the ablation
> that would prove it.

## Prompt 4 — Failure detection and integrity monitoring

> A monocular UAV localization system produces an absolute geo-fix per keyframe by
> matching against a reference database, plus optical-flow propagation in between, fused
> with a Kalman filter and a fixed-lag smoother. Single hypothesis, no particle filter,
> no telemetry to cross-check against.
>
> The dangerous failure is not "no fix" but "confident wrong fix" — e.g. repetitive
> farmland where the matcher latches onto the wrong field.
>
> Survey the state of the art on **integrity monitoring / failure prediction** for visual
> localization: learned and unlearned pose-confidence estimators, inlier-ratio and
> match-distribution statistics as predictors, geometric consistency checks (cycle
> consistency, cross-check with a second matcher, temporal consistency against the motion
> model), RAIM-style ideas imported from GNSS, and conformal-prediction / calibrated
> uncertainty approaches.
>
> Constraints: frozen models only, single 4 GB GPU, must run per keyframe.
>
> Deliverables: (1) the 3–5 cheapest statistics with the best documented separation between
> correct and incorrect fixes; (2) how to combine them into a single accept/reject decision
> with a tunable false-accept rate; (3) how to *evaluate* such a detector properly given
> that ground truth is scarce; (4) whether adopting a multi-hypothesis filter would
> dominate all of the above, and at what compute cost.

## Prompt 5 — Positioning against the literature (for a paper/thesis)

> Characterize where the following system sits in the Absolute Visual Localization (AVL)
> literature for GNSS-denied UAVs, and whether it constitutes a publishable contribution.
>
> System axes: reference map = the operator's own prior fly-over video (not satellite
> tiles / ortho / DSM); geo-referencing of that map = a few manual GPS anchors propagated
> by a 5-DoF pose-graph; input = video, keyframes + optical flow; retrieval = frozen VFM
> descriptors; matching = sparse (ALIKED/RDD + LightGlue); geometry = homography → 5-DoF
> anisotropic affine → GPS; fusion = KF + Huber fixed-lag smoother, single hypothesis;
> sensors = camera only; deployment = desktop GPU, GUI + headless. Training-free throughout.
>
> Tasks:
> 1. Find the closest prior systems, especially any that also use a *self-recorded* map or
>    sparse-anchor geo-referencing rather than public geodata. Be explicit when you find
>    none — absence of a neighbour is the interesting result here.
> 2. Identify the strongest baselines I would be expected to compare against, and which
>    public datasets (with 6-DoF or GPS ground truth over repeated aerial passes) could
>    host such a comparison.
> 3. State the honest novelty claim in one sentence, and then the strongest reviewer
>    objection to it, and what experiment would neutralize that objection.
> 4. List the evaluation metrics and protocol such a paper is expected to report.
>
> Distinguish clearly between claims you verified in a full text, claims from abstracts,
> and your own inference. Give arXiv IDs / DOIs.

## Prompt 6 — Compute budget on a 4 GB GPU

> Pipeline per keyframe on a GTX 1650 (4 GB, fp16 is ~4× *slower* than fp32 on this chip —
> Turing without full tensor-core fp16 throughput for these ops): YOLOv11-seg dynamic-object
> masking → ALIKED (or RDD) local features → DINOv2/v3 global descriptor → LightGlue matching
> against top-K candidates → homography → affine → geo. Optical flow between keyframes.
> All models frozen; VRAM is managed by a load/evict budget manager.
>
> Survey what actually buys latency on this class of hardware without retraining:
> ONNX Runtime vs TensorRT engine conversion (and which of these models convert cleanly),
> INT8 post-training quantization and its accuracy cost for descriptor quality specifically,
> resolution/token-count reduction for ViT backbones, batching strategies across rotation
> and scale variants, candidate prefiltering before expensive matching, and CPU/GPU overlap.
>
> For each: expected speedup with a source, accuracy risk, and implementation effort.
> Explicitly warn me where a published speedup was measured on Ampere/Ada and would not
> transfer to Turing. Rank by (speedup × probability it transfers) ÷ effort.
