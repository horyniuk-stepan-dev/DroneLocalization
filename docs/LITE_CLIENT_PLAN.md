# LITE_CLIENT_PLAN — lightweight localization client (old laptops → Android)

Status: **draft v1, 2026-07-18. No stage started.**
Scope decision (user, 2026-07-18): localization function only, **consume-only**
(reference databases are built on the desktop GPU machine and copied to the
device; the client never builds maps/DBs). UI for the laptop tier stays PyQt6
with a lite preset. Android is aspirational and gated by measurements.

## Why this is viable at all

The architecture is already "expensive keyframe + cheap OF between": heavy
models fire ~1/s, OpenCV LK flow + KF carry the trajectory in between, and the
fixed-lag smoother (tracking.smoother_*, 2026-07-18) holds the trajectory
honest when fixes get sparse. Lite = stretch that ratio and shrink the
per-keyframe cost. Priors already cut scans: A3 rotation prior (1 DINO forward
instead of 4), scale prior EMA, yaw_hint (phones have a compass — free yaw).

The dominant per-keyframe cost is the retrieval backbone (DINOv3/v2 ViT-L,
1024-d). Everything else is small (ALIKED ~1M, LightGlue ~12M, YOLO11n ~3M) or
classical (RANSAC, LK, KF, smoother ≈ 0.5 ms). The hard problem is the
backbone: the DB descriptor space must match the client, so a smaller backbone
means desktop-side DB rebuild and a measured accuracy trade.

## Hardware tiers and budgets

| Tier | Hardware | Keyframe fix | OF | RAM | Disk (app+models) |
|------|----------|--------------|----|-----|-------------------|
| T-A "old laptop" | x86 2C/4T, no dGPU, 8 GB | ≤ 3 s (0.33 Hz) | ≥ 25 fps | ≤ 3 GB | ≤ 1.5 GB |
| T-B "Android"    | mid Snapdragon, 6 GB      | ≤ 5–8 s | ≥ 25 fps | ≤ 2 GB | ≤ 700 MB |

DB pack per mission area is separate (global descriptors: n_frames × dim × 4 B
+ affines + optional SIFT groups — hundreds of MB worst case; float16 halves it).

## Stage 0 — Measure, don't guess (prerequisite for everything)

Force-CPU run of the CURRENT app (device selection env/config knob — add if
missing in ModelManager) on the desktop + on one real old laptop. Extend the
existing Telemetry stages to cover per-model timings: DINO forward, ALIKED,
LightGlue, YOLO, retrieval, RANSAC, per-keyframe total. Output: a table in
this doc + budget allocation per component.
Gate to proceed: none (information stage). Effort: S.

## Stage 1 — Lite runtime profile (config-first, minimal code)

`user_config.lite.json` preset over the EXISTING app:
keyframe_interval 30→60–90; smoother_enabled true (it is the thing that makes
sparse fixes tolerable); normalization target resolution down; retrieval_top_k
12→4; max keypoints 2048→1024; auto_rotation prior-only (raise
rotation_rescan_min_score tolerance); scale_pyramid narrowed to [0.7,1.0,1.4];
depth hint OFF; patchify OFF; VLAD OFF; sift_fallback OFF; debug views OFF.
Code deltas (small, flag-gated): make YOLO masking optional (tracking worker
currently hard-fails without YOLO); keypoint cap from config if not yet.
Gate: mission benchmark (Windows) accuracy degradation ≤ 25% vs baseline on
2 scenes; PDM@K once research-plan 3.2 lands (shared dependency). Effort: S–M.

## Stage 2 — CPU inference backend (ONNX Runtime)

Query-side models exported to ONNX, ORT CPU (AVX2/XNNPACK) as a first-class
ModelManager backend: per-model choice torch-cuda | torch-cpu | ort-cpu
(reuses the engines_cache pattern). Known risk: LightGlue export (adaptive
depth/pruning) — use the established lightglue-onnx recipe or fixed-depth
export. int8 dynamic quantization where parity holds.
Parity tests (partially sandbox-runnable with onnxruntime): descriptor cosine
> 0.99 vs torch; matcher recall on golden pairs; end metric on benchmark.
Gate: keyframe ≤ 3 s on the T-A reference machine. Effort: M.

## Stage 3 — Retrieval backbone diet (the decisive stage)

Consume-only forces client/DB descriptor-space pairing → add `backbone` +
`descriptor_dim` to DB metadata with a hard compatibility check at load
(DB format version bump; sync rule with FlightSimulator per repo convention).
Candidates, each = desktop DB rebuild + benchmark: DINOv2-S/14 (21M, 384-d),
DINOv2-B/14 (86M, 768-d), int8-quantized ViT-L (control, no rebuild).
Local features: keep ALIKED (small) vs XFeat (CPU-first design) — measure;
SIFT+FLANN as a zero-NN emergency path (rootSIFT plumbing already exists).
Gate: accuracy within budget on BOTH Flightmare footage and the
satellite-degraded branch (research-plan 3.2 — without it lite numbers will be
systematically optimistic). Effort: M–L.

## Stage 4 — Lite packaging (T-A deliverable)

PyInstaller profile on the existing BUILD.md/build_executable.py stream:
CPU-only (ideally torch-free if stage 2 covers all models → exe shrinks from
GBs to ~300–500 MB); model pack "lite" (~100–200 MB); PyQt6 GUI with the lite
preset as default config; DB packs copied per mission.
Gate: cold start ≤ 30 s, RAM ≤ 3 GB, live run on an actual old laptop. Effort: S–M.

## Stage 5 — Android (design-only until the entry gate passes)

Entry gate: ORT benchmark on a real phone (trivial benchmark APK or termux)
extrapolated from stage 2 numbers → keyframe ≤ 5–8 s, RAM ≤ 2 GB. If red:
Android remains a ground-station display (laptop computes, phone shows map) —
zero extra engineering.
If green, the port scope is deliberately tiny (consume-only client):
- Inference: ORT Mobile (XNNPACK/NNAPI) or NCNN, same quantized models as stage 2/3.
- Classical: OpenCV Android (LK flow, RANSAC homography).
- State: KF + OutlierDetector + smoother ported (few hundred lines, no deps).
- Retrieval: flat float16 matrix + brute dot product — no FAISS (thousands of
  frames per source do not need an index).
- Sensors: yaw_hint from compass (kills rotation scan), barometer as scale prior hint.
- UI: MapLibre + marker/FOV polygon. No calibration, no builder, no debug views.
Effort: L (only after gate).

## Cross-cutting rules

- Every stage flag-gated; pydantic defaults = current behavior (repo rule).
- Accuracy harness: mission benchmark (Windows) + PDM@K; a lite change without
  a measured accuracy number does not merge.
- DB metadata/versioning changes follow the simulator pairing rule (version
  bump both sides).
- Related plans: PERFORMANCE_ACCURACY_PLAN A4 (TensorRT default — desktop
  path, orthogonal), research 3.2 (PDM@K — shared metric dependency),
  DB_INTERCHANGEABILITY §4 (batched extraction headroom — desktop build side).
