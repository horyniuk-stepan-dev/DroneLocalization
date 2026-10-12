# Relative frame registration — implementation and validation

Branch: `codex/relative-frame-accuracy`.

This change addresses image-to-image registration. Geographic anchor fitting is
unchanged. The measured improvement is in overlap alignment; this is not a claim
that a planar image map recovers the true 3D flight trajectory over terrain.

## Implementation

- Match up to four preceding occupied keyframes, retaining every verified edge.
  Nearby spatial candidates no longer have an arbitrary temporal exclusion gap,
  and existing temporal edges are not inserted twice as spatial edges.
- Run offline LightGlue without adaptive depth exit or point pruning, without
  changing the shared live model's configuration. Increase robust estimation to
  at least 10,000 iterations and 0.999 confidence.
- Check both image footprints, forward and inverse projection errors, inlier
  ratio, convex-hull coverage and principal-axis spread. A diagonal line now has
  zero two-dimensional support in the accuracy profile. Live confidence retains
  the previous axis-based spread metric. Refit after MAD filtering.
- Refine correspondences with initialized LK and a forward/backward check when
  the original video's SHA-256, dimensions and frame-slot identity are available.
  Failed local photometric updates and source-frame read failures retain the
  learned observations; cancellation still propagates. Missing or
  changed videos disable this step with an explicit diagnostic.
- Initialize LOCAL maps by composing full temporal pair transforms. Spatial
  loops cannot seed the map and must agree with the independent seed. Optimize shared image
  observations using symmetric transfer errors in original image pixels, sparse
  least squares and robust point weights. Fix the gauge; do not invent positions
  for disconnected footage. Geographic mode refines around the existing graph's
  anchor results and preserves explicit anchor/linear overrides. Bad links and
  unanchored components are excluded from refinement. GEO preserves the legacy
  result and its provenance for excluded frames; a rejected refinement does not
  abort atomic saving or create a projective dataset on total fallback.
- Compare full-affine (6 parameters) and projective (8 parameters) adjustments
  on the same withheld observations. Prefer affine when its robust validation
  loss is within 5% of projective. Check for folded footprints, excessive scale
  changes and validation regression before saving. Record iteration exhaustion.
- Persist full `frame_homography` matrices in calibration format 3.1. Keep
  `frame_affine` as an exact-center tangent for compatibility. Localization,
  object/FOV projection, layer matching, map markers and local panoramas use the
  full transform. Unsafe centers across/near a reference projective horizon are
  rejected; unsafe FOV polygons are dropped. GeoJSON centers use saved geometry
  and edge residuals use symmetric pixel errors from matched points. Interpolated
  geographic slots keep their affine fallback and existing provenance.
- Measure neighbor disagreement using actual pair-transform predictions in the
  accuracy profile. Save pair rejection reasons and global validation/model
  selection diagnostics alongside the calibration.

## Configuration and compatibility

`user_config.json` has been restored byte-for-byte to its original version.
`propagation.high_accuracy` and `database.accuracy_features` remain **OFF** by
default. Database overlap and the stored point limit remain unchanged (0.5 and
2048 in the supplied configuration). Native feature extraction remains a separate
optional build feature; no GTX 1650 live-latency claim has been validated.

When explicitly enabled, the offline profile uses four temporal neighbors,
up to 512 points per pair, and optional source-verified subpixel refinement.
`point_pair_limit: 0` retains every point for pair/QA storage, not unlimited solver
training. The global solver uses at most 96 spatially spread training points per
link, preserving a separate every-fifth validation subset. Existing physical,
temporal, fit-quality, distance and odometry guards remain active when configured.
Nearby pairs are allowed despite the old loop time gap, with temporal deduplication.

The solver uses analytical symmetric-transfer derivatives (checked independently
by central finite differences). At more than 64 free frames, overlapping windows
of 32 free frames and stride 24 include **all** incident edges, including distant
loop closures. Outside neighbors are fixed during each window solve. It alternates
forward/backward sweeps, stopping below 0.01 px maximum window update or after
8 sweeps. Training and held-out global costs must pass acceptance checks. This is
bounded block-coordinate optimization, not proof of the global optimum. Rejected
windows and remaining motion are recorded. Whole-flight QA is evaluated in chunks.

Existing affine databases remain readable. Re-propagation is needed to store new
geometry; native feature extraction is not required to use this registration path.
The supplied project databases and videos were not modified by benchmarks.

## Post-review full-flight GEO result

`scripts/benchmark_geographic_pipeline.py` ran the complete `main` (1000 m) source
from `testtopboch_max_dino336`: **1089 slots, 524 occupied keyframes, 248 supplied
anchors**. Both modes used the same original database copied into separate scratch
directories. Temporal matching, DINO retrieval, physical/distance/odometry gates,
two-stage pruning, both DEM terrain passes, exact anchor handling, atomic HDF5
saving/reloading and GeoJSON export all ran. The original database SHA-256 is
unchanged. No source recording or user calibration was overwritten.

| Measurement | Legacy | Accuracy |
| --- | --- | --- |
| Common validation median, original pixels | 5.405 | 2.813 |
| Common validation P95, original pixels | 15.397 | 10.514 |
| Non-anchor ground center median, Mercator metres | 2.046 | 1.100 |
| Non-anchor ground center P95, Mercator metres | 4.011 | 2.744 |
| Runtime, seconds | 40.2 | 262.5 |

Transfer comparison uses the same 95,083 observations from accepted constraints,
withheld from global fitting (already seen by pairwise RANSAC/model selection).
Ground-center comparison uses 276 occupied **non-anchor** frames against simulator
`center_mercator`; these units are projected metres, not ground-distance metres.
The supplied dense anchors and DEM constrain drift, so this cannot be generalized
to an unanchored LOCAL flight. The anchor-fitting algorithm was unchanged.

The selected projective candidate used 1,782 links (872 additional verified
spatial overlaps), 471,598 QA points and 152,589 training points. All 524 keyframes
were refined; all 1089 saved slots reloaded. Interpolated slots retain their existing
provenance. All 910 exported graph edges have finite numerical pixel residuals.
The extra point constraints are counted in the alignment report, not exported as
legacy 5-DoF graph edges.

The solver exhausted its eight-sweep budget; the maximum last-sweep window update
was **0.725 px**. Global convergence remains **uncertified** despite improved
held-out error. This is an explicit remaining limitation, not a silent success
flag. High accuracy remains opt-in and database-build settings remain unchanged.

Raw results: [relative_geo_main_2026-10-12.json](benchmarks/relative_geo_main_2026-10-12.json).

## Historical subset checks

`scripts/benchmark_relative_alignment.py` reads existing HDF5 features and source
frames and writes a separate JSON report. It compares the previous LOCAL affine
adjustment with the new pipeline on a temporal subgraph. It does **not** run the
whole flight or global-descriptor loop retrieval. Synthetic integration tests
cover close-loop retrieval, LOCAL and geographic save/reload, exact anchor
preservation, projective panorama/consumer compatibility, invalid geometry,
rotation retries, source identity and cancellation.

Real data: `testtopboch_max_dino336`, existing 1280×720 observations, frame step
30, stored point budget 2048. The larger future-build budget was not used to
produce these benchmark numbers. GPU: RTX 5070 Ti.

Both modes are evaluated on the **same** points and common connected frames,
in original image pixels. Every fifth spatially ordered observation is withheld
from global fitting, but pairwise RANSAC has seen it and model selection uses
it. These are validation residuals, not an independent ground-truth accuracy
certificate. Known-transform synthetic tests also check unused image locations.

| Recording / selected range | Connected frames, old/new | Median px, old → new | P95 px, old → new | Selected model |
| --- | --- | --- | --- | --- |
| 1000 m, first 24 occupied slots | 24 / 24 | 0.881 → 0.674 | 2.988 → 2.070 | Projective |
| 2000 m, first 24 occupied slots | 24 / 24 | 0.747 → 0.787 | 7.312 → 2.220 | Projective |
| 500 m, 16 occupied slots from 1036 | 3 / 3 | 0.515 → 0.426 | 1.488 → 1.729 | Affine |

The 500 m subset remains disconnected after its first three frames. Starting
at slot 1032 yields an isolated gauge and neither mode can produce a connected
map. Extra tested edges cannot replace missing overlap. This failure is retained
in the report rather than hidden by interpolating a trajectory.

These subset numbers describe the pre-review implementation, retained to expose
its negative cases. The two larger projective solves exhausted their 3×150 evaluation budget;
their validation checks passed but solver convergence is not certified. Runtime
is recorded in the JSON; cold model startup and concurrent checks make it an
indicative measurement rather than a controlled performance benchmark.

Post-review regression run: **506 passed, 7 skipped** across geometry,
localization, feature extraction, configuration and real database-builder
characterization. Final focused checks: **53 passed**, including horizon-safe
diagnostics, FOV rejection and the final eight-sweep window solver. Ruff lint,
Ruff format and `git diff --check` pass on the changed Python files. Changed Python
files parse under Python 3.10 grammar; runtime tests use Python 3.11. The existing Loguru telemetry shutdown handler prints a
closed-capture-stream warning after pytest exits successfully.

## Historical subset measurements and accuracy boundary

Ground-center diagnostics intentionally use only the first frame's affine scale
and center to place the local map. Over real terrain, their P95 changed from
798.54 to 1283.40 seed-image pixels at 1000 m and from 143.13 to 179.31 at 2000 m.
Thus better overlap alignment has **not** demonstrated lower long-range ground
drift. These diagnostics mix accumulated registration error, terrain parallax
and the first-frame planar approximation; they must not be presented as GPS
accuracy or omitted when assessing the change.

A single homography cannot represent depth-dependent parallax. Denser visual
overlaps and cross-pass constraints are needed for drift control; accurate 3D
reconstruction over relief additionally needs calibrated camera geometry and
multi-view triangulation/bundle adjustment. Those are not implemented by this
planar-registration branch. The post-review full-flight GEO results below use
the supplied anchors and terrain model; they do not establish unanchored LOCAL
trajectory accuracy. Full GEO replays of the 2000 m and 500 m sources remain untested.

Machine-readable results: [relative_alignment_2026-10-12.json](benchmarks/relative_alignment_2026-10-12.json).

## Post-review scaling measurement

Synthetic 1280×720 sequence, four temporal neighbors, 512 observations per
link, 0.15 px observation noise, both candidate models. Final configuration: 96
training points per link, analytic Jacobian, up to eight overlapping sweeps.

| Frames | Links | Seconds | Peak process working set, MiB | Held-out P95, px |
| --- | --- | --- | --- | --- |
| 100 | 390 | 27.80 | 193.5 | 0.3690 |
| 200 | 790 | 62.37 | 232.5 | 0.3691 |
| 500 | 1990 | 153.31 | 467.9 | 0.3698 |

The 200→500 frame runtime ratio is 2.46 for 2.5× more frames. These are
measurements on this host, with concurrent checks, not an extrapolated 1000-frame
guarantee. Peak memory is the process high-water mark (includes input data and
previous benchmark sizes), not just solver allocations. All three runs hit the
eight-sweep budget; remaining maximum window updates were 0.079–0.128 px. Their
validation passed, but global stationarity is not certified.

Run: `python scripts/benchmark_relative_solver.py --frames 100 200 500 --output scratch/scaling.json`.

Raw results: [relative_solver_scaling_2026-10-12.json](benchmarks/relative_solver_scaling_2026-10-12.json).
