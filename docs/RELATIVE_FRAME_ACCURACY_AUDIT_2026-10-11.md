# Relative frame positioning: accuracy audit

Date: 2026-10-11. Scope: image-to-image registration and the relative frame map. Geographic anchoring, GPS conversion, and live tracking are outside this audit. The user prioritizes accuracy over processing time.

## Recommendation

Make shared image observations the primary optimization target. Preserve full relative geometry through initialization, add redundant overlapping-frame constraints, refine correspondences at original image resolution, and evaluate on observations withheld from fitting. The existing LOCAL point adjustment is a useful foundation and must be extended rather than reimplemented.

For approximately planar imagery, compare full affine and projective multi-view adjustment. For significant terrain/building parallax, a single 2D transform has a physical accuracy ceiling: use camera poses and 3D scene points with bundle adjustment. Extra RANSAC iterations alone cannot remove a model's representation error.

This is an analysis deliverable. Application code and user settings were not changed. Synthetic experiments isolate mathematical limitations; they do not estimate the error of any particular flight.

## Current execution path

1. Database construction extracts local image features and estimates inter-frame homographies for keyframe selection and `frame_poses`: `src/database/frame_processor.py`, `src/database/keyframe_selector.py:138`.
2. Propagation reloads the retained keyframe features and matches them again. It does not simply use the database pose chain as its final calibration: `src/workers/propagation_pipeline.py:817`.
3. `FeatureMatcher` uses the configured local feature family and LightGlue when available. The inspected configuration selects ALIKED. DINOv3/VLAD retrieves candidate overlaps; it does not determine the pixel displacement.
4. `_match_and_build_edge` estimates a homography using PoseLib, with an OpenCV fallback, and optionally shrinks the inlier mask with MAD: `propagation_pipeline.py:1791`, `src/geometry/transformations.py:212`.
5. `homography_to_affine` approximates the homography using five synthetic control points near the image center. `_build_graph_edge` then reduces that affine to translation, two scales, and rotation: `src/geometry/pose_graph/optimizer.py:949` and `:155`.
6. A graph is initialized and optimized from those five-parameter measurements.
7. Only when `converter.mode == "LOCAL"`, retained point correspondences enter full six-parameter affine IRLS adjustment, followed by an additional search for nearby overlapping frames: `propagation_pipeline.py:631`, `src/geometry/relative_alignment.py:82`.

The LOCAL path already fixes a full first-frame gauge, retains graph weights, handles point outliers, checks connectivity, and rejects severe folding/distortion. It does not interpolate unsupported frames into visually measured positions. Those are useful protections.

## Verified limitations, in priority order

### 1. Homography to affine to five parameters loses valid image geometry

The first reduction loses perspective. The second loses shear. A well-estimated homography can therefore become a less accurate constraint before graph optimization even begins. Small graph residuals cannot establish agreement of the actual image content.

The LOCAL affine adjustment can recover shear from retained point matches, but it cannot recover perspective. The ordinary propagation branch does not run that point adjustment. Even in LOCAL, graph construction, initial placement, and pruning happen before point adjustment; distorted initial geometry can affect which constraints survive.

Synthetic evidence at 1920 x 1080, with exact input homographies and no matching noise:

| Input transform | Full-frame RMS after five-parameter conversion | RMS after existing LOCAL affine point refinement |
|---|---:|---:|
| Pure translation | 0.000 px | 0.000 px |
| Shear 0.08 | 26.312 px | 0.028 px |
| Perspective coefficients 0.00002, -0.00001 | 10.243 px | 7.275 px |
| Perspective coefficients 0.00007, -0.00003 | 32.276 px | 22.853 px |

These are errors over a full image grid, not center errors or measurements on real overlapping footage. The exact matrices and the executable experiment are in `scratch/relative_geometry_audit_20261011.py`. All four matrices pass the existing matrix validator. Refinement uses ideal point data over the full footprint to isolate representation error, rather than trying to simulate a real image matcher.

A separate chain experiment repeatedly composes `H = [[1, .02, 20], [0, 1, 10], [0, 0, 1]]`. Comparing exact composition against the current five-parameter BFS gives center errors of 0.00, 2.00, 9.00, and 38.00 px after 1, 5, 10, and 20 transitions. This tests the initializer with isotropy regularization disabled, not the full configured pipeline after loop closures and LOCAL refinement. It shows that center positions also accumulate representation error.

### 2. The graph lacks redundant short-range measurements

`_build_temporal_edges` stops after its first accepted candidate. With current `skip_bridges=false`, it tries only the immediately previous frame with features. Enabling skip bridges provides fallback connectivity, not several simultaneous temporal constraints.

Both descriptor-based loop retrieval and the LOCAL spatial proposal path exclude pairs with a slot gap of at most `loop_closure_min_frame_gap`, currently 10. Consequently many potentially useful nonadjacent short-range overlaps are never measured. A chain alone cannot independently detect an internally consistent bad edge or accumulated bias.

Add edges to several overlapping neighbors, selected by measured overlap and feature support, and keep all independently verified constraints. Build multi-view tracks, retain feature identities, and check three-frame cycles. Treat repeated observations as correlated rather than multiplying confidence blindly.

The main loop gate currently rejects relative rotation above 15 degrees. That can exclude a genuine reverse-direction pass. LOCAL's additional overlap search can recover some such links, but only for frames already in its supported result set. Replace a universal small-angle limit for spatial links with image-footprint validity, geometric support, cycle checks, and appropriate motion knowledge.

### 3. The spread metric misses diagonally collinear observations

`inlier_spread` uses `min(std_x, std_y)` (`src/geometry/point_spread.py:33`). A diagonal line has large variation in both coordinates, although it has no two-dimensional support. A synthetic line from (100,100) to (1000,1000) scores **0.24298**, above both the configured 0.03 gate and 0.15 weight reference, while its covariance's smallest eigenvalue is zero.

This proves a defect in the spread diagnostic, not that every collinear set will pass RANSAC. The LOCAL solver separately rejects exactly rank-deficient point pairs; nearly collinear sets remain a conditioning issue.

Use the smallest eigenvalue of normalized point covariance, hull/grid coverage, and center extrapolation in **both** images. Preserve a distinction between exact degeneracy and legitimate small overlap. Bootstrap transform estimates or propagate point uncertainty to report center and corner uncertainty.

### 4. Temporal edge quality checks are incomplete

The common matching function accepts sufficient inliers and reports a one-way inlier RMSE. Temporal edges subsequently receive transform sanity checks and optional soft weight penalties. They do not use the spatial gate's minimum inlier fraction or hard minimum spread. The live localizer's geometric quality settings are not automatically applied here.

Use a common pair-quality record: finite coordinates, positive image-to-image orientation where appropriate, valid projective denominator across the footprint, symmetric transfer error, robust error percentiles, support in both images, center extrapolation, and uncertainty. Do not treat a minimum inlier count alone as sufficient evidence.

`_refine_mask_mad` changes the mask but returns the original homography. PoseLib/OpenCV already perform internal robust fitting, but there is no explicit final re-estimation on the smaller surviving set. Add robust final refinement and recompute diagnostics for that model; keep a change only if independent validation improves.

### 5. Information is capped before final refinement

The inspected settings request up to 4096 ALIKED keypoints but store at most 2048. Features are score-sorted before saving, so this is a quality-ranked cap, not arbitrary truncation. A high score still does not ensure coverage across the overlap.

LOCAL retains at most **96** spatially distributed point pairs per edge (`relative_alignment.py:68`). That deliberate speed optimization limits how much evidence the final solver can use. More observations are not automatically better; measure 96/256/512/all-valid settings after removing poor and correlated matches.

Local extraction is capped at 1600 px on the longest side in both single and batch paths (`src/models/wrappers/feature_extractor.py:348`, `:453`). Large-image coordinates are rescaled afterwards; this does not restore the image detail lost before detection. Frames already below the cap are unaffected.

LightGlue currently uses adaptive stopping/pruning (0.95/0.99). Its authors recommend disabling these mechanisms for maximum matcher accuracy. This is a useful ablation, not a guarantee of a particular flight-level gain.

Changing extraction resolution or retained keypoint count requires rebuilding features/the database. Changing the final point budget requires code/config support. Propagation alone cannot recover discarded features.

### 6. Subpixel observation refinement and a camera distortion model are absent from this path

The inspected calibration path does not refine matched patches jointly across images or undistort observations using calibrated camera intrinsics. A learned detector can already produce fractional coordinates; that does not replace pairwise or multi-view refinement.

For moderate adjacent-frame changes, evaluate pyramidal patch alignment with forward/backward consistency, masks, and robust rejection, initialized from the geometric transform. For larger changes, evaluate featuremetric multi-view refinement. Preserve original-resolution images or reproducible access to exact source frames.

For real lenses, establish intrinsics/distortion before tuning subpixel thresholds. For synthetic imagery with a distortion-free camera this item may be unnecessary. If residuals depend on scene depth or row timing, investigate parallax or rolling shutter rather than forcing a single planar transform to absorb them.

## Accuracy-first implementation sequence

1. **Establish the benchmark and pair diagnostics.** Correct the spread measure, store feature IDs and match diagnostics, distinguish optimized from unsupported/interpolated poses, and report rejected constraints by reason.
2. **Improve observations and connectivity.** Preserve more well-distributed points, add redundant temporal overlaps, and enable a dedicated offline matcher profile. Use denser sampling where current overlap is insufficient. The current overlap threshold is 0.5; compare 0.7-0.8 as experimental settings rather than universal requirements. Account for the base `frame_step=30` and source FPS; downstream keyframe selection cannot recover skipped video frames.
3. **Refine observations at source resolution.** Validate subpixel patch/feature refinement on held-out correspondences, including blur, exposure changes, and repeated textures. Do not globally raise every threshold to gain match count.
4. **Preserve geometry through the solve.** First reuse the existing affine point solver with a full-affine initializer. Benchmark a projective multi-view solver for planar scenes. Keep relative estimation a dedicated stage with one gauge; geographic fitting can consume its result later.
5. **Use 3D when the data requires it.** For depth-dependent residuals, run relative camera/landmark reconstruction with bundle adjustment, using a calibrated camera model where available. A single affine/homography cannot align terrain and elevated objects simultaneously under translational parallax. Monocular reconstruction still needs a gauge and does not acquire metric scale from images alone.

For a projective planar objective, compare matched observations in their original pixel systems, for example `x_i - project(T_i^-1 T_j x_j)`, and its reverse, with robust losses and observation uncertainty. This avoids lowering map-space residuals merely by shrinking a frame. Anchor the first transform and guard projective horizons and collapsed maps. Store a projective result explicitly; converting it back to a 2x3 affine would lose the improvement.

Candidate offline settings for controlled ablation: LightGlue `depth_confidence=-1`, `width_confidence=-1`; retain at least the requested 4096 keypoints; use native resolution or tested tiles; compare RANSAC confidence 0.999 with a larger iteration budget on difficult pairs. Keep FlashAttention enabled. These are proposed experiments, not settings applied by this audit, and increasing the RANSAC budget does not remedy repeated-texture ambiguity or a wrong model.

## How to establish that accuracy actually improved

Record a baseline and use the same videos, frame IDs, masks, and deterministic settings. Include easy translation, turns/revisits, low-texture fields, repetitive texture, perspective/altitude change, and real relief. Preserve a connected evaluation subset and report coverage/failure rate alongside errors so rejecting difficult frames cannot masquerade as improvement.

Measure symmetric transfer median/P95 on held-out point observations in original image pixels; frame-center and footprint error against synthetic/known ground truth; relative drift at several frame separations; cycle closure error; and the rate of accepted catastrophic links. Training inlier RMSE and optimizer cost are supporting diagnostics, not accuracy ground truth. Existing LOCAL before/after percentiles use the same points optimized by IRLS.

For suitable sharp planar imagery, an initial engineering target could be held-out median below 0.5 px and P95 below 1.5 px. These are proposed acceptance goals, not demonstrated performance or a guarantee for all scenes. Evaluate center error separately because low residual inside a small overlap does not ensure an accurate extrapolated center.

Compare one change at a time: baseline; diagnostics; redundant links; larger feature budget; observation refinement; affine/projective model. Run the 3D alternative when planar residuals remain structured by depth. Promote changes based on held-out geometry and failure rate, with runtime and memory recorded as secondary metrics.

## Validation and reproducibility

- Passed 39 existing tests covering relative affine adjustment, pose graph basics, and MAD homography filtering. Run with plugin autoload disabled and `-s` in this environment. This validates existing behavior, not the proposed redesign.
- An additional `test_relative_map.py` run did not complete collection in this environment and was interrupted; it has no pass/fail result from this audit.
- Ran the deterministic synthetic model-loss and collinearity experiments described above. No trained models, video databases, or geographic anchors are used in them.
- No end-to-end real-video accuracy benchmark was run; no particular recording was selected for this audit.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest tests/unit/geometry/test_relative_alignment.py tests/unit/geometry/pose_graph/test_pose_graph_optimizer.py tests/unit/localization/test_mad_ransac.py -q -s
.\.venv\Scripts\python.exe scratch/relative_geometry_audit_20261011.py
```

Primary references: [LightGlue configuration](https://github.com/cvg/LightGlue#advanced-configuration) for full-depth/non-pruned matching; [Pixel-Perfect SfM](https://github.com/cvg/pixel-perfect-sfm) for featuremetric keypoint and bundle refinement; [COLMAP camera models](https://colmap.github.io/cameras.html) and [COLMAP tutorial](https://colmap.github.io/tutorial.html) for camera-model and reconstruction context. The priority order above is an inference from this repository and the synthetic experiments, not a benchmark comparison of those external systems.
