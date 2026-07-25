# Depth-for-Tilt Feasibility Research

**Date:** 2026-07-25
**Question:** Can Depth Anything V2 (the depth model already in the pipeline) produce a
usable *camera-tilt* signal — i.e. if a frame is captured obliquely, does the depth map
expose a tilt gradient / surface-orientation cue that is useful inside this drone
localization app?

**Verdict:** The intuition is geometrically correct — tilt *does* produce a depth
gradient — but **Depth Anything V2 will not deliver a usable tilt angle in this app's
operating regime.** The homography the pipeline already estimates is the better, cheaper,
and more reliable source of tilt. Depth stays disabled.

---

## Why the idea is sound in principle

A tilted view of a (roughly planar) ground surface leaves a signature in the depth
distribution. The UAV-specific paper **DAPM** formalizes this:

- **Pitch** (optical axis tilt) → *non-linear expansion of depth in the upper region of
  the frame*.
- **Roll** (skyline rotation) → *slanted depth distribution* (a tilted iso-depth field).

So "oblique frame → depth gradient" is real. The problem is extracting a reliable *angle*
from it with **this** model.

## Why Depth Anything V2 specifically fails for tilt

1. **Relative depth is not pose.** DAPM states outright that *relative* depth methods like
   Depth Anything do not directly provide tilt angles. To recover pitch/roll, DAPM trains a
   **dedicated pose branch** on **metric** depth plus ground-truth camera-pose labels.
   Depth Anything is not trained to output pose or surface normals.

2. **Affine ambiguity distorts the plane.** Depth Anything / MiDaS predict depth up to an
   unknown **scale *and* shift**, and — critically — the effective shift differs *across
   surfaces*. These models rank depth well *within* one surface but the geometry *between*
   surfaces drifts (see the CVPR 2025 affine-correction work). A naive plane fit over the
   whole frame therefore yields a biased, unreliable tilt.

3. **Near-nadir is geometrically degenerate — the killer for this app.** DAPM: for
   near-nadir viewing (pitch ≈ 0°) the geometric relationships become degenerate and the
   depth distribution becomes uniform. This app operates **near nadir** (confirmed
   empirically: affine matching succeeds, inliers saturate at the keypoint cap of 2048 with
   high confidence). The tilt signal from depth is weakest exactly where the app lives.

## If per-image tilt / normals were genuinely required

The literature points away from Depth Anything toward dedicated geometric models:

- **Metric3D v2** — foundation model outputting *metric depth + surface normals* zero-shot
  (ranks 1st on normal benchmarks). The correct monocular tool for orientation/normals.
- **DSINE / GroundNet** — dedicated surface-normal / ground-plane-normal networks.
- **DAPM** — a joint UAV depth + pitch/roll/height/FoV model, if that exact bundle is ever
  needed.

All are heavier than the current pipeline, and all still degrade near nadir.

## Recommendation for this app

- **Do not reintroduce depth for tilt.** Any monocular-depth route is the wrong tool in the
  near-nadir regime, and the affine ambiguity biases the geometry regardless of angle.
- **Use the homography instead.** The pipeline already estimates a full frame→map
  homography (PoseLib LO-RANSAC / OpenCV MAGSAC++). For a planar ground scene:
  - Nadir view → the homography is effectively **affine**: the projective row `[h31, h32] ≈ 0`.
  - Tilt → the **projective component `[h31, h32]` becomes non-zero** and its magnitude
    tracks the tilt. This is the geometrically correct "tilt gradient", and it is best
    conditioned precisely near nadir — the opposite of depth.
  - With camera intrinsics `K` (not currently stored in the project) the homography can be
    decomposed via `cv2.decomposeHomographyMat` into rotation + plane normal for a physical
    angle. Without `K`, `|[h31, h32]|` still serves as a relative tilt indicator for free.
- **Cheap next step if desired:** a small, flag-gated instrumentation that logs
  `|[h31, h32]|` per frame from the existing homography (no depth, no new model). Fly an
  intentionally oblique mission; if the indicator rises and correlates with inlier drops,
  a real tilt/rectification feature becomes justified — built on homography decomposition,
  not depth.

## Caveat on the evidence

The sources cover ground / road / general-UAV scenes; no benchmark specifically evaluates
"Depth Anything on nadir aerial frames for tilt" was found. However, near-nadir degeneracy
is a geometric fact, not a dataset artifact, so the conclusion holds for this near-nadir
application.

## Sources

- DAPM: UAV Monocular Depth Estimation from Any Height, Pitch, Roll and FOV — https://arxiv.org/html/2607.21438
- Relative Pose Estimation through Affine Corrections of Monocular Depth Priors (CVPR 2025) — https://arxiv.org/abs/2501.05446
- Metric3D v2: Zero-shot Metric Depth and Surface Normal Estimation — https://arxiv.org/abs/2404.15506
- GroundNet: Monocular Ground Plane Normal Estimation with Geometric Consistency — https://arxiv.org/abs/1811.07222
- DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data — https://arxiv.org/abs/2002.00569
