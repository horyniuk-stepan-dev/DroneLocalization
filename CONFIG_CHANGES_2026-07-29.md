# Config changes — 2026-07-29

Six values changed in `user_config.json`. Everything else left alone, and the
reasons for leaving it are at the bottom — that list matters more than the
changes, because most of this config is already correct.

Baseline: `user_config.json` overrides the Pydantic defaults in **47 places**.
Forty-five of those are deliberate enablements of features that were developed
flag-gated and then switched on. The config is well maintained. Only two
overrides pointed the other way — they *loosened* a safety gate — and those turn
out to be drift, not a decision.

---

## Changed

### 1–2. `tracking.outlier_threshold_std: 80.0 → 4.0`, `max_speed_mps: 350.0 → 120.0`

**This is the one that matters, and it is a bug, not a tuning preference.**

`git log -S` traces both values to commit `f613fde "build executable"`, which
introduced them as:

```json
"outlier_threshold_std": 150.0,
"max_speed_mps": 1000.0,
```

Those are precisely the numbers that `config/localization.py` documents as
defects:

```python
# ВИПРАВЛЕНО: 150.0 фактично ВИМИКАЛО Z-score фільтр (z>150 не буває).
outlier_threshold_std: float = 4.0
# ВИПРАВЛЕНО: 1000 м/с (3.6 млн км/год) не фільтрувало нічого.
max_speed_mps: float = 120.0
```

So the fix was applied to the Pydantic defaults and never reached the file that
actually wins at runtime. `80.0` and `350.0` are partial walk-backs from the
buggy originals, not chosen values. 350 m/s is 1 260 km/h.

**Why 4.0 is safe** — `OutlierDetector.is_outlier` requires *both* a Z-score
above the threshold **and** an absolute deviation above 15 m/s:

```python
is_zscore_outlier = z_score > self.threshold_std and abs(v - mean_v) > 15.0
```

The absolute floor is what prevents over-rejection. Effective threshold:

| speed variance | `threshold_std = 4.0` | was, at `80.0` |
|---|---:|---:|
| 0–3 m/s | 15 m/s (floor binds) | 80–240 m/s |
| 5 m/s | 20 m/s | 400 m/s |
| 10 m/s | 40 m/s | 800 m/s |

Verified against the detector with 5 m/s speed noise in the window:

| event | rejected? |
|---|---|
| ordinary acceleration, +5 m/s | no |
| sharp manoeuvre, +12 m/s | no |
| jump, +60 m/s | **yes** |
| teleport, +340 m/s | **yes** |

**Interaction with the smoother.** The smoother was tuned while these gates were
effectively open, so this is the change most likely to alter observed behaviour.
But the design anticipates it: `localizer.py:728-730` explicitly feeds *rejected*
fixes into the smoother window so Huber weights arbitrate — the smoother is the
safety net *for* an active Z-gate, not a substitute for one. `max_consecutive_outliers: 3`
remains the escape hatch: three consecutive rejections clear the window and
accept, so a genuine relocation costs at most three frames.

### 3. `tracking.of_outlier_gate: false → true`

Closes the structural hole from audit §1.2: `localize_optical_flow` never called
`is_outlier`, so at `keyframe_interval: 30` twenty-nine of every thirty emitted
positions had no check at all. Only worth enabling together with 1–2 — with the
old thresholds the gate would have been decorative.

### 4. `graph_optimization.use_analytic_jacobian: false → true`

The analytic Jacobian is implemented (`optimizer.py:665-798`), returns a proper
sparse CSR matrix, and is verified against 2-point finite differences by
`tests/test_pose_graph_jacobian.py`, which passes. Its docstring claims 3–10×.

Analytic gradients are *more* accurate than finite differences, so the result
should be at least as good, not merely faster. Offline propagation only —
nothing in the realtime path.

### 5. `models.performance.log_latency_stats: false → true`

Measurement only, no behavioural effect (`tracking_worker.py:509-510`). Turned
on because every remaining open question in the audit is a timing question, and
this is what answers them. Logs p50/p95/p99/max every 100 frames.

### 6. `models.performance.dino_cpu_resize: false → true`

The §2.1b flag added earlier today. Enabled **only because you are rebuilding
databases anyway** — it changes the schema fingerprint
(`043ff1f73092d24f → 26e71f8a5e4b5374`), so existing databases will be refused
on open. That refusal is the safety mechanism working as designed.

**This is the least evidence-backed of the six.** The transfer reduction is
certain (24.88 MB → 0.151 MB per forward at 1080p, 165×); what is *not* measured
is how much of the quoted 470 ms global-descriptor cost is transfer versus ViT
compute. It also swaps the resampling filter, so Recall@1 could move either way.

**Revert this one first** if retrieval quality drops after the rebuild — it is
the only change here that touches descriptor values.

---

## Deliberately not changed

| key | value | why left alone |
|---|---|---|
| `models.performance.fp16_enabled` | `false` | Correct for the GTX 1650 — fp16 is slower on Turing consumer cards without tensor-core paths for these ops. |
| `database.frame_step` | `30` | Must equal FlightSimulator's `--frame-step`. Changing one side breaks the pair. |
| `tracking.smoother_*` | as-is | Tuned against live missions. Changing it in the same pass as the outlier gates would make a regression un-attributable. |
| `tracking.of_stride`, `of_half_res`, `temporal_candidate_prior`, `candidate_prefilter` | on | Validated on a live GTX 1650 mission. Not touching validated work. |
| `graph_optimization.*` (44 keys) | as-is | Coherent, deliberate enablement of the staged calibration work. No evidence to second-guess any of it. |
| `gnc_spatial: false` + `two_stage_prune: true` | as-is | Mutually exclusive in code; the current pick is the validated one. |
| `localization.scale_use_depth_hint` | `false` | Costs a DepthAnything forward per 30 keyframes for a *soft* hint, plus VRAM on a 4 GB card. Correctly off. |
| `localization.sift_fallback` | `false` | Needs `store_sift_features: true`, which adds ~1 MB/frame to the database (~1.8 GB per 30-min mission) for a fallback that fires only on hard frames. A continuous rotation prior (BETTER_APPROACHES A.3) is the cheaper answer to the same problem. |
| `localization.use_patchify` | `false` | Storage-heavy (14 patch descriptors/frame) and unvalidated. VLAD is the better-supported alternative if you want retrieval quality. |
| `models.performance.torch_compile`, `propagation_max_workers`, `global_batch_max`, `max_vram_ratio` | as-is | In `hardware_profile.TUNABLE_KEYS` — auto-tune owns them. Hand-setting them fights the auto-tuner. |
| `homography.use_mad_ransac` | `true` | Now actually does what it says. Before today's fix it was inert under `backend: "poselib"`. |
| `models.performance.debug_mode` | `true` | Controls FD-level log suppression. A development preference, not a correctness matter. |

### One thing I did not set, and cannot

`network_api.api_token` is `""` while `network_api.enabled: true` with REST and
WebSocket both listening. Binding is `127.0.0.1`, so exposure is local-only and
the practical risk today is low — but an empty token means no authentication on
either endpoint. I am not going to invent a secret and write it into a
version-controlled file. If these ports ever bind to anything other than
loopback, set a token first.

---

## Verification order

The two blocks are independent — test them separately or a regression will not
be attributable.

**Block A — no rebuild needed (changes 1–5):**

1. `pytest tests/test_localization_characterization.py` — the golden test.
2. One tracked mission. Watch for `OUTLIER DETECTED` in the log: a handful on
   sharp manoeuvres is expected and healthy; a continuous stream means 4.0 is
   too tight for your flight profile — raise toward 6–8 rather than back to 80.
3. Re-run propagation once and compare anchor RMSE against the previous run.
   The analytic Jacobian should give the same or better, faster.

**Block B — needs a rebuild (change 6):**

4. Rebuild one database. Confirm an *old* database is now rejected on open with
   a fingerprint mismatch.
5. Compare Recall@1 (`tests/test_retrieval_metrics.py`) against the pre-rebuild
   figure, and read the new latency percentiles from change 5.

If Recall@1 drops: set `dino_cpu_resize` back to `false` and rebuild. Nothing
else in this list depends on it.

---

## Revision after the first live run (log of 2026-07-29 20:05)

The gate at `max_speed_mps: 120.0` fired constantly on your test footage. You
identified the cause correctly — the test video comes from an online map, not a
drone, so apparent speed is not physical. The log showed something sharper than
"threshold too tight", so the fix is not a bigger number.

### What the log actually showed

**Keyframe path** — 5 fixes rejected at 175–367 m/s, carrying **647, 1191,
1376, 1549 and 1881 inliers**. A kinematic prior was overruling geometry that
RANSAC had verified hundreds of times over.

**Optical-flow path** — 20 rejections, and they split cleanly in two:

| population | flow_quality | count | correct verdict |
|---|---|---:|---|
| genuine LK loss of lock | 0.017 – 0.183 | 15 | reject — the gate is earning its keep |
| real fast content motion | 0.625 – 1.000 | 5 | **accept** — flow is self-consistent |

The two populations do not overlap: highest failure is 0.183, lowest genuine
motion is 0.625.

### Why simply raising the limit was the wrong fix

Simulated at `max_speed_mps: 500`: all 5 keyframes pass, but **11 of the 15
genuine LK failures also slip through** because they sit below the raised cap.
Loosening the number trades the false positives for false negatives — it removes
the gate's only real value.

### What was done instead

New flag **`tracking.outlier_trust_strong_evidence`** (default `false`,
enabled in this config): the kinematic gate is bypassed when *independent*
evidence is strong.

- keyframe: `inliers >= outlier_trust_min_inliers` (100)
- optical flow: `flow_quality >= outlier_trust_min_flow_quality` (0.5)

The reasoning: inliers and flow_quality are direct measurements of *this fix*;
speed is an assumption about the *platform*. When the direct evidence is strong
it should win. 100 sits above every existing inlier threshold in the system
(`min_inliers_accept` 10, `early_stop_inliers` 40, `confidence_max_inliers` 80).
0.5 sits in the empty gap between the two observed flow populations.

Simulated against every event in your log, with `max_speed_mps` back at **120**:

| | outcome |
|---|---|
| keyframe fixes | 5 of 5 rescued |
| OF, self-consistent flow | 5 rescued (q 0.625–1.0) |
| OF, genuine LK failure | 15 still rejected (q 0.017–0.183) |
| OF slipping through on a raised cap | **0** |

So `outlier_threshold_std` and `max_speed_mps` are back at the physical 4.0 and
120.0. Sensitivity to *your* material drops to zero without the gate going
blind — which is what you asked for, reached from the other side.

### If it still churns

Your material has unusually rich texture (map tiles), so inlier counts are high
and the bypass triggers easily. Real drone footage over farmland will produce
far fewer inliers. If keyframes start getting gated there, lower
`outlier_trust_min_inliers` before touching `max_speed_mps` — the bypass is the
better knob, because it discriminates and the speed cap does not.

Simple fallback if you need the test to run *right now* and something is still
wrong: set `tracking.of_outlier_gate: false`. That restores the pre-today
behaviour on the OF path only, leaving keyframes protected.

### Unrelated finding from the same log

`latency_tracker` reported `p95 = 72.7 / 7.0 / 72.6 ms`, `max = 117 / 97 / 199 ms`.
The comment at `localizer.py:122` states a 945 ms keyframe with 470 ms of it in
the global descriptor. **The observed maximum is 199 ms** — that figure is stale
by a factor of 4–5, and every estimate in the audit that leaned on it (including
the case for `dino_cpu_resize`) is correspondingly weaker. Note `p50 = 0.0 ms`
is not an error: `of_stride: 3` skips most OF frames entirely, so half the
samples are near-zero. Worth measuring keyframes separately before drawing
conclusions about where the time goes.
