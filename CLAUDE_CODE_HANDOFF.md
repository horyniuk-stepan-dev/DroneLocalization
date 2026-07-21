# Windows Claude Code handoff — verify, commit & finish (built 2026-07-19)

You are Claude Code running on the user's **Windows** machine with the **full stack**
(torch, PyQt6, CUDA/GPU, lancedb, decord, h5py, faiss, and the real recordings). A prior
**sandbox** session (Linux, no torch/PyQt6/GPU) produced four uncommitted deliverables it
could only *partially* verify. Your job: verify them on the real stack, commit each, then
**finish the in-progress `database_builder` split**.

## Ground rules (read first)
- `CLAUDE.md` (the Operating Manual + Project-Specific Rules) governs everything. Follow it.
  Re-derive; do **not** trust the sandbox session's claims blindly — confirm by running.
- **Do NOT touch or commit the user's separate WIP**: `src/tracking/kalman_filter.py`
  (two-point velocity seed), `tests/test_kalman_filter.py`, and the `user_config.json`
  change (`outlier_threshold_std` 16 → 80). These are the user's, unrelated to this work.
  Stage **only** the exact files listed per task — never `git add -A`.
- Don't change pydantic config defaults; don't toggle `user_config.json`.
- Ruff is pinned **v0.1.9** (pre-commit). Lint changed files: `ruff check --fix` + `ruff format`.
- Run the relevant tests **before every commit**. Commit granularly (one logical change each).
- Report outcomes with the actual numbers/outputs, not just "done".

## Task 0 — establish a green baseline
```
python -m pytest -q
```
Record any **pre-existing** failures so you don't misattribute them to the new code. Confirm
the stack imports: `python -c "import torch, PyQt6, lancedb, h5py, faiss; print('stack ok')"`.

---

## Task 1 — PDM@K retrieval metrics  (pure; just verify + commit)
**Files:** `scripts/retrieval_metrics.py` (new), `tests/test_retrieval_metrics.py` (new).
Pure numpy (no torch) — should pass identically to the sandbox (was 32/32 there).
```
python -m pytest tests/test_retrieval_metrics.py -q          # expect 32 passed
git add scripts/retrieval_metrics.py tests/test_retrieval_metrics.py
git commit -m "feat(eval): add PDM@K/SDM@K/Recall@K retrieval metrics (Track 6 / research-plan 3.2)"
```
Note for the user: `f(R_i)` is a **reconstructed** logistic (the AnyVisLoc paper renders the
closed form as a figure); defaults `l=1.67, alpha=0.85, lambda=6` are the paper's 4:3 values —
for this project's 16:9 frames, `l`/`alpha` should be recomputed before trusting *absolute*
PDM numbers (relative A/B gating is unaffected). Leave as-is unless the user asks.

## Task 2 — Integration flow test  (verify pure + run the real e2e)
**File:** `tests/integration/test_full_cycle.py` (new).
```
python -m pytest tests/integration/test_full_cycle.py -q     # expect 6 passed, 1 skipped
```
Real e2e — needs a dataset dir with: a flight `*.mp4`, a `ground_truth.json` in the
**benchmark schema** (top-level `frame_width`/`frame_height`, `anchors:[frame_id...]`,
`frames:[{frame_id, affine 2x3 px->metric}]`), and a pre-built `benchmark.h5` (build it with
the project builder from the same video if absent):
```
set DRONELOC_E2E_DATASET=D:\path\to\dataset
python -m pytest tests/integration/test_full_cycle.py::test_full_cycle_two_anchor_propagation -q -s
```
- If `ground_truth.json` is the **slots/center_mercator** schema (new simulator) instead of
  `frames/affine`, `load_ground_truth` will KeyError. That's a schema mismatch shared with
  `scripts/benchmark_propagation.py` — **flag it to the user, do not silently convert.**
- If 2-anchor propagation legitimately exceeds the GSD gate, raise `DRONELOC_E2E_TOL_M`
  (threshold tune, not a code bug) and say so.
```
git add tests/integration/test_full_cycle.py
git commit -m "test(integration): full-cycle build->2-anchor->propagation acceptance w/ GSD gate (Track 1 / IMPROVEMENT_PLAN 4.5)"
```

## Task 3 — PropagationPipeline refactor  (verify behaviour 1:1 + commit)
**Files:** `src/workers/propagation_pipeline.py` (new, Qt-free core),
`src/workers/calibration_propagation_worker.py` (rewritten thin QThread wrapper).
The ~1200 lines of graph-propagation logic moved **verbatim** into `PropagationPipeline`
(no PyQt import). The worker is now ~55 lines wiring the 3 signals (progress/completed/error)
to pipeline callbacks and delegating `_propagate`/`stop`/`run`. **Intended behavioural delta: zero.**

Verify:
1. Import (needs PyQt6):
   ```
   python -c "from src.workers.calibration_propagation_worker import CalibrationPropagationWorker; from src.workers.propagation_pipeline import PropagationPipeline; print('ok')"
   ```
2. Run a **real propagation** — either:
   - **GUI:** load a project with a built DB + >=2 calibration anchors, run *propagate*.
     Confirm the progress-dialog texts are exactly: `Передзавантаження фіч у RAM...` (0),
     `Побудова часових ребер (sequential matching)...` (10),
     `Пошук просторових замикань (loop closure)...` (30),
     `Фіксація GPS-якорів (Local Origin)...` (60),
     `Глобальна оптимізація графу (Levenberg-Marquardt)...` (70),
     `Збереження результатів у HDF5...` (85). It completes; the cancel button still stops it.
   - **Benchmark:** `python scripts/benchmark_propagation.py --dataset <dir with benchmark.h5 + ground_truth.json>` runs to completion.
3. Best: diff the propagated HDF5 (`calibration/frame_affine`, `frame_valid`) against a
   pre-refactor build on the same input — must be **identical** (it's a pure move).
If progress texts differ or results change, the emit->callback wiring in the worker is wrong — investigate.
```
git add src/workers/propagation_pipeline.py src/workers/calibration_propagation_worker.py
git commit -m "refactor(propagation): extract Qt-free PropagationPipeline; thin QThread wrapper (Track 1 / IMPROVEMENT_PLAN 1.4)"
```

## Task 4 — keyframe_selector extraction  (verify + commit)
**Files:** `src/database/keyframe_selector.py` (new), `src/database/database_builder.py`
(delegates), `tests/test_keyframe_selector.py` (new, 12 tests).
`is_significant_motion` + `compute_inter_frame_homography` extracted to a pure module (matcher
injected). Builder's `_is_significant_motion`/`_compute_inter_frame_H` now delegate;
`GeometryTransforms` import dropped from the builder. One intended micro-change:
`is_significant_motion` returns plain `bool` (was `np.bool_`) — immaterial to `if save_this_frame:`.
```
python -m pytest tests/test_keyframe_selector.py -q          # expect 12 passed
python -c "from src.database.database_builder import DatabaseBuilder; print('ok')"   # needs torch
```
Then **rebuild a DB from a video with keyframe selection ON** (`database.keyframe_min_translation_px > 0`)
and confirm it's **identical** to a pre-refactor build on the same video: same keyframes selected,
same `frame_poses`, same slots. DB interchangeability is sensitive (see CLAUDE.md / DB_INTERCHANGEABILITY.md).
```
git add src/database/keyframe_selector.py tests/test_keyframe_selector.py src/database/database_builder.py
git commit -m "refactor(db): extract tested keyframe_selector; builder delegates (Track 1 / IMPROVEMENT_PLAN 1.3)"
```

---

## Task 5 — FINISH the `database_builder` split (the real remaining dev work)
Goal (IMPROVEMENT_PLAN п.1.3): decompose the 533-line `build_from_video` monster into cohesive
modules. **Plan gate: a characterization test must pin the invariants BEFORE refactoring.**

### 5a — write the characterization test FIRST (do this before touching build_from_video)
Create `tests/integration/test_db_builder_characterization.py`. Build a small DB with the real
builder (torch/GPU) from a short clip, and assert these **invariants** (currently true — capture
them as the golden net):
- **pose-always** — `global_descriptors/frame_poses[slot]` is a valid, non-zero affine for
  **every** processed slot, including non-keyframes. (build_from_video writes
  `frame_poses[p_idx] = current_pose` unconditionally, ~lines 480-484.)
- **keyframe-selectively** — full local features (`local_features/kp_counts[slot] > 0`) are
  stored **only** for selected keyframes; with selection ON, keyframe count <= processed frames
  and is gated by `is_significant_motion`. (`save_frame_data(p_idx, ...)` called only when
  `save_this_frame`, ~line 486-501.)
- **frame_id <-> slot identity** — DB slot `i` == video frame `i * frame_step`
  (`create_hdf5_structure` writes the `frame_step` attr, ~line 871-873; `save_frame_data` uses
  `p_idx` as the slot, not a running counter).
Gate it with `pytest.importorskip("torch")` + a small fixture video (`tests/fixtures/dummy_video.mp4`
or `long_dummy_video.mp4`; if a random-noise fixture is too degenerate for feature matching to
select keyframes, use a short real flight clip). Run it **green against the current builder** to
capture golden behaviour. Commit the test on its own.

### 5b — extract the remaining modules, keeping 5a + the full suite green after EACH step

**STATUS (updated after the 2026-07-19 Windows session):**
- 5a golden-net: **DONE**, commit `c12320c` (real FlightSimulator clip `flight_clip.mp4`,
  frames 3000-3149; 3 invariants; 6 passed).
- `keypoint_video_writer`: **DONE** (verbatim extraction, 4 unit tests, DB byte-identical vs
  golden, 22 passed). Committed.
- Verified: the GPU build is **data-level deterministic** on this machine (two builds byte-identical),
  so byte-diff vs golden is a valid per-extraction gate.
- **Remaining: `video_frame_source`, `db_writer`, `frame_processor`** — the hard, tightly-coupled trio.

**ORDER — do exactly this: `video_frame_source` -> `db_writer` -> `frame_processor`.**
Rationale: the producer thread touches no HDF5 at all, so it is the most independent and cheapest
first win; `db_writer` must be settled **before** `frame_processor`, because the latter *consumes*
its contract (it writes `frame_poses` and calls `save_frame_data`).

**LOCKED DESIGN DECISION — `DbWriter` owns the HDF5 lifecycle END TO END.**
Open, schema creation, **every** write (including the per-frame `frame_poses` write that
`build_from_video` currently does inline), the lancedb batch **and its finalization**, and close.
After the extraction the builder holds **no** h5 handle and makes **no** direct h5/lance call.
Do **not** take the shortcut of moving only `create_hdf5_structure` + `save_frame_data`: that
leaves ownership of the same file split between the builder and the writer, which is *worse* than
the status quo (indirection added, coupling unchanged). If the clean version looks too big for one
sitting, stop and split it across sittings — do not ship the half-move.

**Why the net is not enough here:** byte-diff protects **correctness, not design**. A rushed
`DbWriter` can be byte-identical and still have a broken ownership contract, and the gate will pass
it. Spend the care on the boundary, not just on the diff.

Procedure per module: extract **one at a time**; after each run 5a + `pytest -q` + a byte-diff of a
rebuilt DB vs golden; keep behaviour identical; commit separately with `refactor(db): ...`.
`build_from_video` ends as a thin orchestrator.
**CRITICAL (DB_INTERCHANGEABILITY):** per-frame outputs and the HDF5/lance schema must not change.
**Caveat on the net:** determinism was confirmed for the *current* config on this machine/driver.
Anything that changes batching (e.g. the Track 5 "real batches in database_builder" item) invalidates
that assumption — re-confirm two-build byte-identity before trusting byte-diff again.

---

## After this (pointers, not for now)
Continue Track 1 per `docs/REMAINING_WORK_PLAN.md` queue: `model_manager` -> vram/registry/loaders,
MainWindow mixins -> controllers, typed `get_cfg`, then mypy in CI. Also outstanding (older,
Windows-only): smoother v2 live-retest (already committed — confirm the trajectory doesn't jitter
to fixes), and the calibration stage-8 mission-benchmark merge gate. See memory notes
`track1-refactor-progress`, `research-integration-plan-progress`.
