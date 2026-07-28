# Threat Model & Security Controls — DroneLocalization

**Status:** living document. Covers the hardening implemented on branch `hardening/p0`
(P0 + P1 + P2-12) and the residual/deferred work. Companion to
`docs/MILITARY_GRADE_HARDENING_PLAN.md` (the roadmap); this document records what
is *built and verified* versus what remains.

> **Design rule honored throughout:** every control ships **flag-gated with the
> default equal to prior behavior**. A stock run is unchanged; hardening is opt-in
> via `user_config.json`. Nothing here silently alters a default.

---

## 1. Threat model

The payload is a GPS-denied localization system. Its working threat model:

| Threat | Assumption | Primary controls |
|---|---|---|
| Contested RF / EW | Network link hostile or absent; operator may need air-gap | Localhost-default telemetry; fail-closed auth; optional TLS |
| Airframe loss / capture | Recovery by an adversary is realistic | Passphrase encryption-at-rest of the whole project (immutable deployment copy); weight-integrity pinning; log redaction |
| Unattended long-duration op | No engineer to restart a crash | Supervisor auto-restart; faulthandler; liveness/stall detection |
| Deadline-bound output | A late fix can be worse than no fix | Deterministic mode; per-frame latency stats; (deferred) drop policy |

---

## 2. Controls implemented (this branch)

| ID | Control | Mechanism | Config flag | Default |
|---|---|---|---|---|
| P0-1 | No pickle-RCE via CESP weights | `weights_only=True` on `torch.load` | — (always) | on |
| P0-2 | Native-crash breadcrumb | `faulthandler` → `logs/faulthandler.log` | — (always) | on |
| P0-3 | Crash auto-recovery (headless) | Supervisor child-process + backoff | `--supervise` (+ `--max-restarts`) | off |
| P0-4 | Telemetry not public by accident | Fail-closed bind on routable host; auto-gen token | `network_api.api_token` | localhost-open, remote fail-closed |
| P0-5 | Mission trail not leaked via logs | Coordinate redaction helper | `models.performance.redact_coords_in_logs` | off |
| P1-7 | Telemetry confidential in transit | Optional TLS (`wss`/`https`), fail-closed | `network_api.tls_enabled` + `tls_certfile`/`tls_keyfile` | off |
| P1-9 | Honest operating state | State machine in `/api/status` | `network_api.expose_operating_state` | off |
| P1-10 | Detect a hung pipeline | WS heartbeat + stall (`LOST`) detection | same flag + `fix_stale_sec`, `heartbeat_interval_sec` | off |
| §4a | Detect content-blind OF coast | Anchor-staleness `DEGRADED` (fresh-keyframe clock) | `network_api.propagation_stale_sec` | off |
| P1-8 | Bounded worst-case latency (partial) | Deterministic cuDNN + latency percentiles | `models.performance.deterministic`, `log_latency_stats` | off |
| P1-6a | Whole project unreadable on capture | Passphrase AES-256-GCM at-rest, every file; immutable copy | `scripts/encrypt_project.py --project/--output` or GUI | off (plaintext master) |
| P2-12 | Weight tamper / swap detection | SHA-256 manifest preflight, fail-closed | `models.performance.weight_integrity_mode` | `off` |

### Behavior notes

- **P0-4 fail-closed:** binding a routable host (e.g. `0.0.0.0`) without a token
  now **refuses to start** (was: warning only). If a remote host is configured
  with no token, `CoordinatesBroker` **auto-generates** one and logs it — the app
  self-heals to secure rather than crashing, and localhost stays tokenless.
- **P1-7 TLS is fail-closed:** `tls_enabled=true` with a missing/invalid
  cert-key pair aborts startup rather than silently serving plaintext.
- **P1-8 is a partial:** the deterministic toggle + latency observability are
  done; the per-frame **deadline + frame-drop policy is deferred** (needs a
  consumer SLA — see §5).
- **P2-12 is a preflight, not per-load hooks:** all pinned weights are checked in
  one pass at startup. This also doubles as a go/no-go self-test on missing weights.

---

## 3. Enabling a "field" profile

To harden a fielded build, set in `user_config.json` (values are examples):

```jsonc
{
  "network_api": {
    "tls_enabled": true,
    "tls_certfile": "certs/telemetry.crt",
    "tls_keyfile":  "certs/telemetry.key",
    "api_token":    "<pin a fixed token, or leave empty to auto-generate on remote bind>",
    "expose_operating_state": true,
    "fix_stale_sec": 3.0,
    "propagation_stale_sec": 8.0
  },
  "models": {
    "performance": {
      "redact_coords_in_logs": true,
      "deterministic": true,
      "log_latency_stats": true,
      "weight_integrity_mode": "enforce"
    }
  }
}
```

And launch headless with supervision:

```
python main.py --headless --project <dir> --source <src> --supervise
```

### Operational setup

- **Weight manifest (P2-12):** after staging weights offline, run
  `python scripts/generate_weights_manifest.py` to write
  `models/weights_manifest.json`. Regenerate whenever weights legitimately change
  (e.g. rebuilding a TensorRT `.engine` on new hardware).
- **TLS cert (P1-7):** for a closed network, a self-signed cert is sufficient
  (`openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=<host>"`).
  Clients must trust/pin it.
- **Token (P0-4):** if `api_token` is empty and a routable host is configured, the
  generated token is logged at WARNING on startup — capture it for clients
  (`Authorization: Bearer <token>` or `?token=<token>`), or pin a fixed one.
- **Map encryption (P1-6a):** build a deployment copy with
  `python scripts/encrypt_project.py --project <src> --output <dst>` (prompts for
  a passphrase twice), or use Файл → «Створити зашифровану копію...» in the GUI.
  Every file in the copy is encrypted, including `project.json`; the plaintext
  master is untouched and stays at the ground station. The copy is **immutable** —
  the app refuses rebuilds, calibration saves, propagation and result exports into
  it. Opening it prompts for the passphrase before anything is loaded; a headless
  run prompts on the terminal. There is **no environment-variable channel**: an
  env var is readable by any same-user process, is inherited by every child, and
  can surface in a crash dump. Keep the passphrase safe — it is never stored and
  cannot be recovered.

---

## 4. Verification evidence (this session)

Verified by executing the actual behavior (not just import), plus the green
`config/headless/network/hardware` unit baseline (44 tests passing):

- P0-1: CESP load path imports with `weights_only=True`.
- P0-2: breadcrumb file written; `faulthandler.is_enabled()` true.
- P0-3: supervisor loop restarts N times → gives up (rc 1); clean child exit → stop (rc 0).
- P0-4: raises on `0.0.0.0` without token; localhost tokenless allowed; broker auto-gen token shared WS↔REST.
- P0-5 / P1: `fmt_coord` redacts when flag on, full precision off.
- P1-7/9/10: live `https` `/api/status` and `wss` heartbeat over a self-signed cert; all 6 state transitions (IDLE→ACQUIRING→TRACKING→DEGRADED→LOST→IDLE); `/api/status` unchanged when flag off.
- P1-8: `LatencyTracker` percentiles exact on a known distribution; interval logging fires on boundary.
- P2-12: manifest excludes cache/non-weights; tamper→MISMATCH; enforce raises on tamper and on missing manifest; warn/off semantics correct.
- P3-15 (first slice): fault injector deterministic per seed; schedule > window >
  probability precedence; each fault (corrupt/black/freeze/shape/delay/eof/exception)
  transforms correctly; FREEZE repeats the *previous* good frame; harness
  transition-report math (dwell, LOST episodes + recovery) correct. **22 tests
  passing** (`tests/test_fault_injection.py`).

**Not verified here (needs a Windows/GPU run):** deterministic cuDNN effect and
latency percentiles under real load; supervisor against a real native crash;
integrity preflight over the real ~640 MB weight set.

---

## 4a. Empirical finding — coverage of the LOST detector (P1-10)

Established by running the P3-15 harness against the real pipeline (RTX 5070 Ti,
project `newzap`, ~3600-frame clip). Each row is a full run; `max_fix_age_sec` is
the peak staleness the fix clock reached against its 3.0 s `fix_stale_sec`.

| Scenario | Injected | max_fix_age | LOST? | Meaning |
|---|---|---|---|---|
| `clean` | — | 0.59 s | no | baseline; a fix lands ~every 0.6 s |
| `stall` | 121 freeze + 70 delay | — | no | a frozen frame is still *valid* → it re-localizes |
| `blackout` | 401 black frames | 0.67 s | no | **bad frame content does not starve the fix clock** |
| `linkstall` | one 6 s silent gap | 5.86 s | **yes — 2.86 s, then recovered** | detector fires on true fix absence and recovers |

**Conclusion — the LOST detector is sound but content-blind:**

- ✅ **Absence of fixes is caught.** A 6 s link stall drove `TRACKING → LOST` at
  the 3 s threshold and recovered to `TRACKING` when frames resumed (2.86 s
  episode). The heartbeat/stall path (P1-10) works as designed.
- ⚠️ **Degraded frame *content* is invisible.** 401 black frames moved the fix
  clock by only 0.08 s versus the clean baseline (0.67 vs 0.59 s) — the pipeline
  keeps emitting a position on unlocalizable frames (most likely optical-flow
  propagation; mechanism inferred, not yet traced). So a camera that degrades to
  garbage *while still delivering frames* would keep the payload reporting
  `TRACKING` with a possibly-wrong fix, and **would not raise LOST**. This is a
  silent-failure mode.

**The existing DEGRADED knobs do NOT close this gap (measured, not assumed).**
The state machine's `DEGRADED` branch keys off `network_api.degraded_min_inliers`
/ `degraded_min_confidence`. A blackout run instrumented for worst-case fix
quality showed **`min_inliers` = 1318, `min_confidence` = 0.506 over the whole
run** — i.e. every fix emitted *during* the 401 black frames carried
healthy-tracking quality numbers. The position is optical-flow propagated from
the last good keyframe and **inherits that keyframe's quality metadata**, so the
inlier/confidence signal the `DEGRADED` branch inspects is stale, not the current
frame's. No threshold can separate blackout from clean (it would need
`> 1318` inliers, which flags healthy frames too).

**Gap CLOSED — anchor-staleness DEGRADED signal (implemented, flag-gated, verified).**
The pipeline already distinguishes a fresh keyframe localization from an
optical-flow coast (`loc_result["is_of"]`). A second staleness clock, symmetric
to the `LOST` fix-clock, now watches the *fresh keyframe anchor*: if tracking
coasts on OF with no fresh anchor for longer than
`network_api.propagation_stale_sec` (default `0.0` = off), the state machine
reports `DEGRADED` — even while the propagated fix clock is still fresh. Design:
`docs/superpowers/specs/2026-07-26-anchor-staleness-degraded-design.md`.

Verified end-to-end on the harness (RTX 5070 Ti, `newzap`, `propagation_stale_sec = 6.0`):

| Run | anchor age (max) | fix age (max) | State outcome |
|---|---|---|---|
| `clean` (8 loops, 31 117 frames) | 4.375 s (< 6) | 0.453 s | `TRACKING` throughout — **0 false DEGRADED** |
| `blackout` (401 black frames) | **24.875 s** (> 6) | 0.672 s (< 3) | `TRACKING → DEGRADED` @20.4 s, recovered @39.4 s |

So blackout — invisible to both the fix clock (0.672 s, no `LOST`) and the
inlier/confidence knobs (frozen at 1318) — now honestly reads `DEGRADED`, while a
healthy 8-loop run does not false-positive. Precedence unchanged: `LOST`
(fix-clock) still outranks anchor-staleness `DEGRADED`.

**Tuning note:** set `propagation_stale_sec` above the *worst-case healthy* anchor
cadence with margin. Here the healthy peak was 4.375 s; 6.0 s worked but leaves
only ~1.6 s of margin — a fielded value of ~8–10 s trades slower DEGRADED
detection for zero false-positive risk. Default stays `0` (off) so the operator
tunes it against their own observed cadence.

**Residual:** a *partially* degraded frame that still yields a false keyframe DB
match (`is_of=False`) would reset the anchor clock and evade `DEGRADED`. Unlikely
for true blackout (no features — confirmed by the frozen inlier count); a pixel
gate would not reliably catch that case either.

---

## 5. Residual risks & deferred work (the bigger tasks)

Ordered by the plan's priority. These are **not** started; each needs either an
operator decision or a focused session with GPU-side validation.

| Item | Why deferred | What it needs to start |
|---|---|---|
| **P1-6 Encryption-at-rest — DONE** (SP1–SP3 + immutable copy). Remaining items are small and listed in §7 | — | — |
| **P1-8 Deadline + drop policy** | Needs a consumer SLA; touches the hot per-frame loop | Target FPS, max acceptable latency at the consumer, drop strategy (drop-oldest vs skip-to-latest) |
| **P2-11 Anti-tamper / dead-man zeroize** | Destructive (wipes map + keys); trigger policy is a decision | Trigger source (timeout / tamper switch / remote command); depends on P1-6 |
| **P2-13 Weight encryption / device binding** | Large; so captured `.engine/.pth` aren't reusable | Key model (shared with P1-6); per-load decrypt path |
| **P2-14 Embedded/edge build** | XL; gated on the hardware decision | Chosen target (e.g. Jetson), onboard-vs-groundstation compute, power/weight budget |
| **P3-15 Soak + fault-injection harness** | **First slice landed** — stream-level fault injection (`src/utils/fault_injection.py`) + harness (`scripts/soak_test.py`) that drives the real pipeline and reports operating-state/latency reaction. Remaining: resource-level scenarios (induced GPU-OOM / thermal) and the actual multi-hour Windows/GPU runs | A GPU box + a built project; run `python scripts/soak_test.py --project <dir> --source flight.mp4 --profile stall` and collect the report |

**Assumption most worth confirming (from the plan):** that adversary capture is
in-scope. If this is a training/civilian build where capture is not a concern,
P1-6 / P2-11 / P2-13 drop sharply in priority and the effort re-centers on
reliability + latency.

---

## 6. Change log

- `hardening/p0` branch: implemented P0 (1–5), P1 (7, 9, 10, and the safe slice of
  8), and P2-12. All flag-gated; `user_config.json` unchanged by the code.
- P3-15 first slice: stream-level fault injection + soak harness
  (`src/utils/fault_injection.py`, `scripts/soak_test.py`,
  `tests/test_fault_injection.py`). A bench test rig, not shipped in the app —
  no config flag, no production-path change.
- P1-6 SP1 (geo-anchor encryption-at-rest): passphrase-derived AES-256-GCM
  foundation (`src/security/at_rest.py`, Scrypt KDF), auto-detected decrypt-on-load
  hook in `MultiAnchorCalibration.load`, operator tool `scripts/encrypt_project.py`,
  new `cryptography` dependency. Fail-closed (no/wrong passphrase → `EncryptionError`);
  plaintext projects byte-for-byte unchanged (auto-detect). 10 unit tests
  (`tests/test_at_rest.py`) + verified end-to-end (encrypted `newzap` calibration
  loads 8 anchors identically to plaintext). Design:
  `docs/superpowers/specs/2026-07-26-encryption-at-rest-geo-anchors-design.md`.
- P1-6 completed (SP2, SP3, encrypted-copy model, immutability). `database.h5`
  decrypts into a `BytesIO` on open (plaintext DBs keep the unchanged lazy path);
  `vectors.lance/` materialises into a PID-stamped temp directory wiped on close,
  with a startup sweep for directories left by a crashed run; the keypoint video
  decrypts to a temp file for the calibration dialog's lifetime.
  `scripts/encrypt_project.py --project/--output` (and Файл → «Створити
  зашифровану копію...») builds a **fully encrypted copy — every file, no
  allowlist** — leaving the plaintext master untouched. `project.json` is
  encrypted too, so it is the marker: detection and the passphrase prompt run
  before the project is loaded. The copy is **immutable**:
  `assert_project_writable` refuses rebuilds, calibration saves, propagation and
  result exports, in the core and again in the GUI with a clear message. The
  `DRONELOC_PASSPHRASE` environment channel was **removed** — the passphrase
  comes from the GUI dialog or a verified console prompt with retries.
  ~60 tests across `test_at_rest`, `test_project_scan`, `test_db_encryption`,
  `test_encrypt_project`; verified end-to-end in both the GUI and headless.
- §4a anchor-staleness `DEGRADED`: closes the content-blind gap found by P3-15.
  New `anchor_fix` worker signal (fired only on a fresh keyframe anchor) →
  broker's `on_anchor_fix` clock → `get_operating_state` DEGRADED branch, gated on
  `network_api.propagation_stale_sec` (default 0 = off). Unit-tested
  (`tests/test_operating_state.py`, 8 tests) + verified end-to-end on the harness
  (blackout → DEGRADED; 8-loop clean → no false positive).

---

## 7. Encryption-at-rest (P1-6) — what is left

The feature is complete and verified in both the GUI and headless. What remains
is small, and none of it blocks field use of an encrypted copy.

### Needs a decision

| Item | Detail |
|---|---|
| `--supervise --headless` on an encrypted project | The supervised child inherits the parent's stdin, so from a terminal it re-prompts on **every restart**, and with no TTY it fails closed on the first start. Unattended supervised operation on an encrypted project does not work today. Fix is a stdin pipe (supervisor prompts once, feeds each child via `subprocess.run(..., input=...)`, plus a non-TTY branch in `get_passphrase`) — **never** a new environment variable. |

### Undeclared dependencies (venv has them, `pyproject.toml` does not)

| Package | Used by | Effect if missing |
|---|---|---|
| `psutil` | `sweep_stale_lance_tempdirs` (owner-PID liveness), `hardware_profile` | The startup sweep degrades to "cannot tell" and never deletes — safe, but stale decrypted indexes accumulate |
| `triton-windows` (3.7.1.post27) | `torch.compile` / inductor on Windows | `_is_torch_compile_supported` returns False and compilation is silently disabled. Third-party community build, not from the PyTorch project — pin the exact version if declared. Verified working with torch 2.12.0.dev20260329+cu128 (inductor compiles on CUDA, values match eager) |

Note: `_is_torch_compile_supported` gates on `import triton` alone, so **any**
importable Triton enables compilation — including a version incompatible with the
installed torch, which is the crash the guard was written to prevent. Re-run the
compile smoke test after any torch upgrade.

### Operator housekeeping

- Copies built before the manifest was encrypted (plaintext `project.json`,
  encrypted artifacts) are still detected and still write-guarded, but their
  manifest — mission name, video path, camera parameters — is readable at rest.
  Rebuild them with the current builder.
- Temp directories created before the PID stamp existed carry no `.owner` file
  and are therefore never swept. Delete them by hand once.

### Unconfirmed

- `moov atom not found` from ffmpeg while loading the keypoint video in the
  calibration dialog. Proven **not** caused by encryption — the master mp4 is
  well-formed (`ftyp`/`free`/`mdat`/`moov`) and the encrypted copy is exactly +52
  bytes (36-byte header + 16-byte GCM tag), i.e. a byte-exact round trip. Still
  needs one calibration open on a plaintext project to confirm it is pre-existing.
