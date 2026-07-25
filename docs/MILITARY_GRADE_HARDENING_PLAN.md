# Military-Grade Hardening Plan — DroneLocalization

**Status:** roadmap + gap analysis. No code changed by this document.
**Scope (per operator decision):** field reliability, security / anti-tamper, edge performance. Standards/certification paperwork is explicitly out of scope for now. Target hardware is undecided, so the plan covers both the current desktop (GTX 1650, 4 GB) and a future embedded/edge port.
**Method note:** every "current state" claim below is grounded in a specific file/line read this session. Effort and severity ratings are engineering judgement (estimates), labelled as such — they are not measured.

---

## 0. Executive verdict

The system is **well-architected and already security-aware, but not yet field-survivable.** For a GPS-denied navigation payload — whose entire reason to exist is operating in a contested, jammed, possibly-capturable environment — the gap to "military grade" is concentrated in five places, in priority order:

1. **Process survivability.** An unhandled or native crash logs once and the process exits. There is no watchdog, no auto-restart, no native-fault capture. In the field, one segfault = mission over, silently.
2. **Data-at-rest / anti-tamper.** Nothing is encrypted at rest. If the airframe is recovered by an adversary, the reference-map database, the model weights, the calibration, and the config are all readable and reusable. This is the single largest divergence from "military grade."
3. **Telemetry confidentiality.** Position broadcast (WebSocket + REST) has token auth in code but it is **off by default** (`api_token = ""`) and there is **no TLS** — position is plaintext on the wire.
4. **Bounded, deterministic latency.** Performance is auto-tuned for throughput, not for a guaranteed per-frame deadline. There is no overload/frame-drop policy and `cudnn.benchmark` deliberately trades determinism for speed.
5. **Verification evidence.** There is a green unit-test baseline but no reliability/soak/fault-injection evidence — the thing that lets someone sign off on fielding it.

What is **already strong** and should not be rebuilt: the localization-level degradation stack (retrieval-only fallback, outlier rejection, Kalman smoothing, FOV-explosion guard, VO anchor-gap guards), the config subsystem (atomic writes, `fsync`, loud fail-to-defaults instead of silent substitution), the model-weight loading (`weights_only=True` at almost every site), and the network layer's defensive defaults (localhost-only, explicit warnings before exposing telemetry).

**Bottom line:** this is a hardening job, not a rewrite. The P0 items below are small, high-leverage, and mostly flag-gated.

---

## 1. What "military grade" means for *this* system

"Military grade" is not a certification you buy; it is a set of properties tied to a threat model. For a GPS-denied localization payload the working threat model is:

- **Contested RF / EW environment.** GPS is denied or spoofed (that is the product's premise); the network link is hostile or absent; the operator may need full air-gap.
- **Airframe loss is expected.** UAVs get shot down, crash, or land in hostile territory. Recovery by an adversary must not hand them the reference map, the models, or the mission trail.
- **Unattended, long-duration operation.** No engineer at a keyboard to restart a crashed process. The payload must degrade, recover, and keep producing a usable position or an honest "I am lost" signal.
- **Deadline-bound output.** A late position estimate is often useless. Worst-case latency matters more than average latency.

The three chosen axes map to that threat model:

| Axis | Property it buys | Failure it prevents |
|---|---|---|
| Field reliability | Survivability & graceful degradation | Silent death, unrecoverable hang, undetected drift |
| Security / anti-tamper | Confidentiality & capture-resistance | Adversary reads/reuses the map, weights, mission trail; telemetry sniffed/spoofed |
| Edge performance | Bounded, deterministic real-time | Missed deadlines, thermal/VRAM collapse, jitter |

*(The fourth axis you did not pick — MIL-STD / DO-178C-style documentation and traceability — becomes relevant only if this goes to a formal acceptance process. It is deferred, not dismissed; several P0/P1 items below produce the artifacts such a process would later demand.)*

---

## 2. Gap analysis — field reliability

**What exists today (verified):**
- Global exception hook: `main.py:75-98` logs unhandled exceptions as `CRITICAL` with traceback, then `sys.exit(1)`.
- Headless graceful shutdown on `SIGINT`: `src/core/headless_runner.py:170`.
- Startup prewarm is fault-tolerant — failure downgrades to lazy load (`main.py:67-72`).
- Config load is robust: corrupt config fails **loudly to built-in defaults** (never silently substitutes another file); atomic write + `fsync` (`config/access.py:83-151`).
- Localization degradation is mature: retrieval-only fallback, `OutlierDetector`, Kalman `TrajectoryFilter`, FOV-explosion guard, VO anchor-gap guards (`docs/architecture.md` flow; `src/localization/localizer.py`, `src/tracking/`).

**Gaps:**

| Requirement | Current state | Gap | Severity |
|---|---|---|---|
| Survive & auto-recover from a crash | Logs then `exit(1)`; no supervisor | No watchdog / auto-restart wrapper | **High** |
| Survive **native** crashes (CUDA/cv2/torch segfault) | No `faulthandler` (grep: none) | Native faults leave no trace, no recovery | **High** |
| Detect a *hung* (not crashed) pipeline | `/api/status` reports tracking/idle+uptime but nothing acts on it | No liveness heartbeat / stall detector | Med-High |
| Honest "I am lost" state | Confidence + outlier reject exist internally | No explicit degraded/lost operating mode surfaced to consumers | Medium |
| Bounded resource growth over long missions | `deque(maxlen=1000)` history; VRAM LRU | Log-file growth, HDF5 handle lifetime, VRAM fragmentation over hours unproven | Medium |
| Deterministic startup on missing deps | Warnings exist | No preflight self-test / go-no-go check | Medium |

**Highest-cost item:** no process-level survivability. Everything downstream (telemetry, tracking) assumes the process is alive; nothing guarantees it.

---

## 3. Gap analysis — security / anti-tamper

**What exists today (verified):**
- Network servers default to **127.0.0.1** and emit an explicit warning if bound to a routable host without a token (`src/network/rest_server.py:23-28`, `src/network/ws_server.py:26-31`).
- Token auth implemented for both REST (Bearer) and WS (query or Bearer): `rest_server.py:37-43`, `ws_server.py:40-53`.
- Model weights loaded with `weights_only=True` at nearly every site (`model_manager.py:460`, `depth_estimator.py:124`, `rdd_wrapper.py:79`) — safe against pickle-RCE.

**Gaps:**

| Requirement | Current state | Gap | Severity |
|---|---|---|---|
| Reference map DB unreadable if captured | HDF5 + LanceDB plaintext on disk | No encryption-at-rest (grep: no crypto beyond a `sha256` schema fingerprint) | **Critical** |
| Model weights unreadable/unusable if captured | `.pth/.onnx/.engine` plaintext (~640 MB) | No weight encryption or binding to device | **High** |
| Telemetry confidential in transit | No TLS anywhere (grep: no `ssl`/certs) | Plaintext WS/REST — position sniffable even with a token | **High** |
| Telemetry auth on by default | `api_token = ""` default (`config/app.py:86`, `user_config.json:19`) | Auth is opt-in; a field build likely ships with it off | **High** |
| No RCE via swapped weight file | One unsafe load: `model_manager.py:730` (CESP) lacks `weights_only=True` | Latent pickle-RCE — **currently dormant** (`cesp.weights_path = null`, so the branch doesn't run) | Medium (latent) |
| Mission trail not leaked via logs | Map-click lat/lon logged at INFO (`main_window.py:169`); anchor/GSD info logged. Continuous live position **not** found logged at INFO | Partial: interactive coords + anchors persist in `app.log`; needs a logging-redaction policy | Medium |
| Remote wipe / dead-man on capture | None | No anti-tamper trigger to zeroize map/keys | Medium (mission-dependent) |
| Video ingest integrity | `video_source.py` supports plaintext `http://` MJPEG | Feed can be sniffed/spoofed on an untrusted link | Low-Med |
| Supply-chain / air-gap | DINOv3 pulled from HuggingFace at first run (`dinov3_wrapper.py:48`) | First-run network dependency; needs verified offline pre-staging + weight hash pinning | Medium |

**Highest-cost item:** encryption-at-rest of the reference-map database. The map is the crown jewel — it is expensive to build, reveals the operational area, and is directly reusable by an adversary. Everything else in this table is secondary to that.

**Honest calibration:** the CESP unsafe-load (`model_manager.py:730`) is real but **dormant** — the code path only runs if someone sets `models.cesp.weights_path`, and it is `null`. It is a one-line fix (`weights_only=True`) and belongs in P0 precisely because it is trivial, but it is not currently exploitable.

---

## 4. Gap analysis — edge performance

**What exists today (verified):**
- Hardware auto-detection + auto-tune on by default (`HardwareProfile.detect()/auto_tune()`, `main.py:115-139`; `user_config.json` `auto_tune=true`).
- Backend tuning applied (TF32, `cudnn.benchmark`, thread counts — `hardware_profile.apply_torch_backends()`).
- `fp16_enabled=false` (correct for GTX 1650, where fp16 is ~4× slower), `torch_compile=true`.
- TensorRT engines (`.engine`) and ONNX LightGlue present; `ModelManager` runs an LRU VRAM budget with eviction/pinning for 4 GB cards.
- Retrieval prefilter + temporal prior + optical-flow stride already committed and validated on a live mission.

**Gaps:**

| Requirement | Current state | Gap | Severity |
|---|---|---|---|
| Guaranteed per-frame deadline | Throughput-tuned, not deadline-tuned | No latency budget / deadline scheduler; late frames not dropped by policy | **High** |
| Deterministic latency (low jitter) | `cudnn.benchmark=True` → nondeterministic kernel choice, variable first-call cost | No deterministic-mode option for worst-case bounding | Medium |
| Graceful behavior under thermal/power limits | None specific | No thermal/throttle awareness or degraded-rate mode | Med (High on embedded) |
| Bounded worst-case VRAM under load | LRU budget exists | Fragmentation/peak over long runs unproven; no hard cap enforcement | Medium |
| Embedded/edge target | Desktop-only today | No Jetson/edge build; ONNX/TensorRT path helps but is unproven off-desktop | Deferred (hardware TBD) |
| Startup-to-first-fix time | Prewarm exists | Cold-start latency (model load) not budgeted for rapid launch | Medium |

**Highest-cost item:** no bounded-latency contract. A navigation payload that sometimes returns a position 400 ms late is, for some consumers, worse than one that returns "no fix" on time. The pipeline needs an explicit deadline + drop policy.

---

## 5. Prioritized roadmap

Ordering is by **cost-of-error relative to effort**, not difficulty. Repo conventions are honored: new behavior ships **flag-gated with defaults = current behavior (off)**; enablement goes in `user_config.json`, never toggled silently; anything touching the git index or requiring a Windows GPU run is handed to you as an exact command. Effort tags are rough estimates.

### P0 — quick, high-leverage, low-risk (days)

1. **Fix the dormant unsafe load.** Add `weights_only=True` at `model_manager.py:730`. Closes a latent pickle-RCE for the cost of one keyword. *(S)*
2. **Add `faulthandler` + crash breadcrumb.** `faulthandler.enable()` writing to a rotating file at startup; capture native segfaults that the Python `excepthook` cannot. *(S)*
3. **Watchdog/supervisor wrapper for headless mode.** A thin external supervisor (or `--supervise` flag) that restarts the pipeline on non-zero exit with backoff and a crash counter. Turns "silent death" into "logged auto-recovery." *(M)*
4. **Ship telemetry auth-on by default for non-localhost.** Refuse to bind a routable host without a token (hard error, not just a warning); generate a token if none is set. Keep localhost frictionless. *(S-M)*
5. **Logging redaction policy.** A flag to suppress/round lat/lon and anchor coordinates in `app.log`; default-on for "field" profile. *(S)*

### P1 — the core military-grade properties (weeks)

6. **Encryption-at-rest for the reference map + calibration.** Encrypt HDF5/LanceDB artifacts (envelope encryption; key from an operator passphrase or hardware key). Decrypt into RAM/really-temp on load. This is the anti-tamper centerpiece. *(L)*
7. **TLS for WS/REST telemetry.** Optional `certfile/keyfile` on both servers; self-signed/pinned for closed networks. Pairs with item 4. *(M)*
8. **Bounded-latency pipeline contract.** Per-frame deadline config + drop/skip policy under overload + a `deterministic` toggle (disables `cudnn.benchmark`) for worst-case bounding. Emit latency histograms. *(M-L)*
9. **Explicit operating-state machine.** Surface `ACQUIRING / TRACKING / DEGRADED / LOST` to telemetry consumers, driven by the confidence/outlier signals that already exist. Consumers must be able to trust the "LOST" signal. *(M)*
10. **Liveness heartbeat + stall detector.** Detect a hung (not crashed) pipeline and feed the watchdog. *(M)*

### P2 — capture-resistance & edge port (mission-dependent)

11. **Anti-tamper / dead-man zeroize.** Optional trigger (timeout, tamper switch, remote command) that wipes decrypted map + keys. *(M-L)*
12. **Verified air-gap + weight pinning.** Offline pre-stage all weights; pin and verify hashes at load; fail closed if a weight mismatches. *(M)*
13. **Model-weight encryption / device binding.** So captured `.engine/.pth` are not directly reusable. *(L)*
14. **Embedded/edge build (only if hardware is chosen).** Jetson (or equivalent) build, TensorRT-first path, thermal-aware degraded mode, power budget. Large, gated on the hardware decision in §6. *(XL)*

### P3 — assurance evidence (parallel, feeds any future certification)

15. **Soak + fault-injection harness.** Multi-hour runs with induced GPU OOM, corrupt frames, link loss, thermal throttle; produce reliability numbers. *(M-L)*
16. **Threat model + security test doc.** Written artifact; the seed for later MIL-STD/RMF work if it comes. *(M)*

---

## 6. Hardware decision (currently open)

Most of P0–P1 and all of P3 are **hardware-independent** — do them regardless. Only item 14 forks:

| | Stay on desktop (GTX 1650 class) | Port to embedded/edge (e.g. Jetson Orin) |
|---|---|---|
| Effort | Low — already runs here | High — new build, TensorRT-first, thermal/power work |
| Real-time | Proven on live mission | Must be re-proven on target |
| SWaP (size/weight/power) | Poor for airborne payload | The reason to do it |
| Anti-tamper | Standard-PC assumptions | Can leverage secure-boot/TPM on some modules |

**Recommendation:** proceed with all hardware-independent hardening now; make the embedded decision on the basis of the actual airframe/payload constraints (weight, power, whether compute is onboard or on a groundstation). Do not block P0/P1 on it.

**Decision inputs needed from you:** onboard vs. groundstation compute? airframe power/weight budget? required frames-per-second and max acceptable latency at the consumer?

---

## 7. Risks, assumptions, and what would change this plan

- **Register-(c) estimates:** all effort/severity tags are judgement, not measurement. A soak test (item 15) could re-rank the reliability items — e.g. if VRAM fragmentation turns out to crash multi-hour runs, that jumps to P0.
- **Assumption — capture is in-scope.** The anti-tamper priority (§3, items 6/11/13) rests on "airframe recovery by an adversary is a real risk." If this is a training/civilian build where capture is not a concern, items 6/11/13 drop sharply in priority and the plan re-centers on reliability + latency. **This is the single assumption most worth confirming.**
- **Assumption — telemetry leaves the box.** If position is only ever consumed on the same host (localhost), TLS (item 7) and auth-by-default (item 4) matter far less. If it goes over any radio link, they are mandatory.
- **Encryption-at-rest cost:** decrypting a large map into RAM has a startup-latency and memory cost that fights item 8 (fast cold start, bounded VRAM). These two will need to be co-designed.
- **What would change the ordering:** a firm embedded-hardware decision pulls item 14 forward and reshapes the latency work around the target's TensorRT/thermal profile.

---

## 8. Provenance (what was verified vs. inferred)

- **Verified from code this session:** the exception hook and exit behavior, absence of `faulthandler`, network auth/localhost defaults and empty `api_token`, absence of any TLS/crypto-at-rest, the single unsafe `torch.load` and its dormant config, coordinate-logging sites and levels, the auto-tune/fp16/torch_compile settings, and the localization degradation stack.
- **Inferred / estimated (register c):** all effort sizes, all severity ratings, the threat model, and the claim that continuous live position is *not* logged (verified only that no INFO-level position log was found — an exhaustive trace of every worker path was not done).
- **Not done:** no code was changed; no benchmark or GPU run was performed (those are Windows-side and yours to run); no exhaustive audit of all 103 modules — reading was scoped to the load-bearing files for the three chosen axes.
