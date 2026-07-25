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
| Airframe loss / capture | Recovery by an adversary is realistic | (Deferred) encryption-at-rest; weight-integrity pinning; log redaction |
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
| P1-8 | Bounded worst-case latency (partial) | Deterministic cuDNN + latency percentiles | `models.performance.deterministic`, `log_latency_stats` | off |
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
    "fix_stale_sec": 3.0
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

**Not verified here (needs a Windows/GPU run):** deterministic cuDNN effect and
latency percentiles under real load; supervisor against a real native crash;
integrity preflight over the real ~640 MB weight set.

---

## 5. Residual risks & deferred work (the bigger tasks)

Ordered by the plan's priority. These are **not** started; each needs either an
operator decision or a focused session with GPU-side validation.

| Item | Why deferred | What it needs to start |
|---|---|---|
| **P1-6 Encryption-at-rest** (map + calibration) | Largest item; wraps the map-load path; key-management is an operator decision | Key model: passphrase-derived vs machine-bound (DPAPI/TPM); decrypt-to-RAM vs temp; co-design with cold-start latency |
| **P1-8 Deadline + drop policy** | Needs a consumer SLA; touches the hot per-frame loop | Target FPS, max acceptable latency at the consumer, drop strategy (drop-oldest vs skip-to-latest) |
| **P2-11 Anti-tamper / dead-man zeroize** | Destructive (wipes map + keys); trigger policy is a decision | Trigger source (timeout / tamper switch / remote command); depends on P1-6 |
| **P2-13 Weight encryption / device binding** | Large; so captured `.engine/.pth` aren't reusable | Key model (shared with P1-6); per-load decrypt path |
| **P2-14 Embedded/edge build** | XL; gated on the hardware decision | Chosen target (e.g. Jetson), onboard-vs-groundstation compute, power/weight budget |
| **P3-15 Soak + fault-injection harness** | Its own project; produces the reliability numbers that re-rank the above | Induced GPU-OOM / corrupt-frame / link-loss / thermal scenarios; multi-hour runs |

**Assumption most worth confirming (from the plan):** that adversary capture is
in-scope. If this is a training/civilian build where capture is not a concern,
P1-6 / P2-11 / P2-13 drop sharply in priority and the effort re-centers on
reliability + latency.

---

## 6. Change log

- `hardening/p0` branch: implemented P0 (1–5), P1 (7, 9, 10, and the safe slice of
  8), and P2-12. All flag-gated; `user_config.json` unchanged by the code.
