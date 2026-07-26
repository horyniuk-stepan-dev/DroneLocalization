# Encryption-at-rest — Sub-project 1: geo-anchors + crypto foundation

**Date:** 2026-07-26
**Branch:** `hardening/p0`
**Roadmap item:** P1-6 (encryption-at-rest), first of three sub-projects.
**Threat:** airframe loss / capture — an adversary recovering the payload must not
read the mission's operational area or map.

## Why this sub-project first

The "map" is four artifacts with different access patterns:

| Artifact | Size | Access | Leaks |
|---|---|---|---|
| `calibration.json` | 17 K | read whole-file at load | affine anchors → exact projected coords |
| `database_graph.geojson` | 100 K | **write-only** (debug export) | per-frame GPS in plaintext |
| `database.h5` | 67 M | h5py lazy random-access all session | keypoints/descriptors |
| `vectors.lance/` | dir | LanceDB opens a filesystem path | global descriptors |

The two small files leak the exact lat/lon of the operation in plaintext
(verified: `database_graph.geojson` holds `[34.926…, 47.820…]` per frame;
`calibration.json` holds Web-Mercator translation `3887719, 6077172`). They are
whole-file reads (or write-only), so they encrypt cleanly with **decrypt-to-RAM**
— no temp files, no lazy-access problem. This sub-project also builds the crypto
+ key foundation that `database.h5` (SP2) and `vectors.lance/` (SP3) reuse.

## Key model (decided)

**Passphrase-derived.** The threat is capture of the whole machine, so a
machine-bound key (DPAPI/TPM) travels with the captured hardware and gives no
protection. A passphrase keeps no key on the device. KDF: **Scrypt**
(n=2¹⁵, r=8, p=1) via the `cryptography` library — memory-hard, no extra
dependency beyond `cryptography`. AEAD: **AES-256-GCM** (authenticated; wrong
passphrase or tampered bytes fail closed).

## Components

### 1. `src/security/at_rest.py` (new — the foundation)

```
MAGIC = b"DLENC1\0"          # 7 bytes, identifies an encrypted container
container = MAGIC | ver(1B) | salt(16B) | nonce(12B) | ciphertext‖tag(16B)
```

- `derive_key(passphrase: str, salt: bytes) -> bytes` — Scrypt → 32-byte key.
- `encrypt_bytes(plaintext: bytes, passphrase: str) -> bytes` — fresh random
  salt+nonce, AES-256-GCM, returns the container.
- `decrypt_bytes(container: bytes, passphrase: str) -> bytes` — parse, derive,
  authenticated-decrypt. `InvalidTag`/short/bad-magic → `EncryptionError`.
- `is_encrypted(data: bytes) -> bool` — `data.startswith(MAGIC)`.
- `EncryptionError(Exception)` — fail-closed; never returns partial plaintext.

The module depends only on `cryptography` (no torch/Qt), so it is unit-testable
in the pure-Python suite.

### 2. Passphrase source — `get_passphrase()` (in `src/security/at_rest.py`)

Order: env `DRONELOC_PASSPHRASE` (headless/supervised — the parent holds it and
passes it to restarted children) → interactive `getpass` if stdin is a TTY →
else raise `EncryptionError` (an encrypted artifact was found but no passphrase
is available). Cached process-wide after first resolution so multiple artifact
loads reuse it.

### 3. Load hook — `src/calibration/multi_anchor_calibration.py`

`MultiAnchorCalibration.load` already reads the whole file into `content`
(line 373). Insert, before the JSON parse:

```python
if is_encrypted(content):
    content = decrypt_bytes(content, get_passphrase())
```

Auto-detection by header means a plaintext project loads exactly as today — the
default path is byte-for-byte unchanged. No config flag gates loading.

### 4. Tool — `scripts/encrypt_project.py`

`--project <dir>` [`--artifacts geo`]. Resolves `calibration.json` +
`database_graph.geojson` from the project (modern `sources/main/` layout, legacy
flat root fallback — mirror `HeadlessRunner`'s resolution). Prompts for the
passphrase twice (confirm). For each artifact: skip if already encrypted;
otherwise `encrypt_bytes`, **verify the round-trip in RAM** (decrypt == original)
before writing the container in place (same path/filename). Prints a summary.
`database_graph.geojson` is never read back at runtime, so it simply stays
encrypted; only an operator decrypts it to view.

### 5. Dependency

Add `cryptography` to `requirements.txt`. Industry-standard; provides AES-256-GCM
and Scrypt. Must be installed in the venv before the tests run.

## Testing

- **Unit `tests/test_at_rest.py`** (pure-Python, needs `cryptography`):
  - encrypt → decrypt round-trip returns the original bytes.
  - wrong passphrase → `EncryptionError` (never partial plaintext).
  - flipped ciphertext / truncated container / bad magic → `EncryptionError`.
  - `is_encrypted` true on a container, false on plaintext JSON.
  - two encryptions of the same plaintext differ (fresh salt+nonce).
  - `get_passphrase` reads the env var; raises when unset and no TTY.
- **Integration (Windows, GPU)** — encrypt `newzap`'s `calibration.json`, run
  headless with `DRONELOC_PASSPHRASE` set: loads and localizes identically to the
  plaintext baseline. Unset the env var → fails closed with a clear message.

## Backward compatibility & safety

- Plaintext projects: unchanged (auto-detect returns false).
- Fail-closed everywhere: no passphrase, wrong passphrase, or tampering raises;
  the app never silently proceeds on unauthenticated data.
- Deleting the plaintext originals is the operator's job (the tool encrypts in
  place after a verified round-trip; it does not leave a `.bak`).

## Out of scope (later sub-projects)

- `database.h5` in-RAM decrypt via h5py core/BytesIO (SP2).
- `vectors.lance/` directory decrypt-to-temp-dir + secure wipe (SP3).
- GUI passphrase dialog; a `security.require_encryption` fail-closed flag that
  refuses to run on plaintext artifacts when a fielded profile mandates
  encryption.

## Residual risk

The passphrase lives in process RAM while running; a live-memory capture of a
powered, unlocked payload could recover it. This is inherent to decrypt-to-use
and out of scope here (mitigation would be P2-11 dead-man zeroize). At-rest
(powered-off or disk-only capture) is fully protected.

---

## Revision 2026-07-26 — encrypted-COPY model + full-project scope (supersedes in-place)

After SP1 landed, the operator clarified the intended end state, which changes
two things:

1. **Encrypted copy, not in-place.** The tool must build a *separate encrypted
   copy* of the project (`--project <src> --output <dst>`), leaving the plaintext
   master untouched. Rationale: matches the deployment model (ground station
   keeps the plaintext master; the drone carries only the encrypted copy →
   capture yields ciphertext), is non-destructive, and removes the
   lose-passphrase-lose-everything danger of in-place. The current in-place
   `scripts/encrypt_project.py` is superseded by this copy builder (the
   `at_rest.py` foundation and the calibration load-hook are unchanged and reused).

2. **All sensitive artifacts, not just geo-anchors.** The encrypted copy encrypts:
   `calibration.json`, `database_graph.geojson`, `database.h5`, `vectors.lance/`,
   and **`database_keypoints.mp4`** (correction: this is used by *calibration*, not
   a throwaway debug artifact, so it is encrypted in the copy — the master keeps
   its plaintext). Non-sensitive files (`project.json`, etc.) are copied as-is.

3. **GUI passphrase dialog (no longer deferred).** The operator runs the GUI, so
   env-var-only passphrase is insufficient — a modal password prompt must appear
   at project load when encrypted artifacts are auto-detected, plus a GUI action
   to build an encrypted copy. This is required for the feature to be usable, not
   a later nicety.

### Remaining work (next session), in priority order

- **GUI passphrase dialog** wired into the project-load path (prompt when an
  encrypted artifact is detected; feed `at_rest.get_passphrase`'s cache).
- **Copy builder** — rewrite `scripts/encrypt_project.py` to `--project/--output`,
  walking the tree, encrypting the five sensitive artifacts, copying the rest.
- **SP2 load hook** — `DatabaseLoader`: decrypt `database.h5` in-RAM (h5py
  `BytesIO`/`core` driver) when the file is an encrypted container.
- **SP3 load hooks** — `vectors.lance/` and `database_keypoints.mp4`:
  decrypt-to-temp-dir, secure-wipe on exit. Accepted trade-off: transient
  plaintext on the temp disk while the app runs (protected at-rest; exposed only
  under live capture, same class as the in-RAM passphrase).

The `src/security/at_rest.py` foundation, `MultiAnchorCalibration.load` hook, and
`tests/test_at_rest.py` from SP1 all carry forward unchanged.
