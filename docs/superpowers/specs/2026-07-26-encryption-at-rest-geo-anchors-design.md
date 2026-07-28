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

- **GUI passphrase dialog — DONE.** See "Revision 2026-07-28" below.
- **Copy builder — DONE.** `scripts/encrypt_project.py` rewritten to
  `--project/--output`: `build_encrypted_copy` walks the tree, encrypts the five
  sensitive artifacts (incl. every file under `vectors.lance/`), copies the rest
  verbatim, refuses an existing output, never touches the source master. Pure
  function + 2 tests (`tests/test_encrypt_project.py`). The copy is fully
  encrypted on the write side; lance/mp4 still need SP3 load hooks to run.
- **SP2 load hook — DONE.** `open_maybe_encrypted_h5` in `database_loader.py`
  peeks the `MAGIC` header: plaintext DBs open on the unchanged lazy path (zero
  overhead), encrypted DBs decrypt whole into a `BytesIO` and open from RAM. Kept
  alive via `DatabaseLoader._decrypted_buf`. 3 tests (`tests/test_db_encryption.py`).
  Note: an encrypted DB is read-only in the field (the propagation close→write→
  reload rewrite path assumes plaintext — fine, calibration/rebuild happens on the
  plaintext master).
- **SP3 load hooks** — `vectors.lance/` and `database_keypoints.mp4`:
  decrypt-to-temp-dir, secure-wipe on exit. Accepted trade-off: transient
  plaintext on the temp disk while the app runs (protected at-rest; exposed only
  under live capture, same class as the in-RAM passphrase).

The `src/security/at_rest.py` foundation, `MultiAnchorCalibration.load` hook, and
`tests/test_at_rest.py` from SP1 all carry forward unchanged.

## Revision 2026-07-28 — GUI passphrase dialog + artifact-path correction

The operator runs the GUI, not the headless runner, so `DRONELOC_PASSPHRASE` and
the `getpass` fallback are both unusable: under a GUI launch stdin still looks
like a TTY, so `get_passphrase` blocks the process on a prompt rendered behind
the window. The GUI must therefore resolve the passphrase itself and inject it.

### Defect fixed first: artifacts are per-source, not at the project root

`ProjectSettings` defaults are `sources/main/database.h5` and
`sources/main/calibration.json`, and `vectors.lance` / `<db_stem>_keypoints.mp4`
are created next to the database. The copy builder matched `SENSITIVE_FILES`
only when `rel_dir == Path(".")`, so for every real project it encrypted
`vectors.lance/**` but left the geo-anchors and the map database in plaintext
inside a copy advertised as encrypted. The root-only fixture hid this.

`_is_sensitive` now matches by basename at **any** depth, plus
`SENSITIVE_SUFFIXES = ("_keypoints.mp4", ".h5")` — a source may declare its own
database filename (`db_area2.h5`), and the keypoint video is named after it, so
a literal-name match let a renamed map escape encryption. Covered by
`test_encrypts_per_source_artifacts` (nested two-source tree, 7 artifacts).

### Passphrase injection API (`src/security/at_rest.py`)

- `set_passphrase(pw)` — fill the process-wide cache; rejects empty.
- `clear_passphrase()` — drop it; called on every project open so a passphrase
  never carries from one project to the next.
- `verify_passphrase(path, pw) -> bool` — full decrypt of one artifact; False on
  wrong passphrase, tampering, or plaintext. Caches nothing.

`get_passphrase` is unchanged — it simply finds the cache filled and never
reaches `getpass`.

### Detection (`src/security/project_scan.py`, Qt-free)

`candidate_artifacts(project_manager)` resolves paths through
`settings.get_enabled_sources()` (never hard-coded roots), ordered
cheapest-to-decrypt first: calibrations → databases → lance/keypoints. A lance
dataset is a directory, so one file inside it is probed. Only the first 64 bytes
of each file are read — a 67 MB database is classified without loading it.
`find_encrypted_artifacts` filters to those carrying the container header.

**An empty result means the project is plaintext and loads exactly as before —
no prompt, no behaviour change.** This is why the feature needs no
`user_config.json` flag: it self-gates on the presence of ciphertext.

### Dialogs (`src/gui/dialogs/passphrase_dialog.py`)

- `PassphraseDialog` — modal, password echo, up to 3 attempts. Each attempt runs
  `verify_passphrase` against `found[0]` (calibration.json, a few KB) and calls
  `set_passphrase` **only after** a proven decryption; a typo can therefore never
  poison later loads in the session.
- `NewPassphraseDialog` — passphrase + confirmation for a new encrypted copy.
  The result is deliberately not cached: the session keeps working against the
  plaintext master.

### Integration (`src/gui/mixins/database_mixin.py`)

`_open_project` calls `clear_passphrase()`, then `load_project`, then
`_prompt_passphrase_if_encrypted()` **before** any artifact is touched. Cancel or
exhausted attempts abort the load with a status-bar message rather than failing
deep inside h5py with an opaque error.

`on_create_encrypted_copy` (menu Файл → «Створити зашифровану копію...») picks a
destination, prompts for a new passphrase, and runs `build_encrypted_copy` in
`EncryptCopyWorker` (`src/workers/encrypt_copy_worker.py`). Whole-file AES-GCM
over hundreds of MB would freeze the GUI thread, hence the `QThread`.

### Testing

`tests/test_project_scan.py` (10 tests) covers detection on plaintext /
encrypted / unloaded projects, per-source path resolution, verification order,
and the set/clear/verify API — including an autouse fixture resetting the
process-wide cache. The `QDialog` classes themselves are verified by a manual
GUI run. 28 tests green across the four encryption suites.

### Residual risk

Verification proves only that `found[0]` decrypts. A copy whose artifacts were
encrypted under different passphrases would pass the dialog and fail later on
the database. `build_encrypted_copy` uses one passphrase per run, so this cannot
arise from the supported workflow.

---

## Revision 2026-07-28b — SP3 load hooks (lance index + keypoint video)

The first GUI run exposed the gap: the encrypted copy built fine (12 artifacts),
but opening it would have failed at `lancedb.connect` on an encrypted index —
`_load_hot_data` has no fallback there. The copy was write-complete and
load-incomplete, so SP3 was promoted from "next session" to blocking.

### `vectors.lance/` — `materialize_maybe_encrypted_lance` (`database_loader.py`)

LanceDB opens a *filesystem directory* and manages its own handles, so there is
no in-RAM route like SP2's. Plaintext index → opened in place, unchanged path,
zero overhead (detected from the first file's header). Encrypted index →
materialised into `tempfile.mkdtemp(prefix="droneloc_lance_")` preserving the
dataset layout, and `lancedb.connect` points at the temp copy. Returns
`(path_to_open, temp_dir_or_None)`; a failed decryption wipes the partial temp
tree before raising.

`DatabaseLoader._lance_tempdir` holds it; `close()` drops `lance_table` **before**
`wipe_tree` — an open dataset would otherwise pin the decrypted files on disk.

### `<db_stem>_keypoints.mp4` — `_materialize_keypoints_video` (`calibration_mixin.py`)

The video is opened by path by the reader, so an encrypted one is decrypted via
`decrypt_to_tempfile` before `CalibrationDialog` is constructed and wiped in a
`finally` around `exec()` — the plaintext exists only while the dialog is open.
A missing video stays a non-error (the dialog already handles absence).

### Testing

4 new tests in `tests/test_db_encryption.py`: plaintext index opens in place;
encrypted index materialises with layout preserved, wipes on close, leaves the
source ciphertext untouched; wrong passphrase leaves no `droneloc_lance_*` temp
dir behind; empty directory opens in place. 32 green across the four suites.

### Residual risk

Decrypted lance files and the keypoint video exist as plaintext on the temp disk
while the app runs — the accepted decrypt-to-use trade-off already recorded for
SP3. `wipe_tree`/`wipe_file` are best-effort on SSD/CoW filesystems. At-rest
(powered-off) protection is unaffected.

**Leak found in the first end-to-end GUI run and fixed:** the wipe is driven by
`DatabaseLoader.close()`, but `MainWindow.closeEvent` never closed the databases,
so even a *clean* exit left a populated `droneloc_lance_*` directory (≈500 KB of
plaintext global descriptors) on the temp disk. `closeEvent` now closes
`db_manager`/`database` before delegating to Qt. Regression test:
`test_close_wipes_lance_tempdir`.

Still open: a hard crash (kill, power loss) skips `closeEvent` entirely and
leaves the directory behind. Sweeping stale `droneloc_lance_*` directories at
startup is not implemented — it needs care not to delete a concurrently running
instance's directory.
