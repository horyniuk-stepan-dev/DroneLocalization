# Config / Logging Unification — handoff (2026-07-15)

## Verdict
Config and logging usage are already uniform across `src/`, `config/`, `main.py`.
One real config-file drift was found and **fixed**. Two items remain, both needing
your Windows machine (the sandbox can't write the git index or delete files on D:).

## Done
- **`user_config.example.json`**: was 9 keys stale vs the Pydantic `AppConfig` model.
  Added at model DEFAULT values (preserving the 2 intentional example diffs
  `local_extractor="rdd"`, `use_tensorrt_for_yolo=true`):
  - `graph_optimization`: `temporal_edge_gate`, `temporal_gate_max_rotation_deg`,
    `temporal_gate_max_scale_ratio`, `temporal_gate_max_shift_frac`,
    `anchor_gap_check`, `anchor_gap_max_dev_m`, `anchor_gap_downweight`
  - `propagation`: `skip_bridges`, `mnn_fallback`
  - Result: 333/333 keys, matches the model. `user_config.json` already matched.

## Verified clean
- Logging: every module uses `logger = get_logger(__name__)` (loguru). No stdlib
  `logging`. Only 2 `print()` in `config/access.py` (load/save error paths, run
  before `setup_logging()` — defensible).
- Config: canonical `from config import ...` everywhere; no `config.config` shim
  usage; reads via `get_cfg()` (~198 sites); direct `APP_SETTINGS.` only for
  legit writes/bootstrap.

## TODO (needs you, on Windows)
1. **Line endings** — 38 of 107 `.py` files are CRLF; `.gitattributes` mandates
   `eol=lf`. Fix cleanly with:
   ```
   git add --renormalize .
   git commit -m "Normalize line endings to LF per .gitattributes"
   ```
   (Do NOT hand-rewrite the 38 files — that creates the exact noise .gitattributes
   exists to prevent.)
2. **Delete cruft**:
   ```
   git rm config/_wtest.tmp config/models.py.bak-preextractorfix \
          src/database/database_builder.py.corrupt-bak __deltest.tmp
   ```

## Optional hardening
- Add a test asserting `user_config.example.json == AppConfig()` defaults except
  the 2 intentional keys — the current sync tests never check the example file,
  which is why it drifted.
- Switch the 2 `print()` in `config/access.py` to `logger` for strict uniformity.

## Flag to confirm
- `user_config.json` has `temporal_edge_gate=true` and `anchor_gap_check=true`
  (model defaults are `false`). Confirm these were deliberately enabled.
