# Building a standalone (offline) Windows executable

Produces a self-contained `DroneLocalization.exe` (PyInstaller `--onedir`) that runs on a
machine with **no Python, no internet, and no model downloads**. The only external
requirement is an NVIDIA GPU driver for CUDA; without one the app falls back to CPU.

## What gets bundled

- The Python 3.11 runtime, all packages, and torch + CUDA runtime DLLs.
- `models/` — RDD, YOLO (`.pt`/`.onnx`/`.engine`), Depth-Anything weights.
- `third_party/` — RDD and Depth-Anything-V2 source.
- Model caches under `_internal/.cache/`: DINOv2 hub repo + checkpoint; ALIKED / SuperPoint /
  LightGlue checkpoints; the DINOv3 HuggingFace snapshot; **and** the DINOv3
  `trust_remote_code` modules.
- GUI resources (the map template).

At runtime the frozen build points `TORCH_HOME`/`HF_HOME` at the bundled cache and forces
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so nothing reaches the network. Writable data
(logs, `user_config.json`) goes to `%LOCALAPPDATA%\DroneLocalization`.

## Prerequisites

PyInstaller only bundles caches that already exist on the **build machine**, so build on the
fully working dev setup with every model downloaded.

1. Python 3.11 venv (`pyproject.toml` pins `>=3.10,<3.12`):

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   pip install pyinstaller
   ```

2. Pre-populate every model cache — run one full localization end-to-end so all models
   download into `%USERPROFILE%\.cache`:

   ```
   python main.py    :: open a project, run a video through localization once
   ```

   Confirm these exist afterward:
   - `models\depth_anything_v2_vits.pth`, `models\RDD-v2.pth`, `models\yolo11n-seg.pt`
   - `%USERPROFILE%\.cache\torch\hub\facebookresearch_dinov2_main`
   - `%USERPROFILE%\.cache\torch\hub\checkpoints\*.pth` (dinov2, aliked, superpoint, lightglue)
   - `%USERPROFILE%\.cache\huggingface\hub\models--facebook--dinov3-vitl16-pretrain-sat493m`
   - `%USERPROFILE%\.cache\huggingface\modules\transformers_modules\...` (DINOv3 code)

3. Pin the DINOv3 revision (silences the `trust_remote_code` warning and makes the offline
   module path deterministic). Find the cached commit hash:

   ```
   dir %USERPROFILE%\.cache\huggingface\hub\models--facebook--dinov3-vitl16-pretrain-sat493m\snapshots
   ```

   Put that folder name into `user_config.json`:

   ```json
   "global_descriptor": { "dinov3": { "hf_revision": "<that-commit-hash>" } }
   ```

   (or set the default in `config/models.py` -> `Dinov3ModelConfig.hf_revision`).

## Build

```
del DroneLocalization.spec           :: remove the stale, divergent spec first
python scripts\build_executable.py   :: default: --windowed, models included
```

- Do **not** pass `--no-models` for an autonomous build.
- Add `--console` for a one-off debug build that shows a terminal window.
- Output: `dist\DroneLocalization\DroneLocalization.exe` (expect several GB).

## Distribute

- Portable: zip `dist\DroneLocalization\` — runs from anywhere writable, no install.
- Installer: `iscc create_installer.iss` -> `dist\Install_DroneLocalization.exe` (installs to
  Program Files; works now that writable data is redirected to `%LOCALAPPDATA%`).

## Verify on a clean machine

On a PC with **no Python and no internet**:
1. Launch the exe; confirm it starts and the map view renders.
2. Run a localization; tail `%LOCALAPPDATA%\DroneLocalization\logs\app.log` — there should be
   no "downloading" / "connection" messages.
3. Confirm DINOv3, DINOv2, the local extractor (RDD/ALIKED), and (if used) depth all load.

## Notes / limits

- Cross-GPU: `use_tensorrt_for_yolo` defaults to **False** so YOLO runs on any GPU via the
  `.pt`; a bundled `.engine` is GPU-specific and won't load elsewhere.
- The NVIDIA driver is the one thing that can't be bundled; without a supported GPU the app
  runs on CPU (slower).
- The image is large (~several GB) because of torch + CUDA + weights — expected.
