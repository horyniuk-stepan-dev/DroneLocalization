#!/usr/bin/env python3
"""
Build script for Drone Topometric Localization → standalone EXE (PyInstaller).

Usage:
    python scripts/build_executable.py            # Full build with models + DINO caches
    python scripts/build_executable.py --no-models # Build without .pth/.engine files
    python scripts/build_executable.py --console   # Keep console window visible (debug)

Output:  dist/DroneLocalization/DroneLocalization.exe
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = ROOT_DIR / "main.py"
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"
SPEC_DIR = ROOT_DIR

# Data directories to bundle
GUI_RESOURCES = ROOT_DIR / "src" / "gui" / "resources"
CONFIG_DIR = ROOT_DIR / "config"
MODELS_DIR = ROOT_DIR / "models"
THIRD_PARTY_DIR = ROOT_DIR / "third_party"

# Cache directories for pre-downloaded models (DINOv2, DINOv3, LightGlue, etc.).
# Preferred source: the project-local cache models/.cache (populated by
# config.paths.ensure_model_cache_env in dev). Fallback: the legacy
# user-profile cache, so builds keep working before migration.


def _cache_root(sub: str) -> Path:
    project = MODELS_DIR / ".cache" / sub
    return project if project.is_dir() else Path.home() / ".cache" / sub


TORCH_HUB_CACHE = _cache_root("torch") / "hub"
HF_HUB_CACHE = _cache_root("huggingface") / "hub"

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _find_package_dir(package_name: str) -> Path | None:
    """Locate an installed package in the current venv."""
    try:
        mod = __import__(package_name)
        return Path(mod.__file__).parent
    except Exception:
        return None


def _find_cuda_libs() -> list[tuple[str, str]]:
    """Find CUDA runtime DLLs from the torch installation."""
    torch_dir = _find_package_dir("torch")
    if torch_dir is None:
        return []

    cuda_dirs = [
        torch_dir / "lib",
        torch_dir.parent / "nvidia",  # nvidia-* packages
    ]

    binaries = []
    for d in cuda_dirs:
        if not d.exists():
            continue
        for dll in d.rglob("*.dll"):
            binaries.append((str(dll), str(dll.parent.relative_to(torch_dir.parent))))
        for so in d.rglob("*.so*"):
            binaries.append((str(so), str(so.parent.relative_to(torch_dir.parent))))

    return binaries


def _dir_size_mb(path: Path) -> float:
    """Calculate total size of a directory in MB."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def _collect_dino_caches() -> list[tuple[str, str, str, float]]:
    """Find cached DINO / LightGlue / ALIKED models to bundle.

    Returns list of (source_path, dest_in_bundle, description, size_mb).
    """
    entries = []

    # DINOv2: torch.hub repo (Python code for model architecture)
    dinov2_repo = TORCH_HUB_CACHE / "facebookresearch_dinov2_main"
    if dinov2_repo.exists():
        size = _dir_size_mb(dinov2_repo)
        entries.append(
            (
                str(dinov2_repo),
                ".cache/torch/hub/facebookresearch_dinov2_main",
                f"DINOv2 hub repo ({size:.0f} MB)",
                size,
            )
        )

    # DINOv2 checkpoint (the actual weights)
    dinov2_ckpt = TORCH_HUB_CACHE / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    if dinov2_ckpt.exists():
        size = dinov2_ckpt.stat().st_size / 1024**2
        entries.append(
            (
                str(dinov2_ckpt),
                ".cache/torch/hub/checkpoints",
                f"DINOv2 vitl14 weights ({size:.0f} MB)",
                size,
            )
        )

    # Other torch.hub checkpoints used by the project
    hub_checkpoints = {
        "aliked-n16.pth": "ALIKED-n16 weights",
        "aliked_lightglue_v0-1_arxiv.pth": "ALIKED-LightGlue weights",
        "superpoint_lightglue_v0-1_arxiv.pth": "SuperPoint-LightGlue weights",
        "superpoint_v1.pth": "SuperPoint v1 weights",
    }
    for fname, desc in hub_checkpoints.items():
        ckpt = TORCH_HUB_CACHE / "checkpoints" / fname
        if ckpt.exists():
            size = ckpt.stat().st_size / 1024**2
            entries.append(
                (
                    str(ckpt),
                    ".cache/torch/hub/checkpoints",
                    f"{desc} ({size:.0f} MB)",
                    size,
                )
            )

    # DINOv3: HuggingFace model cache
    dinov3_hf = HF_HUB_CACHE / "models--facebook--dinov3-vitl16-pretrain-sat493m"
    if dinov3_hf.exists():
        size = _dir_size_mb(dinov3_hf)
        entries.append(
            (
                str(dinov3_hf),
                ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-sat493m",
                f"DINOv3 HuggingFace cache ({size:.0f} MB)",
                size,
            )
        )

    # DINOv3 trust_remote_code: custom model code lives outside hub/, in
    # ~/.cache/huggingface/modules/transformers_modules/ -- required offline.
    hf_modules = HF_HUB_CACHE.parent / "modules"
    if hf_modules.exists():
        size = _dir_size_mb(hf_modules)
        entries.append(
            (
                str(hf_modules),
                ".cache/huggingface/modules",
                f"HF remote-code modules ({size:.0f} MB)",
                size,
            )
        )

    return entries


def build(include_models: bool = True, console: bool = False) -> None:
    """Run PyInstaller with all required options."""
    os.chdir(ROOT_DIR)

    print("=" * 70)
    print("  Drone Topometric Localization — PyInstaller Build")
    print("=" * 70)
    print(f"  Root:      {ROOT_DIR}")
    print(f"  Models:    {'included' if include_models else 'EXCLUDED'}")
    print(f"  Console:   {'visible' if console else 'hidden (windowed)'}")
    print("=" * 70)

    # ---- Collect --add-data entries ---- #
    datas: list[str] = []

    # GUI resources (map template, etc.)
    if GUI_RESOURCES.exists():
        datas.append(f"--add-data={GUI_RESOURCES};src/gui/resources")
        print(f"  + GUI resources: {GUI_RESOURCES}")

    # Config directory
    if CONFIG_DIR.exists():
        datas.append(f"--add-data={CONFIG_DIR / 'config.py'};config")
        print(f"  + Config: config/config.py")

    # Neural network model weights (models/ directory)
    if include_models and MODELS_DIR.exists():
        datas.append(f"--add-data={MODELS_DIR};models")
        total_mb = sum(f.stat().st_size for f in MODELS_DIR.iterdir() if f.is_file()) / 1024**2
        print(f"  + Models dir: {MODELS_DIR} ({total_mb:.0f} MB)")

    # DINO / LightGlue / ALIKED cached models (torch.hub + HuggingFace)
    if include_models:
        dino_entries = _collect_dino_caches()
        for src, dst, desc, _ in dino_entries:
            datas.append(f"--add-data={src};{dst}")
            print(f"  + {desc}")
        if not dino_entries:
            print("  [!] No cached DINO/LightGlue models found -- they'll download on first run")

    # Third-party: Depth-Anything-V2 (Python source, needed at runtime)
    da_v2_dir = THIRD_PARTY_DIR / "Depth-Anything-V2" / "depth_anything_v2"
    if da_v2_dir.exists():
        datas.append(f"--add-data={da_v2_dir};depth_anything_v2")
        print(f"  + Depth-Anything-V2: {da_v2_dir}")

    # Third-party: RDD (Python source + configs)
    rdd_dir = THIRD_PARTY_DIR / "rdd"
    if rdd_dir.exists():
        datas.append(f"--add-data={rdd_dir / 'RDD'};third_party/rdd/RDD")
        rdd_configs = rdd_dir / "configs"
        if rdd_configs.exists():
            datas.append(f"--add-data={rdd_configs};third_party/rdd/configs")
        print(f"  + RDD: {rdd_dir}")

    # ---- Collect --add-binary entries (CUDA DLLs) ---- #
    binaries: list[str] = []
    cuda_libs = _find_cuda_libs()
    if cuda_libs:
        print(f"  + CUDA libraries: {len(cuda_libs)} files")
        for src, dst in cuda_libs:
            binaries.append(f"--add-binary={src};{dst}")

    # ---- Hidden imports ---- #
    # These modules are imported dynamically or are needed by dependencies
    hidden_imports = [
        # PyQt6
        "PyQt6.QtWebEngineWidgets",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebChannel",
        "PyQt6.QtNetwork",
        "PyQt6.QtPrintSupport",
        # PyTorch ecosystem
        "torch",
        "torch.nn",
        "torch.utils",
        "torch.utils.data",
        "torchvision",
        "torchvision.transforms",
        "torchvision.models",
        # Computer vision
        "cv2",
        "ultralytics",
        # Feature matching
        "lightglue",
        "lightglue.utils",
        "lightglue.aliked",
        "lightglue.superpoint",
        "kornia",
        "kornia.feature",
        # Database & serialization
        "h5py",
        "lancedb",
        "pyarrow",
        "pyarrow.pandas_compat",
        "pandas",
        "faiss",
        # Geometry & projection
        "pyproj",
        "poselib",
        "scipy",
        "scipy.optimize",
        "scipy.spatial",
        "filterpy",
        "filterpy.kalman",
        # Data validation
        "pydantic",
        "pydantic.fields",
        "pydantic_core",
        # Networking
        "websockets",
        "websockets.server",
        "websockets.client",
        "aiohttp",
        # Serialization
        "orjson",
        "geojson",
        # Logging
        "loguru",
        # Utilities
        "tqdm",
        "PIL",
        "PIL.Image",
        "yaml",
        # pkg_resources / jaraco (known PyInstaller issue)
        "pkg_resources",
        "jaraco.text",
        "jaraco.functools",
        "jaraco.context",
        # supervision (ultralytics dependency)
        "supervision",
        # numpy
        "numpy",
    ]

    # ---- Excludes (reduce size & fix crashes) ---- #
    excludes = [
        # onnx.reference crashes PyInstaller's isolated subprocess (exit code 3221225477)
        "onnx.reference",
        "onnx.reference.ops",
        # Unused heavy packages
        "matplotlib",
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "ruff",
        "coverage",
        "sphinx",
        "docutils",
    ]

    # ---- Build argument list ---- #
    # Clean old build artifacts manually (--clean inside PyInstaller can cause issues)
    for cleanup_dir in [BUILD_DIR / "DroneLocalization", DIST_DIR / "DroneLocalization"]:
        if cleanup_dir.exists():
            print(f"  Cleaning: {cleanup_dir}")
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    args = [
        str(MAIN_SCRIPT),
        "--name=DroneLocalization",
        "--noconfirm",
        "--onedir",
        "--console" if console else "--windowed",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        f"--specpath={SPEC_DIR}",
        # Runtime hook: fix torch DLL loading + cache redirect
        f"--runtime-hook={ROOT_DIR / 'scripts' / 'pyi_rth_torch.py'}",
    ]

    # Icon (if exists)
    icon_path = ROOT_DIR / "src" / "gui" / "resources" / "icon.ico"
    if icon_path.exists():
        args.append(f"--icon={icon_path}")

    # Add data/binary/hidden-import/exclude args
    args.extend(datas)
    args.extend(binaries)
    for hi in hidden_imports:
        args.append(f"--hidden-import={hi}")
    for ex in excludes:
        args.append(f"--exclude-module={ex}")

    # ---- Run PyInstaller ---- #
    print()
    print(f"  Running PyInstaller with {len(args)} arguments...")
    print(f"  This may take 10-20 minutes for PyTorch projects.")
    print()

    import PyInstaller.__main__

    PyInstaller.__main__.run(args)

    # ---- Post-build: set up cache symlinks / env redirect ---- #
    output_dir = DIST_DIR / "DroneLocalization"
    _create_cache_redirect(output_dir)

    print("\n  Cleaning up duplicate MSVC runtime DLLs from PyQt6...")
    pyqt_bin = output_dir / "_internal" / "PyQt6" / "Qt6" / "bin"
    if pyqt_bin.exists():
        for dll_name in [
            "VCRUNTIME140.dll",
            "VCRUNTIME140_1.dll",
            "MSVCP140.dll",
            "MSVCP140_1.dll",
        ]:
            dll_file = pyqt_bin / dll_name
            if dll_file.exists():
                dll_file.unlink()
                print(f"  Removed: {dll_file}")

    # ---- Post-build summary ---- #
    output_exe = output_dir / "DroneLocalization.exe"

    print()
    print("=" * 70)
    if output_exe.exists():
        # Calculate total size
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        print(f"  [OK] BUILD SUCCESSFUL")
        print(f"  Output: {output_dir}")
        print(f"  EXE:    {output_exe}")
        print(f"  Size:   {total_size / 1024**3:.2f} GB")
        print()
        print(f"  The bundled DINO models are in: {output_dir / '.cache'}")
        print(f"  To distribute: zip the entire DroneLocalization/ folder.")
    else:
        print(f"  [FAIL] BUILD FAILED -- EXE not found at {output_exe}")
        print(f"     Check the log above for errors.")
    print("=" * 70)


def _create_cache_redirect(output_dir: Path) -> None:
    """Create a startup hook that redirects torch.hub and HF cache to the bundled .cache/.

    This ensures the bundled DINO models are found without internet access.
    """
    cache_dir = output_dir / ".cache"
    if not cache_dir.exists():
        return  # No caches were bundled

    hook_content = '''"""Runtime hook: redirect torch.hub and HuggingFace cache to bundled .cache/."""
import os
import sys

# PyInstaller sets _MEIPASS for --onedir; for --onefile it's a temp dir
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
    cache_dir = os.path.join(app_dir, ".cache")

    if os.path.isdir(cache_dir):
        # torch.hub cache
        torch_hub = os.path.join(cache_dir, "torch", "hub")
        if os.path.isdir(torch_hub):
            os.environ.setdefault("TORCH_HOME", os.path.join(cache_dir, "torch"))

        # HuggingFace cache
        hf_hub = os.path.join(cache_dir, "huggingface", "hub")
        if os.path.isdir(hf_hub):
            os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub)
            os.environ.setdefault("TRANSFORMERS_CACHE", hf_hub)
'''
    hook_path = output_dir / "_pyi_rth_cache_redirect.py"
    hook_path.write_text(hook_content, encoding="utf-8")
    print(f"  Created cache redirect hook: {hook_path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DroneLocalization EXE")
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Skip bundling .pth/.engine and cached DINO model files. "
        "Users must ensure models are available (download or copy manually).",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Keep the console window visible (useful for debugging).",
    )
    cli_args = parser.parse_args()

    build(
        include_models=not cli_args.no_models,
        console=cli_args.console,
    )
