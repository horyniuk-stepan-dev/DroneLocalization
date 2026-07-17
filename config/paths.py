"""Filesystem locations for writable application data (logs, user config).

In a frozen build the executable typically lives under Program Files, which is
read-only for standard users — so writing logs/config relative to the current
working directory fails (PermissionError) and can crash startup. When frozen we
redirect writable data to %LOCALAPPDATA%\\DroneLocalization (always writable).
In dev we keep the current working directory, preserving existing behaviour.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "DroneLocalization"


def user_data_dir() -> Path:
    """Base directory for writable files. Creates it (only) when frozen."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home()
        target = root / APP_DIR_NAME
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            target = Path.home() / APP_DIR_NAME
            target.mkdir(parents=True, exist_ok=True)
        return target
    return Path.cwd()


def models_root() -> Path:
    """Single storage root for model weights and hub caches.

    Dev: <repo>/models, anchored to this file (not the cwd), so scripts
    find weights from any working directory. Frozen: <_MEIPASS>/models.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "models"
    return Path(__file__).resolve().parents[1] / "models"


def ensure_model_cache_env() -> None:
    """Redirect torch.hub / HuggingFace caches into models/.cache (dev only).

    Keeps hub-downloaded models (DINOv2/v3, LightGlue, ALIKED, SuperPoint)
    inside the project instead of the user profile. setdefault: an explicit
    user override always wins; in frozen builds the runtime hook already
    points these at the bundled cache, so we do nothing.
    """
    if getattr(sys, "frozen", False):
        return
    cache = models_root() / ".cache"
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "huggingface" / "hub"))
