"""Filesystem locations for writable application data (logs, user config).

In a frozen build the executable typically lives under Program Files, which is
read-only for standard users — so writing logs/config relative to the current
working directory fails (PermissionError) and can crash startup. When frozen we
redirect writable data to %LOCALAPPDATA%\\DroneLocalization (always writable).

**Dev — anchored to the repo, NOT to the cwd (2026-07-21).** Раніше тут був
``Path.cwd()``, і це означало, що ``user_config.json`` читався лише тоді, коли
застосунок запускали САМЕ з кореня репозиторію. Запуск з іншої теки (IDE з
власним working directory, ярлик, ``python D:\\...\\main.py`` з іншого місця)
мовчки давав вбудовані дефолти замість налаштувань користувача — а це 30+
розбіжностей, серед них вимкнений smoother, edge-гейти й torch_compile.
``models_root()`` нижче вже був cwd-незалежним; тепер обидва однакові.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "DroneLocalization"
USER_CONFIG_NAME = "user_config.json"


def repo_root() -> Path:
    """Корінь репозиторію — прив'язка до цього файлу, не до cwd."""
    return Path(__file__).resolve().parents[1]


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
    return repo_root()


def user_config_candidates() -> list[Path]:
    """Шляхи пошуку ``user_config.json`` у порядку пріоритету; перший наявний виграє.

    Dev: корінь репозиторію, плюс cwd — щоб не зламати сценарій, де користувач
    свідомо тримає окремий конфіг у робочій теці (якщо cwd і є коренем — один
    шлях, без дублювання).

    Frozen: %LOCALAPPDATA% (туди ж пишемо) має пріоритет як користувацький
    override; далі — конфіг поруч із .exe. Другий шлях потрібен тому, що
    ``DroneLocalization.spec`` НЕ пакує ``user_config.json`` у білд: без нього
    свіжовстановлений застосунок стартував би на дефолтах, мовчки ігноруючи
    налаштування, з якими його збирали.
    """
    primary = user_data_dir() / USER_CONFIG_NAME
    if getattr(sys, "frozen", False):
        beside_exe = Path(sys.executable).resolve().parent / USER_CONFIG_NAME
        return [primary] if beside_exe == primary else [primary, beside_exe]
    cwd_cfg = Path.cwd().resolve() / USER_CONFIG_NAME
    return [primary] if cwd_cfg == primary else [primary, cwd_cfg]


def models_root() -> Path:
    """Single storage root for model weights and hub caches.

    Dev: <repo>/models, anchored to this file (not the cwd), so scripts
    find weights from any working directory. Frozen: <_MEIPASS>/models.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "models"
    return repo_root() / "models"


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
