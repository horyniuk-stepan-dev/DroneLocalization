"""Config access helpers and default singleton instances."""

from typing import Any, cast

from config.app import AppConfig
from config.models import Dinov2ModelConfig, Dinov3ModelConfig
from config.paths import models_root, user_data_dir


def get_cfg(config: Any, path: str, default: Any = None) -> Any:
    """Централізований доступ до конфігу з dot-path.
    Працює як зі словниками, так і з Pydantic-моделями.
    """
    keys = path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                current = default
                break
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            current = default
            break

    # Resolve "models/..." values against the single models root
    # (config.paths.models_root) at READ time only — stored config stays
    # relative, so user_config.json never accumulates absolute paths.
    # Dev: repo-anchored, cwd-independent. Frozen: only when the file is
    # really bundled (preserves the old fallback behaviour).
    import os
    import sys
    if isinstance(current, str) and (current.startswith("models/") or current.startswith("models\\")):
        resolved = os.path.join(str(models_root().parent), current)
        if not getattr(sys, "frozen", False) or os.path.exists(resolved):
            return resolved

    return current


def get_active_descriptor_cfg(config: Any) -> "Dinov2ModelConfig | Dinov3ModelConfig":
    """Повертає конфіг активного глобального дескриптора (DINOv2 або DINOv3).

    Працює як з Pydantic-об'єктом (APP_SETTINGS), так і зі словником (APP_CONFIG).
    Fallback: DINOv2 за замовчуванням.
    """
    gd = get_cfg(config, "global_descriptor")
    if gd is None:
        return Dinov2ModelConfig()

    # Pydantic object — використовуємо метод active()
    if hasattr(gd, "active"):
        return cast("Dinov2ModelConfig | Dinov3ModelConfig", gd.active())

    # dict — реконструюємо з вкладених словників
    if isinstance(gd, dict):
        backend = gd.get("backend", "dinov2")
        if backend == "dinov3":
            return Dinov3ModelConfig(**(gd.get("dinov3") or {}))
        return Dinov2ModelConfig(**(gd.get("dinov2") or {}))

    return Dinov2ModelConfig()


import json
import os
import tempfile

CONFIG_FILE_PATH = str(user_data_dir() / "user_config.json")

def load_user_config() -> AppConfig:
    """Завантажує налаштування користувача з файлу, якщо він існує. Інакше повертає дефолтні."""
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return AppConfig(**data)
        except Exception as e:
            print(f"Failed to load user config from {CONFIG_FILE_PATH}: {e}. Using defaults.")
    return AppConfig()

def save_user_config(config: AppConfig) -> None:
    """Зберігає налаштування атомарно: пишемо в тимчасовий файл поруч і
    робимо os.replace, щоб перерваний запис не пошкодив основний конфіг."""
    target = os.path.abspath(CONFIG_FILE_PATH)
    directory = os.path.dirname(target)
    tmp_path = None
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".user_config.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = f.name
            f.write(config.model_dump_json(indent=4))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception as e:
        print(f"Failed to save user config to {target}: {e}")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# Екземпляр конфігу за замовчуванням (зчитаний з файлу або дефолтний).
APP_SETTINGS = load_user_config()
# Також надаємо доступ як до словника для зворотньої сумісності
APP_CONFIG = APP_SETTINGS.model_dump()
