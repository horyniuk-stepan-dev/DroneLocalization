import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log") -> None:
    """Налаштування системи логування для всієї програми."""
    logger.remove()

    # Standart output (pretty console)
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard file output (text)
    logger.add(
        str(log_path),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    # JSON sink for structured logging/metrics
    json_path = log_path.with_name("metrics.jsonl")
    logger.add(
        str(json_path),
        level=log_level,
        serialize=True,
        rotation="10 MB",
        retention="14 days",
    )


def get_logger(name: str | None = None) -> Any:
    """Отримання екземпляра логера."""
    if name:
        return logger.bind(name=name)
    return logger


@contextmanager
def silent_output(force: bool = False):
    """
    Context manager to suppress output (stdout/stderr).
    By default, it uses a safe Python-level override.
    If debug_mode is False (via APP_SETTINGS) or force is True, 
    it attempts a more aggressive FD-level redirection which catches C++ logs.
    """
    from config.config import APP_SETTINGS
    import io

    # Determine if we should be truly silent (FD-level) or just Python-silent
    # We use FD-level only if debug_mode is False to avoid the previous "permanent silence" issues
    # or if explicitly forced.
    is_debug = getattr(getattr(APP_SETTINGS, "models", None), "performance", None).debug_mode if APP_SETTINGS else True
    aggressive = not is_debug or force

    if not aggressive:
        # Safe mode: Only catch Python prints
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = save_stdout
            sys.stderr = save_stderr
        return

    # Aggressive mode: FD-level redirection (os.dup2)
    # catches C++ output from OpenCV, TensorRT, etc.
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stdout_fd = os.dup(1)
    save_stderr_fd = os.dup(2)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(save_stdout_fd, 1)
        os.dup2(save_stderr_fd, 2)
        os.close(null_fd)
        os.close(save_stdout_fd)
        os.close(save_stderr_fd)
