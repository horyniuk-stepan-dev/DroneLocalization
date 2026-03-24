import sys
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
