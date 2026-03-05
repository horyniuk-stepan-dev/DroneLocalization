import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_level="INFO", log_file="logs/app.log"):
    """Налаштування системи логування для всієї програми"""
    logger.remove()

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. ЗАХИСТ ВІД NONETYPE: Виводимо в консоль, ТІЛЬКИ якщо вона фізично існує
    if sys.stderr is not None:
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | {name}:{function}:{line} - <level>{message}</level>",
            level="INFO"
        )

    # 2. ЗАВЖДИ пишемо логи у файл (дуже корисно для відлагодження готового .exe)
    logger.add(
        "logs/app_run.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def get_logger(name=None):
    """Отримання екземпляра логера"""
    if name:
        return logger.bind(name=name)
    return logger