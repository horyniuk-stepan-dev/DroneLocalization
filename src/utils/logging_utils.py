"""
Logging configuration
"""

from loguru import logger
import sys


def setup_logging(log_level="INFO", log_file=None):
    """Setup application logging"""
    # TODO: Configure loguru
    # TODO: Add file handler if specified
    # TODO: Set format
    pass


def get_logger(name=None):
    """Get logger instance"""
    return logger
