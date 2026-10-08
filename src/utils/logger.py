"""
Centralized logging configuration for the Fake Job Detection System.
Provides formatted console and file logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = "logs/app.log",
    module_name: Optional[str] = None,
) -> logging.Logger:
    """
    Configures and returns a logger instance with consistent formatting.

    Args:
        log_level: Logging level (e.g. logging.INFO, logging.DEBUG)
        log_file: Path to output log file (optional)
        module_name: Name of the logger, defaults to root logger if None

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(module_name or "fake_job_detection")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not initialize file logging at {log_file}: {e}")

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Convenience accessor for child loggers across packages.

    Args:
        module_name: Usually __name__ of the calling module

    Returns:
        logging.Logger
    """
    return logging.getLogger(f"fake_job_detection.{module_name}")
