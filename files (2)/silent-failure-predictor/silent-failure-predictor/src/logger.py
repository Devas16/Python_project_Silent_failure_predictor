"""
logger.py
---------
Centralized logging configuration using Loguru.
All modules should import the logger from here to ensure
consistent formatting and output across the project.
"""

import os
import sys
from loguru import logger
from src.config_loader import load_config

def setup_logger() -> logger:
    """
    Configure and return a Loguru logger instance.

    Reads logging settings from config.yaml and sets up:
    - Console output (colored, human-readable)
    - File output (with rotation and retention policies)

    Returns:
        Configured loguru logger instance.
    """
    config = load_config()
    log_cfg = config.get("logging", {})

    log_level = log_cfg.get("level", "INFO")
    log_dir   = log_cfg.get("log_dir", "logs/")
    log_file  = log_cfg.get("log_file", "app.log")
    rotation  = log_cfg.get("rotation", "10 MB")
    retention = log_cfg.get("retention", "7 days")
    fmt       = log_cfg.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{line} | {message}"
    )

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler — colorized
    logger.add(
        sys.stdout,
        level=log_level,
        format=fmt,
        colorize=True,
    )

    # File handler — with rotation & retention
    logger.add(
        os.path.join(log_dir, log_file),
        level=log_level,
        format=fmt,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    logger.info(f"Logger initialized | level={log_level} | file={os.path.join(log_dir, log_file)}")
    return logger


# Module-level logger — import this in other modules
log = setup_logger()
