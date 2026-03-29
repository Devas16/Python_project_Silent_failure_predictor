"""
config_loader.py
----------------
Loads and exposes the project configuration from config/config.yaml.
Provides a single cached access point so the YAML is parsed only once.
"""

import os
import yaml
from functools import lru_cache
from typing import Any, Dict

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),   # src/
    "..",                         # project root
    "config",
    "config.yaml",
)


@lru_cache(maxsize=1)
def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    The result is cached after the first call so subsequent imports
    of the config are free.

    Args:
        path: Absolute or relative path to the YAML config file.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Configuration file not found: {abs_path}\n"
            "Make sure config/config.yaml exists at the project root."
        )

    with open(abs_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return config


def get(key: str, default: Any = None) -> Any:
    """
    Convenience helper — fetch a top-level config key.

    Args:
        key:     Top-level key in config.yaml (e.g. "data", "logging").
        default: Value returned when the key is missing.

    Returns:
        Config section dict or scalar, or *default* if not found.
    """
    return load_config().get(key, default)
