"""Settings-read shorthand for tests that assert only effective values."""

from pathlib import Path

from pixelup.app_config import AppConfig, load_app_config_result


def load_app_config(path: Path | None = None) -> AppConfig:
    """Use the real load boundary, including preservation and quarantine behavior."""
    return load_app_config_result(path).config
