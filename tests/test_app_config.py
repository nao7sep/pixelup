from pathlib import Path

import pytest

from pixelup.app_config import AppConfig, load_app_config, save_app_config
from pixelup.paths import OutputFormat


def test_app_config_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        max_concurrent_jobs=3,
        close_tab_on_success=False,
        output_format=OutputFormat.WEBP,
        quality=82,
        tile=512,
        device="cpu",
        auto_download=False,
    )

    save_app_config(config, path)

    assert load_app_config(path) == config


def test_missing_app_config_uses_defaults(tmp_path: Path) -> None:
    assert load_app_config(tmp_path / "missing.json") == AppConfig()


def test_invalid_config_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_app_config(path)
