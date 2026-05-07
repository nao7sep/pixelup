from pathlib import Path

from pixelup.app_config import AppConfig, load_app_config, save_app_config
from pixelup.paths import OutputFormat


def test_app_config_round_trips_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
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
    assert load_app_config(tmp_path / "missing.toml") == AppConfig()
