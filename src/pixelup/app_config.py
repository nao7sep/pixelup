from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pixelup.config import APP_NAME
from pixelup.paths import OutputFormat

CONFIG_PATH = Path.home() / f".{APP_NAME}" / "config.toml"


@dataclass(frozen=True, slots=True)
class AppConfig:
    max_concurrent_jobs: int = 1
    close_tab_on_success: bool = True
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = 0
    device: str = "auto"
    auto_download: bool = True


def load_app_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return AppConfig(
        max_concurrent_jobs=max(1, int(data.get("max_concurrent_jobs", 1))),
        close_tab_on_success=bool(data.get("close_tab_on_success", True)),
        output_format=OutputFormat(str(data.get("output_format", OutputFormat.PNG.value))),
        quality=min(100, max(0, int(data.get("quality", 95)))),
        tile=max(0, int(data.get("tile", 0))),
        device=str(data.get("device", "auto")),
        auto_download=bool(data.get("auto_download", True)),
    )


def save_app_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"max_concurrent_jobs = {config.max_concurrent_jobs}",
                f"close_tab_on_success = {_toml_bool(config.close_tab_on_success)}",
                f'output_format = "{config.output_format.value}"',
                f"quality = {config.quality}",
                f"tile = {config.tile}",
                f'device = "{config.device}"',
                f"auto_download = {_toml_bool(config.auto_download)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"
