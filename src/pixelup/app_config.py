from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelup.config import resolve_state_dir
from pixelup.paths import OutputFormat

CONFIG_PATH = resolve_state_dir() / "config.json"


@dataclass(frozen=True, slots=True)
class AppConfig:
    max_concurrent_jobs: int = 1
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = 0
    device: str = "auto"
    auto_download: bool = True


def load_app_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"PixelUp config must be a JSON object: {path}")
    return AppConfig(
        max_concurrent_jobs=max(1, int(data.get("max_concurrent_jobs", 1))),
        output_format=OutputFormat(str(data.get("output_format", OutputFormat.PNG.value))),
        quality=min(100, max(0, int(data.get("quality", 95)))),
        tile=max(0, int(data.get("tile", 0))),
        device=str(data.get("device", "auto")),
        auto_download=bool(data.get("auto_download", True)),
    )


def save_app_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_json(config: AppConfig) -> dict[str, Any]:
    return {
        "auto_download": config.auto_download,
        "device": config.device,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "output_format": config.output_format.value,
        "quality": config.quality,
        "tile": config.tile,
    }
