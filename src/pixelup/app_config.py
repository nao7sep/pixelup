from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelup.config import resolve_state_dir
from pixelup.devices import DEFAULT_DEVICE, DEVICE_VALUES
from pixelup.paths import OutputFormat


def config_path() -> Path:
    """Resolve ``config.json`` under the storage root, lazily on every call.

    The root is resolved here rather than frozen into a module-level constant at
    import time: import-time resolution captures a half-set environment and
    freezes ``PIXELUP_HOME`` for the life of the process. Resolving on first use
    means a ``PIXELUP_HOME`` set before launch is honored and tests can vary it.
    """
    return resolve_state_dir() / "config.json"

# Valid domains for the persisted settings. The UI controls and this loader both
# reference these, so a value can never be representable in one place but not the
# other.
MIN_CONCURRENT_JOBS = 1
MAX_CONCURRENT_JOBS = 8
MIN_QUALITY = 0
MAX_QUALITY = 100
MIN_TILE = 0
MAX_TILE = 4096
TILE_STEP = 256


@dataclass(frozen=True, slots=True)
class AppConfig:
    max_concurrent_jobs: int = MIN_CONCURRENT_JOBS
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = MIN_TILE
    device: str = DEFAULT_DEVICE
    auto_download: bool = True


def load_app_config(path: Path | None = None) -> AppConfig:
    if path is None:
        path = config_path()
    if not path.exists():
        return AppConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"PixelUp config must be a JSON object: {path}")
    defaults = AppConfig()
    return AppConfig(
        max_concurrent_jobs=_clamp_int(
            data.get("max_concurrent_jobs"),
            defaults.max_concurrent_jobs,
            MIN_CONCURRENT_JOBS,
            MAX_CONCURRENT_JOBS,
        ),
        output_format=_coerce_output_format(data.get("output_format"), defaults.output_format),
        quality=_clamp_int(data.get("quality"), defaults.quality, MIN_QUALITY, MAX_QUALITY),
        tile=_clamp_int(data.get("tile"), defaults.tile, MIN_TILE, MAX_TILE),
        device=_coerce_device(data.get("device"), defaults.device),
        auto_download=bool(data.get("auto_download", defaults.auto_download)),
    )


def save_app_config(config: AppConfig, path: Path | None = None) -> None:
    if path is None:
        path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _coerce_device(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).lower()
    return text if text in DEVICE_VALUES else default


def _coerce_output_format(value: Any, default: OutputFormat) -> OutputFormat:
    if value is None:
        return default
    try:
        return OutputFormat(str(value).lower())
    except ValueError:
        return default


def _to_json(config: AppConfig) -> dict[str, Any]:
    return {
        "auto_download": config.auto_download,
        "device": config.device,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "output_format": config.output_format.value,
        "quality": config.quality,
        "tile": config.tile,
    }
