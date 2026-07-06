from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelup.config import quarantine_corrupt_file, resolve_state_dir, write_managed_text
from pixelup.devices import DEFAULT_DEVICE, DEVICE_VALUES
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY, normalize_font_family
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
# Tiling is on by default so peak memory scales with the tile, not the image: a
# whole-image pass (tile=0) can exhaust GPU/MPS memory and hard-crash on large
# inputs. 256 keeps peak memory low enough to run on modest GPUs and smaller-memory
# machines; output is effectively identical to larger tiles, and a power user can
# raise it in settings for a small speed gain.
DEFAULT_TILE = 256


@dataclass(frozen=True, slots=True)
class AppConfig:
    max_concurrent_jobs: int = MIN_CONCURRENT_JOBS
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = DEFAULT_TILE
    device: str = DEFAULT_DEVICE
    auto_download: bool = True
    font_family: str = DEFAULT_UI_FONT_FAMILY


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    """The outcome of loading ``config.json``: the settings, plus whether a corrupt
    file had to be quarantined.

    ``quarantined_to`` is the path the corrupt original was moved aside to
    (``<stem>-<ms-utc>.invalid``) when the file was unreadable, else ``None``. It
    exists so the startup shell can surface a *non-fatal* user-facing notice — the
    corrupt file held only disposable preferences, so resetting to defaults is safe,
    and the user should know their tweaks were replaced and where the old file went.
    The quarantine-vs-notice decision stays here, out of the GUI: the window merely
    reports what this pure loader already decided.
    """

    config: AppConfig
    quarantined_to: Path | None = None


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load the settings, resetting a corrupt file to defaults.

    Thin accessor over :func:`load_app_config_result` for the many callers that only
    need the ``AppConfig`` and not the quarantine event.
    """
    return load_app_config_result(path).config


def load_app_config_result(path: Path | None = None) -> ConfigLoadResult:
    """Load ``config.json``, quarantining-then-resetting a corrupt file rather than crashing.

    A present-but-corrupt settings file must never take the app down at startup: the
    storage-path conventions require the load path to *halt or quarantine-then-reset*,
    and for ``config.json`` — pure, disposable preferences — quarantine-then-reset is
    the right choice, because being unable to launch over a bad settings file is the
    worse failure. So on unreadable JSON or a non-object shape, the corrupt file is
    moved aside to ``<stem>-<ms-utc>.invalid`` (bytes preserved via
    :func:`quarantine_corrupt_file`), the built-in defaults are written back through
    the normal save path, and the result records where the original went so the caller
    can surface a non-fatal notice. A missing file is the normal first run: defaults,
    no quarantine.
    """
    if path is None:
        path = config_path()
    if not path.exists():
        return ConfigLoadResult(AppConfig())
    # Only unreadable-as-JSON content is treated as corrupt-and-resettable. A read
    # that fails with an OSError (a permission or I/O problem) is a transient access
    # failure, not corruption, and must not cost the user their real settings, so it
    # is left to propagate rather than quarantining a file we simply could not read.
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("config is not a JSON object")
    except ValueError:
        quarantined_to = quarantine_corrupt_file(path)
        defaults = AppConfig()
        save_app_config(defaults, path)
        return ConfigLoadResult(defaults, quarantined_to=quarantined_to)
    defaults = AppConfig()
    return ConfigLoadResult(
        AppConfig(
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
            font_family=normalize_font_family(data.get("font_family"), defaults.font_family),
        )
    )


def save_app_config(config: AppConfig, path: Path | None = None) -> None:
    """Persist the settings through the single managed-text atomic-write choke point.

    ``config.json`` is PixelUp's one managed text store, and the write goes through
    :func:`pixelup.config.write_managed_text` — the atomic temp-then-rename that is
    both the durability floor and the exact point the data-backup layer records the
    bytes just written. This is the app's RECORD write site: every save of
    ``config.json`` is captured (deduped) in ``backups.sqlite3`` (data-backup-conventions).
    """
    if path is None:
        path = config_path()
    write_managed_text(path, json.dumps(_to_json(config), indent=2, sort_keys=True) + "\n")


def ensure_app_config(path: Path | None = None) -> bool:
    """Create ``config.json`` from the built-in defaults on first run, only when it is absent.

    So the settings file exists on disk immediately rather than only after the first save
    (storage-path conventions, "Materializing settings on first run"). The single trigger is
    absence: an existing file is never inspected or overwritten, so a good or hand-edited file is
    never at risk. It is written through :func:`save_app_config` — the same serializer the normal
    save path uses — not a hand-built literal. Returns ``True`` when a file was created.
    """
    if path is None:
        path = config_path()
    if path.exists():
        return False
    save_app_config(AppConfig(), path)
    return True


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
        "font_family": config.font_family,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "output_format": config.output_format.value,
        "quality": config.quality,
        "tile": config.tile,
    }
