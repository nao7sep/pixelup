from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pixelup.config import quarantine_corrupt_file, resolve_state_dir, write_managed_text
from pixelup.devices import DEVICE_VALUES
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY, normalize_font_family
from pixelup.jobs import (
    JobSettings,
    job_settings_log_payload,
)
from pixelup.parameters import (
    ALPHA_MODE_VALUES,
    MAX_DENOISE_STRENGTH,
    MAX_QUALITY,
    MIN_DENOISE_STRENGTH,
    MIN_QUALITY,
    SCALE_VALUES,
    TARGET_PROFILE_VALUES,
    TILE_VALUES,
)
from pixelup.paths import OutputFormat


def config_path() -> Path:
    """Resolve ``config.json`` under the storage root, lazily on every call.

    The root is resolved here rather than frozen into a module-level constant at
    import time: import-time resolution captures a half-set environment and
    freezes ``PIXELUP_HOME`` for the life of the process. Resolving on first use
    means a ``PIXELUP_HOME`` set before launch is honored and tests can vary it.
    """
    return resolve_state_dir() / "config.json"

# Valid domain of the settings this module still owns. The settings dialog and this
# loader both reference these, so a value can never be representable in one place but
# not the other. The image-processing parameters' own domains live beside JobSettings
# (see pixelup.jobs), which is where their meaning now lives.
MIN_CONCURRENT_JOBS = 1
MAX_CONCURRENT_JOBS = 8


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Everything PixelUp persists in ``config.json``.

    Two kinds of thing, one home each. The two scalars are the Settings modal's
    whole content — what the main window does not show. ``parameters`` is the main
    window's Parameters panel, persisted whole: the panel is the only place those
    values are edited, and ``JobSettings()`` is the only place their built-in
    defaults are written, so there is no second defaults layer to drift against.
    """

    max_concurrent_jobs: int = MIN_CONCURRENT_JOBS
    font_family: str = DEFAULT_UI_FONT_FAMILY
    parameters: JobSettings = field(default_factory=JobSettings)


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
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        config = _decode_app_config(data)
    except ValueError:
        quarantined_to = quarantine_corrupt_file(path)
        defaults = AppConfig()
        save_app_config(defaults, path)
        return ConfigLoadResult(defaults, quarantined_to=quarantined_to)
    return ConfigLoadResult(config)


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


def _decode_app_config(value: Any) -> AppConfig:
    """Decode consumed fields without coercing malformed persisted data.

    Missing fields retain their built-ins and unknown fields are inert. A present
    field must have the exact type and domain the serializer writes; otherwise the
    caller treats the entire store as unreadable and quarantines it.
    """
    data = _object(value, "config")
    defaults = AppConfig()
    return AppConfig(
        max_concurrent_jobs=_optional_int_range(
            data,
            "max_concurrent_jobs",
            defaults.max_concurrent_jobs,
            MIN_CONCURRENT_JOBS,
            MAX_CONCURRENT_JOBS,
        ),
        font_family=_optional_font_family(data, defaults.font_family),
        parameters=(
            _decode_parameters(data["parameters"])
            if "parameters" in data
            else defaults.parameters
        ),
    )


def _decode_parameters(value: Any) -> JobSettings:
    data = _object(value, "parameters")
    defaults = JobSettings()
    output_format = _optional_choice(
        data,
        "output_format",
        defaults.output_format.value,
        tuple(item.value for item in OutputFormat),
    )
    return JobSettings(
        scale=_optional_int_choice(data, "scale", defaults.scale, SCALE_VALUES),
        face_enhance=_optional_bool(data, "face_enhance", defaults.face_enhance),
        denoise_strength=_optional_float_range(
            data,
            "denoise_strength",
            defaults.denoise_strength,
            MIN_DENOISE_STRENGTH,
            MAX_DENOISE_STRENGTH,
        ),
        alpha_mode=_optional_choice(
            data, "alpha_mode", defaults.alpha_mode, ALPHA_MODE_VALUES
        ),
        device=_optional_choice(data, "device", defaults.device, DEVICE_VALUES),
        output_format=OutputFormat(output_format),
        quality=_optional_int_range(
            data, "quality", defaults.quality, MIN_QUALITY, MAX_QUALITY
        ),
        tile=_optional_int_choice(data, "tile", defaults.tile, TILE_VALUES),
        strip_metadata=_optional_bool(data, "strip_metadata", defaults.strip_metadata),
        target_profile=_optional_choice(
            data,
            "target_profile",
            defaults.target_profile,
            TARGET_PROFILE_VALUES,
            allow_none=True,
        ),
    )


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    return value


def _optional_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} is not a boolean")
    return value


def _optional_int_range(
    data: dict[str, Any], key: str, default: int, low: int, high: int
) -> int:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} is outside its valid integer range")
    return value


def _optional_float_range(
    data: dict[str, Any], key: str, default: float, low: float, high: float
) -> float:
    if key not in data:
        return default
    value = data[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= value <= high
    ):
        raise ValueError(f"{key} is outside its valid numeric range")
    return float(value)


def _optional_int_choice(
    data: dict[str, Any], key: str, default: int, choices: tuple[int, ...]
) -> int:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
        raise ValueError(f"{key} is not a recognized integer choice")
    return value


def _optional_choice(
    data: dict[str, Any],
    key: str,
    default: Any,
    choices: tuple[Any, ...],
    *,
    allow_none: bool = False,
) -> Any:
    if key not in data:
        return default
    value = data[key]
    if value is None and allow_none and None in choices:
        return None
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{key} is not a recognized string choice")
    return value


def _optional_font_family(data: dict[str, Any], default: str) -> str:
    if "font_family" not in data:
        return default
    value = data["font_family"]
    if not isinstance(value, str):
        raise ValueError("font_family is not a string")
    return normalize_font_family(value, default)


def _to_json(config: AppConfig) -> dict[str, Any]:
    return {
        "font_family": config.font_family,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "parameters": _parameters_to_json(config.parameters),
    }


def _parameters_to_json(parameters: JobSettings) -> dict[str, Any]:
    return {
        "alpha_mode": parameters.alpha_mode,
        "denoise_strength": parameters.denoise_strength,
        "device": parameters.device,
        "face_enhance": parameters.face_enhance,
        "output_format": parameters.output_format.value,
        "quality": parameters.quality,
        "scale": parameters.scale,
        "strip_metadata": parameters.strip_metadata,
        "target_profile": parameters.target_profile,
        "tile": parameters.tile,
    }


def config_log_payload(config: AppConfig) -> dict[str, object]:
    return {
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "font_family": config.font_family,
        "parameters": job_settings_log_payload(config.parameters),
    }
