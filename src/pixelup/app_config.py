from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from pixelup.config import quarantine_corrupt_file, resolve_state_dir, write_managed_text
from pixelup.devices import DEVICE_VALUES
from pixelup.errors import ErrorCode, PixelupError
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY
from pixelup.i18n.languages import SYSTEM, TAGS
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
from pixelup.session_log import log
from pixelup.text_cleanup import single_line


def config_path() -> Path:
    """Resolve ``config.json`` under the storage root, lazily on every call.

    The root is resolved here rather than frozen into a module-level constant at
    import time: import-time resolution captures a half-set environment and
    freezes ``PIXELUP_DATA_DIR`` for the life of the process. Resolving on first use
    means a ``PIXELUP_DATA_DIR`` set before launch is honored and tests can vary it.
    """
    return resolve_state_dir() / "config.json"


# Two PixelUp windows share one config.json. This bounds how long a save waits for
# the other window's own in-flight save (backup_store.py's SQLite busy_timeout for
# the same two-instance case is 5s; matched here for a consistent worst-case wait).
_CONFIG_LOCK_TIMEOUT_SECONDS = 5


def _config_lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


# Valid domain of the settings this module still owns. The settings dialog and this
# loader both reference these, so a value can never be representable in one place but
# not the other. The image-processing parameters' own domains live beside JobSettings
# (see pixelup.jobs), which is where their meaning now lives.
MIN_CONCURRENT_JOBS = 1
MAX_CONCURRENT_JOBS = 8


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Effective settings; only changed sets are persisted in ``config.json``.

    Two kinds of thing, one home each. The scalars are the Settings modal's whole
    content — what the main window does not show. ``parameters`` is the main
    window's Parameters panel, persisted whole: the panel is the only place those
    values are edited, and ``JobSettings()`` is the only place their built-in
    defaults are written, so there is no second defaults layer to drift against.
    """

    max_concurrent_jobs: int = MIN_CONCURRENT_JOBS
    font_family: str = DEFAULT_UI_FONT_FAMILY
    # The interface language: "system" to follow the computer, or a language tag
    # (localization-conventions).
    language: str = SYSTEM
    parameters: JobSettings = field(default_factory=JobSettings)


_CONFIG_SET_KEYS = tuple(item.name for item in fields(AppConfig))
_warned_invalid_sets: set[tuple[Path, str]] = set()


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    """The outcome of loading ``config.json``: the settings, plus whether a corrupt
    file had to be quarantined.

    ``quarantined_to`` is the path the corrupt original was moved aside to
    (``<stem>-<ms-utc>.invalid``) when the file was unreadable, else ``None``. It
    exists so the startup shell can surface a *non-fatal* user-facing notice — the
    settings fall back to built-ins in memory without writing a replacement file,
    and the user should know their tweaks are inactive and where the old file went.
    The quarantine-vs-notice decision stays here, out of the GUI: the window merely
    reports what this pure loader already decided.
    """

    config: AppConfig
    quarantined_to: Path | None = None


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load effective settings, quarantining a corrupt file without writing a replacement.

    Thin accessor over :func:`load_app_config_result` for the many callers that only
    need the ``AppConfig`` and not the quarantine event.
    """
    return load_app_config_result(path).config


def load_app_config_result(path: Path | None = None) -> ConfigLoadResult:
    """Load effective sets, preserving unreadable JSON through quarantine.

    Missing files and quarantined files both leave the store absent. A malformed
    set falls back on its own; it does not cost the user the other settings.
    Access failures propagate without moving or overwriting the file.
    """
    if path is None:
        path = config_path()
    data, quarantined_to = _read_config_map(path)
    return ConfigLoadResult(_decode_app_config(data, path), quarantined_to)


def _read_config_map(path: Path) -> tuple[dict[str, Any], Path | None]:
    if not path.exists():
        return {}, None
    try:
        data = _object(json.loads(path.read_text(encoding="utf-8")), "config")
    except ValueError:
        # Keep the rename outside the read-failure handler: a failed quarantine
        # must propagate rather than overwrite bytes it exists to preserve.
        pass
    else:
        return data, None
    return {}, quarantine_corrupt_file(path)


def save_app_config_merged(
    candidate: AppConfig,
    previous: AppConfig,
    path: Path | None = None,
) -> AppConfig:
    """Save an edit to ``config.json`` without discarding a sibling window's own edit.

    A caller keeps its own full in-memory ``AppConfig`` and edits it by building
    ``candidate = replace(previous, <one field>=<new value>)``. Saving that whole
    object naively (PU-3) loses any field a second PixelUp window already changed
    and saved since ``previous`` was loaded, because ``candidate`` still carries
    ``previous``'s stale copy of it. This takes a short-lived cross-process
    :class:`filelock.FileLock` on ``config.json`` — the same tolerated-two-instance
    pattern :mod:`pixelup.models`, :mod:`pixelup.output_reservation`, and
    :mod:`pixelup.backup_store` already use for their own shared files — reads the
    file fresh under the lock, applies onto *that* only the fields ``candidate``
    actually changed relative to ``previous``, and writes the file from the result
    (config-sets-conventions). The returned ``AppConfig`` becomes the caller's new
    in-memory copy, so it also picks up whatever the other window wrote to fields
    this edit did not touch.
    """
    if path is None:
        path = config_path()
    lock = FileLock(str(_config_lock_path(path)), timeout=_CONFIG_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except Timeout as exc:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "Timed out saving settings; another PixelUp window is saving them.",
            details={"path": str(path), "lock_timeout": _CONFIG_LOCK_TIMEOUT_SECONDS},
        ) from exc
    try:
        stored, _quarantined_to = _read_config_map(path)
        merged = replace(
            _decode_app_config(stored, path),
            **{
                key: getattr(candidate, key)
                for key in _CONFIG_SET_KEYS
                if getattr(candidate, key) != getattr(previous, key)
            },
        )
        data = _stored_sets(merged)
        if data != stored:
            write_managed_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return merged
    finally:
        lock.release()


def _stored_sets(config: AppConfig) -> dict[str, Any]:
    """Every set that differs from its built-in, whole (config-sets-conventions)."""
    defaults = AppConfig()
    return {
        key: value
        for key, value in _to_json(config).items()
        if getattr(config, key) != getattr(defaults, key)
    }


def _decode_app_config(data: dict[str, Any], path: Path) -> AppConfig:
    defaults = AppConfig()
    decoded: dict[str, Any] = {}
    for key in _CONFIG_SET_KEYS:
        if key not in data:
            continue
        try:
            decoded[key] = _decode_set(key, data[key], defaults)
        except ValueError as exc:
            warning_key = (path, key)
            if warning_key not in _warned_invalid_sets:
                _warned_invalid_sets.add(warning_key)
                log.warning("config.invalid_set", path=str(path), key=key, reason=str(exc))
    return AppConfig(**decoded)


def _decode_set(key: str, value: Any, defaults: AppConfig) -> Any:
    if key == "parameters":
        return _decode_parameters(value)
    if key == "font_family":
        if not isinstance(value, str):
            raise ValueError("font_family is not a string")
        return single_line(value)
    if key == "language":
        return _optional_choice({key: value}, key, defaults.language, (SYSTEM, *TAGS))
    return _optional_int_range(
        {key: value}, key, defaults.max_concurrent_jobs, MIN_CONCURRENT_JOBS, MAX_CONCURRENT_JOBS
    )


def _decode_parameters(value: Any) -> JobSettings:
    data = _object(value, "parameters")
    defaults = JobSettings()
    if not _parameters_to_json(defaults).keys() <= data.keys():
        raise ValueError("parameters is missing a member")
    output_format = _optional_choice(
        data,
        "output_format",
        defaults.output_format.value,
        tuple(item.value for item in OutputFormat),
    )
    return JobSettings(
        scale=_optional_int_choice(data, "scale", defaults.scale, SCALE_VALUES),
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


def _to_json(config: AppConfig) -> dict[str, Any]:
    return {
        "font_family": config.font_family,
        "language": config.language,
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "parameters": _parameters_to_json(config.parameters),
    }


def _parameters_to_json(parameters: JobSettings) -> dict[str, Any]:
    return {
        "alpha_mode": parameters.alpha_mode,
        "denoise_strength": parameters.denoise_strength,
        "device": parameters.device,
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
        "language": config.language,
        "parameters": job_settings_log_payload(config.parameters),
    }
