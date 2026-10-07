from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from pixelup.config import quarantine_corrupt_file, resolve_state_dir, write_managed_text
from pixelup.devices import DEVICE_VALUES
from pixelup.errors import ErrorCode, PixelupError
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY
from pixelup.formats import CONFIG_FORMAT_VERSION, format_version
from pixelup.i18n.languages import SYSTEM, is_preference
from pixelup.i18n.message import Message
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
    """Effective settings; ``config.json`` holds only the sets that differ from their
    built-ins, each whole (config-sets-conventions).

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


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    """The outcome of loading ``config.json``: the settings, plus whether a corrupt
    file had to be quarantined or a newer PixelUp's file was left alone.

    ``quarantined_to`` is the path the corrupt original was moved aside to
    (``<stem>-<ms-utc>.invalid``) when the file was unreadable, else ``None``.
    ``newer_format`` is the file's format version when a newer PixelUp wrote it,
    else ``None``. Either way the settings fall back to built-ins in memory, and the
    startup shell surfaces a *non-fatal* notice so the user knows their settings
    are inactive. The decision stays here, out of the GUI: the window merely
    reports what this pure loader already decided.
    """

    config: AppConfig
    quarantined_to: Path | None = None
    newer_format: int | None = None


@dataclass(frozen=True, slots=True)
class ConfigSaveResult:
    config: AppConfig
    quarantined_to: Path | None = None


@dataclass(frozen=True, slots=True)
class _StoredConfig:
    """What ``config.json`` holds, as read: its map, or why it holds nothing usable.
    ``exists`` is whether a file is still there to compare a save against."""

    data: dict[str, Any]
    exists: bool
    quarantined_to: Path | None = None
    newer_format: int | None = None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(_config_lock_path(path)), timeout=_CONFIG_LOCK_TIMEOUT_SECONDS):
        stored = _read_config_map(path)
    return ConfigLoadResult(
        _decode_app_config(stored.data, path), stored.quarantined_to, stored.newer_format
    )


def _read_config_map(path: Path) -> _StoredConfig:
    """Read the stored map. A file a newer PixelUp wrote is intact data this build
    cannot read, so it is left exactly in place (store-recovery-conventions)."""
    if not path.exists():
        return _StoredConfig({}, exists=False)
    try:
        data = _object(json.loads(path.read_text(encoding="utf-8")), "config")
        version = format_version(data.get(_FORMAT_VERSION_KEY))
    except ValueError:
        return _StoredConfig({}, exists=False, quarantined_to=quarantine_corrupt_file(path))
    if version > CONFIG_FORMAT_VERSION:
        return _StoredConfig({}, exists=True, newer_format=version)
    return _StoredConfig(data, exists=True)


def save_app_config_merged(
    candidate: AppConfig,
    previous: AppConfig,
    path: Path | None = None,
) -> ConfigSaveResult:
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
    (config-sets-conventions). The returned config becomes the caller's new
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
        stored = _read_config_map(path)
        if stored.newer_format is not None:
            raise PixelupError(
                ErrorCode.STORE_NEWER_FORMAT,
                Message("error.configNewer"),
                details={"path": str(path), "format_version": stored.newer_format},
            )
        merged = replace(
            _decode_app_config(stored.data, path),
            **{
                key: getattr(candidate, key)
                for key in _SET_DECODERS
                if getattr(candidate, key) != getattr(previous, key)
            },
        )
        sets = _stored_sets(merged)
        # With no file and no set to store, the absence already says every set is
        # at its built-in, so nothing is written (config-sets-conventions).
        data = {_FORMAT_VERSION_KEY: CONFIG_FORMAT_VERSION, **sets} if sets or stored.exists else {}
        if data != stored.data:
            write_managed_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return ConfigSaveResult(merged, stored.quarantined_to)
    finally:
        lock.release()


_FORMAT_VERSION_KEY = "format_version"


def _stored_sets(config: AppConfig) -> dict[str, Any]:
    """Every set that differs from its built-in, whole (config-sets-conventions)."""
    defaults = AppConfig()
    return {
        key: value
        for key, value in _to_json(config).items()
        if getattr(config, key) != getattr(defaults, key)
    }


def _decode_app_config(data: dict[str, Any], path: Path) -> AppConfig:
    decoded: dict[str, Any] = {}
    for key, decode in _SET_DECODERS.items():
        if key not in data:
            continue
        try:
            decoded[key] = decode(data[key])
        except ValueError as exc:
            log.warning("config.invalid_set", path=str(path), key=key, reason=str(exc))
    return AppConfig(**decoded)


def _decode_font_family(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("font_family is not a string")
    return single_line(value)


def _decode_language(value: Any) -> str:
    if not is_preference(value):
        raise ValueError("language is not a recognized preference")
    return value


def _decode_max_concurrent_jobs(value: Any) -> int:
    return _int_range(value, "max_concurrent_jobs", MIN_CONCURRENT_JOBS, MAX_CONCURRENT_JOBS)


def _decode_parameters(value: Any) -> JobSettings:
    data = _object(value, "parameters")
    if not _parameters_to_json(JobSettings()).keys() <= data.keys():
        raise ValueError("parameters is missing a member")
    output_format = _choice(
        data["output_format"], "output_format", tuple(item.value for item in OutputFormat)
    )
    return JobSettings(
        scale=_int_choice(data["scale"], "scale", SCALE_VALUES),
        denoise_strength=_float_range(
            data["denoise_strength"],
            "denoise_strength",
            MIN_DENOISE_STRENGTH,
            MAX_DENOISE_STRENGTH,
        ),
        alpha_mode=_choice(data["alpha_mode"], "alpha_mode", ALPHA_MODE_VALUES),
        device=_choice(data["device"], "device", DEVICE_VALUES),
        output_format=OutputFormat(output_format),
        quality=_int_range(data["quality"], "quality", MIN_QUALITY, MAX_QUALITY),
        tile=_int_choice(data["tile"], "tile", TILE_VALUES),
        strip_metadata=_bool(data["strip_metadata"], "strip_metadata"),
        target_profile=_choice(data["target_profile"], "target_profile", TARGET_PROFILE_VALUES),
    )


_SET_DECODERS: dict[str, Callable[[Any], Any]] = {
    "font_family": _decode_font_family,
    "language": _decode_language,
    "max_concurrent_jobs": _decode_max_concurrent_jobs,
    "parameters": _decode_parameters,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} is not a boolean")
    return value


def _int_range(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} is outside its valid integer range")
    return value


def _float_range(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{name} is outside its valid numeric range")
    return float(value)


def _int_choice(value: Any, name: str, choices: tuple[int, ...]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
        raise ValueError(f"{name} is not a recognized integer choice")
    return value


def _choice(value: Any, name: str, choices: tuple[str | None, ...]) -> Any:
    if not (value is None or isinstance(value, str)) or value not in choices:
        raise ValueError(f"{name} is not a recognized string choice")
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
