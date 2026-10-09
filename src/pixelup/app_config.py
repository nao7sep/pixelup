from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
class SettingsFile:
    """What PixelUp knows of ``config.json``, which it owns while it runs.

    One PixelUp process is the supported writer (a second instance is neither
    prevented nor coordinated), so the file is read once at launch and each save
    writes from this state rather than reading the file again. Hand edits made while
    PixelUp runs are replaced by its next save (developer decision).

    ``data`` is the stored map as last read or written. ``unapplied`` names the keys
    in it this build does not use: sets that failed validation and keys it does not
    know. Their stored values are written back unchanged until the user saves that
    same set, so loading alone never erases a saved setting (config-sets-conventions,
    developer decision). ``newer_format`` is the version of a file a newer PixelUp
    wrote; such a file is never written.
    """

    path: Path
    exists: bool = False
    data: Mapping[str, Any] = field(default_factory=dict)
    unapplied: frozenset[str] = frozenset()
    newer_format: int | None = None


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    """The outcome of loading ``config.json``: the settings, the file state saves
    start from, and what the startup shell tells the user.

    ``quarantined_to`` is the path the unreadable original was moved aside to
    (``<stem>-<utc>.invalid``), else ``None``. ``file.newer_format`` is set when a
    newer PixelUp wrote the file. Either way the settings fall back to built-ins in
    memory. ``rejected`` names the sets whose stored values failed validation and
    run on their built-ins while their stored values stay in the file. The startup
    shell surfaces a non-fatal notice for each case; the decision stays here, out of
    the GUI.
    """

    config: AppConfig
    file: SettingsFile
    quarantined_to: Path | None = None
    rejected: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfigSaveResult:
    config: AppConfig
    file: SettingsFile


def load_app_config(path: Path | None = None) -> AppConfig:
    """Load effective settings, quarantining a corrupt file without writing a replacement.

    Thin accessor over :func:`load_app_config_result` for the many callers that only
    need the ``AppConfig``.
    """
    return load_app_config_result(path).config


def load_app_config_result(path: Path | None = None) -> ConfigLoadResult:
    """Load effective sets, preserving unreadable JSON through quarantine.

    Missing files and quarantined files both leave the store absent. A malformed
    set falls back on its own; it does not cost the user the other settings, and
    its stored value is kept. Access failures propagate without moving or
    overwriting the file.
    """
    if path is None:
        path = config_path()
    if not path.exists():
        return ConfigLoadResult(AppConfig(), SettingsFile(path))
    try:
        data = _object(json.loads(path.read_text(encoding="utf-8")), "config")
        version = format_version(data.get(_FORMAT_VERSION_KEY))
    except ValueError:
        return ConfigLoadResult(
            AppConfig(), SettingsFile(path), quarantined_to=quarantine_corrupt_file(path)
        )
    if version > CONFIG_FORMAT_VERSION:
        # Intact data this build cannot read: left exactly in place, and never
        # written (store-recovery-conventions).
        return ConfigLoadResult(AppConfig(), SettingsFile(path, exists=True, newer_format=version))
    config, rejected = _decode_app_config(data, path)
    unknown = {key for key in data if key != _FORMAT_VERSION_KEY and key not in _SET_DECODERS}
    return ConfigLoadResult(
        config,
        SettingsFile(
            path,
            exists=True,
            data=MappingProxyType(dict(data)),
            unapplied=frozenset(unknown | set(rejected)),
        ),
        rejected=rejected,
    )


def save_app_config(
    candidate: AppConfig,
    previous: AppConfig,
    file: SettingsFile,
) -> ConfigSaveResult:
    """Save ``candidate`` whole over ``config.json``, from the state PixelUp holds.

    Every set that differs from its built-in is written whole, and every other
    known set is left out (config-sets-conventions). A stored value this build did
    not apply is written back unchanged unless this save changes that same set
    (``candidate`` differs from ``previous`` there), which is the user replacing it.
    """
    if file.newer_format is not None:
        raise PixelupError(
            ErrorCode.STORE_NEWER_FORMAT,
            Message("error.configNewer"),
            details={"path": str(file.path), "format_version": file.newer_format},
        )
    kept = {
        key: file.data[key]
        for key in file.unapplied
        if key not in _SET_DECODERS or getattr(candidate, key) == getattr(previous, key)
    }
    sets = _stored_sets(candidate)
    # With no file and nothing to store, the absence already says every set is at
    # its built-in, so nothing is written (config-sets-conventions).
    data = (
        {_FORMAT_VERSION_KEY: CONFIG_FORMAT_VERSION, **kept, **sets}
        if sets or kept or file.exists
        else {}
    )
    if data != file.data:
        write_managed_text(file.path, json.dumps(data, indent=2, sort_keys=True) + "\n")
    saved = SettingsFile(
        file.path,
        exists=file.exists or bool(data),
        data=MappingProxyType(data),
        unapplied=frozenset(kept),
    )
    return ConfigSaveResult(candidate, saved)


_FORMAT_VERSION_KEY = "format_version"


def _stored_sets(config: AppConfig) -> dict[str, Any]:
    """Every set that differs from its built-in, whole (config-sets-conventions)."""
    defaults = AppConfig()
    return {
        key: value
        for key, value in _to_json(config).items()
        if getattr(config, key) != getattr(defaults, key)
    }


def _decode_app_config(data: dict[str, Any], path: Path) -> tuple[AppConfig, tuple[str, ...]]:
    """The effective settings, and the keys of the sets that failed validation."""
    decoded: dict[str, Any] = {}
    rejected: list[str] = []
    for key, decode in _SET_DECODERS.items():
        if key not in data:
            continue
        try:
            decoded[key] = decode(data[key])
        except ValueError as exc:
            log.warning("config.invalid_set", path=str(path), key=key, reason=str(exc))
            rejected.append(key)
    return AppConfig(**decoded), tuple(rejected)


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
