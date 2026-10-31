from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pixelup.config import resolve_state_dir, write_managed_text
from pixelup.devices import DEVICE_VALUES
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY
from pixelup.formats import (
    CONFIG_FORMAT,
    PARAMETERS_FORMAT,
    Format,
    NewerFormatError,
    RefusedFile,
    RefusedFilesError,
    json_content,
    json_text,
)
from pixelup.i18n.languages import SYSTEM, is_preference
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
from pixelup.time_zones import SYSTEM_ZONE, is_zone_preference


def config_path() -> Path:
    """Resolve ``config.json`` under the storage root, lazily on every call.

    The root is resolved here rather than frozen into a module-level constant at
    import time: import-time resolution captures a half-set environment and
    freezes ``PIXELUP_DATA_DIR`` for the life of the process. Resolving on first use
    means a ``PIXELUP_DATA_DIR`` set before launch is honored and tests can vary it.
    """
    return resolve_state_dir() / CONFIG_FILE_NAME


def parameters_path() -> Path:
    """Resolve ``parameters.json`` under the storage root, lazily like ``config_path``."""
    return resolve_state_dir() / PARAMETERS_FILE_NAME


CONFIG_FILE_NAME = "config.json"
PARAMETERS_FILE_NAME = "parameters.json"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Effective settings; each file holds only the sets that differ from their
    built-ins, each whole (config-sets-conventions).

    Each set has one editor: general preferences in Settings, the models folder
    in Managed models, and ``parameters`` in the main
    window's Parameters panel, persisted whole: the panel is the only place those
    values are edited, and ``JobSettings()`` is the only place their built-in
    defaults are written, so there is no second defaults layer to drift against.

    ``parameters`` is stored in ``parameters.json`` and every other set in
    ``config.json``: the panel saves as the user works while the preferences
    change rarely, so each file's backup history and reset follow its own editor
    (data-lifecycle-conventions, things share a file only when they share a
    lifecycle).
    """

    font_family: str = DEFAULT_UI_FONT_FAMILY
    # The interface language: "system" to follow the computer, or a language tag
    # (localization-conventions).
    language: str = SYSTEM
    # The zone the shown times are read in: "system" to follow the computer, or an
    # IANA zone (timestamp-conventions).
    display_zone: str = SYSTEM_ZONE
    parameters: JobSettings = field(default_factory=JobSettings)
    models_folder: str = ""


@dataclass(frozen=True, slots=True)
class SettingsFile:
    """What PixelUp knows of one settings file, which it owns while it runs.

    Startup admits one PixelUp writer per data root, so the file is read once at
    launch and each save writes from this state rather than reading the file again.
    Hand edits made while
    PixelUp runs are replaced by its next save (developer decision).

    ``data`` is the stored content, without its header, as last read or written.
    ``unapplied`` names the keys
    in it this build does not use: sets that failed validation and keys it does not
    know. Their stored values are written back unchanged until the user saves that
    same set, so loading alone never erases a saved setting (config-sets-conventions,
    developer decision).
    """

    path: Path
    exists: bool = False
    data: Mapping[str, Any] = field(default_factory=dict)
    unapplied: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class SettingsFiles:
    """``config.json`` and ``parameters.json``, as PixelUp last read or wrote them."""

    config: SettingsFile
    parameters: SettingsFile

    @classmethod
    def absent(cls, root: Path | None = None) -> SettingsFiles:
        """Both files as not yet written, under ``root`` or the storage root."""
        if root is None:
            return cls(SettingsFile(config_path()), SettingsFile(parameters_path()))
        return cls(SettingsFile(root / CONFIG_FILE_NAME), SettingsFile(root / PARAMETERS_FILE_NAME))

    def __iter__(self) -> Iterator[SettingsFile]:
        return iter((self.config, self.parameters))

    def holding(self, key: str) -> SettingsFile:
        """The file that stores the set ``key``."""
        return self.parameters if key in _PARAMETER_DECODERS else self.config


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    """The outcome of loading the settings files: the settings, the file state saves
    start from, and what the startup shell tells the user.

    ``rejected`` names the sets whose stored values failed validation and run on
    their built-ins while their stored values stay in their file. The startup shell
    surfaces a non-fatal notice for them; the decision stays here, out of the GUI.
    """

    config: AppConfig
    files: SettingsFiles = field(default_factory=SettingsFiles.absent)
    rejected: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfigSaveResult:
    config: AppConfig
    files: SettingsFiles


def load_app_config_result(root: Path | None = None) -> ConfigLoadResult:
    """Load effective sets from both files under ``root`` (the storage root by default).

    Raises ``RefusedFilesError`` naming each file that is unreadable or that a newer
    PixelUp wrote: settings stop the launch, and neither file is changed
    (store-recovery-conventions, What a refused file does).
    """
    files = SettingsFiles.absent(root)
    loads: list[_FileLoad] = []
    refused: list[RefusedFile] = []
    for file, store in ((files.config, _CONFIG_STORE), (files.parameters, _PARAMETERS_STORE)):
        try:
            loads.append(_load_settings_file(file.path, store))
        except NewerFormatError:
            refused.append(RefusedFile(file.path, newer=True))
        except ValueError:
            refused.append(RefusedFile(file.path, newer=False))
    if refused:
        raise RefusedFilesError(tuple(refused))
    config_load, parameters_load = loads
    return ConfigLoadResult(
        AppConfig(**config_load.sets, **parameters_load.sets),
        SettingsFiles(config_load.file, parameters_load.file),
        rejected=config_load.rejected + parameters_load.rejected,
    )


def save_app_config(
    candidate: AppConfig,
    previous: AppConfig,
    files: SettingsFiles,
) -> ConfigSaveResult:
    """Save ``candidate`` whole over its files, from the state PixelUp holds.

    Each file is written only when what it would hold changed, so a Settings save
    leaves ``parameters.json`` alone and a Parameters save or reset leaves
    ``config.json`` alone. Both files passed their header check at launch, which a
    refused file stops, so every file written here is one this build reads.
    """
    stores = ((files.config, _CONFIG_STORE), (files.parameters, _PARAMETERS_STORE))
    saved = []
    for file, store in stores:
        data, kept = _file_data(candidate, previous, file, store)
        if data != file.data:
            write_managed_text(file.path, json_text(store.format, data))
            file = SettingsFile(
                file.path,
                exists=file.exists or bool(data),
                data=MappingProxyType(data),
                unapplied=frozenset(kept),
            )
        saved.append(file)
    return ConfigSaveResult(candidate, SettingsFiles(*saved))


@dataclass(frozen=True, slots=True)
class _FileLoad:
    file: SettingsFile
    sets: dict[str, Any] = field(default_factory=dict)
    rejected: tuple[str, ...] = ()


def _load_settings_file(path: Path, store: _Store) -> _FileLoad:
    """Load one file's sets.

    A missing file leaves the store absent. Raises ``NewerFormatError`` for a file
    a newer PixelUp wrote and ``ValueError`` for one that cannot be read; access
    failures propagate too, and none of them touches the file. A malformed set
    falls back on its own; it does not cost the user the other settings, and its
    stored value is kept.
    """
    if not path.exists():
        return _FileLoad(SettingsFile(path))
    data = json_content(json.loads(path.read_text(encoding="utf-8")), store.format, path)
    sets, rejected = _decode_sets(data, path, store.decoders)
    unknown = {key for key in data if key not in store.decoders}
    return _FileLoad(
        SettingsFile(
            path,
            exists=True,
            data=MappingProxyType(dict(data)),
            unapplied=frozenset(unknown | set(rejected)),
        ),
        sets,
        rejected=rejected,
    )


def _file_data(
    candidate: AppConfig,
    previous: AppConfig,
    file: SettingsFile,
    store: _Store,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """What one file's content holds after saving ``candidate``, and the unapplied
    keys kept.

    Every set that differs from its built-in is written whole, and every other
    known set is left out (config-sets-conventions). A stored value this build did
    not apply is written back unchanged unless this save changes that same set
    (``candidate`` differs from ``previous`` there), which is the user replacing it.
    """
    kept = {
        key: file.data[key]
        for key in file.unapplied
        if key not in store.decoders or getattr(candidate, key) == getattr(previous, key)
    }
    sets = {key: value for key, value in _stored_sets(candidate).items() if key in store.decoders}
    # Written only when it differs from the content last read or written, so with
    # no file and nothing to store, the absence already says every set is at its
    # built-in and nothing is written (config-sets-conventions).
    return {**kept, **sets}, kept


def _stored_sets(config: AppConfig) -> dict[str, Any]:
    """Every set that differs from its built-in, whole (config-sets-conventions)."""
    defaults = AppConfig()
    return {
        key: value
        for key, value in _to_json(config).items()
        if getattr(config, key) != getattr(defaults, key)
    }


def _decode_sets(
    data: dict[str, Any], path: Path, decoders: Mapping[str, Callable[[Any], Any]]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """One file's valid sets, and the keys of the sets that failed validation."""
    decoded: dict[str, Any] = {}
    rejected: list[str] = []
    for key, decode in decoders.items():
        if key not in data:
            continue
        try:
            decoded[key] = decode(data[key])
        except ValueError as exc:
            log.warning("config.invalid_set", path=str(path), key=key, reason=str(exc))
            rejected.append(key)
    return decoded, tuple(rejected)


def _decode_font_family(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("font_family is not a string")
    return single_line(value)


def _decode_models_folder(value: Any) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("models_folder is not a path string")
    return value


def _decode_language(value: Any) -> str:
    if not is_preference(value):
        raise ValueError("language is not a recognized preference")
    return value


def _decode_display_zone(value: Any) -> str:
    if not is_zone_preference(value):
        raise ValueError("display_zone is not a recognized zone")
    return value


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


_CONFIG_DECODERS: dict[str, Callable[[Any], Any]] = {
    "font_family": _decode_font_family,
    "language": _decode_language,
    "display_zone": _decode_display_zone,
    "models_folder": _decode_models_folder,
}
_PARAMETER_DECODERS: dict[str, Callable[[Any], Any]] = {
    "parameters": _decode_parameters,
}


@dataclass(frozen=True, slots=True)
class _Store:
    """One settings file's format and the sets it holds."""

    format: Format
    decoders: Mapping[str, Callable[[Any], Any]]


_CONFIG_STORE = _Store(CONFIG_FORMAT, _CONFIG_DECODERS)
_PARAMETERS_STORE = _Store(PARAMETERS_FORMAT, _PARAMETER_DECODERS)


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
        "display_zone": config.display_zone,
        "parameters": _parameters_to_json(config.parameters),
        "models_folder": config.models_folder,
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
        "font_family": config.font_family,
        "language": config.language,
        "display_zone": config.display_zone,
        "parameters": job_settings_log_payload(config.parameters),
        "models_folder": config.models_folder,
    }
