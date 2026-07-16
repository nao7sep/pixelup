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

    Two kinds of thing, one home each. The three scalars are the Settings modal's
    whole content — what the main window does not show. ``parameters`` is the main
    window's Parameters panel, persisted whole: the panel is the only place those
    values are edited, and ``JobSettings()`` is the only place their built-in
    defaults are written, so there is no second defaults layer to drift against.
    """

    max_concurrent_jobs: int = MIN_CONCURRENT_JOBS
    auto_download: bool = True
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
            auto_download=bool(data.get("auto_download", defaults.auto_download)),
            font_family=normalize_font_family(data.get("font_family"), defaults.font_family),
            parameters=_load_parameters(data.get("parameters"), defaults.parameters),
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


def _load_parameters(value: Any, defaults: JobSettings) -> JobSettings:
    """Load the Parameters panel field by field, each falling back to its built-in.

    Same lenient-load contract as the scalars above: a missing, absent-typed, or
    out-of-domain field quietly becomes its ``JobSettings()`` default rather than
    failing the load, because config.json is disposable preferences and one bad
    field must not cost the user the other nine. Only whole-file corruption is
    escalated (quarantine-then-reset, see load_app_config_result); a ``parameters``
    key that is not an object is just a field that cannot be read, so the whole
    panel falls back to the built-ins.
    """
    if not isinstance(value, dict):
        return defaults
    return JobSettings(
        scale=_coerce_scale(value.get("scale"), defaults.scale),
        face_enhance=bool(value.get("face_enhance", defaults.face_enhance)),
        denoise_strength=_clamp_float(
            value.get("denoise_strength"),
            defaults.denoise_strength,
            MIN_DENOISE_STRENGTH,
            MAX_DENOISE_STRENGTH,
        ),
        alpha_mode=_coerce_alpha_mode(value.get("alpha_mode"), defaults.alpha_mode),
        device=_coerce_device(value.get("device"), defaults.device),
        output_format=_coerce_output_format(value.get("output_format"), defaults.output_format),
        quality=_clamp_int(value.get("quality"), defaults.quality, MIN_QUALITY, MAX_QUALITY),
        tile=_coerce_tile(value.get("tile"), defaults.tile),
        strip_metadata=bool(value.get("strip_metadata", defaults.strip_metadata)),
        target_profile=_coerce_target_profile(
            value.get("target_profile"), defaults.target_profile
        ),
    )


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _coerce_scale(value: Any, default: int) -> int:
    """Read a persisted scale, falling back to the built-in unless it names a real choice.

    Enumerated, not clamped — the difference matters. ``quality`` is a range, where
    snapping a stray value to the nearest end is a faithful reading of what the user
    meant. Scale is a choice between 2x and 4x: a 3 is not "close to"
    either one, so it falls back to the built-in the way an unknown ``alpha_mode``
    does. Membership is therefore tested against the value itself rather than an
    ``int()`` of it, so a hand-edited 2.5 cannot truncate into a selectable 2 the user
    never picked; a numeric string is still read, matching the other loaders' tolerance.
    """
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return default
    for scale in SCALE_VALUES:
        if value == scale:
            return scale
    return default


def _coerce_tile(value: Any, default: int) -> int:
    """Read a persisted tile size, falling back to the built-in unless it names a real choice.

    Enumerated for the same reason as scale, and it changed camps: tile used to be a
    clamped range, so a hand-edited 3000 became 2816 or 4096 — a size no one picked and
    the panel can no longer show. The sizes are doublings, not points on a continuum, so
    a stray value is not "near" one of them. 0 is a real choice here (the whole-image
    pass), not the bottom of a range.
    """
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return default
    for tile in TILE_VALUES:
        if value == tile:
            return tile
    return default


def _coerce_device(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).lower()
    return text if text in DEVICE_VALUES else default


def _coerce_alpha_mode(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).lower()
    return text if text in ALPHA_MODE_VALUES else default


def _coerce_target_profile(value: Any, default: str | None) -> str | None:
    # ``None`` is itself a valid target profile ("Default", no conversion), and it is
    # also the built-in, so a missing key and an explicit null land on the same value
    # through the default.
    if value is None:
        return default
    text = str(value).lower()
    return text if text in TARGET_PROFILE_VALUES else default


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
        "auto_download": config.auto_download,
        "font_family": config.font_family,
        "parameters": job_settings_log_payload(config.parameters),
    }
