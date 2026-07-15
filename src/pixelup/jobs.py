from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from pixelup.devices import DEFAULT_DEVICE
from pixelup.paths import OutputFormat, default_output_path
from pixelup.upscale import UpscaleOptions, effective_denoise_strength

# The valid domain of every image-processing parameter, named and gathered here
# beside the JobSettings they bound. The Parameters panel builds its controls from
# these and the config loader clamps/coerces incoming values against them, so a
# value can never be representable in one place but not the other.
MIN_QUALITY = 0
MAX_QUALITY = 100
MIN_TILE = 0
MAX_TILE = 4096
TILE_STEP = 256
MIN_DENOISE_STRENGTH = 0.0
MAX_DENOISE_STRENGTH = 1.0
DENOISE_STRENGTH_STEP = 0.1

# Ordered (label, value) pairs for the two enumerated parameters, in the shape
# DEVICE_CHOICES already established: labels are for UI display, values are what a
# job carries and what config persistence stores. Keep these the only place either
# set is enumerated for the panel and the loader.
SCALE_CHOICES: tuple[tuple[str, int], ...] = (
    ("2x", 2),
    ("4x", 4),
)
SCALE_VALUES: tuple[int, ...] = tuple(value for _label, value in SCALE_CHOICES)
ALPHA_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Real-ESRGAN", "realesrgan"),
    ("Bicubic", "bicubic"),
)
ALPHA_MODE_VALUES: tuple[str, ...] = tuple(value for _label, value in ALPHA_MODE_CHOICES)
TARGET_PROFILE_CHOICES: tuple[tuple[str, str | None], ...] = (
    ("Default", None),
    ("sRGB", "srgb"),
    ("Display P3", "p3"),
    ("Adobe RGB", "adobergb"),
)
TARGET_PROFILE_VALUES: tuple[str | None, ...] = tuple(
    value for _label, value in TARGET_PROFILE_CHOICES
)

# Tiling is on by default so peak memory scales with the tile, not the image: a
# whole-image pass (tile=0) can exhaust GPU/MPS memory and hard-crash on large
# inputs. 256 keeps peak memory low enough to run on modest GPUs and smaller-memory
# machines; output is effectively identical to larger tiles, and a power user can
# raise it — or deliberately choose 0, which stays selectable (MIN_TILE) — in the
# Parameters panel.
DEFAULT_TILE = 256

# 4x is the scale PixelUp has always opened on, and the one every bundled model is
# trained for (the x2 model is the lone exception, and plan_warnings covers the
# mismatch), so it stays the built-in. 2x remains selectable in the Parameters panel.
DEFAULT_SCALE = 4


@dataclass(frozen=True, slots=True)
class JobSettings:
    """The image-processing parameters a job runs with — and, constructed bare, the
    single source of PixelUp's built-in parameter defaults.

    ``JobSettings()`` *is* the built-ins: the Parameters panel's "Reset parameters"
    restores exactly this, and the config loader falls back to it field by field.
    There is deliberately no second defaults layer to drift against — the field
    defaults below are the only place a built-in parameter value is written.
    """

    scale: int = DEFAULT_SCALE
    face_enhance: bool = False
    denoise_strength: float = 0.5
    alpha_mode: str = "realesrgan"
    device: str = DEFAULT_DEVICE
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = DEFAULT_TILE
    strip_metadata: bool = False
    target_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ImageEntry:
    input_path: Path
    input_size: tuple[int, int] | None


@dataclass(slots=True)
class Job:
    """One queued unit of work: an input, a model, and the panel as it stood.

    ``settings`` is the enqueue snapshot — the Parameters panel captured whole at the
    moment the job was created, scale included. The panel may move on afterwards; a
    queued job never does.
    """

    id: int
    input_path: Path
    model: str
    output_path: Path
    settings: JobSettings
    auto_download: bool
    status: str = "pending"
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def settings_for_model(settings: JobSettings, model: str) -> JobSettings:
    # Single source of the "denoise applies only to the general model" rule (see upscale).
    normalized = effective_denoise_strength(model, settings.denoise_strength)
    if normalized == settings.denoise_strength:
        return settings
    return replace(settings, denoise_strength=normalized)


def create_jobs(
    *,
    input_paths: list[Path],
    models: list[str],
    settings: JobSettings,
    existing_jobs: list[Job],
    auto_download: bool,
    job_ids: Iterator[int],
) -> list[Job]:
    """Plan a batch of jobs from the panel snapshot in ``settings``.

    ``settings`` carries every parameter the batch runs with — scale among them — so
    the snapshot each job freezes is exactly the one the caller passed, with nothing
    arriving alongside it to drift out of sync.
    """
    # One reservation set for the whole batch, keyed by resolved absolute path, so
    # inputs whose stems differ only in case (Photo.png vs photo.png) disambiguate
    # against each other and not just against pre-existing files.
    reserved: set[Path] = set()
    for job in existing_jobs:
        _reserve_output_bundle(reserved, job.output_path)

    jobs: list[Job] = []
    for input_path in input_paths:
        for model in models:
            model_settings = settings_for_model(settings, model)
            output_path = default_output_path(
                input_path,
                model=model,
                scale=model_settings.scale,
                output_format=model_settings.output_format,
                reserved=reserved,
            )
            _reserve_output_bundle(reserved, output_path)
            jobs.append(
                Job(
                    id=next(job_ids),
                    input_path=input_path,
                    model=model,
                    output_path=output_path,
                    settings=model_settings,
                    auto_download=auto_download,
                )
            )
    return jobs


def retry_failed_jobs(jobs: list[Job]) -> list[int]:
    # One reservation set for the whole batch (see create_jobs): case-only
    # sibling inputs must disambiguate against each other, not just live files.
    reserved: set[Path] = set()
    for job in jobs:
        if job.status != "failed":
            _reserve_output_bundle(reserved, job.output_path)

    retried: list[int] = []
    for job in jobs:
        if job.status != "failed":
            continue
        job.output_path = default_output_path(
            job.input_path,
            model=job.model,
            scale=job.settings.scale,
            output_format=job.settings.output_format,
            reserved=reserved,
        )
        _reserve_output_bundle(reserved, job.output_path)
        job.status = "pending"
        job.message = ""
        job.warnings = []
        retried.append(job.id)
    return retried


def _reserve_output_bundle(reserved: set[Path], output_path: Path) -> None:
    reserved.add(output_path)
    reserved.add(output_path.with_suffix(".json"))


def options_for_job(job: Job) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=job.input_path,
        output_arg=str(job.output_path),
        model=job.model,
        scale=job.settings.scale,
        tile=job.settings.tile,
        tile_pad=10,
        pre_pad=0,
        # Full precision by default: half precision on MPS can produce black/NaN
        # Real-ESRGAN output, and tiling already bounds the extra memory cost.
        fp32=True,
        face_enhance=job.settings.face_enhance,
        denoise_strength=job.settings.denoise_strength,
        alpha_mode=job.settings.alpha_mode,
        gpu_id=None,
        device=job.settings.device,
        output_format=job.settings.output_format,
        quality=job.settings.quality,
        background="white",
        strip_metadata=job.settings.strip_metadata,
        target_profile=job.settings.target_profile,
        overwrite=False,
        auto_download=job.auto_download,
        download_timeout=600,
        lock_timeout=600,
    )


def coerce_output_format(value: OutputFormat | str | object) -> OutputFormat:
    if isinstance(value, OutputFormat):
        return value
    if isinstance(value, str):
        return OutputFormat(value)
    raise ValueError(f"Unsupported output format: {value!r}")


def job_settings_log_payload(settings: JobSettings) -> dict[str, object]:
    return {
        "scale": settings.scale,
        "face_enhance": settings.face_enhance,
        "denoise_strength": settings.denoise_strength,
        "alpha_mode": settings.alpha_mode,
        "device": settings.device,
        "output_format": coerce_output_format(settings.output_format).value,
        "quality": settings.quality,
        "tile": settings.tile,
        "strip_metadata": settings.strip_metadata,
        "target_profile": settings.target_profile,
    }


def job_status_summary(statuses: Iterable[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for status in statuses:
        total += 1
        counts[status] += 1
    if total == 0:
        return "No jobs"
    queued = counts["pending"] + counts["running"] + counts["cancelling"]
    parts: list[str] = []
    if counts["succeeded"]:
        parts.append(f"{counts['succeeded']} done")
    if counts["failed"]:
        parts.append(f"{counts['failed']} failed")
    if counts["cancelled"]:
        parts.append(f"{counts['cancelled']} cancelled")
    if queued:
        parts.append(f"{queued} queued")
    return ", ".join(parts)


def job_log_payload(job: Job) -> dict[str, object]:
    # No top-level "scale": it lives in the settings payload now, and logging it twice
    # would be two places to drift.
    return {
        "input_path": str(job.input_path),
        "model": job.model,
        "output_path": str(job.output_path),
        "settings": job_settings_log_payload(job.settings),
        "auto_download": job.auto_download,
    }
