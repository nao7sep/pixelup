from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from pixelup.app_config import AppConfig
from pixelup.paths import OutputFormat, default_output_path
from pixelup.upscale import UpscaleOptions, effective_denoise_strength


@dataclass(frozen=True, slots=True)
class JobSettings:
    face_enhance: bool = False
    denoise_strength: float = 0.5
    alpha_mode: str = "realesrgan"
    device: str = "auto"
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = 0
    strip_metadata: bool = False
    target_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ImageEntry:
    input_path: Path
    input_size: tuple[int, int] | None


@dataclass(slots=True)
class Job:
    id: int
    input_path: Path
    model: str
    scale: int
    output_path: Path
    settings: JobSettings
    auto_download: bool
    status: str = "pending"
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def job_settings_defaults(config: AppConfig) -> JobSettings:
    return JobSettings(
        device=config.device,
        output_format=config.output_format,
        quality=config.quality,
        tile=config.tile,
    )


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
    scale: int,
    settings: JobSettings,
    existing_jobs: list[Job],
    auto_download: bool,
    job_ids: Iterator[int],
) -> list[Job]:
    reserved_by_input: dict[Path, set[Path]] = defaultdict(set)
    for job in existing_jobs:
        _reserve_output_bundle(reserved_by_input[job.input_path], job.output_path)

    jobs: list[Job] = []
    for input_path in input_paths:
        reserved = reserved_by_input[input_path]
        for model in models:
            model_settings = settings_for_model(settings, model)
            output_path = default_output_path(
                input_path,
                model=model,
                scale=scale,
                output_format=model_settings.output_format,
                reserved=reserved,
            )
            _reserve_output_bundle(reserved, output_path)
            jobs.append(
                Job(
                    id=next(job_ids),
                    input_path=input_path,
                    model=model,
                    scale=scale,
                    output_path=output_path,
                    settings=model_settings,
                    auto_download=auto_download,
                )
            )
    return jobs


def retry_failed_jobs(jobs: list[Job]) -> list[int]:
    reserved_by_input: dict[Path, set[Path]] = defaultdict(set)
    for job in jobs:
        if job.status != "failed":
            _reserve_output_bundle(reserved_by_input[job.input_path], job.output_path)

    retried: list[int] = []
    for job in jobs:
        if job.status != "failed":
            continue
        reserved = reserved_by_input[job.input_path]
        job.output_path = default_output_path(
            job.input_path,
            model=job.model,
            scale=job.scale,
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
        scale=job.scale,
        tile=job.settings.tile,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
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


def config_log_payload(config: AppConfig) -> dict[str, object]:
    return {
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "output_format": config.output_format.value,
        "quality": config.quality,
        "tile": config.tile,
        "device": config.device,
        "auto_download": config.auto_download,
        "font_family": config.font_family,
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
    return {
        "input_path": str(job.input_path),
        "model": job.model,
        "scale": job.scale,
        "output_path": str(job.output_path),
        "settings": job_settings_log_payload(job.settings),
        "auto_download": job.auto_download,
    }
