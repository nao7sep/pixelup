from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from pixelup.app_config import AppConfig
from pixelup.paths import OutputFormat, default_output_path
from pixelup.upscale import UpscaleOptions


@dataclass(frozen=True, slots=True)
class AdvancedSettings:
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
    advanced: AdvancedSettings
    auto_download: bool
    status: str = "pending"
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def advanced_defaults(config: AppConfig) -> AdvancedSettings:
    return AdvancedSettings(
        device=config.device,
        output_format=config.output_format,
        quality=config.quality,
        tile=config.tile,
    )


def advanced_for_model(settings: AdvancedSettings, model: str) -> AdvancedSettings:
    if model != "realesr-general-x4v3" and settings.denoise_strength != 1.0:
        return replace(settings, denoise_strength=1.0)
    return settings


def create_jobs(
    *,
    input_paths: list[Path],
    models: list[str],
    scale: int,
    advanced: AdvancedSettings,
    existing_jobs: list[Job],
    auto_download: bool,
    job_ids: Iterator[int],
) -> list[Job]:
    reserved_by_input: dict[Path, set[Path]] = defaultdict(set)
    for job in existing_jobs:
        reserved_by_input[job.input_path].add(job.output_path)

    jobs: list[Job] = []
    for input_path in input_paths:
        reserved = reserved_by_input[input_path]
        for model in models:
            model_advanced = advanced_for_model(advanced, model)
            output_path = default_output_path(
                input_path,
                model=model,
                scale=scale,
                output_format=model_advanced.output_format,
                reserved=reserved,
            )
            reserved.add(output_path)
            jobs.append(
                Job(
                    id=next(job_ids),
                    input_path=input_path,
                    model=model,
                    scale=scale,
                    output_path=output_path,
                    advanced=model_advanced,
                    auto_download=auto_download,
                )
            )
    return jobs


def retry_failed_jobs(jobs: list[Job]) -> list[int]:
    reserved_by_input: dict[Path, set[Path]] = defaultdict(set)
    for job in jobs:
        if job.status != "failed":
            reserved_by_input[job.input_path].add(job.output_path)

    retried: list[int] = []
    for job in jobs:
        if job.status != "failed":
            continue
        reserved = reserved_by_input[job.input_path]
        job.output_path = default_output_path(
            job.input_path,
            model=job.model,
            scale=job.scale,
            output_format=job.advanced.output_format,
            reserved=reserved,
        )
        reserved.add(job.output_path)
        job.status = "pending"
        job.message = ""
        job.warnings = []
        retried.append(job.id)
    return retried


def options_for_job(job: Job) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=job.input_path,
        output_arg=str(job.output_path),
        model=job.model,
        scale=job.scale,
        tile=job.advanced.tile,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
        face_enhance=job.advanced.face_enhance,
        denoise_strength=job.advanced.denoise_strength,
        alpha_mode=job.advanced.alpha_mode,
        gpu_id=None,
        device=job.advanced.device,
        output_format=job.advanced.output_format,
        quality=job.advanced.quality,
        background="white",
        strip_metadata=job.advanced.strip_metadata,
        target_profile=job.advanced.target_profile,
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


def advanced_log_payload(settings: AdvancedSettings) -> dict[str, object]:
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
    }


def job_log_payload(job: Job) -> dict[str, object]:
    return {
        "input_path": str(job.input_path),
        "model": job.model,
        "scale": job.scale,
        "output_path": str(job.output_path),
        "advanced": advanced_log_payload(job.advanced),
        "auto_download": job.auto_download,
    }
