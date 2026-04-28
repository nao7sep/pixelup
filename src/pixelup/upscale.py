from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pixelup.config import RuntimeDirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import read_image_size
from pixelup.models import (
    DownloadCallback,
    WaitingCallback,
    download_model,
    known_model,
    require_model_present,
)
from pixelup.paths import (
    OutputContext,
    OutputFormat,
    RunTimestamp,
    infer_output_format,
    resolve_output_path,
)


@dataclass(frozen=True, slots=True)
class UpscaleOptions:
    input_path: Path
    output_arg: str
    model: str
    scale: int
    tile: int
    tile_pad: int
    pre_pad: int
    fp32: bool
    face_enhance: bool
    denoise_strength: float
    alpha_mode: str
    gpu_id: int | None
    device: str
    output_format: OutputFormat | None
    quality: int
    background: str
    strip_metadata: bool
    target_profile: str | None
    overwrite: bool
    auto_download: bool
    download_timeout: int
    lock_timeout: int
    dry_run: bool


@dataclass(frozen=True, slots=True)
class UpscalePlan:
    input_path: Path
    output_path: Path
    model: str
    scale: int
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    output_format: OutputFormat
    runtime_dirs: RuntimeDirs
    device: str
    dry_run: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "ok": True,
            "input": str(self.input_path),
            "output": str(self.output_path),
            "model": self.model,
            "scale": self.scale,
            "input_size": list(self.input_size),
            "output_size": list(self.output_size),
            "format": self.output_format.value,
            "device": self.device,
            "dry_run": self.dry_run,
            "models_dir": str(self.runtime_dirs.models_dir),
            "temp_dir": str(self.runtime_dirs.temp_dir),
        }


def build_plan(
    options: UpscaleOptions,
    runtime_dirs: RuntimeDirs,
    *,
    check_model: bool = True,
) -> UpscalePlan:
    validate_options(options)
    input_path = options.input_path.expanduser().resolve()
    if not input_path.exists():
        raise PixelupError(
            ErrorCode.INPUT_NOT_FOUND,
            "Input image does not exist.",
            details={"input": str(input_path)},
        )
    if not input_path.is_file():
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            "Input path is not a file.",
            details={"input": str(input_path)},
        )

    input_size = read_image_size(input_path)
    output_format = infer_output_format(options.output_arg, options.output_format)
    timestamp = RunTimestamp.now()
    context = OutputContext(
        input_path=input_path,
        output_arg=options.output_arg,
        model=options.model,
        scale=options.scale,
        output_format=output_format,
        input_size=input_size,
        face_enhance=options.face_enhance,
        denoise_strength=options.denoise_strength,
        timestamp=timestamp,
    )
    output_path = resolve_output_path(context)
    validate_output_path(output_path, overwrite=options.overwrite)
    if check_model:
        for name in required_model_names(options):
            require_model_present(runtime_dirs.models_dir, name)
    device = resolve_device(options.device, options.gpu_id)
    return UpscalePlan(
        input_path=input_path,
        output_path=output_path,
        model=options.model,
        scale=options.scale,
        input_size=input_size,
        output_size=context.output_size,
        output_format=output_format,
        runtime_dirs=runtime_dirs,
        device=device,
        dry_run=options.dry_run,
    )


def run_upscale(
    options: UpscaleOptions,
    runtime_dirs: RuntimeDirs,
    *,
    on_download: DownloadCallback | None = None,
    on_waiting: WaitingCallback | None = None,
) -> dict[str, object]:
    plan = build_plan(
        options,
        runtime_dirs,
        check_model=options.dry_run or not options.auto_download,
    )
    if options.dry_run:
        payload = plan.to_payload()
        payload["message"] = "Dry run plan is valid."
        return payload
    if options.auto_download:
        for name in required_model_names(options):
            if known_model(name) is None:
                require_model_present(runtime_dirs.models_dir, name)
                continue
            download_model(
                runtime_dirs.models_dir,
                name,
                download_timeout=options.download_timeout,
                lock_timeout=options.lock_timeout,
                on_download=on_download,
                on_waiting=on_waiting,
            )
    else:
        for name in required_model_names(options):
            require_model_present(runtime_dirs.models_dir, name)
    raise PixelupError(
        ErrorCode.INTERNAL_ERROR,
        "Real-ESRGAN inference is not implemented in this phase.",
        hint="Use --dry-run in phase 1, or continue with the next implementation phase.",
    )


def validate_options(options: UpscaleOptions) -> None:
    if options.scale not in {2, 4}:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--scale must be 2 or 4.")
    if options.tile < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--tile must be 0 or greater.")
    if options.tile_pad < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--tile-pad must be 0 or greater.")
    if options.pre_pad < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--pre-pad must be 0 or greater.")
    if not 0 <= options.denoise_strength <= 1:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "--denoise-strength must be between 0 and 1.",
        )
    if options.denoise_strength != 1.0 and options.model != "realesr-general-x4v3":
        raise PixelupError(
            ErrorCode.DENOISE_STRENGTH_UNSUPPORTED,
            "--denoise-strength is only valid with model 'realesr-general-x4v3'.",
            details={"model": options.model, "denoise_strength": options.denoise_strength},
        )
    if options.alpha_mode not in {"realesrgan", "bicubic"}:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "--alpha-mode must be 'realesrgan' or 'bicubic'.",
        )
    if not 0 <= options.quality <= 100:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--quality must be between 0 and 100.")
    if options.target_profile not in {None, "srgb", "p3", "adobergb"}:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "--target-profile must be one of 'srgb', 'p3', or 'adobergb'.",
        )
    if options.device not in {"auto", "mps", "cuda", "cpu"}:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "--device must be one of 'auto', 'mps', 'cuda', or 'cpu'.",
        )
    if options.download_timeout <= 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--download-timeout must be positive.")
    if options.lock_timeout < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "--lock-timeout must be 0 or greater.")


def required_model_names(options: UpscaleOptions) -> list[str]:
    names = [options.model]
    if options.model == "realesr-general-x4v3" and options.denoise_strength != 1.0:
        names.append("realesr-general-wdn-x4v3")
    if options.face_enhance:
        names.append("GFPGANv1.4")
    return names


def validate_output_path(path: Path, *, overwrite: bool) -> None:
    parent = path.parent
    if not parent.exists():
        raise PixelupError(
            ErrorCode.OUTPUT_DIR_MISSING,
            "Output parent directory does not exist.",
            details={"output": str(path), "parent": str(parent)},
        )
    if not parent.is_dir():
        raise PixelupError(
            ErrorCode.OUTPUT_DIR_MISSING,
            "Output parent path is not a directory.",
            details={"output": str(path), "parent": str(parent)},
        )
    if path.exists() and not overwrite:
        raise PixelupError(
            ErrorCode.OUTPUT_EXISTS,
            "Output file already exists.",
            hint="Use --overwrite to replace the existing file.",
            details={"output": str(path)},
        )
    if not os.access(parent, os.W_OK):
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Output parent directory is not writable.",
            details={"output": str(path), "parent": str(parent)},
        )


def resolve_device(device: str, gpu_id: int | None) -> str:
    if device != "auto":
        return device
    if _is_apple_silicon():
        return "mps"
    if gpu_id is not None:
        return "cuda"
    return "cpu"


def _is_apple_silicon() -> bool:
    return os.uname().sysname == "Darwin" and os.uname().machine == "arm64"
