from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pixelup.config import RuntimeDirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import (
    image_from_bgr_array,
    load_source_metadata,
    read_image_size,
    save_output_image,
)
from pixelup.inference import InferenceConfig, model_architecture_spec, run_inference
from pixelup.models import (
    DownloadCallback,
    WaitingCallback,
    download_model,
    known_model,
    model_present,
    require_model_present,
)
from pixelup.paths import (
    OutputContext,
    OutputFormat,
    RunTimestamp,
    infer_output_format,
    resolve_output_path,
)

StartCallback = Callable[["UpscalePlan", int], None]
ProgressCallback = Callable[[str], None]
WarningCallback = Callable[[str], None]
TileCallback = Callable[[int, int], None]


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
            require_model_present(
                runtime_dirs.models_dir,
                name,
                auto_download_disabled=not options.auto_download,
            )
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
    on_start: StartCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_warning: WarningCallback | None = None,
    on_tile: TileCallback | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    plan = build_plan(
        options,
        runtime_dirs,
        check_model=not options.auto_download,
    )
    for warning in plan_warnings(options, plan):
        if on_warning:
            on_warning(warning)
    if options.dry_run:
        payload = plan.to_payload()
        payload["message"] = "Dry run plan is valid."
        payload["models_present"] = {
            name: model_present(runtime_dirs.models_dir, name)
            for name in required_model_names(options)
        }
        return payload
    if on_start:
        on_start(plan, count_tiles(plan.input_size, options.tile))
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
            require_model_present(
                runtime_dirs.models_dir,
                name,
                auto_download_disabled=True,
            )

    output_array = run_inference(
        InferenceConfig(
            input_path=plan.input_path,
            models_dir=runtime_dirs.models_dir,
            model=options.model,
            scale=options.scale,
            tile=options.tile,
            tile_pad=options.tile_pad,
            pre_pad=options.pre_pad,
            fp32=options.fp32,
            face_enhance=options.face_enhance,
            denoise_strength=options.denoise_strength,
            alpha_mode=options.alpha_mode,
            gpu_id=options.gpu_id,
            device=plan.device,
        ),
        on_progress=on_progress,
        on_tile=on_tile,
    )
    if on_progress:
        on_progress("encode")
    output_size = save_output_image(
        image_from_bgr_array(output_array),
        output_path=plan.output_path,
        output_format=plan.output_format,
        quality=options.quality,
        background=options.background,
        temp_dir=runtime_dirs.temp_dir,
        source_metadata=load_source_metadata(plan.input_path),
        strip_metadata=options.strip_metadata,
        target_profile=options.target_profile,
    )
    return {
        "ok": True,
        "input": str(plan.input_path),
        "output": str(plan.output_path),
        "model": plan.model,
        "scale": plan.scale,
        "input_size": list(plan.input_size),
        "output_size": list(output_size),
        "format": plan.output_format.value,
        "ms": round((time.perf_counter() - started) * 1000),
    }


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


def plan_warnings(options: UpscaleOptions, plan: UpscalePlan) -> list[str]:
    warnings: list[str] = []
    if format_mismatch := _format_extension_mismatch(plan.output_path, plan.output_format):
        warnings.append(
            "Output path extension "
            f"'.{format_mismatch}' does not match requested format "
            f"'{plan.output_format.value}'."
        )
    native_scale = model_architecture_spec(options.model, requested_scale=options.scale).netscale
    if native_scale != options.scale:
        warnings.append(
            f"Model '{options.model}' is trained for {native_scale}x, "
            f"but --scale is {options.scale}x; Real-ESRGAN will rescale the output."
        )
    return warnings


def _format_extension_mismatch(path: Path, output_format: OutputFormat) -> str | None:
    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        return None
    if suffix == "jpeg":
        suffix = "jpg"
    expected = "jpg" if output_format == OutputFormat.JPG else output_format.value
    return suffix if suffix != expected else None


def count_tiles(input_size: tuple[int, int], tile: int) -> int:
    if tile <= 0:
        return 1
    width, height = input_size
    return max(1, math.ceil(width / tile) * math.ceil(height / tile))


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
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if gpu_id is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"
