from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pixelup.config import RuntimeDirs
from pixelup.devices import DEVICE_VALUES, resolve_device
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
    require_model_present,
)
from pixelup.paths import (
    OutputContext,
    OutputFormat,
    infer_output_format,
    resolve_output_path,
)
from pixelup.session_log import log

StartCallback = Callable[["UpscalePlan", int], None]
ProgressCallback = Callable[[str], None]
WarningCallback = Callable[[str], None]
TileCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


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


GENERAL_DENOISE_MODEL = "realesr-general-x4v3"
DENOISE_NEUTRAL = 1.0


def model_supports_denoise(model: str) -> bool:
    """Denoise strength blends in the separately-trained wdn weights, which exist only for the
    general x4v3 model; every other architecture ignores it."""
    return model == GENERAL_DENOISE_MODEL


def effective_denoise_strength(model: str, denoise_strength: float) -> float:
    """The denoise strength a model will actually act on: the caller's value for the general model,
    the neutral ``DENOISE_NEUTRAL`` (no wdn blend) for every other model. Normalizing a non-neutral
    value on a model that ignores denoise — rather than rejecting it — keeps a leftover slider value
    harmless for a direct caller, the same coercion the GUI already applies per model."""
    return denoise_strength if model_supports_denoise(model) else DENOISE_NEUTRAL


@dataclass(frozen=True, slots=True)
class UpscalePlan:
    input_path: Path
    output_path: Path
    model: str
    scale: int
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    output_format: OutputFormat
    device: str


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
    context = OutputContext(
        input_path=input_path,
        output_arg=options.output_arg,
        model=options.model,
        scale=options.scale,
        output_format=output_format,
        input_size=input_size,
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
        device=device,
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
    should_cancel: CancelCheck | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    plan = build_plan(
        options,
        runtime_dirs,
        check_model=not options.auto_download,
    )
    log.info(
        "upscale.planned",
        input=str(plan.input_path),
        output=str(plan.output_path),
        model=plan.model,
        scale=plan.scale,
        input_size=list(plan.input_size),
        output_size=list(plan.output_size),
        output_format=plan.output_format.value,
        device=plan.device,
        auto_download=options.auto_download,
    )
    for warning in plan_warnings(options, plan):
        log.warning("upscale.warning", input=str(plan.input_path), text=warning)
        if on_warning:
            on_warning(warning)
    if on_start:
        on_start(plan, count_tiles(plan.input_size, options.tile))
    if should_cancel and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
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
                should_cancel=should_cancel,
            )

    inference_started = time.perf_counter()
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
        should_cancel=should_cancel,
    )
    log.info(
        "upscale.inference_done",
        model=options.model,
        device=plan.device,
        face_enhance=options.face_enhance,
        duration_ms=round((time.perf_counter() - inference_started) * 1000),
    )
    if should_cancel and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
    if on_progress:
        on_progress("encode")
    output_size = save_output_image(
        image_from_bgr_array(output_array),
        output_path=plan.output_path,
        output_format=plan.output_format,
        quality=options.quality,
        background=options.background,
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
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Scale must be 2x or 4x.")
    if options.tile < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Tile size must be 0 or greater.")
    if options.tile_pad < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Tile padding must be 0 or greater.")
    if options.pre_pad < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Pre-padding must be 0 or greater.")
    if not 0 <= options.denoise_strength <= 1:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Denoise strength must be between 0 and 1.",
        )
    # Denoise on a non-general model is not an error: it simply does not apply and is normalized
    # to the neutral value (see effective_denoise_strength). Rejecting a non-neutral value here
    # only ever bit direct callers — the GUI already coerces it away per model before validating.
    if options.alpha_mode not in {"realesrgan", "bicubic"}:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Alpha mode must be Real-ESRGAN or Bicubic.",
        )
    if not 0 <= options.quality <= 100:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Quality must be between 0 and 100.")
    if options.target_profile not in {None, "srgb", "p3", "adobergb"}:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Target profile must be one of sRGB, Display P3, or Adobe RGB.",
        )
    if options.device not in DEVICE_VALUES:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "Device must be one of Auto, MPS, CUDA, or CPU.",
        )
    if options.download_timeout <= 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Download timeout must be positive.")
    if options.lock_timeout < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "Lock timeout must be 0 or greater.")


def required_model_names(options: UpscaleOptions) -> list[str]:
    names = [options.model]
    if model_supports_denoise(options.model) and options.denoise_strength != 1.0:
        names.append("realesr-general-wdn-x4v3")
    if options.face_enhance:
        names.extend(
            [
                "GFPGANv1.4",
                "facexlib-detection-retinaface-resnet50",
                "facexlib-parsing-parsenet",
            ]
        )
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
            f"but the selected scale is {options.scale}x; Real-ESRGAN will rescale the output."
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
            hint="Remove the existing file, then retry the job.",
            details={"output": str(path)},
        )
    if not os.access(parent, os.W_OK):
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Output parent directory is not writable.",
            details={"output": str(path), "parent": str(parent)},
        )

