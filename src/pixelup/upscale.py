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
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.imaging import (
    PublishedCallback,
    image_from_bgr_array,
    load_source_metadata,
    read_image_size,
    save_output_image,
)
from pixelup.inference import InferenceConfig, model_architecture_spec, run_inference
from pixelup.model_management import required_artifact_names
from pixelup.models import require_model_present
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
from pixelup.paths import (
    OutputContext,
    OutputFormat,
    absolute_user_path,
    infer_output_format,
    resolve_output_path,
)
from pixelup.session_log import log

StartCallback = Callable[["UpscalePlan", int], None]
ProgressCallback = Callable[[str], None]
WarningCallback = Callable[[Message], None]
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
    lock_timeout: int


@dataclass(frozen=True, slots=True)
class UpscalePlan:
    input_path: Path
    read_path: Path
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
    input_path = absolute_user_path(options.input_path)
    read_path = input_path.resolve()
    if not read_path.exists():
        raise PixelupError(
            ErrorCode.INPUT_NOT_FOUND,
            Message("error.inputMissing"),
            details={"input": str(input_path)},
        )
    if not read_path.is_file():
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            Message("error.inputNotFile"),
            details={"input": str(input_path)},
        )

    input_size = read_image_size(read_path)
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
        read_path=read_path,
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
    on_start: StartCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_warning: WarningCallback | None = None,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
    on_output_published: PublishedCallback | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    plan = build_plan(
        options,
        runtime_dirs,
        check_model=True,
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
    )
    for warning in plan_warnings(options, plan):
        log.warning("upscale.warning", input=str(plan.input_path), text=english().of(warning))
        if on_warning:
            on_warning(warning)
    if on_start:
        on_start(plan, count_tiles(plan.input_size, options.tile))
    if should_cancel and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
    inference_started = time.perf_counter()
    output_array = run_inference(
        InferenceConfig(
            input_path=plan.read_path,
            models_dir=runtime_dirs.models_dir,
            model=options.model,
            scale=options.scale,
            tile=options.tile,
            tile_pad=options.tile_pad,
            pre_pad=options.pre_pad,
            fp32=options.fp32,
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
        source_metadata=load_source_metadata(plan.read_path),
        strip_metadata=options.strip_metadata,
        target_profile=options.target_profile,
        on_published=on_output_published,
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
    # Every domain below comes from pixelup.parameters, the same leaf the Parameters
    # panel and the config loader read. Hardcoding them here (as this once did) let
    # the panel offer a value this function would reject at runtime.
    if options.scale not in SCALE_VALUES:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("error.scaleInvalid"))
    if options.tile not in TILE_VALUES:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.tileInvalid"),
        )
    if options.tile_pad < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("error.tilePadInvalid"))
    if options.pre_pad < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("error.prePadInvalid"))
    if not MIN_DENOISE_STRENGTH <= options.denoise_strength <= MAX_DENOISE_STRENGTH:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.denoiseInvalid"),
        )
    # Denoise on a non-general model is not an error: it simply does not apply and is normalized
    # to the neutral value (see effective_denoise_strength). Rejecting a non-neutral value here
    # only ever bit direct callers — the GUI already coerces it away per model before validating.
    if options.alpha_mode not in ALPHA_MODE_VALUES:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.alphaModeInvalid"),
        )
    if not MIN_QUALITY <= options.quality <= MAX_QUALITY:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("error.qualityInvalid"))
    if options.target_profile not in TARGET_PROFILE_VALUES:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.targetProfileInvalid"),
        )
    if options.device not in DEVICE_VALUES:
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            Message("error.deviceInvalid"),
        )
    if options.lock_timeout < 0:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, Message("error.lockTimeoutInvalid"))


def required_model_names(options: UpscaleOptions) -> list[str]:
    return list(
        required_artifact_names(
            (options.model,),
            denoise_strength=options.denoise_strength,
        )
    )


def plan_warnings(options: UpscaleOptions, plan: UpscalePlan) -> list[Message]:
    warnings: list[Message] = []
    if format_mismatch := _format_extension_mismatch(plan.output_path, plan.output_format):
        warnings.append(
            Message.of(
                "warning.extensionMismatch",
                extension=format_mismatch,
                format=plan.output_format.value,
            )
        )
    native_scale = model_architecture_spec(options.model, requested_scale=options.scale).netscale
    if native_scale != options.scale:
        # The scales are passed as text: "4x" is a factor, not a quantity to group.
        warnings.append(
            Message.of(
                "warning.scaleMismatch",
                model=options.model,
                native=str(native_scale),
                scale=str(options.scale),
            )
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
            Message("error.outputDirMissing"),
            details={"output": str(path), "parent": str(parent)},
        )
    if not parent.is_dir():
        raise PixelupError(
            ErrorCode.OUTPUT_DIR_MISSING,
            Message("error.outputDirNotDirectory"),
            details={"output": str(path), "parent": str(parent)},
        )
    if path.exists() and not overwrite:
        raise PixelupError(
            ErrorCode.OUTPUT_EXISTS,
            Message("error.outputExists"),
            hint=Message("error.hintRemoveExisting"),
            details={"output": str(path)},
        )
    if not os.access(parent, os.W_OK):
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.outputDirUnwritable"),
            details={"output": str(path), "parent": str(parent)},
        )
