from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelup.devices import to_torch_device
from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message
from pixelup.model_management import (
    DENOISE_NEUTRAL,
    GENERAL_DENOISE_MODEL,
    GENERAL_MODEL,
    model_supports_denoise,
)
from pixelup.model_registry import ModelArchitecture, model_info
from pixelup.models import model_file

ProgressCallback = Callable[[str], None]
TileCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]

@dataclass(frozen=True, slots=True)
class InferenceConfig:
    models_dir: Path
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


def run_inference(
    config: InferenceConfig,
    pixels: Any,
    *,
    on_progress: ProgressCallback | None = None,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
    try:
        try:
            return _run_inference(
                config,
                pixels,
                on_progress=on_progress,
                on_tile=on_tile,
                should_cancel=should_cancel,
            )
        except PixelupError:
            raise
        except MemoryError as exc:
            raise out_of_memory_error() from exc
        except RuntimeError as exc:
            if _is_out_of_memory(exc):
                raise out_of_memory_error() from exc
            raise PixelupError(
                ErrorCode.INTERNAL_ERROR,
                "Inference failed.",
                details={"reason": str(exc)},
            ) from exc
        except Exception as exc:
            raise PixelupError(
                ErrorCode.INTERNAL_ERROR,
                "Inference failed.",
                details={"reason": str(exc)},
            ) from exc
    except BaseException as exc:
        _clear_failure_frames(exc)
        raise
    finally:
        _release_device_cache(config.device)


def _clear_failure_frames(failure: BaseException) -> None:
    # Completed native frames may retain the model, source and tiles. Clear their
    # locals before emptying the device cache, while retaining diagnostic stacks
    # and the original cause. traceback leaves the active settlement frame alone.
    seen: set[int] = set()
    while id(failure) not in seen:
        seen.add(id(failure))
        traceback.clear_frames(failure.__traceback__)
        next_failure = failure.__cause__ or failure.__context__
        if next_failure is None:
            break
        failure = next_failure


def _run_inference(
    config: InferenceConfig,
    pixels: Any,
    *,
    on_progress: ProgressCallback | None = None,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
    _check_cancelled(should_cancel)
    _emit(on_progress, "load_model")
    _import_torch()
    concrete_device = config.device
    torch_device = to_torch_device(concrete_device, config.gpu_id)
    upsampler = _create_upsampler(
        config,
        concrete_device=concrete_device,
        torch_device=torch_device,
        on_tile=on_tile,
        should_cancel=should_cancel,
    )

    _check_cancelled(should_cancel)
    _emit(on_progress, "upscale")
    output, _ = upsampler.enhance(
        pixels, outscale=config.scale, alpha_upsampler=config.alpha_mode
    )
    return output


def _release_device_cache(concrete_device: str) -> None:
    """Hand the GPU memory torch keeps cached back to the system once a job's model
    work ends, so the long encode and the next job do not start with it held. On
    Apple Silicon that memory is the system's own RAM."""
    if concrete_device == "mps":
        _import_torch().mps.empty_cache()
    elif concrete_device == "cuda":
        _import_torch().cuda.empty_cache()


def _create_upsampler(
    config: InferenceConfig,
    *,
    concrete_device: str,
    torch_device: Any,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
    # Native inference libraries belong to the job worker, not window startup.
    from pixelup.realesrgan_runtime import RealESRGANer

    model_paths: str | list[str] = str(model_file(config.models_dir, config.model))
    dni_weight = None
    if model_supports_denoise(config.model) and config.denoise_strength != DENOISE_NEUTRAL:
        model_paths = [
            str(model_file(config.models_dir, GENERAL_MODEL)),
            str(model_file(config.models_dir, GENERAL_DENOISE_MODEL)),
        ]
        dni_weight = [config.denoise_strength, 1 - config.denoise_strength]

    spec = model_info(config.model).architecture
    return RealESRGANer(
        scale=spec.netscale,
        model_path=model_paths,
        dni_weight=dni_weight,
        model=_build_network(spec),
        tile=config.tile,
        tile_pad=config.tile_pad,
        pre_pad=config.pre_pad,
        half=not config.fp32 and concrete_device != "cpu",
        device=torch_device,
        gpu_id=config.gpu_id if concrete_device == "cuda" else None,
        on_tile=on_tile,
        should_cancel=should_cancel,
    )


def _build_network(spec: ModelArchitecture) -> Any:
    from pixelup.realesrgan_models import RRDBNet, SRVGGNetCompact

    network = RRDBNet if spec.kind == "rrdb" else SRVGGNetCompact
    return network(**spec.params)


def _import_torch() -> Any:
    import torch

    return torch


def out_of_memory_error() -> PixelupError:
    """A job stopped for lack of memory, with what the user can change."""
    return PixelupError(
        ErrorCode.OUT_OF_MEMORY,
        Message("error.outOfMemory"),
        hint=Message("error.hintOutOfMemory"),
    )


# How MPS reports exhaustion: a plain RuntimeError, not torch.OutOfMemoryError,
# once an allocation passes torch's MPS ceiling or Metal's per-buffer limit.
_MPS_OUT_OF_MEMORY = ("MPS backend out of memory", "Invalid buffer size")


def _is_out_of_memory(exc: RuntimeError) -> bool:
    torch = _import_torch()
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc)
    return any(marker in text for marker in _MPS_OUT_OF_MEMORY)


def _emit(callback: ProgressCallback | None, phase: str) -> None:
    if callback:
        callback(phase)


def _check_cancelled(should_cancel: CancelCheck | None) -> None:
    if should_cancel is not None and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
