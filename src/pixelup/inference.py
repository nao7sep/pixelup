from __future__ import annotations

import math
import re
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from pixelup.devices import resolve_device, to_torch_device
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
    _import_torch()  # refuses a torch that would load weights unsafely
    concrete_device = resolve_device(config.device, config.gpu_id)
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
    try:
        output, _ = upsampler.enhance(
            pixels, outscale=config.scale, alpha_upsampler=config.alpha_mode
        )
    finally:
        del upsampler
        _release_device_cache(concrete_device)
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
    needs_cancel_subclass = on_tile is not None or should_cancel is not None
    cls = _tile_reporting_upsampler_class(RealESRGANer) if needs_cancel_subclass else RealESRGANer
    upsampler = cls(
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
    )
    if cls is not RealESRGANer:
        upsampler._pixelup_on_tile = on_tile
        upsampler._pixelup_should_cancel = should_cancel
    return upsampler


def _tile_reporting_upsampler_class(base: type) -> type:
    # Subclass of PixelUp's RealESRGANer subset that emits a per-tile callback and
    # honors should_cancel. tile_process()'s override delegates to the adapted
    # implementation and only assumes:
    #   - self.img.shape is (batch, channel, height, width)
    #   - self.tile_size is the tile edge length
    #   - the upstream loop calls self.model(input_tile) exactly once per tile
    # If a future release breaks any of these, the override silently falls back
    # to plain super().tile_process() and emits no per-tile events. The actual
    # upscale always proceeds. Callback exceptions are swallowed.

    class _TileReportingUpsampler(base):
        def tile_process(self) -> Any:
            callback: TileCallback | None = getattr(self, "_pixelup_on_tile", None)
            should_cancel: CancelCheck | None = getattr(self, "_pixelup_should_cancel", None)
            if callback is None and should_cancel is None:
                return super().tile_process()
            try:
                _, _, height, width = self.img.shape
                tile_size = self.tile_size
                total = math.ceil(width / tile_size) * math.ceil(height / tile_size)
            except Exception:
                return super().tile_process()

            original_model = self.model
            counter = [0]

            def wrapped_model(*args: Any, **kwargs: Any) -> Any:
                if should_cancel is not None and should_cancel():
                    raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
                result = original_model(*args, **kwargs)
                counter[0] += 1
                if callback is not None:
                    try:
                        callback(counter[0], total)
                    except PixelupError:
                        raise
                    except Exception:
                        pass
                return result

            self.model = wrapped_model
            try:
                with redirect_stdout(StringIO()):
                    return super().tile_process()
            finally:
                self.model = original_model

    return _TileReportingUpsampler


def _build_network(spec: ModelArchitecture) -> Any:
    from pixelup.realesrgan_models import RRDBNet, SRVGGNetCompact

    network = RRDBNet if spec.kind == "rrdb" else SRVGGNetCompact
    return network(**spec.params)


# PixelUp's adapted RealESRGANer calls torch.load without a weights_only argument.
# Model-load
# safety therefore rests on torch.load's default weights_only=True (the code-execution
# gate, holding from torch 2.6 onward); a model that pixelup downloaded is additionally
# verified against its pinned SHA-256 at download (models.verify_model_file), while a
# user-placed file is trusted as-is — either way weights_only is what prevents a
# malicious pickle from executing. We assert that 2.6 floor here so a downgrade fails
# loudly rather than silently restoring full-pickle execution. Every shipped model is a
# plain state_dict that loads under weights_only=True; a future model needing a
# non-tensor global would be allowed
# surgically via torch.serialization.add_safe_globals — never with weights_only=False.
_MIN_SAFE_TORCH = (2, 6)


def _import_torch() -> Any:
    import torch

    _assert_safe_torch_load(torch)
    return torch


def _assert_safe_torch_load(torch_module: Any) -> None:
    match = re.match(r"(\d+)\.(\d+)", str(torch_module.__version__))
    if match is None:
        return  # unparseable build string: fall back to the SHA-256 pin, do not brick
    if (int(match.group(1)), int(match.group(2))) < _MIN_SAFE_TORCH:
        raise PixelupError(
            ErrorCode.INTERNAL_ERROR,
            "PixelUp requires torch >= 2.6, where torch.load defaults to the safe "
            "weights_only=True; the installed torch is older and would load model "
            "weights with full unpickling.",
            details={"torch_version": str(torch_module.__version__)},
        )


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
