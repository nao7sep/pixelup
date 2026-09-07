from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from pixelup.devices import resolve_device, to_torch_device
from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import register_image_plugins
from pixelup.models import model_file
from pixelup.realesrgan_models import RRDBNet, SRVGGNetCompact
from pixelup.realesrgan_runtime import RealESRGANer

ProgressCallback = Callable[[str], None]
TileCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]

@dataclass(frozen=True, slots=True)
class InferenceConfig:
    input_path: Path
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


@dataclass(frozen=True, slots=True)
class ModelArchitectureSpec:
    kind: str
    netscale: int
    params: Mapping[str, int | str]


def model_architecture_spec(model: str, *, requested_scale: int = 4) -> ModelArchitectureSpec:
    match model:
        case "RealESRGAN_x4plus" | "RealESRNet_x4plus":
            return _rrdb_spec(scale=4, num_block=23)
        case "RealESRGAN_x2plus":
            return _rrdb_spec(scale=2, num_block=23)
        case "RealESRGAN_x4plus_anime_6B":
            return _rrdb_spec(scale=4, num_block=6)
        case "realesr-animevideov3":
            return _srvgg_spec(scale=4, num_conv=16)
        case "realesr-general-x4v3":
            return _srvgg_spec(scale=4, num_conv=32)
        case _:
            return _rrdb_spec(scale=requested_scale, num_block=23)


def run_inference(
    config: InferenceConfig,
    *,
    on_progress: ProgressCallback | None = None,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
    try:
        return _run_inference(
            config,
            on_progress=on_progress,
            on_tile=on_tile,
            should_cancel=should_cancel,
        )
    except PixelupError:
        raise
    except RuntimeError as exc:
        if _is_out_of_memory(exc):
            raise PixelupError(
                ErrorCode.OUT_OF_MEMORY,
                "Inference ran out of memory.",
                user_hint="Try a smaller tile size, such as 512.",
            ) from exc
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
    *,
    on_progress: ProgressCallback | None = None,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
    _check_cancelled(should_cancel)
    _emit(on_progress, "load_model")
    _import_torch()  # surface a friendly error if torch is not installed
    image = _read_input_image(config.input_path)
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
    output, _ = upsampler.enhance(image, outscale=config.scale, alpha_upsampler=config.alpha_mode)
    return output


def _rrdb_spec(*, scale: int, num_block: int) -> ModelArchitectureSpec:
    return ModelArchitectureSpec(
        kind="rrdb",
        netscale=scale,
        params={
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": num_block,
            "num_grow_ch": 32,
            "scale": scale,
        },
    )


def _srvgg_spec(*, scale: int, num_conv: int) -> ModelArchitectureSpec:
    return ModelArchitectureSpec(
        kind="srvgg",
        netscale=scale,
        params={
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_conv": num_conv,
            "upscale": scale,
            "act_type": "prelu",
        },
    )


def _create_upsampler(
    config: InferenceConfig,
    *,
    concrete_device: str,
    torch_device: Any,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
    model_paths: str | list[str] = str(model_file(config.models_dir, config.model))
    dni_weight = None
    if config.model == "realesr-general-x4v3" and config.denoise_strength != 1.0:
        model_paths = [
            str(model_file(config.models_dir, "realesr-general-x4v3")),
            str(model_file(config.models_dir, "realesr-general-wdn-x4v3")),
        ]
        dni_weight = [config.denoise_strength, 1 - config.denoise_strength]

    spec = model_architecture_spec(config.model, requested_scale=config.scale)
    needs_tile_subclass = config.tile > 0 and (on_tile is not None or should_cancel is not None)
    cls = _tile_reporting_upsampler_class(RealESRGANer) if needs_tile_subclass else RealESRGANer
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
    # Subclass of PixelUp's RealESRGANer subset that emits a per-tile callback.
    # The override delegates to the adapted implementation and only assumes:
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


def _build_network(spec: ModelArchitectureSpec) -> Any:
    if spec.kind == "rrdb":
        return RRDBNet(**spec.params)
    if spec.kind == "srvgg":
        return SRVGGNetCompact(**spec.params)
    raise PixelupError(
        ErrorCode.INTERNAL_ERROR,
        "Unsupported Real-ESRGAN model architecture.",
        details={"kind": spec.kind},
    )

def _read_input_image(path: Path) -> Any:
    np = _import_numpy()
    register_image_plugins()
    try:
        with Image.open(path) as image:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            array = np.array(image)
    except UnidentifiedImageError as exc:
        raise PixelupError(
            ErrorCode.INPUT_INVALID_FORMAT,
            "Input is not a readable image format.",
            details={"input": str(path)},
        ) from exc
    except PermissionError as exc:
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            "Input image is not readable.",
            details={"input": str(path), "reason": str(exc)},
        ) from exc
    except OSError as exc:
        raise PixelupError(
            ErrorCode.INPUT_UNREADABLE,
            "Input image could not be opened.",
            details={"input": str(path), "reason": str(exc)},
        ) from exc

    channels = array.shape[2]
    if channels == 3:
        return array[:, :, ::-1].copy()
    if channels == 4:
        return array[:, :, [2, 1, 0, 3]].copy()
    raise PixelupError(
        ErrorCode.INPUT_INVALID_FORMAT,
        "Input image has an unsupported channel layout.",
        details={"input": str(path), "channels": channels},
    )


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
    try:
        import torch
    except ImportError as exc:
        raise _missing_inference_dependency("torch", exc) from exc
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


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise _missing_inference_dependency("numpy", exc) from exc
    return np


def _missing_inference_dependency(package: str, exc: ImportError) -> PixelupError:
    return PixelupError(
        ErrorCode.INTERNAL_ERROR,
        f"Inference dependency '{package}' is not installed.",
        details={"dependency": package, "reason": str(exc)},
    )


def _is_out_of_memory(exc: RuntimeError) -> bool:
    torch = _import_torch()
    return isinstance(exc, torch.OutOfMemoryError)


def _emit(callback: ProgressCallback | None, phase: str) -> None:
    if callback:
        callback(phase)


def _check_cancelled(should_cancel: CancelCheck | None) -> None:
    if should_cancel is not None and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
