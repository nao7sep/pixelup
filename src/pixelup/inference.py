from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import register_image_plugins
from pixelup.models import model_file

ProgressCallback = Callable[[str], None]

_INFERENCE_DEPS_HINT = (
    "Install the PixelUp inference stack: torch, torchvision, realesrgan, "
    "basicsr-fixed, opencv-python, and gfpgan when using --face-enhance."
)


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
    face_enhance: bool
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
) -> Any:
    try:
        return _run_inference(config, on_progress=on_progress)
    except PixelupError:
        raise
    except RuntimeError as exc:
        if _is_out_of_memory(exc):
            raise PixelupError(
                ErrorCode.OUT_OF_MEMORY,
                "Inference ran out of memory.",
                hint="Try a smaller --tile value such as --tile 512.",
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
) -> Any:
    _emit(on_progress, "load_model")
    torch = _import_torch()
    image = _read_input_image(config.input_path)
    device = _torch_device(torch, config.device, config.gpu_id)
    upsampler = _create_upsampler(config, torch_device=device)

    _emit(on_progress, "upscale")
    if config.face_enhance:
        _emit(on_progress, "face_enhance")
        return _run_face_enhance(config, upsampler, image)
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


def _create_upsampler(config: InferenceConfig, *, torch_device: Any) -> Any:
    try:
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise _missing_inference_dependency("realesrgan", exc) from exc

    model_paths: str | list[str] = str(model_file(config.models_dir, config.model))
    dni_weight = None
    if config.model == "realesr-general-x4v3" and config.denoise_strength != 1.0:
        model_paths = [
            str(model_file(config.models_dir, "realesr-general-x4v3")),
            str(model_file(config.models_dir, "realesr-general-wdn-x4v3")),
        ]
        dni_weight = [config.denoise_strength, 1 - config.denoise_strength]

    spec = model_architecture_spec(config.model, requested_scale=config.scale)
    return RealESRGANer(
        scale=spec.netscale,
        model_path=model_paths,
        dni_weight=dni_weight,
        model=_build_network(spec),
        tile=config.tile,
        tile_pad=config.tile_pad,
        pre_pad=config.pre_pad,
        half=not config.fp32 and config.device != "cpu",
        device=torch_device,
        gpu_id=config.gpu_id if config.device == "cuda" else None,
    )


def _build_network(spec: ModelArchitectureSpec) -> Any:
    if spec.kind == "rrdb":
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as exc:
            raise _missing_inference_dependency("basicsr-fixed", exc) from exc
        return RRDBNet(**spec.params)
    if spec.kind == "srvgg":
        try:
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        except ImportError as exc:
            raise _missing_inference_dependency("realesrgan", exc) from exc
        return SRVGGNetCompact(**spec.params)
    raise PixelupError(
        ErrorCode.INTERNAL_ERROR,
        "Unsupported Real-ESRGAN model architecture.",
        details={"kind": spec.kind},
    )


def _run_face_enhance(config: InferenceConfig, upsampler: Any, image: Any) -> Any:
    try:
        from gfpgan import GFPGANer
    except ImportError as exc:
        raise PixelupError(
            ErrorCode.FACE_ENHANCE_UNAVAILABLE,
            "GFPGAN is not installed.",
            hint=_INFERENCE_DEPS_HINT,
        ) from exc

    face_enhancer = GFPGANer(
        model_path=str(model_file(config.models_dir, "GFPGANv1.4")),
        upscale=config.scale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )
    _, _, output = face_enhancer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    return output


def _read_input_image(path: Path) -> Any:
    np = _import_numpy()
    register_image_plugins()
    try:
        with Image.open(path) as image:
            if image.mode not in {"RGB", "RGBA", "L"}:
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

    if getattr(array, "ndim", 0) == 2:
        return array
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


def _torch_device(torch: Any, device: str, gpu_id: int | None) -> Any:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps and mps.is_available())
    if device == "auto":
        if mps_available:
            return torch.device("mps")
        if gpu_id is not None:
            device = "cuda"
        else:
            return torch.device("cpu")
    if device == "mps":
        if not mps_available:
            raise PixelupError(ErrorCode.INVALID_ARGUMENT, "MPS is not available.")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise PixelupError(ErrorCode.INVALID_ARGUMENT, "CUDA is not available.")
        if gpu_id is not None and gpu_id >= torch.cuda.device_count():
            raise PixelupError(
                ErrorCode.INVALID_ARGUMENT,
                "CUDA GPU index is not available.",
                details={"gpu_id": gpu_id, "device_count": torch.cuda.device_count()},
            )
        return torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise PixelupError(
        ErrorCode.INVALID_ARGUMENT,
        "--device must be one of 'auto', 'mps', 'cuda', or 'cpu'.",
    )


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise _missing_inference_dependency("torch", exc) from exc
    return torch


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
        hint=_INFERENCE_DEPS_HINT,
        details={"dependency": package, "reason": str(exc)},
    )


def _is_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _emit(callback: ProgressCallback | None, phase: str) -> None:
    if callback:
        callback(phase)
