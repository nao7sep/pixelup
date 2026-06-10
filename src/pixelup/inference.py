from __future__ import annotations

import math
import threading
import warnings
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import register_image_plugins
from pixelup.models import model_file

ProgressCallback = Callable[[str], None]
TileCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]

# GFPGANer hardcodes the facexlib model directory when it builds its
# FaceRestoreHelper, so the only way to point face-detection and parsing
# weights at the PixelUp models directory is to substitute the
# FaceRestoreHelper symbol GFPGANer reads during construction. That
# substitution mutates a module global, so it is held only for the duration
# of construction and serialized across worker threads.
_GFPGAN_HELPER_LOCK = threading.Lock()


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
                hint="Try a smaller tile size, such as 512.",
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
    torch = _import_torch()
    image = _read_input_image(config.input_path)
    device = _torch_device(torch, config.device, config.gpu_id)
    upsampler = _create_upsampler(
        config,
        torch_device=device,
        on_tile=on_tile,
        should_cancel=should_cancel,
    )

    _check_cancelled(should_cancel)
    _emit(on_progress, "upscale")
    if config.face_enhance:
        _emit(on_progress, "face_enhance")
        return _run_face_enhance(config, upsampler, image, torch_device=device)
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
    torch_device: Any,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Any:
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
        half=not config.fp32 and config.device != "cpu",
        device=torch_device,
        gpu_id=config.gpu_id if config.device == "cuda" else None,
    )
    if cls is not RealESRGANer:
        upsampler._pixelup_on_tile = on_tile  # type: ignore[attr-defined]
        upsampler._pixelup_should_cancel = should_cancel  # type: ignore[attr-defined]
    return upsampler


def _tile_reporting_upsampler_class(base: type) -> type:
    # Subclass of realesrgan.RealESRGANer that emits a per-tile callback.
    #
    # Pinned against realesrgan==0.3.0. The override delegates to the upstream
    # tile_process and only assumes:
    #   - self.img.shape is (batch, channel, height, width)
    #   - self.tile_size is the tile edge length
    #   - the upstream loop calls self.model(input_tile) exactly once per tile
    # If a future release breaks any of these, the override silently falls back
    # to plain super().tile_process() and emits no per-tile events. The actual
    # upscale always proceeds. Callback exceptions are swallowed.
    class _TileReportingUpsampler(base):  # type: ignore[misc, valid-type]
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


def _run_face_enhance(
    config: InferenceConfig,
    upsampler: Any,
    image: Any,
    *,
    torch_device: Any,
) -> Any:
    try:
        import gfpgan.utils as gfpgan_utils
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from gfpgan import GFPGANer
    except ImportError as exc:
        raise PixelupError(
            ErrorCode.FACE_ENHANCE_UNAVAILABLE,
            "GFPGAN is not installed.",
        ) from exc

    class _PixelupFaceRestoreHelper(FaceRestoreHelper):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["model_rootpath"] = str(config.models_dir)
            super().__init__(*args, **kwargs)

    face_enhancer = _build_face_enhancer(
        gfpgan_utils,
        GFPGANer,
        _PixelupFaceRestoreHelper,
        config,
        upsampler,
        torch_device,
    )
    # enhance() is NOT wrapped in redirect_stdout/redirect_stderr. This runs on
    # a worker thread that can execute concurrently with other jobs, and
    # redirecting the process-global streams here would race with them and leave
    # sys.stdout/sys.stderr pointing at a dead buffer. The plain upscale path
    # likewise does not suppress enhance() output, and surfacing it lets GFPGAN's
    # own inference-failure message through.
    _, _, output = face_enhancer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    return output


def _build_face_enhancer(
    gfpgan_utils: Any,
    gfpganer_cls: Any,
    helper_cls: type,
    config: InferenceConfig,
    upsampler: Any,
    torch_device: Any,
) -> Any:
    # The helper substitution and its matching restore must happen as one atomic
    # step: GFPGANer reads gfpgan.utils.FaceRestoreHelper while it builds its own
    # face helper, so the lock keeps a concurrent face-enhance job from
    # constructing against a half-applied or already-restored global. This
    # serialized window is also the only place it is safe to silence the noisy
    # model-loading output, since redirecting the process-global stdout/stderr is
    # only race-free while the lock is held.
    with _GFPGAN_HELPER_LOCK:
        original_helper = gfpgan_utils.FaceRestoreHelper
        gfpgan_utils.FaceRestoreHelper = helper_cls
        try:
            with (
                redirect_stdout(StringIO()),
                redirect_stderr(StringIO()),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                return gfpganer_cls(
                    model_path=str(model_file(config.models_dir, "GFPGANv1.4")),
                    upscale=config.scale,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=upsampler,
                    device=torch_device,
                )
        finally:
            gfpgan_utils.FaceRestoreHelper = original_helper


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


def _torch_device(torch: Any, device: str, gpu_id: int | None) -> Any:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps and mps.is_available())
    if device == "auto":
        if mps_available:
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
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
        "Device must be one of Auto, MPS, CUDA, or CPU.",
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
        details={"dependency": package, "reason": str(exc)},
    )


def _is_out_of_memory(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _emit(callback: ProgressCallback | None, phase: str) -> None:
    if callback:
        callback(phase)


def _check_cancelled(should_cancel: CancelCheck | None) -> None:
    if should_cancel is not None and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
