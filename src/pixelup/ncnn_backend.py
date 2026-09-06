from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pixelup.errors import ErrorCode, PixelupError

TileInference = Callable[[np.ndarray], np.ndarray]
TileCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class NcnnModelFiles:
    param: Path
    weights: Path
    native_scale: int


class NcnnNetwork:
    """One loaded ncnn network owned by a single inference job."""

    def __init__(
        self,
        files: NcnnModelFiles,
        *,
        use_gpu: bool,
        gpu_id: int | None,
        fp32: bool,
    ) -> None:
        try:
            import ncnn
        except ImportError as exc:
            raise PixelupError(
                ErrorCode.INTERNAL_ERROR,
                "The ncnn inference runtime is not installed.",
                details={"reason": str(exc)},
            ) from exc

        self._ncnn = ncnn
        self._net = ncnn.Net()
        self._net.opt.use_vulkan_compute = use_gpu
        self._net.opt.use_fp16_packed = use_gpu and not fp32
        self._net.opt.use_fp16_storage = use_gpu and not fp32
        self._net.opt.use_fp16_arithmetic = use_gpu and not fp32
        if use_gpu:
            count = int(ncnn.get_gpu_count())
            index = int(ncnn.get_default_gpu_index()) if gpu_id is None else gpu_id
            if count <= 0 or index < 0 or index >= count:
                self.close()
                raise PixelupError(
                    ErrorCode.INVALID_ARGUMENT,
                    "A Vulkan-capable GPU is not available.",
                    details={"gpu_id": index, "device_count": count},
                )
            self._net.set_vulkan_device(index)
        if self._net.load_param(str(files.param)) != 0:
            self.close()
            raise _model_load_error(files.param)
        if self._net.load_model(str(files.weights)) != 0:
            self.close()
            raise _model_load_error(files.weights)

    def __call__(self, value: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        with self._net.create_extractor() as extractor:
            if extractor.input("in0", self._ncnn.Mat(contiguous).clone()) != 0:
                raise PixelupError(
                    ErrorCode.INTERNAL_ERROR,
                    "ncnn rejected the inference input.",
                )
            result, output = extractor.extract("out0")
        if result != 0:
            raise PixelupError(
                ErrorCode.INTERNAL_ERROR,
                "ncnn could not produce an inference result.",
                details={"result": int(result)},
            )
        return np.array(output, dtype=np.float32, copy=True)

    def close(self) -> None:
        net = getattr(self, "_net", None)
        if net is not None:
            net.clear()
            self._net = None

    def __enter__(self) -> NcnnNetwork:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def upscale_bgr(
    image: np.ndarray,
    *,
    native_scale: int,
    output_scale: int,
    tile: int,
    tile_pad: int,
    pre_pad: int,
    alpha_mode: str,
    infer_tile: TileInference,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> np.ndarray:
    """Upscale a BGR/BGRA uint8 image around one ncnn tile inference callable."""
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise PixelupError(
            ErrorCode.INPUT_INVALID_FORMAT,
            "Input image has an unsupported channel layout.",
        )
    _check_cancelled(should_cancel)
    height, width = image.shape[:2]
    rgb = image[:, :, :3][:, :, ::-1].astype(np.float32) / 255.0
    rgb_chw = np.transpose(rgb, (2, 0, 1))
    output_rgb = _upscale_chw(
        rgb_chw,
        native_scale=native_scale,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        infer_tile=infer_tile,
        on_tile=on_tile,
        should_cancel=should_cancel,
    )
    output_bgr = np.transpose(output_rgb[[2, 1, 0]], (1, 2, 0))

    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        if alpha_mode == "realesrgan":
            alpha_chw = np.repeat((alpha.astype(np.float32) / 255.0)[None, :, :], 3, axis=0)
            output_alpha_rgb = _upscale_chw(
                alpha_chw,
                native_scale=native_scale,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                infer_tile=infer_tile,
                should_cancel=should_cancel,
            )
            output_alpha = (
                output_alpha_rgb[0] * 0.299
                + output_alpha_rgb[1] * 0.587
                + output_alpha_rgb[2] * 0.114
            )
        else:
            output_alpha = np.asarray(
                Image.fromarray(alpha).resize(
                    (width * native_scale, height * native_scale),
                    Image.Resampling.BICUBIC,
                ),
                dtype=np.float32,
            ) / 255.0
        output_bgr = np.dstack((output_bgr, output_alpha))

    output = np.clip(output_bgr * 255.0, 0.0, 255.0).round().astype(np.uint8)
    if output_scale != native_scale:
        output = _resize_bgr(output, (width * output_scale, height * output_scale))
    return output


def _upscale_chw(
    value: np.ndarray,
    *,
    native_scale: int,
    tile: int,
    tile_pad: int,
    pre_pad: int,
    infer_tile: TileInference,
    on_tile: TileCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> np.ndarray:
    original_height, original_width = value.shape[1:]
    prepared = _pad_bottom_right(value, pre_pad, pre_pad)
    if native_scale == 2:
        height, width = prepared.shape[1:]
        prepared = _pad_bottom_right(prepared, height % 2, width % 2)
    height, width = prepared.shape[1:]

    tile_size = tile if tile > 0 else max(height, width)
    tiles_x = max(1, math.ceil(width / tile_size))
    tiles_y = max(1, math.ceil(height / tile_size))
    total = tiles_x * tiles_y
    output = np.zeros((3, height * native_scale, width * native_scale), dtype=np.float32)

    done = 0
    for y_index in range(tiles_y):
        for x_index in range(tiles_x):
            _check_cancelled(should_cancel)
            start_x = x_index * tile_size
            end_x = min(start_x + tile_size, width)
            start_y = y_index * tile_size
            end_y = min(start_y + tile_size, height)
            padded_start_x = max(start_x - tile_pad, 0)
            padded_end_x = min(end_x + tile_pad, width)
            padded_start_y = max(start_y - tile_pad, 0)
            padded_end_y = min(end_y + tile_pad, height)
            input_tile = prepared[
                :, padded_start_y:padded_end_y, padded_start_x:padded_end_x
            ]
            tile_height, tile_width = input_tile.shape[1:]
            pad_y = tile_height % 2 if native_scale == 2 else 0
            pad_x = tile_width % 2 if native_scale == 2 else 0
            inferred = infer_tile(_pad_bottom_right(input_tile, pad_y, pad_x))
            expected_shape = (
                3,
                (tile_height + pad_y) * native_scale,
                (tile_width + pad_x) * native_scale,
            )
            if inferred.shape != expected_shape:
                raise PixelupError(
                    ErrorCode.INTERNAL_ERROR,
                    "ncnn returned an unexpected output shape.",
                    details={"expected": list(expected_shape), "actual": list(inferred.shape)},
                )

            output_start_x = start_x * native_scale
            output_end_x = end_x * native_scale
            output_start_y = start_y * native_scale
            output_end_y = end_y * native_scale
            tile_start_x = (start_x - padded_start_x) * native_scale
            tile_start_y = (start_y - padded_start_y) * native_scale
            tile_end_x = tile_start_x + (end_x - start_x) * native_scale
            tile_end_y = tile_start_y + (end_y - start_y) * native_scale
            output[
                :, output_start_y:output_end_y, output_start_x:output_end_x
            ] = inferred[:, tile_start_y:tile_end_y, tile_start_x:tile_end_x]
            done += 1
            if on_tile is not None:
                try:
                    on_tile(done, total)
                except PixelupError:
                    raise
                except Exception:
                    pass

    _check_cancelled(should_cancel)
    return output[:, : original_height * native_scale, : original_width * native_scale]


def _pad_bottom_right(value: np.ndarray, bottom: int, right: int) -> np.ndarray:
    if bottom == 0 and right == 0:
        return value
    mode = "reflect" if value.shape[1] > bottom and value.shape[2] > right else "edge"
    return np.pad(value, ((0, 0), (0, bottom), (0, right)), mode=mode)


def _resize_bgr(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if value.shape[2] == 4:
        rgba = value[:, :, [2, 1, 0, 3]]
        resized = np.asarray(Image.fromarray(rgba).resize(size, Image.Resampling.LANCZOS))
        return resized[:, :, [2, 1, 0, 3]].copy()
    rgb = value[:, :, ::-1]
    resized = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))
    return resized[:, :, ::-1].copy()


def _model_load_error(path: Path) -> PixelupError:
    return PixelupError(
        ErrorCode.MODEL_CORRUPT,
        "The ncnn model could not be loaded.",
        user_hint="Reinstall the model from Managed models.",
        details={"path": str(path)},
    )


def _check_cancelled(should_cancel: CancelCheck | None) -> None:
    if should_cancel is not None and should_cancel():
        raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
