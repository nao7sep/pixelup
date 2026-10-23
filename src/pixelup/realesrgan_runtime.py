# Real-ESRGAN: BSD 3-Clause License
#
# Copyright (c) 2021, Xintao Wang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The small Real-ESRGAN inference subset used by PixelUp, adapted and modified
from Real-ESRGAN 0.3.0 under the license above."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from pixelup.errors import ErrorCode, PixelupError


class RealESRGANer:
    """Load a Real-ESRGAN model and upscale complete images or tiles.

    PixelUp's change to the upstream runtime: each finished tile is converted to
    8-bit and pasted into the Pillow image the encoder will save, instead of
    collecting the whole result as float on the device and then copying it into
    arrays. Every conversion is per pixel, so the result is unchanged while the
    output exists only once, as that image (volume-handling-conventions). PixelUp
    hands the runtime 8-bit pixels; wider sources are converted before inference.
    """

    def __init__(
        self,
        scale: int,
        model_path: str | list[str],
        dni_weight: list[float] | None = None,
        model: Any = None,
        tile: int = 256,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = False,
        device: Any = None,
        gpu_id: int | None = None,
        on_tile: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.on_tile = on_tile
        self.should_cancel = should_cancel
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale: int | None = None
        self.mod_pad_h = 0
        self.mod_pad_w = 0
        # How finished pixels are written: "rgb" or "gray".
        self._finish = "rgb"
        self.half = half
        self.device = _resolve_device(device, gpu_id)

        if isinstance(model_path, list):
            if dni_weight is None or len(model_path) != len(dni_weight):
                raise ValueError("model_path and dni_weight must have equal lengths")
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            loadnet = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)

        keyname = "params_ema" if "params_ema" in loadnet else "params"
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(
        self,
        net_a_path: str,
        net_b_path: str,
        weights: list[float],
        key: str = "params",
    ) -> dict[str, Any]:
        net_a = torch.load(net_a_path, map_location=torch.device("cpu"), weights_only=True)
        net_b = torch.load(net_b_path, map_location=torch.device("cpu"), weights_only=True)
        for name, value_a in net_a[key].items():
            net_a[key][name] = weights[0] * value_a + weights[1] * net_b[key][name]
        return net_a

    def pre_process(self, image: np.ndarray[Any, Any]) -> None:
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        self.img = tensor.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h = 0
            self.mod_pad_w = 0
            _, _, height, width = self.img.size()
            if height % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - height % self.mod_scale
            if width % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - width % self.mod_scale
            self.img = F.pad(
                self.img,
                (0, self.mod_pad_w, 0, self.mod_pad_h),
                "reflect",
            )

    def tile_process(self) -> None:
        _, _, height, width = self.img.shape
        self._begin_output()
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                if self.should_cancel is not None and self.should_cancel():
                    raise PixelupError(ErrorCode.JOB_CANCELLED, "Job cancelled.")
                offset_x = tile_x * self.tile_size
                offset_y = tile_y * self.tile_size
                input_start_x = offset_x
                input_end_x = min(offset_x + self.tile_size, width)
                input_start_y = offset_y
                input_end_y = min(offset_y + self.tile_size, height)
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                with torch.no_grad():
                    output_tile = self.model(input_tile)

                output_start_x = input_start_x * self.scale
                output_start_y = input_start_y * self.scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
                self._place(
                    output_tile[
                        :,
                        :,
                        output_start_y_tile:output_end_y_tile,
                        output_start_x_tile:output_end_x_tile,
                    ],
                    output_start_y,
                    output_start_x,
                )
                del output_tile
                if self.on_tile is not None:
                    try:
                        self.on_tile(tile_y * tiles_x + tile_x + 1, tiles_x * tiles_y)
                    except PixelupError:
                        raise
                    except Exception:
                        pass

    def _begin_output(self) -> None:
        """The output image at its final size: the padding added before inference
        (``pre_pad`` and the mod padding, both right and bottom) is never written."""
        _, _, height, width = self.img.shape
        crop_h = self.pre_pad + (self.mod_pad_h if self.mod_scale is not None else 0)
        crop_w = self.pre_pad + (self.mod_pad_w if self.mod_scale is not None else 0)
        size = ((width - crop_w) * self.scale, (height - crop_h) * self.scale)
        self.output = Image.new("L" if self._finish == "gray" else "RGB", size)

    def _place(self, tensor: torch.Tensor, top: int, left: int) -> None:
        """Convert finished model output to 8-bit and paste the part that falls
        inside the output image at (``top``, ``left``)."""
        out_width, out_height = self.output.size
        height = min(tensor.shape[2], out_height - top)
        width = min(tensor.shape[3], out_width - left)
        if height <= 0 or width <= 0:
            return
        pixels = _tensor_to_rgb(tensor[:, :, :height, :width])
        if self._finish == "gray":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        self.output.paste(Image.fromarray(_quantize(pixels)), (left, top))

    def post_process(self) -> Image.Image:
        return self.output

    @torch.no_grad()
    def enhance(
        self,
        image: np.ndarray[Any, Any],
        outscale: float | None = None,
        alpha_upsampler: str = "realesrgan",
    ) -> tuple[Image.Image, str]:
        """Upscale 8-bit BGR, BGRA or grayscale pixels into an RGB, RGBA or L image."""
        if image.dtype != np.uint8:
            raise ValueError("The runtime takes 8-bit pixels.")
        input_height, input_width = image.shape[0:2]
        # One float copy of the input, scaled and reordered in place.
        image = image.astype(np.float32)
        np.divide(image, 255, out=image)
        if len(image.shape) == 2:
            image_mode = "L"
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_mode = "RGBA"
            alpha = image[:, :, 3].copy()
            image = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2RGB)
            if alpha_upsampler == "realesrgan":
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            image_mode = "RGB"
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)

        self._finish = "gray" if image_mode == "L" else "rgb"
        self.pre_process(image)
        del image
        self.tile_process()
        output = self.post_process()
        del self.img

        if image_mode == "RGBA":
            if alpha_upsampler == "realesrgan":
                self._finish = "gray"
                self.pre_process(alpha)
                self.tile_process()
                output_alpha = self.post_process()
                del self.img
            else:
                height, width = alpha.shape[0:2]
                output_alpha = Image.fromarray(
                    _quantize(
                        cv2.resize(
                            alpha,
                            (width * self.scale, height * self.scale),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    )
                )
            output.putalpha(output_alpha)
            del output_alpha
        self.output = None

        if outscale is not None and outscale != float(self.scale):
            # Rare (a model's native scale above the chosen one): the resize needs an
            # array, so this path briefly holds the image twice.
            pixels = np.asarray(output)
            mode = output.mode
            output = None
            output = Image.fromarray(
                cv2.resize(
                    pixels,
                    (int(input_width * outscale), int(input_height * outscale)),
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                mode,
            )
        return output, image_mode


def _resolve_device(device: Any, gpu_id: int | None) -> Any:
    if device is not None:
        return device
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    array = tensor.data[0].float().cpu().clamp_(0, 1).numpy()
    return np.ascontiguousarray(np.transpose(array, (1, 2, 0)))


def _quantize(pixels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return (pixels.clip(0, 1) * 255.0).round().astype(np.uint8)
