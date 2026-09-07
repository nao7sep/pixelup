"""The small Real-ESRGAN inference subset used by PixelUp.

Adapted from Real-ESRGAN 0.3.0 under its BSD-3-Clause license. The source
notice and license text are in the repository's ``THIRD_PARTY_NOTICES`` file.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F


class RealESRGANer:
    """Load a Real-ESRGAN model and upscale complete images or tiles."""

    def __init__(
        self,
        scale: int,
        model_path: str | list[str],
        dni_weight: list[float] | None = None,
        model: Any = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = False,
        device: Any = None,
        gpu_id: int | None = None,
    ) -> None:
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale: int | None = None
        self.half = half
        self.device = _resolve_device(device, gpu_id)

        if isinstance(model_path, list):
            if dni_weight is None or len(model_path) != len(dni_weight):
                raise ValueError("model_path and dni_weight must have equal lengths")
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            loadnet = torch.load(model_path, map_location=torch.device("cpu"))

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
        net_a = torch.load(net_a_path, map_location=torch.device("cpu"))
        net_b = torch.load(net_b_path, map_location=torch.device("cpu"))
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

    def process(self) -> None:
        self.output = self.model(self.img)

    def tile_process(self) -> None:
        batch, channel, height, width = self.img.shape
        output_shape = (
            batch,
            channel,
            height * self.scale,
            width * self.scale,
        )
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
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
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
                self.output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self) -> torch.Tensor:
        if self.mod_scale is not None:
            _, _, height, width = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : height - self.mod_pad_h * self.scale,
                0 : width - self.mod_pad_w * self.scale,
            ]
        if self.pre_pad != 0:
            _, _, height, width = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : height - self.pre_pad * self.scale,
                0 : width - self.pre_pad * self.scale,
            ]
        return self.output

    @torch.no_grad()
    def enhance(
        self,
        image: np.ndarray[Any, Any],
        outscale: float | None = None,
        alpha_upsampler: str = "realesrgan",
    ) -> tuple[np.ndarray[Any, Any], str]:
        input_height, input_width = image.shape[0:2]
        image = image.astype(np.float32)
        max_range = 65535 if np.max(image) > 256 else 255
        image = image / max_range
        if len(image.shape) == 2:
            image_mode = "L"
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_mode = "RGBA"
            alpha = image[:, :, 3]
            image = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2RGB)
            if alpha_upsampler == "realesrgan":
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            image_mode = "RGB"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.pre_process(image)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_image = _tensor_to_bgr(self.post_process())
        if image_mode == "L":
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

        if image_mode == "RGBA":
            if alpha_upsampler == "realesrgan":
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = _tensor_to_bgr(self.post_process())
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                height, width = alpha.shape[0:2]
                output_alpha = cv2.resize(
                    alpha,
                    (width * self.scale, height * self.scale),
                    interpolation=cv2.INTER_LINEAR,
                )
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
            output_image[:, :, 3] = output_alpha

        if max_range == 65535:
            output = (output_image * 65535.0).round().astype(np.uint16)
        else:
            output = (output_image * 255.0).round().astype(np.uint8)
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output,
                (int(input_width * outscale), int(input_height * outscale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return output, image_mode


def _resolve_device(device: Any, gpu_id: int | None) -> Any:
    if device is not None:
        return device
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    array = tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    return np.transpose(array[[2, 1, 0], :, :], (1, 2, 0))
