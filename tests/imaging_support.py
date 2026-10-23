"""Convert the inference test doubles' BGR/BGRA arrays into Pillow images."""

from typing import Any

from PIL import Image


def image_from_bgr_array(array: Any) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array)
    channels = array.shape[2]
    assert channels in {3, 4}, "test double must return BGR or BGRA"
    order = [2, 1, 0, 3] if channels == 4 else [2, 1, 0]
    return Image.fromarray(array[:, :, order])
