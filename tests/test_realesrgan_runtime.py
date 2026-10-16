from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pixelup.realesrgan_models import RRDBNet, SRVGGNetCompact
from pixelup.realesrgan_runtime import RealESRGANer


class _NearestModel(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(tensor, scale_factor=4, mode="nearest")


def test_adapted_networks_keep_upstream_weight_key_layouts() -> None:
    rrdb = RRDBNet(3, 3, scale=4, num_feat=8, num_block=1, num_grow_ch=4)
    srvgg = SRVGGNetCompact(num_feat=8, num_conv=1, upscale=4)

    assert list(rrdb.state_dict())[:4] == [
        "conv_first.weight",
        "conv_first.bias",
        "body.0.rdb1.conv1.weight",
        "body.0.rdb1.conv1.bias",
    ]
    assert list(srvgg.state_dict()) == [
        "body.0.weight",
        "body.0.bias",
        "body.1.weight",
        "body.2.weight",
        "body.2.bias",
        "body.3.weight",
        "body.4.weight",
        "body.4.bias",
    ]


def test_adapted_runtime_preserves_rgb_rescale_behavior(tmp_path: Path) -> None:
    model_path = tmp_path / "nearest.pth"
    torch.save({"params": {}}, model_path)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=_NearestModel(),
        pre_pad=0,
        device=torch.device("cpu"),
    )
    image = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )

    output, mode = upsampler.enhance(image, outscale=2, alpha_upsampler="bicubic")

    assert mode == "RGB"
    assert output.tolist() == [
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]],
        [[0, 1, 2], [0, 1, 2], [4, 5, 6], [3, 4, 5]],
        [[8, 9, 10], [7, 8, 9], [11, 12, 13], [11, 12, 13]],
        [[7, 8, 9], [7, 8, 9], [10, 11, 12], [10, 11, 12]],
    ]


def test_adapted_runtime_preserves_rgba_and_tile_behavior(tmp_path: Path) -> None:
    model_path = tmp_path / "nearest.pth"
    torch.save({"params": {}}, model_path)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=_NearestModel(),
        tile=2,
        tile_pad=0,
        pre_pad=0,
        device=torch.device("cpu"),
    )
    rgb = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    image = np.dstack((rgb, np.array([[0, 64], [128, 255]], dtype=np.uint8)))

    output, mode = upsampler.enhance(image, outscale=4, alpha_upsampler="bicubic")

    assert mode == "RGBA"
    assert output.shape == (8, 8, 4)
    assert output[0, 0].tolist() == [1, 2, 3, 0]
    assert output[-1, -1].tolist() == [10, 11, 12, 255]


def _image(kind: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    if kind == "gray16":
        return rng.integers(300, 65536, (13, 19), dtype=np.uint16)
    shapes = {"rgb": (13, 19, 3), "gray": (17, 11), "rgba": (15, 21, 4)}
    return rng.integers(0, 256, shapes[kind], dtype=np.uint8)


@pytest.mark.parametrize("kind", ["rgb", "gray", "rgba", "gray16"])
@pytest.mark.parametrize("alpha", ["realesrgan", "bicubic"])
def test_tiles_written_one_by_one_match_the_whole_image(
    tmp_path: Path, kind: str, alpha: str
) -> None:
    # Each finished tile goes straight into the integer output image; with a model
    # that reads no neighbours, any tiling must give exactly the whole-image result.
    model_path = tmp_path / "nearest.pth"
    torch.save({"params": {}}, model_path)

    def run(tile: int) -> tuple[np.ndarray, str]:
        upsampler = RealESRGANer(
            scale=4, model_path=str(model_path), model=_NearestModel(),
            tile=tile, tile_pad=3, pre_pad=10, device=torch.device("cpu"),
        )
        output, mode = upsampler.enhance(_image(kind), outscale=4, alpha_upsampler=alpha)
        assert upsampler.output is None  # the output image is handed over, not kept
        return output, mode

    whole, whole_mode = run(0)
    for tile in (4, 5, 8):
        tiled, mode = run(tile)
        assert mode == whole_mode
        assert tiled.dtype == whole.dtype
        assert np.array_equal(tiled, whole)
