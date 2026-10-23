import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
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
    assert output.mode == "RGB"
    # The image is RGB; read it back in the BGR order the input was given in.
    assert np.asarray(output)[:, :, ::-1].tolist() == [
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
    assert output.mode == "RGBA"
    assert output.size == (8, 8)
    # Input BGR (1, 2, 3) is RGB (3, 2, 1).
    assert list(output.getpixel((0, 0))) == [3, 2, 1, 0]
    assert list(output.getpixel((7, 7))) == [12, 11, 10, 255]


def _image(kind: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    shapes = {"rgb": (13, 19, 3), "gray": (17, 11), "rgba": (15, 21, 4)}
    return rng.integers(0, 256, shapes[kind], dtype=np.uint8)


@pytest.mark.parametrize("kind", ["rgb", "gray", "rgba"])
@pytest.mark.parametrize("alpha", ["realesrgan", "bicubic"])
def test_tiles_written_one_by_one_match_a_single_tile(
    tmp_path: Path, kind: str, alpha: str
) -> None:
    # Each finished tile goes straight into the output image; with a model that
    # reads no neighbours, any tiling must give exactly the one-tile result.
    model_path = tmp_path / "nearest.pth"
    torch.save({"params": {}}, model_path)

    def run(tile: int) -> tuple[Image.Image, str]:
        upsampler = RealESRGANer(
            scale=4, model_path=str(model_path), model=_NearestModel(),
            tile=tile, tile_pad=3, pre_pad=10, device=torch.device("cpu"),
        )
        output, mode = upsampler.enhance(_image(kind), outscale=4, alpha_upsampler=alpha)
        assert upsampler.output is None  # the output image is handed over, not kept
        return output, mode

    whole, whole_mode = run(64)  # one tile larger than the image
    for tile in (4, 5, 8):
        tiled, mode = run(tile)
        assert mode == whole_mode
        assert tiled.mode == whole.mode
        assert np.array_equal(np.asarray(tiled), np.asarray(whole))


def test_the_runtime_takes_only_8_bit_pixels(tmp_path: Path) -> None:
    # PixelUp converts wider sources to 8-bit before inference (rgb_pixels).
    model_path = tmp_path / "nearest.pth"
    torch.save({"params": {}}, model_path)
    upsampler = RealESRGANer(
        scale=4, model_path=str(model_path), model=_NearestModel(), device=torch.device("cpu")
    )
    with pytest.raises(ValueError):
        upsampler.enhance(np.zeros((4, 4), dtype=np.uint16))


@pytest.mark.parametrize("unsafe_path", ["single", "dni_a", "dni_b"])
def test_checkpoint_loading_stays_safe_when_environment_changes_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_path: str
) -> None:
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    safe = tmp_path / "safe.pth"
    unsafe = tmp_path / "unsafe.pth"
    torch.save({"params": {}}, safe)
    # A Path requires a pickle global; tensors and state dictionaries do not.
    torch.save({"params": {}, "unexpected_global": Path("untrusted")}, unsafe)
    model_path = str(unsafe)
    if unsafe_path == "dni_a":
        model_path = [str(unsafe), str(safe)]
    elif unsafe_path == "dni_b":
        model_path = [str(safe), str(unsafe)]
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        RealESRGANer(
            scale=4, model_path=model_path, dni_weight=[0.5, 0.5],
            model=_NearestModel(), device=torch.device("cpu"),
        )


def _runtime(tmp_path: Path, **kwargs) -> RealESRGANer:
    model_path = tmp_path / "nearest.pth"
    torch.save({"params": {}}, model_path)
    return RealESRGANer(
        scale=4, model_path=str(model_path), model=_NearestModel(),
        tile=2, tile_pad=0, pre_pad=0, device=torch.device("cpu"), **kwargs,
    )


def test_tile_callback_reports_written_tiles_and_cancel_stops_next_tile(tmp_path: Path) -> None:
    from pixelup.errors import ErrorCode, PixelupError

    events = []
    upsampler = _runtime(
        tmp_path,
        on_tile=lambda completed, total: events.append((completed, total)),
        should_cancel=lambda: len(events) == 1,
    )
    upsampler.pre_process(np.ones((2, 4, 3), dtype=np.float32))
    with pytest.raises(PixelupError) as caught:
        upsampler.tile_process()
    assert caught.value.code == ErrorCode.JOB_CANCELLED
    assert events == [(1, 2)]
    assert upsampler.output.getpixel((0, 0)) == (255, 255, 255)
    assert upsampler.output.getpixel((8, 0)) == (0, 0, 0)


@pytest.mark.parametrize("structured", [False, True])
def test_tile_callback_error_policy(tmp_path: Path, structured: bool) -> None:
    from pixelup.errors import ErrorCode, PixelupError

    calls = []
    failure = PixelupError(ErrorCode.JOB_CANCELLED, "callback cancelled") if structured else (
        RuntimeError("callback diagnostic")
    )

    def callback(completed, total):
        calls.append((completed, total))
        raise failure

    upsampler = _runtime(tmp_path, on_tile=callback)
    image = np.full((2, 4, 3), 255, dtype=np.uint8)
    if structured:
        with pytest.raises(PixelupError) as caught:
            upsampler.enhance(image)
        assert caught.value is failure
        assert calls == [(1, 2)]
    else:
        output, _ = upsampler.enhance(image)
        assert calls == [(1, 2), (2, 2)]
        assert output.getpixel((15, 0)) == (255, 255, 255)


def test_bicubic_alpha_matches_clipped_cubic_reference(tmp_path: Path) -> None:
    alpha = np.array([[0, 0, 255, 255], [255, 255, 0, 0]], dtype=np.uint8)
    pixels = np.dstack((np.zeros((2, 4, 3), dtype=np.uint8), alpha))
    output, _ = _runtime(tmp_path).enhance(pixels, alpha_upsampler="bicubic")
    cubic = cv2.resize(alpha.astype(np.float32) / 255, (16, 8), interpolation=cv2.INTER_CUBIC)
    assert cubic.min() < 0 and cubic.max() > 1  # clipping matters at the edge
    expected = (cubic.clip(0, 1) * 255).round().astype(np.uint8)
    assert np.array_equal(np.asarray(output.getchannel("A")), expected)
