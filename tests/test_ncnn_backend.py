from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pixelup.errors import PixelupError
from pixelup.ncnn_backend import NcnnModelFiles, NcnnNetwork, upscale_bgr


def _nearest(scale: int):
    def infer(value: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(value, scale, axis=1), scale, axis=2)

    return infer


def test_tiled_upscale_matches_whole_image_for_odd_dimensions() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)

    whole = upscale_bgr(
        image,
        native_scale=2,
        output_scale=2,
        tile=0,
        tile_pad=1,
        pre_pad=1,
        alpha_mode="bicubic",
        infer_tile=_nearest(2),
    )
    events: list[tuple[int, int]] = []
    tiled = upscale_bgr(
        image,
        native_scale=2,
        output_scale=2,
        tile=3,
        tile_pad=1,
        pre_pad=1,
        alpha_mode="bicubic",
        infer_tile=_nearest(2),
        on_tile=lambda done, total: events.append((done, total)),
    )

    assert np.array_equal(tiled, whole)
    assert tiled.shape == (10, 14, 3)
    assert events == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


def test_alpha_can_use_model_or_bicubic_without_changing_output_size() -> None:
    image = np.zeros((2, 3, 4), dtype=np.uint8)
    image[:, :, :3] = (10, 20, 30)
    image[:, :, 3] = ((0, 64, 255), (255, 64, 0))
    calls: list[tuple[int, int, int]] = []

    def infer(value: np.ndarray) -> np.ndarray:
        calls.append(value.shape)
        return _nearest(4)(value)

    model_alpha = upscale_bgr(
        image,
        native_scale=4,
        output_scale=2,
        tile=0,
        tile_pad=0,
        pre_pad=0,
        alpha_mode="realesrgan",
        infer_tile=infer,
    )
    bicubic_alpha = upscale_bgr(
        image,
        native_scale=4,
        output_scale=2,
        tile=0,
        tile_pad=0,
        pre_pad=0,
        alpha_mode="bicubic",
        infer_tile=infer,
    )

    assert model_alpha.shape == (4, 6, 4)
    assert bicubic_alpha.shape == (4, 6, 4)
    assert calls == [(3, 2, 3), (3, 2, 3), (3, 2, 3)]


def test_cancellation_is_checked_between_tiles() -> None:
    calls = 0

    def should_cancel() -> bool:
        return calls >= 1

    def infer(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _nearest(4)(value)

    with pytest.raises(PixelupError) as excinfo:
        upscale_bgr(
            np.zeros((8, 8, 3), dtype=np.uint8),
            native_scale=4,
            output_scale=4,
            tile=4,
            tile_pad=0,
            pre_pad=0,
            alpha_mode="bicubic",
            infer_tile=infer,
            should_cancel=should_cancel,
        )

    assert calls == 1
    assert excinfo.value.code == "job_cancelled"


def test_unexpected_network_shape_is_an_internal_error() -> None:
    with pytest.raises(PixelupError) as excinfo:
        upscale_bgr(
            np.zeros((2, 2, 3), dtype=np.uint8),
            native_scale=4,
            output_scale=4,
            tile=0,
            tile_pad=0,
            pre_pad=0,
            alpha_mode="bicubic",
            infer_tile=lambda _value: np.zeros((3, 4, 4), dtype=np.float32),
        )

    assert excinfo.value.code == "internal_error"
    assert excinfo.value.details == {"expected": [3, 8, 8], "actual": [3, 4, 4]}


def test_network_refuses_unavailable_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNet:
        def __init__(self) -> None:
            self.opt = SimpleNamespace()

        def clear(self) -> None:
            pass

    fake = ModuleType("ncnn")
    fake.Net = FakeNet  # type: ignore[attr-defined]
    fake.get_gpu_count = lambda: 0  # type: ignore[attr-defined]
    fake.get_default_gpu_index = lambda: -1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ncnn", fake)

    with pytest.raises(PixelupError) as excinfo:
        NcnnNetwork(
            NcnnModelFiles(Path("model.param"), Path("model.bin"), 4),
            use_gpu=True,
            gpu_id=None,
            fp32=False,
        )

    assert excinfo.value.code == "invalid_argument"
