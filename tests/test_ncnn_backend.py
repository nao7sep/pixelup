from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pixelup.errors import PixelupError
from pixelup.ncnn_backend import (
    NcnnModelFiles,
    NcnnNetwork,
    NcnnUpscaleConfig,
    interpolate_ncnn_weights,
    run_ncnn_upscale,
    upscale_bgr,
)


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


def test_interpolate_ncnn_weights_preserves_tags_and_blends_tensors(tmp_path: Path) -> None:
    param = tmp_path / "model.param"
    primary = tmp_path / "primary.bin"
    companion = tmp_path / "companion.bin"
    destination = tmp_path / "blended.bin"
    tag = bytes.fromhex("476b3001")
    param.write_text(
        "7767517\n"
        "4 4\n"
        "Input in0 0 1 in0\n"
        "Convolution conv 1 1 in0 hidden 0=2 5=1 6=3\n"
        "PReLU prelu 1 1 hidden activated 0=2\n"
        "BinaryOp add 2 1 activated in0 out0 0=0\n",
        encoding="utf-8",
    )

    def encoded(weight: float) -> bytes:
        return b"".join(
            (
                tag,
                np.full(3, weight, dtype="<f2").tobytes(),
                np.full(2, weight + 1, dtype="<f4").tobytes(),
                np.full(2, weight + 2, dtype="<f4").tobytes(),
            )
        )

    primary.write_bytes(encoded(1.0))
    companion.write_bytes(encoded(0.0))
    interpolate_ncnn_weights(
        param,
        primary,
        companion,
        destination,
        primary_weight=0.25,
    )

    output = destination.read_bytes()
    assert output[:4] == tag
    assert np.array_equal(np.frombuffer(output, dtype="<f2", count=3, offset=4), [0.25] * 3)
    assert np.array_equal(np.frombuffer(output, dtype="<f4", count=2, offset=10), [1.25] * 2)
    assert np.array_equal(np.frombuffer(output, dtype="<f4", count=2, offset=18), [2.25] * 2)


@pytest.mark.parametrize("strength", (-0.01, 1.01))
def test_interpolate_ncnn_weights_rejects_invalid_strength(
    tmp_path: Path, strength: float
) -> None:
    with pytest.raises(PixelupError) as excinfo:
        interpolate_ncnn_weights(
            tmp_path / "model.param",
            tmp_path / "primary.bin",
            tmp_path / "companion.bin",
            tmp_path / "output.bin",
            primary_weight=strength,
        )

    assert excinfo.value.code == "invalid_argument"


def test_interpolate_ncnn_weights_rejects_an_incompatible_companion(tmp_path: Path) -> None:
    param = tmp_path / "model.param"
    primary = tmp_path / "primary.bin"
    companion = tmp_path / "companion.bin"
    param.write_text(
        "7767517\n2 2\nInput in0 0 1 in0\nPReLU prelu 1 1 in0 out0 0=1\n",
        encoding="utf-8",
    )
    primary.write_bytes(np.array([1.0], dtype="<f4").tobytes())
    companion.write_bytes(b"")

    with pytest.raises(PixelupError) as excinfo:
        interpolate_ncnn_weights(
            param,
            primary,
            companion,
            tmp_path / "output.bin",
            primary_weight=0.5,
        )

    assert excinfo.value.code == "model_corrupt"
    assert not (tmp_path / "output.bin").exists()


def test_run_ncnn_upscale_owns_denoise_temp_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pixelup.ncnn_backend as backend

    tag = bytes.fromhex("476b3001")
    param_text = (
        "7767517\n2 2\nInput in0 0 1 in0\n"
        "Convolution conv 1 1 in0 out0 0=1 5=0 6=1\n"
    )

    def model_files(name: str, value: float) -> NcnnModelFiles:
        param = tmp_path / f"{name}.param"
        weights = tmp_path / f"{name}.bin"
        param.write_text(param_text, encoding="utf-8")
        weights.write_bytes(tag + np.array([value], dtype="<f2").tobytes())
        return NcnnModelFiles(param, weights, 2)

    captured_weights: list[bytes] = []

    class FakeNetwork:
        def __init__(self, files: NcnnModelFiles, **_kwargs: object) -> None:
            captured_weights.append(files.weights.read_bytes())

        def __call__(self, value: np.ndarray) -> np.ndarray:
            return _nearest(2)(value)

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(backend, "NcnnNetwork", FakeNetwork)
    temp_dir = tmp_path / "temp"
    progress: list[str] = []
    output = run_ncnn_upscale(
        np.zeros((2, 3, 3), dtype=np.uint8),
        NcnnUpscaleConfig(
            model=model_files("primary", 1.0),
            denoise_companion=model_files("companion", 0.0),
            denoise_strength=0.25,
            output_scale=2,
            tile=0,
            tile_pad=0,
            pre_pad=0,
            alpha_mode="bicubic",
            use_gpu=False,
            gpu_id=None,
            fp32=True,
        ),
        temp_dir=temp_dir,
        on_progress=progress.append,
    )

    assert output.shape == (4, 6, 3)
    assert progress == ["load_model", "upscale"]
    assert np.frombuffer(captured_weights[0], dtype="<f2", count=1, offset=4)[0] == 0.25
    assert list(temp_dir.iterdir()) == []


def test_run_ncnn_upscale_rejects_denoise_without_companion(tmp_path: Path) -> None:
    files = NcnnModelFiles(tmp_path / "model.param", tmp_path / "model.bin", 4)

    with pytest.raises(PixelupError) as excinfo:
        run_ncnn_upscale(
            np.zeros((2, 2, 3), dtype=np.uint8),
            NcnnUpscaleConfig(
                model=files,
                denoise_companion=None,
                denoise_strength=0.5,
                output_scale=4,
                tile=0,
                tile_pad=0,
                pre_pad=0,
                alpha_mode="bicubic",
                use_gpu=False,
                gpu_id=None,
                fp32=True,
            ),
            temp_dir=tmp_path / "temp",
        )

    assert excinfo.value.code == "invalid_argument"
