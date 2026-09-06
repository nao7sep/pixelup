from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelup.config import RuntimeDirs
from pixelup.models import model_file
from pixelup.ncnn_backend import NcnnModelFiles, NcnnUpscaleConfig, run_ncnn_upscale
from pixelup.paths import OutputFormat
from pixelup.upscale import UpscaleOptions, run_upscale


def test_real_esrgan_small_image_smoke(tmp_path: Path) -> None:
    if os.environ.get("PIXELUP_RUN_REAL_INFERENCE") != "1":
        pytest.skip("set PIXELUP_RUN_REAL_INFERENCE=1 to run the real inference smoke test")

    models_env = os.environ.get("PIXELUP_REAL_INFERENCE_MODELS_DIR")
    if not models_env:
        pytest.skip("set PIXELUP_REAL_INFERENCE_MODELS_DIR to a directory with model weights")

    model = "realesr-general-x4v3"
    models_dir = Path(models_env).expanduser().resolve()
    if not model_file(models_dir, model).is_file():
        pytest.skip(f"{model} weights are not present in {models_dir}")

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (1, 1), (24, 64, 128)).save(input_path)

    result = run_upscale(
        UpscaleOptions(
            input_path=input_path,
            output_arg=str(output_path),
            model=model,
            scale=4,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            fp32=True,
            face_enhance=False,
            denoise_strength=1.0,
            alpha_mode="realesrgan",
            gpu_id=None,
            device="cpu",
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            strip_metadata=False,
            target_profile=None,
            overwrite=False,
            lock_timeout=600,
        ),
        RuntimeDirs(models_dir=models_dir, temp_dir=temp_dir),
    )

    assert result["ok"] is True
    assert result["input_size"] == [1, 1]
    assert result["output_size"] == [4, 4]
    assert output_path.is_file()


def test_real_ncnn_small_image_smoke(tmp_path: Path) -> None:
    if os.environ.get("PIXELUP_RUN_REAL_NCNN_INFERENCE") != "1":
        pytest.skip("set PIXELUP_RUN_REAL_NCNN_INFERENCE=1 to run real ncnn inference")

    models_env = os.environ.get("PIXELUP_REAL_NCNN_MODELS_DIR")
    if not models_env:
        pytest.skip("set PIXELUP_REAL_NCNN_MODELS_DIR to converted ncnn model pairs")

    models_dir = Path(models_env).expanduser().resolve()
    stem = models_dir / "realesr-general-x4v3.ncnn"
    files = NcnnModelFiles(stem.with_suffix(".ncnn.param"), stem.with_suffix(".ncnn.bin"), 4)
    if not files.param.is_file() or not files.weights.is_file():
        pytest.skip(f"general ncnn model pair is not present in {models_dir}")

    image = np.full((1, 1, 3), (128, 64, 24), dtype=np.uint8)
    output = run_ncnn_upscale(
        image,
        NcnnUpscaleConfig(
            model=files,
            denoise_companion=None,
            denoise_strength=1.0,
            output_scale=4,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            alpha_mode="realesrgan",
            use_gpu=False,
            gpu_id=None,
            fp32=True,
        ),
        temp_dir=tmp_path,
    )

    assert output.shape == (4, 4, 3)
