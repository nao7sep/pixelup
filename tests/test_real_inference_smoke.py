from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from pixelup.config import RuntimeDirs
from pixelup.models import model_file
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
            auto_download=False,
            download_timeout=600,
            lock_timeout=600,
            dry_run=False,
        ),
        RuntimeDirs(models_dir=models_dir, temp_dir=temp_dir),
    )

    assert result["ok"] is True
    assert result["input_size"] == [1, 1]
    assert result["output_size"] == [4, 4]
    assert output_path.is_file()
