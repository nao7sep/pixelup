from pathlib import Path

import pytest
from PIL import Image

from pixelup.config import RuntimeDirs
from pixelup.errors import PixelupError
from pixelup.paths import OutputFormat
from pixelup.upscale import UpscaleOptions, build_plan


def options(
    input_path: Path,
    output_arg: str,
    *,
    model: str = "RealESRGAN_x4plus",
) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=input_path,
        output_arg=output_arg,
        model=model,
        scale=4,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
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
        dry_run=True,
    )


def test_build_plan_validates_input_output_and_model(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    Image.new("RGB", (3, 2), "white").save(input_path)

    plan = build_plan(options(input_path, str(output_path)), RuntimeDirs(models_dir, temp_dir))

    assert plan.input_size == (3, 2)
    assert plan.output_size == (12, 8)
    assert plan.output_path == output_path
    assert plan.model == "RealESRGAN_x4plus"


def test_build_plan_rejects_missing_model(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(input_path)
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()

    with pytest.raises(PixelupError) as excinfo:
        build_plan(
            options(input_path, str(tmp_path / "output.png")),
            RuntimeDirs(models_dir, temp_dir),
        )

    assert excinfo.value.code == "model_not_found"
