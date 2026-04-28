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
    dry_run: bool = True,
    tile: int = 0,
) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=input_path,
        output_arg=output_arg,
        model=model,
        scale=4,
        tile=tile,
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
        download_timeout=600,
        lock_timeout=600,
        dry_run=dry_run,
    )


def test_build_plan_validates_input_output_and_model(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()
    (models_dir / "custom-model.pth").write_bytes(b"weights")
    Image.new("RGB", (3, 2), "white").save(input_path)

    plan = build_plan(
        options(input_path, str(output_path), model="custom-model"),
        RuntimeDirs(models_dir, temp_dir),
    )

    assert plan.input_size == (3, 2)
    assert plan.output_size == (12, 8)
    assert plan.output_path == output_path
    assert plan.model == "custom-model"


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


def test_run_upscale_calls_inference_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module
    from pixelup.upscale import run_upscale

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()
    (models_dir / "custom-model.pth").write_bytes(b"weights")
    Image.new("RGB", (2, 2), "white").save(input_path)
    events: list[tuple[str, object]] = []

    def fake_run_inference(*args: object, on_progress: object = None) -> object:
        assert on_progress is not None
        return object()

    monkeypatch.setattr(upscale_module, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        upscale_module,
        "image_from_bgr_array",
        lambda array: Image.new("RGB", (8, 8), "red"),
    )

    result = run_upscale(
        options(input_path, str(output_path), model="custom-model", dry_run=False, tile=1),
        RuntimeDirs(models_dir, temp_dir),
        on_start=lambda plan, tiles: events.append(("start", tiles)),
        on_progress=lambda phase: events.append(("progress", phase)),
    )

    assert result["ok"] is True
    assert result["output_size"] == [8, 8]
    assert output_path.is_file()
    assert ("start", 4) in events
    assert ("progress", "encode") in events
