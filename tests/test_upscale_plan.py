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
    tile: int = 0,
    scale: int = 4,
    output_format: OutputFormat | None = OutputFormat.PNG,
    auto_download: bool = False,
    face_enhance: bool = False,
) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=input_path,
        output_arg=output_arg,
        model=model,
        scale=scale,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
        face_enhance=face_enhance,
        denoise_strength=1.0,
        alpha_mode="realesrgan",
        gpu_id=None,
        device="cpu",
        output_format=output_format,
        quality=95,
        background="white",
        strip_metadata=False,
        target_profile=None,
        overwrite=False,
        auto_download=auto_download,
        download_timeout=600,
        lock_timeout=600,
    )


def _stub_successful_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(upscale_module, "run_inference", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        upscale_module,
        "image_from_bgr_array",
        lambda array: Image.new("RGB", (8, 8), "red"),
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


def test_build_plan_directory_output_defaults_to_png(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "out"
    models_dir.mkdir()
    temp_dir.mkdir()
    output_dir.mkdir()
    (models_dir / "custom-model.pth").write_bytes(b"weights")
    Image.new("RGB", (3, 2), "white").save(input_path)

    plan = build_plan(
        options(
            input_path,
            str(output_dir),
            model="custom-model",
            output_format=None,
        ),
        RuntimeDirs(models_dir, temp_dir),
    )

    assert plan.output_format == OutputFormat.PNG
    assert plan.output_path == output_dir / "input-custom-model-4x.png"


def test_build_plan_rejects_missing_unknown_model(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(input_path)
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()

    with pytest.raises(PixelupError) as excinfo:
        build_plan(
            options(input_path, str(tmp_path / "output.png"), model="custom-model"),
            RuntimeDirs(models_dir, temp_dir),
        )

    assert excinfo.value.code == "model_not_found"


def test_run_upscale_calls_inference_and_writes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    _stub_successful_inference(monkeypatch)

    result = run_upscale(
        options(input_path, str(output_path), model="custom-model", tile=1),
        RuntimeDirs(models_dir, temp_dir),
        on_start=lambda plan, tiles: events.append(("start", tiles)),
        on_progress=lambda phase: events.append(("progress", phase)),
    )

    assert result["ok"] is True
    assert result["output_size"] == [8, 8]
    assert output_path.is_file()
    assert ("start", 4) in events
    assert ("progress", "encode") in events


def test_run_upscale_without_auto_download_errors_on_missing_model(tmp_path: Path) -> None:
    from pixelup.upscale import run_upscale

    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()
    Image.new("RGB", (1, 1), "white").save(input_path)

    with pytest.raises(PixelupError) as excinfo:
        run_upscale(
            options(input_path, str(tmp_path / "output.png")),
            RuntimeDirs(models_dir, temp_dir),
        )

    assert excinfo.value.code == "model_not_found"


def test_face_enhance_requires_gfpgan_and_facexlib_helper_models(tmp_path: Path) -> None:
    from pixelup.upscale import required_model_names

    input_path = tmp_path / "input.png"

    assert required_model_names(
        options(
            input_path,
            str(tmp_path / "output.png"),
            model="realesr-general-x4v3",
            face_enhance=True,
        )
    ) == [
        "realesr-general-x4v3",
        "GFPGANv1.4",
        "facexlib-detection-retinaface-resnet50",
        "facexlib-parsing-parsenet",
    ]


def test_run_upscale_warns_for_forced_format_extension_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup.upscale import run_upscale

    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()
    (models_dir / "custom-model.pth").write_bytes(b"weights")
    Image.new("RGB", (1, 1), "white").save(input_path)
    warnings: list[str] = []
    _stub_successful_inference(monkeypatch)

    run_upscale(
        options(
            input_path,
            str(tmp_path / "output.png"),
            model="custom-model",
            output_format=OutputFormat.JPG,
        ),
        RuntimeDirs(models_dir, temp_dir),
        on_warning=warnings.append,
    )

    assert warnings == [
        "Output path extension '.png' does not match requested format 'jpg'."
    ]


def test_run_upscale_warns_for_model_native_scale_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module
    from pixelup.upscale import run_upscale

    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    models_dir.mkdir()
    temp_dir.mkdir()
    (models_dir / "RealESRGAN_x2plus.pth").write_bytes(b"weights")
    Image.new("RGB", (1, 1), "white").save(input_path)
    warnings: list[str] = []
    _stub_successful_inference(monkeypatch)
    monkeypatch.setattr(upscale_module, "require_model_present", lambda *args, **kwargs: None)

    run_upscale(
        options(
            input_path,
            str(tmp_path / "output.png"),
            model="RealESRGAN_x2plus",
            scale=4,
        ),
        RuntimeDirs(models_dir, temp_dir),
        on_warning=warnings.append,
    )

    assert warnings == [
        "Model 'RealESRGAN_x2plus' is trained for 2x, but --scale is 4x; "
        "Real-ESRGAN will rescale the output."
    ]


def test_auto_device_detection_uses_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    from pixelup.upscale import resolve_device

    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("auto", None) == "cpu"
    assert resolve_device("auto", 0) == "cpu"

    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    assert resolve_device("auto", None) == "cuda"
    assert resolve_device("auto", 0) == "cuda"

    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    assert resolve_device("auto", None) == "mps"
    assert resolve_device("auto", 0) == "mps"


def test_forced_device_validation_uses_torch_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    from pixelup.upscale import resolve_device

    fake_torch = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(PixelupError) as excinfo:
        resolve_device("mps", None)
    assert excinfo.value.code == "invalid_argument"

    with pytest.raises(PixelupError) as excinfo:
        resolve_device("cuda", None)
    assert excinfo.value.code == "invalid_argument"

    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
    )
    assert resolve_device("cuda", 0) == "cuda"

    with pytest.raises(PixelupError) as excinfo:
        resolve_device("cuda", 2)
    assert excinfo.value.code == "invalid_argument"
