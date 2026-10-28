from pathlib import Path

import pytest
from imaging_support import image_from_bgr_array
from PIL import Image

from pixelup.config import RuntimeDirs
from pixelup.errors import PixelupError
from pixelup.i18n.localizer import english
from pixelup.i18n.message import Message
from pixelup.paths import OutputFormat
from pixelup.upscale import UpscaleOptions, build_plan


def options(
    input_path: Path,
    output_path: Path,
    *,
    model: str = "RealESRGAN_x4plus",
    tile: int = 256,
    scale: int = 4,
    output_format: OutputFormat = OutputFormat.PNG,
) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=input_path,
        output_path=output_path,
        model=model,
        scale=scale,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
        denoise_strength=1.0,
        alpha_mode="realesrgan",
        gpu_id=None,
        device="cpu",
        output_format=output_format,
        quality=95,
        background="white",
        strip_metadata=False,
        target_profile=None,
    )


def _stub_successful_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(
        upscale_module,
        "run_inference",
        lambda *args, **kwargs: Image.new("RGB", (8, 8), "red"),
    )


def test_build_plan_validates_input_output_and_model(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    Image.new("RGB", (3, 2), "white").save(input_path)

    plan = build_plan(
        options(input_path, output_path, model="RealESRGAN_x4plus"),
        RuntimeDirs(models_dir),
    )

    assert plan.input_size == (3, 2)
    assert plan.output_size == (12, 8)
    assert plan.output_path == output_path
    assert plan.model == "RealESRGAN_x4plus"



def test_build_plan_keeps_literal_input_and_uses_resolved_path_only_for_reads(
    tmp_path: Path,
    file_symlink_capability: None,
) -> None:
    source_dir = tmp_path / "source"
    chosen_dir = tmp_path / "chosen"
    models_dir = tmp_path / "models"
    source_dir.mkdir()
    chosen_dir.mkdir()
    models_dir.mkdir()
    target = source_dir / "target.png"
    Image.new("RGB", (3, 2), "white").save(target)
    chosen = chosen_dir / "dropped-link.png"
    chosen.symlink_to(target)
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    output = chosen_dir / "dropped-link-realesrgan-x4plus-4x.png"

    plan = build_plan(
        options(chosen, output, model="RealESRGAN_x4plus"),
        RuntimeDirs(models_dir),
    )

    assert plan.input_path == chosen
    assert plan.read_path == target
    assert plan.output_path == output


def test_build_plan_rejects_a_model_pixelup_does_not_offer(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(input_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with pytest.raises(PixelupError) as excinfo:
        build_plan(
            options(input_path, tmp_path / "output.png", model="not-a-pixelup-model"),
            RuntimeDirs(models_dir),
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
    models_dir.mkdir()
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    # 300px at the smallest offered tile (128) is genuinely tiled: ceil(300/128) = 3
    # per axis, so on_start must report 9. This used to be a 2x2 image at tile=1 —
    # a size no longer in the domain, since tile is now a choice among doublings.
    Image.new("RGB", (300, 300), "white").save(input_path)
    events: list[tuple[str, object]] = []
    _stub_successful_inference(monkeypatch)

    result = run_upscale(
        options(input_path, output_path, model="RealESRGAN_x4plus", tile=128),
        RuntimeDirs(models_dir),
        on_start=lambda plan, tiles: events.append(("start", tiles)),
        on_progress=lambda phase: events.append(("progress", phase)),
    )

    assert result["ok"] is True
    # 8x8 is the stubbed inference's output, measured from the written file — it does
    # not track the input size, so growing the input above does not move it.
    assert result["output_size"] == [8, 8]
    assert output_path.is_file()
    assert ("start", 9) in events
    assert ("progress", "encode") in events


def test_run_upscale_errors_on_missing_model(tmp_path: Path) -> None:
    from pixelup.upscale import run_upscale

    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    Image.new("RGB", (1, 1), "white").save(input_path)

    with pytest.raises(PixelupError) as excinfo:
        run_upscale(
            options(input_path, tmp_path / "output.png"),
            RuntimeDirs(models_dir),
        )

    assert excinfo.value.code == "model_not_found"


def test_run_upscale_warns_for_forced_format_extension_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup.upscale import run_upscale

    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    Image.new("RGB", (1, 1), "white").save(input_path)
    warnings: list[Message] = []
    _stub_successful_inference(monkeypatch)

    run_upscale(
        options(
            input_path,
            tmp_path / "output.png",
            model="RealESRGAN_x4plus",
            output_format=OutputFormat.JPG,
        ),
        RuntimeDirs(models_dir),
        on_warning=warnings.append,
    )

    assert [english().of(warning) for warning in warnings] == [
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
    models_dir.mkdir()
    (models_dir / "RealESRGAN_x2plus.pth").write_bytes(b"weights")
    Image.new("RGB", (1, 1), "white").save(input_path)
    warnings: list[Message] = []
    _stub_successful_inference(monkeypatch)
    monkeypatch.setattr(upscale_module, "require_model_present", lambda *args, **kwargs: None)

    run_upscale(
        options(
            input_path,
            tmp_path / "output.png",
            model="RealESRGAN_x2plus",
            scale=4,
        ),
        RuntimeDirs(models_dir),
        on_warning=warnings.append,
    )

    assert [english().of(warning) for warning in warnings] == [
        "Model 'RealESRGAN_x2plus' is trained for 2x, but the selected scale is 4x; "
        "Real-ESRGAN will rescale the output."
    ]


def test_run_upscale_uses_one_source_capture_when_the_path_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import json
    from types import SimpleNamespace

    import numpy as np

    from pixelup import inference
    from pixelup.imaging import _profile_bytes
    from pixelup.sidecar import write_sidecar
    from pixelup.upscale import run_upscale

    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    exif = Image.Exif()
    exif[0x8769] = {0x9003: "2020:01:02 03:04:05", 0xA002: 3, 0xA003: 2}
    original_pixels = Image.new("RGB", (3, 2), (20, 40, 60))
    original_profile = _profile_bytes("p3")
    original_pixels.save(source, exif=exif, icc_profile=original_profile)
    original_bytes = source.read_bytes()
    enhanced = []

    def enhance(pixels, **kwargs):
        enhanced.append(pixels.copy())
        return image_from_bgr_array(np.repeat(np.repeat(pixels, 4, axis=0), 4, axis=1)), None

    monkeypatch.setattr(inference, "_import_torch", lambda: None)
    monkeypatch.setattr(inference, "to_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(inference, "_create_upsampler", lambda *args, **kwargs:
                        SimpleNamespace(enhance=enhance))

    def replace_after_capture(plan, tiles):
        replacement_exif = Image.Exif()
        replacement_exif[0x8769] = {0x9003: "2025:06:07 08:09:10"}
        replacement = tmp_path / "replacement.png"
        Image.new("RGB", (5, 4), (200, 180, 160)).save(
            replacement, exif=replacement_exif, icc_profile=_profile_bytes("adobergb")
        )
        replacement.replace(source)

    opts = options(source, output, model="RealESRGAN_x4plus")
    result = run_upscale(opts, RuntimeDirs(models_dir),
                         on_start=replace_after_capture)
    write_sidecar(input_path=source, output_path=output, options=opts,
                  result=result, warnings=[])

    assert source.read_bytes() != original_bytes
    assert enhanced[0].shape == (2, 3, 3)
    assert enhanced[0][0, 0].tolist() == [60, 40, 20]
    assert result["input_size"] == [3, 2]
    assert result["input_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert result["input_size_bytes"] == len(original_bytes)
    with Image.open(output) as saved:
        assert saved.size == (12, 8)
        assert saved.getpixel((0, 0)) == (20, 40, 60)
        assert saved.info["icc_profile"] == original_profile
        assert saved.getexif().get_ifd(0x8769)[0x9003] == "2020:01:02 03:04:05"
    sidecar = json.loads(output.with_suffix(".json").read_text())
    assert sidecar["input"] == {"filename": "input.png", "dimensions": [3, 2],
                               "sha256": hashlib.sha256(original_bytes).hexdigest(),
                               "size_bytes": len(original_bytes)}


def test_source_buffers_close_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref
    from types import SimpleNamespace

    from pixelup import imaging, inference
    from pixelup.upscale import run_upscale

    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    models = tmp_path / "models"
    models.mkdir()
    (models / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("L", (3, 2), 120).save(source, exif=exif)
    stream_refs = []
    image_refs = []
    original_open = Image.open
    original_transpose = imaging.ImageOps.exif_transpose
    original_rgb = imaging.rgb_pixels

    def observed_open(stream, **kwargs):
        stream_refs.append(weakref.ref(stream))
        result = original_open(stream, **kwargs)
        image_refs.append(weakref.ref(result))
        return result

    def observed_transpose(image):
        result = original_transpose(image)
        image_refs.append(weakref.ref(result))
        return result

    def observed_rgb(image):
        result = original_rgb(image)
        image_refs.append(weakref.ref(result))
        return result

    def construct(*args, **kwargs):
        assert len(stream_refs) == 1
        assert all(reference() is None for reference in stream_refs + image_refs)
        return SimpleNamespace(
            enhance=lambda pixels, **kwargs: (image_from_bgr_array(pixels), None)
        )

    monkeypatch.setattr(Image, "open", observed_open)
    monkeypatch.setattr(imaging.ImageOps, "exif_transpose", observed_transpose)
    monkeypatch.setattr(imaging, "rgb_pixels", observed_rgb)
    monkeypatch.setattr(inference, "_import_torch", lambda: None)
    monkeypatch.setattr(inference, "to_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(inference, "_create_upsampler", construct)
    run_upscale(options(source, output, model="RealESRGAN_x4plus"),
                RuntimeDirs(models))


def test_the_planned_record_keeps_the_input_version_when_inference_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import json
    import sqlite3

    from pixelup import upscale as upscale_module
    from pixelup.errors import ErrorCode
    from pixelup.session_log import configure_session_logging, flush_records
    from pixelup.upscale import run_upscale

    database = configure_session_logging()
    input_path = tmp_path / "input.png"
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    Image.new("RGB", (4, 4), "white").save(input_path)
    original = input_path.read_bytes()

    def edit_input_then_fail(*args: object, **kwargs: object) -> object:
        Image.new("RGB", (4, 4), "black").save(input_path)
        raise PixelupError(ErrorCode.INTERNAL_ERROR, "Inference failed.")

    monkeypatch.setattr(upscale_module, "run_inference", edit_input_then_fail)

    with pytest.raises(PixelupError):
        run_upscale(
            options(input_path, tmp_path / "output.png", model="RealESRGAN_x4plus"),
            RuntimeDirs(models_dir),
        )

    flush_records()
    connection = sqlite3.connect(database)
    try:
        (fields,) = connection.execute(
            "SELECT fields FROM logs WHERE message = 'upscale.planned'"
        ).fetchone()
    finally:
        connection.close()
    planned = json.loads(fields)
    assert input_path.read_bytes() != original
    assert planned["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert planned["input_size_bytes"] == len(original)


@pytest.mark.parametrize("input_size", [(8191, 1), (1, 8191)])
@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_build_plan_accepts_largest_reachable_webp_axis_and_other_formats(
    tmp_path: Path, input_size: tuple[int, int], output_format: OutputFormat
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"header dimensions supplied separately")
    plan = build_plan(
        options(source, tmp_path / f"output.{output_format}", scale=2,
                output_format=output_format),
        RuntimeDirs(tmp_path / "models"), check_model=False, input_size=input_size,
    )
    assert max(plan.output_size) == 16382


@pytest.mark.parametrize("input_size", [(8192, 1), (1, 8192)])
@pytest.mark.parametrize("entry", ["build_plan", "run_upscale"])
def test_webp_overflow_rejected_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    input_size: tuple[int, int], entry: str,
) -> None:
    from types import SimpleNamespace

    from pixelup import upscale

    source = tmp_path / "input.png"
    source.write_bytes(b"header dimensions supplied separately")
    monkeypatch.setattr(
        upscale, "load_source_image", lambda _path: SimpleNamespace(size=input_size)
    )
    inferred = []
    monkeypatch.setattr(upscale, "run_inference", lambda *args, **kwargs: inferred.append(args))
    opts = options(source, tmp_path / "output.webp", scale=2, output_format=OutputFormat.WEBP)
    with pytest.raises(PixelupError) as failure:
        if entry == "build_plan":
            build_plan(opts, RuntimeDirs(tmp_path / "models"),
                       check_model=False, input_size=input_size)
        else:
            upscale.run_upscale(opts, RuntimeDirs(tmp_path / "models"))
    assert failure.value.code == "invalid_argument"
    assert failure.value.message.key == "error.webpDimensions"
    assert failure.value.details["maximum"] == 16383
    assert inferred == []


@pytest.mark.parametrize("output_format", [OutputFormat.PNG, OutputFormat.JPG])
@pytest.mark.parametrize("input_size", [(8192, 1), (1, 8192)])
def test_webp_axis_limit_does_not_restrict_other_output_formats(
    tmp_path: Path, output_format: OutputFormat, input_size: tuple[int, int]
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"header dimensions supplied separately")
    plan = build_plan(
        options(source, tmp_path / f"output.{output_format}", scale=2,
                output_format=output_format),
        RuntimeDirs(tmp_path / "models"), check_model=False, input_size=input_size,
    )
    assert max(plan.output_size) == 16384


def test_run_resolves_auto_once_and_native_inference_receives_that_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from pixelup import inference, upscale

    source = tmp_path / "input.png"
    Image.new("RGB", (2, 2), "red").save(source)
    models = tmp_path / "models"
    models.mkdir()
    (models / "RealESRGAN_x4plus.pth").write_bytes(b"weights")
    resolved = []
    native = []
    validated = []
    original_validate = upscale.validate_options

    def resolve(device, gpu_id):
        resolved.append((device, gpu_id))
        assert len(resolved) == 1
        return "cpu"

    def validate(opts):
        validated.append(opts)
        original_validate(opts)

    def construct(config, **kwargs):
        native.append((config.device, kwargs["concrete_device"], kwargs["torch_device"]))
        return SimpleNamespace(enhance=lambda pixels, **_kwargs: (Image.new("RGB", (8, 8)), None))

    monkeypatch.setattr(upscale, "resolve_device", resolve)
    monkeypatch.setattr(upscale, "validate_options", validate)
    monkeypatch.setattr(inference, "_import_torch", lambda: None)
    monkeypatch.setattr(inference, "to_torch_device", lambda device, _gpu_id: f"torch:{device}")
    monkeypatch.setattr(inference, "_create_upsampler", construct)
    opts = replace(options(source, tmp_path / "output.png"), device="auto")
    result = upscale.run_upscale(opts, RuntimeDirs(models))
    assert result["ok"]
    assert resolved == [("auto", None)]
    assert native == [("cpu", "cpu", "torch:cpu")]
    assert validated == [opts]


def test_public_build_plan_validates_options_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pixelup import upscale

    source = tmp_path / "input.png"
    source.write_bytes(b"header dimensions supplied separately")
    validated = []
    original_validate = upscale.validate_options

    def validate(opts):
        validated.append(opts)
        original_validate(opts)

    monkeypatch.setattr(upscale, "validate_options", validate)
    opts = options(source, tmp_path / "output.png")
    build_plan(opts, RuntimeDirs(tmp_path / "models"), check_model=False, input_size=(2, 2))
    assert validated == [opts]


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_run_preflight_accepts_supported_axes_without_allocating_large_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    vertical: bool, output_format: OutputFormat,
) -> None:
    from pixelup import upscale
    from pixelup.imaging import SourceImage, SourceMetadata

    source = tmp_path / "input.png"
    source.write_bytes(b"source capture substituted")
    axis = 8191 if output_format == OutputFormat.WEBP else 8192
    size = (1, axis) if vertical else (axis, 1)
    capture = SourceImage(
        pixels=object(), size=size, metadata=SourceMetadata(), sha256="a" * 64, size_bytes=12,
    )
    monkeypatch.setattr(upscale, "load_source_image", lambda _path: capture)
    monkeypatch.setattr(upscale, "require_model_present", lambda *args: None)
    inferred = []

    def infer(config, pixels, **kwargs):
        inferred.append(pixels)
        return Image.new("RGB", (2, 2), "red")

    monkeypatch.setattr(upscale, "run_inference", infer)
    result = upscale.run_upscale(
        options(source, tmp_path / f"output.{output_format}", scale=2,
                output_format=output_format),
        RuntimeDirs(tmp_path / "models"),
    )
    assert inferred == [capture.pixels]
    assert result["input_size"] == list(size)
    assert result["ok"]
