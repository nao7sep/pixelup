from __future__ import annotations

from pathlib import Path

import pytest

from pixelup.errors import ErrorCode, PixelupError
from pixelup.paths import OutputFormat
from pixelup.upscale import (
    UpscaleOptions,
    _format_extension_mismatch,
    count_tiles,
    validate_options,
    validate_output_path,
)


def make_options(**overrides: object) -> UpscaleOptions:
    base: dict[str, object] = dict(
        input_path=Path("in.png"),
        output_arg="out.png",
        model="RealESRGAN_x4plus",
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
        download_timeout=600,
        lock_timeout=600,
    )
    base.update(overrides)
    return UpscaleOptions(**base)  # type: ignore[arg-type]


def test_validate_options_accepts_valid_options() -> None:
    validate_options(make_options())


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"scale": 3}, ErrorCode.INVALID_ARGUMENT),
        ({"tile": -1}, ErrorCode.INVALID_ARGUMENT),
        ({"tile_pad": -1}, ErrorCode.INVALID_ARGUMENT),
        ({"pre_pad": -1}, ErrorCode.INVALID_ARGUMENT),
        ({"model": "realesr-general-x4v3", "denoise_strength": 1.5}, ErrorCode.INVALID_ARGUMENT),
        ({"model": "realesr-general-x4v3", "denoise_strength": -0.1}, ErrorCode.INVALID_ARGUMENT),
        ({"alpha_mode": "nearest"}, ErrorCode.INVALID_ARGUMENT),
        ({"quality": 101}, ErrorCode.INVALID_ARGUMENT),
        ({"quality": -1}, ErrorCode.INVALID_ARGUMENT),
        ({"target_profile": "rec2020"}, ErrorCode.INVALID_ARGUMENT),
        ({"device": "vulkan"}, ErrorCode.INVALID_ARGUMENT),
        ({"download_timeout": 0}, ErrorCode.INVALID_ARGUMENT),
        ({"lock_timeout": -1}, ErrorCode.INVALID_ARGUMENT),
        (
            {"model": "RealESRGAN_x4plus", "denoise_strength": 0.5},
            ErrorCode.DENOISE_STRENGTH_UNSUPPORTED,
        ),
    ],
)
def test_validate_options_rejects_bad_values(overrides: dict[str, object], code: ErrorCode) -> None:
    with pytest.raises(PixelupError) as exc_info:
        validate_options(make_options(**overrides))
    assert exc_info.value.code == code
    # Messages surface verbatim in the GUI queue, so they must reference the
    # on-screen control rather than a non-existent CLI flag.
    assert "--" not in exc_info.value.message
    assert exc_info.value.hint is None or "--" not in exc_info.value.hint


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"scale": 3}, "Scale must be 2x or 4x."),
        ({"tile": -1}, "Tile size must be 0 or greater."),
        ({"quality": 101}, "Quality must be between 0 and 100."),
        ({"device": "vulkan"}, "Device must be one of Auto, MPS, CUDA, or CPU."),
        (
            {"target_profile": "rec2020"},
            "Target profile must be one of sRGB, Display P3, or Adobe RGB.",
        ),
        ({"alpha_mode": "nearest"}, "Alpha mode must be Real-ESRGAN or Bicubic."),
    ],
)
def test_validate_options_uses_gui_neutral_messages(
    overrides: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(PixelupError) as exc_info:
        validate_options(make_options(**overrides))
    assert exc_info.value.message == expected_message


def test_validate_options_allows_denoise_only_for_general_model() -> None:
    validate_options(make_options(model="realesr-general-x4v3", denoise_strength=0.5))


def test_count_tiles_returns_one_when_tiling_disabled() -> None:
    assert count_tiles((1000, 800), 0) == 1
    assert count_tiles((1000, 800), -5) == 1


def test_count_tiles_uses_ceiling_grid() -> None:
    assert count_tiles((100, 100), 64) == 4
    assert count_tiles((128, 128), 64) == 4
    assert count_tiles((10, 10), 512) == 1


@pytest.mark.parametrize(
    ("name", "output_format", "expected"),
    [
        ("out.png", OutputFormat.PNG, None),
        ("out.jpg", OutputFormat.JPG, None),
        ("out.jpeg", OutputFormat.JPG, None),
        ("out", OutputFormat.PNG, None),
        ("out.png", OutputFormat.JPG, "png"),
        ("out.webp", OutputFormat.PNG, "webp"),
    ],
)
def test_format_extension_mismatch(
    name: str, output_format: OutputFormat, expected: str | None
) -> None:
    assert _format_extension_mismatch(Path(name), output_format) == expected


def test_validate_output_path_rejects_missing_parent(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "out.png"
    with pytest.raises(PixelupError) as exc_info:
        validate_output_path(target, overwrite=False)
    assert exc_info.value.code == ErrorCode.OUTPUT_DIR_MISSING


def test_validate_output_path_rejects_parent_that_is_a_file(tmp_path: Path) -> None:
    parent = tmp_path / "afile"
    parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PixelupError) as exc_info:
        validate_output_path(parent / "out.png", overwrite=False)
    assert exc_info.value.code == ErrorCode.OUTPUT_DIR_MISSING


def test_validate_output_path_rejects_existing_file_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "out.png"
    target.write_bytes(b"")
    with pytest.raises(PixelupError) as exc_info:
        validate_output_path(target, overwrite=False)
    assert exc_info.value.code == ErrorCode.OUTPUT_EXISTS


def test_validate_output_path_allows_existing_file_with_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "out.png"
    target.write_bytes(b"")
    validate_output_path(target, overwrite=True)


def test_validate_output_path_rejects_unwritable_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(upscale_module.os, "access", lambda *args, **kwargs: False)
    with pytest.raises(PixelupError) as exc_info:
        validate_output_path(tmp_path / "out.png", overwrite=False)
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE
