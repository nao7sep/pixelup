from __future__ import annotations

from pathlib import Path

import pytest

from pixelup.errors import ErrorCode, PixelupError
from pixelup.paths import OutputFormat
from pixelup.upscale import (
    DENOISE_NEUTRAL,
    UpscaleOptions,
    _format_extension_mismatch,
    count_tiles,
    effective_denoise_strength,
    model_supports_denoise,
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
        ({"tile": -1}, "Tile size must be one of the offered sizes."),
        ({"tile": 2816}, "Tile size must be one of the offered sizes."),
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


def test_validate_options_allows_denoise_for_general_model() -> None:
    validate_options(make_options(model="realesr-general-x4v3", denoise_strength=0.5))


def test_validate_options_accepts_denoise_on_non_general_model() -> None:
    # Denoise simply does not apply to a non-general model: a leftover value is normalized away
    # (effective_denoise_strength), not rejected. This previously raised for a direct caller — the
    # GUI path masked it by coercing denoise to neutral per model first.
    validate_options(make_options(model="RealESRGAN_x4plus", denoise_strength=0.5))


@pytest.mark.parametrize(
    ("model", "denoise", "expected"),
    [
        ("realesr-general-x4v3", 0.5, 0.5),  # the general model keeps the caller's value
        ("realesr-general-x4v3", 1.0, 1.0),
        ("RealESRGAN_x4plus", 0.5, DENOISE_NEUTRAL),  # every other model normalizes to neutral
        ("RealESRGAN_x4plus", 1.0, DENOISE_NEUTRAL),
        ("realesr-animevideov3", 0.0, DENOISE_NEUTRAL),
    ],
)
def test_effective_denoise_strength_normalizes_off_the_general_model(
    model: str, denoise: float, expected: float
) -> None:
    assert effective_denoise_strength(model, denoise) == expected


def test_model_supports_denoise_only_for_the_general_model() -> None:
    assert model_supports_denoise("realesr-general-x4v3") is True
    assert model_supports_denoise("RealESRGAN_x4plus") is False


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


# validate_options must READ pixelup.parameters, never restate it. It used to
# hardcode five domains as literals — scale {2,4}, alpha {"realesrgan","bicubic"},
# target profile, and the quality/denoise bounds — because the constants lived in
# jobs.py, which imports upscale.py, so importing them back would have cycled. The
# panel and the loader read the real constants, so any edit to them (a third scale,
# a new alpha mode) would have produced a value the panel offers and this function
# rejects at runtime. The constants now live in the leaf pixelup.parameters and both
# sides import it.
#
# These tests fail if the coupling is ever broken, by extending each domain and
# asserting the validator follows. A test that only checked today's literals would
# pass just as happily against a hardcoded copy — which is the whole bug.
def test_validate_options_follows_the_shared_scale_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(upscale_module, "SCALE_VALUES", (2, 4, 8))
    validate_options(make_options(scale=8))


def test_validate_options_follows_the_shared_alpha_mode_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(upscale_module, "ALPHA_MODE_VALUES", ("realesrgan", "bicubic", "lanczos"))
    validate_options(make_options(alpha_mode="lanczos"))


def test_validate_options_follows_the_shared_target_profile_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(
        upscale_module, "TARGET_PROFILE_VALUES", (None, "srgb", "p3", "adobergb", "rec2020")
    )
    validate_options(make_options(target_profile="rec2020"))


def test_validate_options_follows_the_shared_quality_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(upscale_module, "MAX_QUALITY", 200)
    validate_options(make_options(quality=150))


def test_validate_options_follows_the_shared_denoise_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup import upscale as upscale_module

    monkeypatch.setattr(upscale_module, "MAX_DENOISE_STRENGTH", 2.0)
    validate_options(make_options(model="realesr-general-x4v3", denoise_strength=1.5))


def test_the_panel_and_the_validator_cannot_disagree() -> None:
    # The leaf is the single home: upscale.py must hold the very same objects, not
    # equal copies. Identity, not equality — a copied literal can start equal.
    from pixelup import parameters
    from pixelup import upscale as upscale_module

    assert upscale_module.SCALE_VALUES is parameters.SCALE_VALUES
    assert upscale_module.ALPHA_MODE_VALUES is parameters.ALPHA_MODE_VALUES
    assert upscale_module.TARGET_PROFILE_VALUES is parameters.TARGET_PROFILE_VALUES
