from pathlib import Path

import pytest

from pixelup.errors import PixelupError
from pixelup.paths import (
    OutputContext,
    OutputFormat,
    default_output_path,
    infer_output_format,
    model_filename_token,
    resolve_output_path,
)


def context(output_arg: str) -> OutputContext:
    return OutputContext(
        input_path=Path("/tmp/Girl Image.png"),
        output_arg=output_arg,
        model="RealESRGAN_x4plus_anime_6B",
        scale=4,
        output_format=OutputFormat.PNG,
        input_size=(800, 600),
    )


def test_directory_output_uses_model_and_scale_filename(tmp_path: Path) -> None:
    resolved = resolve_output_path(
        OutputContext(
            input_path=tmp_path / "girl.png",
            output_arg=str(tmp_path),
            model="RealESRGAN_x4plus_anime_6B",
            scale=4,
            output_format=OutputFormat.PNG,
            input_size=(800, 600),
        )
    )

    assert resolved == tmp_path / "girl-realesrgan-x4plus-anime-6b-4x.png"


def test_default_output_path_adds_collision_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    first = tmp_path / "girl-realesr-general-x4v3-4x.png"
    first.write_bytes(b"existing")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_reserved_paths_are_treated_as_collisions(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    reserved = {tmp_path / "girl-realesr-general-x4v3-4x.png"}

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        reserved=reserved,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_avoids_sidecar_json_collision(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    # An existing sidecar JSON at the default stem must push the new bundle to "-2".
    (tmp_path / "girl-realesr-general-x4v3-4x.json").write_bytes(b"{}")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_treats_reserved_sidecar_as_collision(tmp_path: Path) -> None:
    input_path = tmp_path / "girl.png"
    reserved = {tmp_path / "girl-realesr-general-x4v3-4x.json"}

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        reserved=reserved,
    )

    assert resolved == tmp_path / "girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_treats_case_only_existing_file_as_collision(tmp_path: Path) -> None:
    # A sibling that differs only in case is one file on macOS/Windows, so the
    # planner must disambiguate even on a case-sensitive filesystem.
    input_path = tmp_path / "Girl.png"
    (tmp_path / "girl-realesr-general-x4v3-4x.png").write_bytes(b"existing")

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
    )

    assert resolved == tmp_path / "Girl-realesr-general-x4v3-4x-2.png"


def test_default_output_path_treats_case_only_reserved_name_as_collision(tmp_path: Path) -> None:
    input_path = tmp_path / "Girl.png"
    reserved = {tmp_path / "girl-realesr-general-x4v3-4x.png"}

    resolved = default_output_path(
        input_path,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        reserved=reserved,
    )

    assert resolved == tmp_path / "Girl-realesr-general-x4v3-4x-2.png"


def test_model_filename_token_is_lowercase_with_hyphens() -> None:
    assert model_filename_token("RealESRGAN_x4plus_anime_6B") == "realesrgan-x4plus-anime-6b"


def test_output_templates_are_not_supported() -> None:
    with pytest.raises(PixelupError) as excinfo:
        resolve_output_path(context("/tmp/{stem}.png"))

    assert excinfo.value.code == "invalid_argument"


def test_directory_output_defaults_to_png_when_format_is_unset(tmp_path: Path) -> None:
    assert infer_output_format(str(tmp_path), None) == OutputFormat.PNG
