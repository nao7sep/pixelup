from pathlib import Path

import pytest

from pixelup.errors import PixelupError
from pixelup.paths import (
    OutputContext,
    OutputFormat,
    RunTimestamp,
    infer_output_format,
    resolve_output_path,
)


def context(output_arg: str, *, face: bool = False, denoise: float = 1.0) -> OutputContext:
    return OutputContext(
        input_path=Path("/tmp/girl.png"),
        output_arg=output_arg,
        model="realesr-general-x4v3",
        scale=4,
        output_format=OutputFormat.PNG,
        input_size=(800, 600),
        face_enhance=face,
        denoise_strength=denoise,
        timestamp=RunTimestamp("20260428", "032202", "20260428-032202-utc"),
    )


def test_default_directory_pattern_uses_model_alias_and_final_width(tmp_path: Path) -> None:
    resolved = resolve_output_path(context(str(tmp_path)))

    assert resolved == tmp_path / "girl__general_4x__3200px.png"


def test_empty_placeholder_drops_separator_to_the_right() -> None:
    resolved = resolve_output_path(context("/tmp/{stem}__{denoise}__{model_short}.{ext}"))

    assert resolved == Path("/tmp/girl__general.png").resolve()


def test_empty_placeholder_drops_separator_to_the_left_when_right_missing() -> None:
    resolved = resolve_output_path(context("/tmp/{stem}__{face}.{ext}"))

    assert resolved == Path("/tmp/girl.png").resolve()


def test_non_empty_optional_placeholders_are_preserved() -> None:
    resolved = resolve_output_path(
        context("/tmp/{stem}__{denoise}__{face}__{datetime}.{ext}", face=True, denoise=0.4)
    )

    assert resolved == Path("/tmp/girl__0.4__face__20260428-032202-utc.png").resolve()


def test_unknown_placeholder_is_invalid() -> None:
    with pytest.raises(PixelupError) as excinfo:
        resolve_output_path(context("/tmp/{stem}_{unknown}.{ext}"))

    assert excinfo.value.code == "invalid_argument"


def test_directory_output_defaults_to_png_when_format_is_unset(tmp_path: Path) -> None:
    assert infer_output_format(str(tmp_path), None) == OutputFormat.PNG


def test_ext_placeholder_defaults_to_png_when_format_is_unset() -> None:
    assert infer_output_format("/tmp/{stem}.{ext}", None) == OutputFormat.PNG
