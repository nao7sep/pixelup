from pathlib import Path

import pytest
from PIL import Image

from pixelup.errors import PixelupError
from pixelup.imaging import SourceMetadata, save_output_image
from pixelup.paths import OutputFormat


def test_save_output_image_writes_atomically_and_flattens_jpg_alpha(tmp_path: Path) -> None:
    output = tmp_path / "out.jpg"
    temp_dir = tmp_path / "temp"
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 128))

    size = save_output_image(
        image,
        output_path=output,
        output_format=OutputFormat.JPG,
        quality=95,
        background="white",
        temp_dir=temp_dir,
        source_metadata=SourceMetadata(),
        strip_metadata=True,
        target_profile=None,
    )

    assert size == (2, 2)
    assert output.is_file()
    with Image.open(output) as saved:
        assert saved.mode == "RGB"
        assert saved.format == "JPEG"
    assert list(temp_dir.iterdir()) == []


def test_save_output_image_rejects_invalid_background(tmp_path: Path) -> None:
    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            output_path=tmp_path / "out.jpg",
            output_format=OutputFormat.JPG,
            quality=95,
            background="not-a-color",
            temp_dir=tmp_path / "temp",
            source_metadata=SourceMetadata(),
            strip_metadata=False,
            target_profile=None,
        )

    assert excinfo.value.code == "invalid_argument"


def test_save_output_image_can_embed_srgb_profile(tmp_path: Path) -> None:
    output = tmp_path / "out.png"

    save_output_image(
        Image.new("RGB", (1, 1), "white"),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        temp_dir=tmp_path / "temp",
        source_metadata=SourceMetadata(),
        strip_metadata=False,
        target_profile="srgb",
    )

    with Image.open(output) as saved:
        assert saved.info["icc_profile"]


def test_save_output_image_rejects_pending_non_srgb_profiles(tmp_path: Path) -> None:
    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=tmp_path / "out.png",
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            temp_dir=tmp_path / "temp",
            source_metadata=SourceMetadata(),
            strip_metadata=False,
            target_profile="p3",
        )

    assert excinfo.value.code == "internal_error"
