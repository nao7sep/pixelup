from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from pixelup.errors import PixelupError
from pixelup.imaging import SourceMetadata, _profile_bytes, save_output_image
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


def test_target_profiles_are_available_without_system_lookup() -> None:
    for name in ("srgb", "p3", "adobergb"):
        profile = ImageCms.ImageCmsProfile(BytesIO(_profile_bytes(name)))

        assert ImageCms.getProfileDescription(profile)


def test_save_output_image_can_embed_p3_profile(tmp_path: Path) -> None:
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
        target_profile="p3",
    )

    with Image.open(output) as saved:
        assert saved.info["icc_profile"]


def test_save_output_image_strips_metadata_and_profile(tmp_path: Path) -> None:
    output = tmp_path / "out.jpg"

    save_output_image(
        Image.new("RGB", (1, 1), "white"),
        output_path=output,
        output_format=OutputFormat.JPG,
        quality=95,
        background="white",
        temp_dir=tmp_path / "temp",
        source_metadata=SourceMetadata(icc_profile=_profile_bytes("srgb"), xmp=b"<xmp />"),
        strip_metadata=True,
        target_profile=None,
    )

    with Image.open(output) as saved:
        assert "icc_profile" not in saved.info
        assert "xmp" not in saved.info


def test_save_output_image_preserves_png_xmp(tmp_path: Path) -> None:
    output = tmp_path / "out.png"
    xmp = b"<x:xmpmeta></x:xmpmeta>"

    save_output_image(
        Image.new("RGB", (1, 1), "white"),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        temp_dir=tmp_path / "temp",
        source_metadata=SourceMetadata(xmp=xmp),
        strip_metadata=False,
        target_profile=None,
    )

    with Image.open(output) as saved:
        assert saved.info["xmp"] == xmp


def test_save_output_image_cleans_temp_file_on_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.png"
    temp_dir = tmp_path / "temp"
    original_save = Image.Image.save

    def fail_after_partial_write(self: Image.Image, fp: object, **kwargs: object) -> None:
        Path(fp).write_bytes(b"partial")
        raise ValueError("boom")

    monkeypatch.setattr(Image.Image, "save", fail_after_partial_write)
    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            temp_dir=temp_dir,
            source_metadata=SourceMetadata(),
            strip_metadata=False,
            target_profile=None,
        )

    monkeypatch.setattr(Image.Image, "save", original_save)
    assert excinfo.value.code == "output_unwritable"
    assert not output.exists()
    assert list(temp_dir.iterdir()) == []
