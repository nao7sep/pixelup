import errno
import os
import re
import xml.etree.ElementTree as ElementTree
from io import BytesIO
from itertools import product
from pathlib import Path
from struct import pack

import pytest
from imaging_support import image_from_bgr_array
from PIL import Image, ImageCms

from pixelup.errors import PixelupError
from pixelup.icc_profiles import D50_XYZ, _description_tag, _profile_with_tags, _xyz_tag
from pixelup.imaging import (
    SourceMetadata,
    _profile_bytes,
    load_source_image,
    rgb_pixels,
    save_output_image,
)
from pixelup.paths import OutputFormat


def test_save_output_image_writes_atomically_and_flattens_jpg_alpha(tmp_path: Path) -> None:
    output = tmp_path / "out.jpg"
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 128))

    size = save_output_image(
        image,
        output_path=output,
        output_format=OutputFormat.JPG,
        quality=95,
        background="white",
        source_metadata=SourceMetadata(),
        strip_metadata=True,
        target_profile=None,
    )

    assert size == (2, 2)
    assert output.is_file()
    with Image.open(output) as saved:
        assert saved.mode == "RGB"
        assert saved.format == "JPEG"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_output_image_stages_a_random_name_beside_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The random stage stays in the destination directory, so publication
    # always stays on the destination volume.
    output_dir = tmp_path / "output-volume"
    output_dir.mkdir()
    output = output_dir / "photo-x4plus-4x.png"
    original_open = os.open
    captured: list[Path] = []

    def capture_open(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            captured.append(Path(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", capture_open)
    save_output_image(
        Image.new("RGB", (1, 1), "white"),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        source_metadata=SourceMetadata(),
        strip_metadata=True,
        target_profile=None,
    )
    monkeypatch.setattr(os, "open", original_open)

    assert len(captured) == 1
    assert re.fullmatch(r"pixelup-[0-9a-f]{16}\.tmp", captured[0].name)
    assert captured[0].parent == output.parent


def test_save_output_image_rejects_invalid_background(tmp_path: Path) -> None:
    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            output_path=tmp_path / "out.jpg",
            output_format=OutputFormat.JPG,
            quality=95,
            background="not-a-color",
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
    original_save = Image.Image.save

    def fail_after_partial_write(self: Image.Image, fp: object, **kwargs: object) -> None:
        fp.write(b"partial")  # type: ignore[attr-defined]
        fp.flush()  # type: ignore[attr-defined]
        raise ValueError("boom")

    monkeypatch.setattr(Image.Image, "save", fail_after_partial_write)
    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=False,
            target_profile=None,
        )

    monkeypatch.setattr(Image.Image, "save", original_save)
    assert excinfo.value.code == "output_unwritable"
    assert not output.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_output_image_does_not_replace_a_late_competing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.png"
    original_save = Image.Image.save

    def save_then_compete(self: Image.Image, fp: object, **kwargs: object) -> None:
        original_save(self, fp, **kwargs)
        output.write_bytes(b"competitor")

    monkeypatch.setattr(Image.Image, "save", save_then_compete)

    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=True,
            target_profile=None,
        )

    assert excinfo.value.code == "output_exists"
    assert output.read_bytes() == b"competitor"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_output_image_preserves_an_exact_boundary_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.png"
    real_link = os.link

    def compete_then_link(source: Path, target: Path, **kwargs: object) -> None:
        output.write_bytes(b"exact-boundary-winner")
        real_link(source, target, **kwargs)

    monkeypatch.setattr("pixelup.imaging.os.link", compete_then_link)

    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=True,
            target_profile=None,
        )

    assert excinfo.value.code == "output_exists"
    assert output.read_bytes() == b"exact-boundary-winner"


@pytest.mark.parametrize("late_name", ["out.jpg", "out.json"])
def test_save_output_image_removes_only_its_claim_when_a_companion_wins_at_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_name: str,
) -> None:
    output = tmp_path / "out.png"
    late = tmp_path / late_name
    real_link = os.link

    def companion_then_link(source: Path, target: Path, **kwargs: object) -> None:
        late.write_bytes(b"publication-winner")
        real_link(source, target, **kwargs)

    monkeypatch.setattr("pixelup.imaging.os.link", companion_then_link)

    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=True,
            target_profile=None,
        )

    assert excinfo.value.code == "output_exists"
    assert not output.exists()
    assert late.read_bytes() == b"publication-winner"


@pytest.mark.parametrize("late_name", ["out.jpg", "out.json"])
def test_save_output_image_revalidates_the_whole_bundle_after_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    late_name: str,
) -> None:
    output = tmp_path / "out.png"
    late = tmp_path / late_name
    original_save = Image.Image.save

    def save_then_compete(self: Image.Image, fp: object, **kwargs: object) -> None:
        original_save(self, fp, **kwargs)
        late.write_bytes(b"late-winner")

    monkeypatch.setattr(Image.Image, "save", save_then_compete)

    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=True,
            target_profile=None,
        )

    assert excinfo.value.code == "output_exists"
    assert late.read_bytes() == b"late-winner"
    assert not output.exists()


def test_save_output_image_treats_a_broken_symlink_as_occupied(
    tmp_path: Path, file_symlink_capability: None
) -> None:
    output = tmp_path / "out.png"
    output.symlink_to(tmp_path / "missing.png")

    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=True,
            target_profile=None,
        )

    assert excinfo.value.code == "output_exists"
    assert output.is_symlink()
    target = os.readlink(output)
    if os.name == "nt":
        target = target.removeprefix("\\\\?\\")
    assert Path(target) == tmp_path / "missing.png"


def test_save_output_image_failed_native_publication_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.png"

    def unsupported_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EPERM, "hard links unsupported")

    def rename_failure(source: Path, target: Path) -> None:
        assert source.is_file()
        assert not target.exists()
        raise OSError(errno.EIO, "native publication failed")

    monkeypatch.setattr("pixelup.imaging.os.link", unsupported_link)
    monkeypatch.setattr("pixelup.output_reservation.os.link", unsupported_link)
    monkeypatch.setattr("pixelup.output_publication.rename_no_replace", rename_failure)

    with pytest.raises(PixelupError) as excinfo:
        save_output_image(
            Image.new("RGB", (1, 1), "white"),
            output_path=output,
            output_format=OutputFormat.PNG,
            quality=95,
            background="white",
            source_metadata=SourceMetadata(),
            strip_metadata=True,
            target_profile=None,
        )

    assert excinfo.value.code == "output_unwritable"
    assert list(tmp_path.iterdir()) == []


def test_save_output_image_uses_native_no_replace_rename_without_hard_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.png"

    def unsupported_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr("pixelup.imaging.os.link", unsupported_link)

    size = save_output_image(
        Image.new("RGB", (1, 1), "white"),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        source_metadata=SourceMetadata(),
        strip_metadata=True,
        target_profile=None,
    )

    assert size == (1, 1)
    with Image.open(output) as saved:
        assert saved.size == (1, 1)


def _camera_exif(*, color_space: int, interop_index: str) -> bytes:
    exif = Image.Exif()
    exif[0x8769] = {
        0xA002: 2,
        0xA003: 1,
        0xA001: color_space,
        0x9003: "2020:01:02 03:04:05",
        0xA005: {1: interop_index},
    }
    return exif.tobytes()


ADOBE_RGB_EXIF = _camera_exif(color_space=0xFFFF, interop_index="R03")
MID_GREEN = (100, 150, 50)


def _save_png(
    tmp_path: Path,
    metadata: SourceMetadata,
    *,
    strip_metadata: bool = False,
    target_profile: str | None = None,
    size: tuple[int, int] = (2, 1),
) -> Image.Image:
    output = tmp_path / "out.png"
    save_output_image(
        Image.new("RGB", size, MID_GREEN),
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        source_metadata=metadata,
        strip_metadata=strip_metadata,
        target_profile=target_profile,
    )
    with Image.open(output) as saved:
        saved.load()
        return saved


def _profile_name(image: Image.Image) -> str:
    profile = ImageCms.ImageCmsProfile(BytesIO(image.info["icc_profile"]))
    return ImageCms.getProfileDescription(profile).strip()


def test_kept_exif_and_xmp_carry_the_output_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "out.jpg"
    xmp = (
        b'<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        b' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:ImageWidth="2"'
        b' tiff:ImageLength="1"/>'
    )

    save_output_image(
        Image.new("RGB", (8, 4), "white"),
        output_path=output,
        output_format=OutputFormat.JPG,
        quality=95,
        background="white",
        source_metadata=SourceMetadata(
            exif=_camera_exif(color_space=1, interop_index="R98"), xmp=xmp
        ),
        strip_metadata=False,
        target_profile=None,
    )

    with Image.open(output) as saved:
        exif_ifd = saved.getexif().get_ifd(0x8769)
        assert (exif_ifd[0xA002], exif_ifd[0xA003]) == (8, 4)
        assert exif_ifd[0x9003] == "2020:01:02 03:04:05"
        description = ElementTree.fromstring(saved.info["xmp"])
        assert description.get("{http://ns.adobe.com/tiff/1.0/}ImageWidth") == "8"
        assert description.get("{http://ns.adobe.com/tiff/1.0/}ImageLength") == "4"


def test_kept_metadata_dates_are_one_local_moment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta, timezone

    class _LocalMoment(datetime):
        # Already in the local zone the clock below reports, whatever the computer's is.
        def astimezone(self, tz: object = None) -> datetime:  # type: ignore[override]
            return self

    class _Clock:
        @staticmethod
        def now() -> datetime:
            return _LocalMoment(2026, 10, 5, 21, 30, 15, 250000, timezone(timedelta(hours=9)))

    monkeypatch.setattr("pixelup.imaging.datetime", _Clock)
    xmp = (
        b'<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        b' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        b' xmp:ModifyDate="2021-01-01T00:00:00Z"/>'
    )

    saved = _save_png(
        tmp_path,
        SourceMetadata(exif=_camera_exif(color_space=1, interop_index="R98"), xmp=xmp),
    )

    exif = saved.getexif()
    assert exif[0x0132] == "2026:10:05 21:30:15"
    assert exif.get_ifd(0x8769)[0x9010] == "+09:00"
    assert exif.get_ifd(0x8769)[0x9003] == "2020:01:02 03:04:05"
    stamp = b"2026-10-05T21:30:15+09:00"
    description = ElementTree.fromstring(saved.info["xmp"])
    assert description.get("{http://ns.adobe.com/xap/1.0/}ModifyDate") == stamp.decode()
    assert description.get("{http://ns.adobe.com/xap/1.0/}MetadataDate") == stamp.decode()


def test_untagged_adobe_rgb_source_is_labelled_adobe_rgb(tmp_path: Path) -> None:
    saved = _save_png(tmp_path, SourceMetadata(exif=ADOBE_RGB_EXIF))

    assert _profile_name(saved) == "PixelUp Adobe RGB (1998)"
    assert saved.getpixel((0, 0)) == MID_GREEN
    assert saved.getexif().get_ifd(0x8769)[0xA001] == 0xFFFF


def test_untagged_adobe_rgb_source_converts_from_adobe_rgb(tmp_path: Path) -> None:
    from_srgb = _save_png(tmp_path, SourceMetadata(), target_profile="srgb").getpixel((0, 0))
    (tmp_path / "out.png").unlink()

    saved = _save_png(tmp_path, SourceMetadata(exif=ADOBE_RGB_EXIF), target_profile="srgb")

    assert _profile_name(saved) == "sRGB built-in"
    assert saved.getpixel((0, 0)) != from_srgb
    exif = saved.getexif()
    assert exif.get_ifd(0x8769)[0xA001] == 1
    assert exif.get_ifd(0xA005)[1] == "R98"


def test_exif_colour_space_follows_a_conversion_away_from_srgb(tmp_path: Path) -> None:
    srgb_exif = _camera_exif(color_space=1, interop_index="R98")

    saved = _save_png(tmp_path, SourceMetadata(exif=srgb_exif), target_profile="adobergb")

    exif = saved.getexif()
    assert exif.get_ifd(0x8769)[0xA001] == 0xFFFF
    assert exif.get_ifd(0xA005)[1] == "R03"


def test_stripped_adobe_rgb_source_is_converted_to_srgb(tmp_path: Path) -> None:
    saved = _save_png(tmp_path, SourceMetadata(exif=ADOBE_RGB_EXIF), strip_metadata=True)

    assert "icc_profile" not in saved.info
    assert "exif" not in saved.info
    assert saved.getpixel((0, 0)) != MID_GREEN


def test_unreadable_source_exif_is_carried_as_it_is(tmp_path: Path) -> None:
    unreadable = b"Exif\x00\x00MM\x00*\x00\x00\xff\xff"

    saved = _save_png(tmp_path, SourceMetadata(exif=unreadable))

    assert saved.info["exif"] == unreadable


def _profile_for(space: bytes, pcs: bytes, tags: list[tuple[str, bytes]]) -> bytes:
    profile = bytearray(_profile_with_tags([("desc", _description_tag("Input")), *tags]))
    profile[16:24] = space + pcs
    return bytes(profile)


# A gamma 2.2 grayscale profile, and a CMYK profile whose any ink darkens to neutral grey.
GRAY_PROFILE = _profile_for(
    b"GRAY",
    b"XYZ ",
    [("wtpt", _xyz_tag(D50_XYZ)), ("kTRC", b"curv" + bytes(4) + pack(">IH", 1, 0x0233))],
)
_LUT_TABLE = bytes(range(256))
CMYK_PROFILE = _profile_for(
    b"CMYK",
    b"Lab ",
    [
        ("wtpt", _xyz_tag(D50_XYZ)),
        (
            "A2B0",
            b"mft1"
            + bytes(4)
            + bytes((4, 3, 2, 0))
            + b"".join(pack(">i", v) for v in (65536, 0, 0, 0, 65536, 0, 0, 0, 65536))
            + _LUT_TABLE * 4
            + b"".join(
                bytes((255 - 255 * max(corner), 128, 128)) for corner in product((0, 1), repeat=4)
            )
            + _LUT_TABLE * 3,
        ),
    ],
)


@pytest.mark.parametrize(
    ("mode", "pixel", "profile", "suffix"),
    [
        ("CMYK", (255, 0, 0, 0), CMYK_PROFILE, ".jpg"),
        ("LA", (255, 128), GRAY_PROFILE, ".png"),
    ],
)
@pytest.mark.parametrize(
    ("strip_metadata", "target_profile", "output_profile"),
    [
        (False, None, "sRGB built-in"),
        (False, "adobergb", "PixelUp Adobe RGB (1998)"),
        (True, None, None),
    ],
)
def test_non_rgb_input_is_decoded_through_its_profile_and_output_profile_describes_it(
    tmp_path: Path,
    mode: str,
    pixel: tuple[int, ...],
    profile: bytes,
    suffix: str,
    strip_metadata: bool,
    target_profile: str | None,
    output_profile: str | None,
) -> None:
    source = tmp_path / f"in{suffix}"
    Image.new(mode, (2, 1), pixel).save(source, icc_profile=profile, exif=ADOBE_RGB_EXIF)
    output = tmp_path / "out.png"

    with Image.open(source) as image:
        decoded = rgb_pixels(image)
    metadata = load_source_image(source).metadata
    save_output_image(
        decoded,
        output_path=output,
        output_format=OutputFormat.PNG,
        quality=95,
        background="white",
        source_metadata=metadata,
        strip_metadata=strip_metadata,
        target_profile=target_profile,
    )

    assert decoded.mode == ("RGBA" if "A" in mode else "RGB")
    red, green, blue = decoded.getpixel((0, 0))[:3]
    # Converted through the profile: the cyan ink and the grey level read as neutral grey.
    assert red == green == blue
    assert metadata.icc_profile == _profile_bytes("srgb")
    with Image.open(output) as saved:
        if output_profile is None:
            assert "icc_profile" not in saved.info
        else:
            assert _profile_name(saved) == output_profile
            # The carried EXIF names the output's colour space, not the source's.
            color_space = saved.getexif().get_ifd(0x8769)[0xA001]
            assert color_space == (1 if target_profile is None else 0xFFFF)


def test_inference_reads_a_non_rgb_input_through_its_profile(tmp_path: Path) -> None:
    from pixelup.imaging import load_source_image

    source = tmp_path / "in.jpg"
    Image.new("CMYK", (2, 1), (255, 0, 0, 0)).save(source, icc_profile=CMYK_PROFILE)

    blue, green, red = load_source_image(source).pixels[0, 0]

    assert blue == green == red


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("strip_metadata", [False, True])
def test_oriented_pixels_survive_metadata_policy(
    tmp_path: Path, orientation: int, strip_metadata: bool
) -> None:
    from pixelup.imaging import load_source_image, read_image_size

    source = tmp_path / "source.png"
    pixels = Image.new("RGB", (3, 2))
    pixels.putdata([(10, 20, 30), (40, 50, 60), (70, 80, 90),
                    (100, 110, 120), (130, 140, 150), (160, 170, 180)])
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x8769] = {0x9003: "2020:01:02 03:04:05"}
    xmp = (
        b'<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        b' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="'
        + str(orientation).encode() + b'"/>'
    )
    from PIL import PngImagePlugin

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_itxt("XML:com.adobe.xmp", xmp.decode())
    pixels.save(source, exif=exif, pnginfo=pnginfo)
    transforms = {
        2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    expected = pixels.transpose(transforms[orientation]) if orientation != 1 else pixels
    assert read_image_size(source) == expected.size
    decoded = image_from_bgr_array(load_source_image(source).pixels)
    output = tmp_path / "out.png"
    save_output_image(
        decoded, output_path=output, output_format=OutputFormat.PNG, quality=95,
        background="white", source_metadata=load_source_image(source).metadata,
        strip_metadata=strip_metadata, target_profile=None,
    )
    with Image.open(output) as saved:
        assert saved.size == expected.size
        assert list(saved.get_flattened_data()) == list(expected.get_flattened_data())
        if strip_metadata:
            assert "exif" not in saved.info
            assert "xmp" not in saved.info
        else:
            assert saved.getexif()[0x0112] == 1
            assert saved.getexif().get_ifd(0x8769)[0x9003] == "2020:01:02 03:04:05"
            description = ElementTree.fromstring(saved.info["xmp"])
            assert description.get("{http://ns.adobe.com/tiff/1.0/}Orientation") == "1"


@pytest.mark.parametrize("profile", [None, GRAY_PROFILE], ids=["untagged", "gray-icc"])
def test_16_bit_grayscale_retains_the_tonal_range(tmp_path: Path, profile: bytes | None) -> None:
    import numpy as np

    from pixelup.imaging import load_source_image

    source = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 16384, 32768, 49152, 65535]], dtype=np.uint16)).save(
        source, icc_profile=profile
    )
    values = load_source_image(source).pixels[0, :, 0].tolist()
    assert values[0] == 0
    assert values[-1] == 255
    assert all(left < right for left, right in zip(values, values[1:], strict=False))
    if profile is None:
        assert values == [0, 64, 128, 191, 255]


@pytest.mark.parametrize("mode,transparency,expected", [
    ("P", 0, [0, 255, 255]),
    ("P", bytes((0, 128, 255)), [0, 128, 255]),
    ("L", 0, [0, 255, 255]),
    ("RGB", (0, 0, 0), [0, 255, 255]),
])
@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_png_transparency_survives_decode_and_profile_conversion(
    tmp_path: Path, mode: str, transparency: object, expected: list[int],
    output_format: OutputFormat,
) -> None:
    from pixelup.imaging import load_source_image

    source = tmp_path / "source.png"
    pixels = Image.new(mode, (3, 1))
    if mode == "P":
        pixels.putpalette([0, 0, 0, 120, 120, 120, 240, 240, 240] + [0] * 759)
        pixels.putdata([0, 1, 2])
    elif mode == "L":
        pixels.putdata([0, 120, 240])
    else:
        pixels.putdata([(0, 0, 0), (120, 120, 120), (240, 240, 240)])
    pixels.save(source, transparency=transparency, icc_profile=_profile_bytes("p3"))
    decoded = image_from_bgr_array(load_source_image(source).pixels)
    assert decoded.mode == "RGBA"
    assert list(decoded.getchannel("A").get_flattened_data()) == expected
    output = tmp_path / ("out." + output_format.value)
    save_output_image(
        decoded, output_path=output, output_format=output_format, quality=100,
        background="white", source_metadata=load_source_image(source).metadata,
        strip_metadata=True, target_profile=None,
    )
    with Image.open(output) as saved:
        if output_format == OutputFormat.JPG:
            assert saved.mode == "RGB"
            assert all(channel >= 250 for channel in saved.getpixel((0, 0)))
        else:
            assert saved.mode == "RGBA"
            assert list(saved.getchannel("A").get_flattened_data()) == expected


@pytest.mark.parametrize("dtype", ["<u2", ">u2"])
def test_16_bit_grayscale_conversion_is_endian_safe(dtype: str) -> None:
    import numpy as np

    image = Image.fromarray(np.array([[0, 16384, 32768, 49152, 65535]], dtype=dtype))
    with rgb_pixels(image) as decoded:
        assert [pixel[0] for pixel in decoded.get_flattened_data()] == [0, 64, 128, 191, 255]


def test_gray_icc_16_bit_midtones_convert_from_normalized_levels(tmp_path: Path) -> None:
    import numpy as np

    from pixelup.imaging import load_source_image

    source = tmp_path / "gray.png"
    Image.fromarray(np.array([[0, 16384, 32768, 49152, 65535]], dtype=np.uint16)).save(
        source, icc_profile=GRAY_PROFILE,
    )
    captured = load_source_image(source)
    reference = Image.fromarray(np.array([[0, 64, 128, 191, 255]], dtype=np.uint8))
    reference.info["icc_profile"] = GRAY_PROFILE
    with rgb_pixels(reference) as expected:
        assert captured.pixels[:, :, ::-1].tolist() == np.asarray(expected).tolist()
    assert captured.metadata.icc_profile == _profile_bytes("srgb")
    assert captured.metadata.decoded_to_srgb is True


def test_16_bit_transparency_keys_precede_quantization_and_orientation(tmp_path: Path) -> None:
    import numpy as np

    from pixelup.imaging import load_source_image

    source = tmp_path / "gray.png"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(np.array([[32768, 32769], [0, 65535]], dtype=np.uint16)).save(
        source, transparency=32768, exif=exif,
    )
    captured = load_source_image(source)

    assert captured.pixels[:, :, 0].tolist() == [[0, 128], [255, 128]]
    assert captured.pixels[:, :, 3].tolist() == [[255, 0], [255, 255]]


def test_source_capture_keeps_original_exif_offsets_before_transpose(tmp_path: Path) -> None:
    from pixelup.imaging import load_source_image

    note = b"MAKER-NOTE-BYTES"
    thumbnail = b"\xff\xd8thumbnail\xff\xd9"
    header = b"II*\x00" + pack("<I", 8)
    ifd0 = (
        pack("<H", 2) + pack("<HHIHH", 0x0112, 3, 1, 6, 0)
        + pack("<HHII", 0x8769, 4, 1, 38) + pack("<I", 72)
    )
    exif_ifd = pack("<H", 1) + pack("<HHII", 0x927C, 7, len(note), 56) + bytes(4)
    ifd1 = (
        pack("<H", 3) + pack("<HHIHH", 0x0112, 3, 1, 8, 0)
        + pack("<HHII", 0x0201, 4, 1, 114)
        + pack("<HHII", 0x0202, 4, 1, len(thumbnail)) + bytes(4)
    )
    original = b"Exif\x00\x00" + header + ifd0 + exif_ifd + note + ifd1 + thumbnail
    source = tmp_path / "source.png"
    Image.new("RGB", (3, 2), "red").save(source, exif=original)
    captured = load_source_image(source)
    assert captured.metadata.exif == original
    output = tmp_path / "out.png"
    save_output_image(
        image_from_bgr_array(captured.pixels), output_path=output,
        output_format=OutputFormat.PNG, quality=95, background="white",
        source_metadata=captured.metadata, strip_metadata=False, target_profile=None,
    )
    with Image.open(output) as saved:
        exif = saved.getexif()
        assert exif[0x0112] == 1
        assert exif.get_ifd(0x8769)[0x927C] == note
        preview = exif.get_ifd(-1)
        assert preview[0x0112] == 8
        assert (preview[0x0201], preview[0x0202]) == (114, len(thumbnail))
        block = saved.info["exif"]
        assert block[6 + 56:6 + 72] == note
        assert block[6 + 114:6 + 114 + len(thumbnail)] == thumbnail


def test_heif_opener_applies_orientation_once(tmp_path: Path) -> None:
    from PIL import ImageOps
    from pillow_heif import register_heif_opener

    from pixelup.imaging import (
        load_source_image,
        read_image_orientation,
        read_image_size,
    )

    register_heif_opener()
    pixels = Image.new("RGB", (12, 8))
    pixels.paste("red", (0, 0, 6, 8))
    pixels.paste("blue", (6, 0, 12, 8))
    exif = Image.Exif()
    exif[0x0112] = 6
    source = tmp_path / "source.heic"
    pixels.save(source, format="HEIF", quality=100, exif=exif.tobytes())
    with Image.open(source) as ordinary:
        assert ordinary.info["original_orientation"] == 6
        assert ordinary.getexif()[0x0112] == 1
        expected = ImageOps.exif_transpose(ordinary)
    assert read_image_size(source) == (8, 12)
    assert read_image_orientation(source) == 1
    captured = load_source_image(source)
    with image_from_bgr_array(captured.pixels) as decoded:
        assert decoded.size == expected.size == (8, 12)
        assert decoded.tobytes() == expected.tobytes()


def test_header_probe_does_not_decode_png_pixels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from PIL import PngImagePlugin

    from pixelup.imaging import read_image_size

    source = tmp_path / "source.png"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (3, 2)).save(source, exif=exif)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *args:
                        pytest.fail("header size probe must not decode the raster"))
    assert read_image_size(source) == (2, 3)


@pytest.mark.parametrize("orientation", [b'tiff:Orientation="6"',
                                         b"><tiff:Orientation>6</tiff:Orientation>"])
def test_xmp_only_orientation_agrees_with_the_header_probe_without_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orientation: bytes,
) -> None:
    from PIL import PngImagePlugin

    from pixelup.imaging import load_source_image, read_image_size

    xmp = (
        b'<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
        b' xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
        + orientation
        + (b"</rdf:Description>" if orientation.startswith(b">") else b"/>")
    )
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_itxt("XML:com.adobe.xmp", xmp.decode())
    source = tmp_path / "source.png"
    Image.new("RGB", (3, 2)).save(source, pnginfo=pnginfo)
    captured = load_source_image(source)
    assert captured.size == (2, 3)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *args:
                        pytest.fail("header size probe must not decode the raster"))
    assert read_image_size(source) == captured.size


@pytest.mark.parametrize("metadata,expected", [
    ("absent", 1), ("exif", 6), ("xmp-attribute", 6), ("xmp-element", 6), ("both", 2),
])
def test_orientation_reader_uses_metadata_without_decoding_or_verifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: str, expected: int,
) -> None:
    from PIL import PngImagePlugin

    from pixelup.imaging import read_image_orientation

    source = tmp_path / "source.png"
    kwargs = {}
    if metadata in {"exif", "both"}:
        exif = Image.Exif()
        exif[0x0112] = 2 if metadata == "both" else 6
        kwargs["exif"] = exif
    if metadata in {"xmp-attribute", "xmp-element", "both"}:
        description = (
            '<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            ' xmlns:tiff="http://ns.adobe.com/tiff/1.0/"'
        )
        xmp = description + (
            "><tiff:Orientation>6</tiff:Orientation></rdf:Description>"
            if metadata == "xmp-element" else ' tiff:Orientation="6"/>'
        )
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_itxt("XML:com.adobe.xmp", xmp)
        kwargs["pnginfo"] = pnginfo
    Image.new("RGB", (3, 2)).save(source, **kwargs)

    def unexpected(*args):
        pytest.fail("orientation reading must remain metadata-only")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", unexpected)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "verify", unexpected)
    assert read_image_orientation(source) == expected


@pytest.mark.parametrize("exists", [False, True])
def test_orientation_reader_preserves_input_error_causes(tmp_path: Path, exists: bool) -> None:
    from PIL import UnidentifiedImageError

    from pixelup.errors import ErrorCode
    from pixelup.imaging import read_image_orientation

    source = tmp_path / "source.png"
    if exists:
        source.write_bytes(b"invalid image content")
    with pytest.raises(PixelupError) as error:
        read_image_orientation(source)
    assert error.value.code == (
        ErrorCode.INPUT_INVALID_FORMAT if exists else ErrorCode.INPUT_UNREADABLE
    )
    assert isinstance(
        error.value.__cause__, UnidentifiedImageError if exists else FileNotFoundError
    )
    assert error.value.details["input"] == str(source)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file read permissions")
def test_orientation_reader_reports_permission_denial(tmp_path: Path) -> None:
    from pixelup.errors import ErrorCode
    from pixelup.imaging import read_image_orientation

    source = tmp_path / "source.png"
    Image.new("RGB", (3, 2)).save(source)
    source.chmod(0)
    try:
        with pytest.raises(PixelupError) as error:
            read_image_orientation(source)
        assert error.value.code == ErrorCode.INPUT_UNREADABLE
        assert isinstance(error.value.__cause__, PermissionError)
    finally:
        source.chmod(0o600)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ordinary permission bits")
@pytest.mark.parametrize("source_mode", [0o600, 0o640, 0o604, 0o644])
def test_image_output_does_not_broaden_source_read_permissions(
    tmp_path: Path, source_mode: int
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), "red").save(source)
    source.chmod(source_mode)
    captured = load_source_image(source)
    ordinary = tmp_path / "ordinary"
    ordinary.write_bytes(b"")
    ordinary_reads = ordinary.stat().st_mode & 0o444
    output = tmp_path / "out.png"

    save_output_image(
        Image.new("RGB", (2, 2), "red"), output_path=output,
        output_format=OutputFormat.PNG, quality=95, background="white",
        source_metadata=captured.metadata, strip_metadata=True, target_profile=None,
        read_permissions=captured.read_permissions,
    )

    assert captured.read_permissions == source_mode & 0o444
    assert output.stat().st_mode & 0o444 == ordinary_reads & captured.read_permissions
    assert output.stat().st_mode & 0o7111 == 0
    with Image.open(output) as decoded:
        assert decoded.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX ordinary permission bits")
def test_image_staging_is_private_before_any_encoded_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_save = Image.Image.save
    observed: list[int] = []

    def checked_save(image, file, *args, **kwargs):
        descriptor = file.fileno()
        status = os.fstat(descriptor)
        assert status.st_size == 0
        assert status.st_mode & 0o077 == 0
        observed.append(descriptor)
        return original_save(image, file, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", checked_save)
    save_output_image(
        Image.new("RGB", (2, 2), "red"), output_path=tmp_path / "out.png",
        output_format=OutputFormat.PNG, quality=95, background="white",
        source_metadata=SourceMetadata(), strip_metadata=True, target_profile=None,
    )
    assert len(observed) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX ordinary permission bits")
def test_source_mode_and_bytes_are_captured_from_the_same_open_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    source = tmp_path / "source.png"
    replacement = tmp_path / "replacement.png"
    Image.new("RGB", (2, 2), "red").save(source)
    Image.new("RGB", (3, 3), "blue").save(replacement)
    original_bytes = source.read_bytes()
    source.chmod(0o600)
    replacement.chmod(0o644)
    original_open = Path.open
    captures = []

    def open_then_replace(path, *args, **kwargs):
        opened = original_open(path, *args, **kwargs)
        if path == source:
            captures.append(path)
            replacement.replace(source)
        return opened

    monkeypatch.setattr(Path, "open", open_then_replace)
    captured = load_source_image(source)

    assert captures == [source]
    assert captured.size == (2, 2)
    assert captured.sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert captured.read_permissions == 0o400
    assert source.stat().st_mode & 0o444 == 0o444
