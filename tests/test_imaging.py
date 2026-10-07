import errno
import os
import re
import xml.etree.ElementTree as ElementTree
from io import BytesIO
from itertools import product
from pathlib import Path
from struct import pack

import pytest
from PIL import Image, ImageCms

from pixelup.errors import PixelupError
from pixelup.icc_profiles import D50_XYZ, _description_tag, _profile_with_tags, _xyz_tag
from pixelup.imaging import (
    SourceMetadata,
    _profile_bytes,
    load_source_metadata,
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


def test_save_output_image_temp_file_uses_stem_nanoid_shape_beside_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The staged temp name is <stem>-<nanoid>.tmp, derived from the target
    # output's stem, and it is staged in the output's own directory so the
    # final no-clobber publication always stays on one volume.
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
    assert re.fullmatch(r"photo-x4plus-4x-[A-Za-z0-9_-]{21}\.tmp", captured[0].name)
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


def test_save_output_image_without_hard_links_removes_a_partly_copied_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.png"

    def unsupported_link(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EPERM, "hard links unsupported")

    def disk_full(source: object, target: object) -> None:
        target.write(b"partial")  # type: ignore[attr-defined]
        target.flush()  # type: ignore[attr-defined]
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("pixelup.imaging.os.link", unsupported_link)
    monkeypatch.setattr("pixelup.output_reservation.os.link", unsupported_link)
    monkeypatch.setattr("pixelup.imaging._copy_staged", disk_full)

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


def test_save_output_image_falls_back_to_an_exclusive_claim_without_hard_links(
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
    metadata = load_source_metadata(source)
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
    from pixelup.inference import _read_input_image

    source = tmp_path / "in.jpg"
    Image.new("CMYK", (2, 1), (255, 0, 0, 0)).save(source, icc_profile=CMYK_PROFILE)

    blue, green, red = _read_input_image(source)[0, 0]

    assert blue == green == red
