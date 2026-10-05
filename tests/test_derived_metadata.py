import struct
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from pixelup.derived_metadata import (
    OutputColor,
    exif_declares_adobe_rgb,
    exif_for_output,
    xmp_for_output,
)

EXIF_IFD = 0x8769
GPS_IFD = 0x8825
INTEROP_IFD = 0xA005
MODIFIED = datetime(2026, 10, 5, 21, 30, 15, tzinfo=timezone(timedelta(hours=9)))
STAMP = b"2026-10-05T21:30:15+09:00"


def camera_exif(*, color_space: int = 1, interop_index: str | None = "R98") -> bytes:
    exif = Image.Exif()
    exif[0x0110] = "Camera"
    exif[0x0132] = "2021:01:01 00:00:00"
    exif_ifd: dict[int, object] = {
        0xA002: 400,
        0xA003: 300,
        0xA001: color_space,
        0x9003: "2020:01:02 03:04:05",
        0x9010: "-05:00",
        0x9011: "+01:00",
    }
    if interop_index is not None:
        exif_ifd[INTEROP_IFD] = {1: interop_index, 2: b"0100"}
    exif[EXIF_IFD] = exif_ifd
    exif[GPS_IFD] = {1: "N", 2: (35.0, 39.0, 0.0)}
    return exif.tobytes()


def read_exif(data: bytes) -> Image.Exif:
    exif = Image.Exif()
    exif.load(data)
    return exif


def test_exif_carries_the_output_dimensions_and_keeps_capture_facts() -> None:
    source = camera_exif()

    output = exif_for_output(source, size=(1600, 1200), color=None, modified=MODIFIED)

    exif = read_exif(output)
    exif_ifd = exif.get_ifd(EXIF_IFD)
    assert (exif_ifd[0xA002], exif_ifd[0xA003]) == (1600, 1200)
    assert exif_ifd[0xA001] == 1
    assert (exif_ifd[0x9003], exif_ifd[0x9011]) == ("2020:01:02 03:04:05", "+01:00")
    assert exif.get_ifd(GPS_IFD)[1] == "N"
    assert exif[0x0110] == "Camera"
    # The modification date is the output's, in local wall-clock time with its offset.
    assert (exif[0x0132], exif_ifd[0x9010]) == ("2026:10:05 21:30:15", "+09:00")
    # Patched in place: nothing moves, so offsets into the block stay valid.
    assert len(output) == len(source)


def test_exif_without_dates_gains_them_and_keeps_every_other_value() -> None:
    exif = Image.Exif()
    exif[0x0110] = "Camera"
    exif[0x0131] = "Editor"
    source = exif.tobytes()

    output = read_exif(exif_for_output(source, size=(8, 4), color=None, modified=MODIFIED))

    assert (output[0x0110], output[0x0131]) == ("Camera", "Editor")
    assert output[0x0132] == "2026:10:05 21:30:15"
    assert output.get_ifd(EXIF_IFD) == {0x9010: "+09:00"}


def test_exif_needs_a_modification_time_with_its_offset() -> None:
    with pytest.raises(ValueError):
        exif_for_output(camera_exif(), size=(1, 1), color=None, modified=datetime(2026, 1, 1))


def test_exif_dimension_past_the_short_range_becomes_a_long() -> None:
    # Little-endian, hand-built so PixelXDimension and PixelYDimension are SHORTs.
    header = b"II*\x00" + struct.pack("<I", 8)
    ifd0 = struct.pack("<H", 1) + struct.pack("<HHII", EXIF_IFD, 4, 1, 26) + bytes(4)
    exif_ifd = (
        struct.pack("<H", 2)
        + struct.pack("<HHIHH", 0xA002, 3, 1, 300, 0)
        + struct.pack("<HHIHH", 0xA003, 3, 1, 200, 0)
        + bytes(4)
    )
    source = header + ifd0 + exif_ifd

    output = exif_for_output(source, size=(70000, 800), color=None, modified=MODIFIED)

    exif = read_exif(output)
    exif_ifd_values = exif.get_ifd(EXIF_IFD)
    assert (exif_ifd_values[0xA002], exif_ifd_values[0xA003]) == (70000, 800)
    assert (exif[0x0132], exif_ifd_values[0x9010]) == ("2026:10:05 21:30:15", "+09:00")
    assert output.startswith(b"Exif\x00\x00II")


@pytest.mark.parametrize(
    ("target", "color_space", "interop_index"),
    [("srgb", 1, "R98"), ("adobergb", 0xFFFF, "R03"), ("p3", 0xFFFF, None)],
)
def test_exif_colour_space_follows_a_conversion(
    target: str, color_space: int, interop_index: str | None
) -> None:
    source = camera_exif(color_space=0xFFFF, interop_index="R03")

    output = exif_for_output(
        source, size=(1600, 1200), color=OutputColor(target, "Profile"), modified=MODIFIED
    )

    exif = read_exif(output)
    assert exif.get_ifd(EXIF_IFD)[0xA001] == color_space
    interop = exif.get_ifd(INTEROP_IFD)
    assert interop.get(1) == interop_index
    assert interop[2] == b"0100"


def test_exif_adobe_rgb_is_recognized_from_dcf_or_the_nonstandard_value() -> None:
    assert exif_declares_adobe_rgb(camera_exif(color_space=0xFFFF, interop_index="R03"))
    assert exif_declares_adobe_rgb(camera_exif(color_space=2, interop_index=None))
    assert not exif_declares_adobe_rgb(camera_exif(color_space=1, interop_index="R98"))
    assert not exif_declares_adobe_rgb(camera_exif(color_space=0xFFFF, interop_index=None))
    assert not exif_declares_adobe_rgb(b"Exif\x00\x00not a tiff")


def test_unreadable_exif_is_rejected() -> None:
    with pytest.raises(ValueError):
        exif_for_output(
            b"Exif\x00\x00MM\x00*\x00\x00\xff\xff", size=(1, 1), color=None, modified=MODIFIED
        )


def test_xmp_carries_the_output_dimensions_in_either_form() -> None:
    source = (
        b"<rdf:Description tiff:ImageWidth=\"400\" tiff:ImageLength='300'"
        b' exif:DateTimeOriginal="2020-01-02T03:04:05" xmp:ModifyDate="2021-01-01T00:00:00Z">'
        b"<xmp:MetadataDate>2021-01-01T00:00:00Z</xmp:MetadataDate>"
        b"<exif:PixelXDimension>400</exif:PixelXDimension>"
        b"<exif:PixelYDimension>300</exif:PixelYDimension>"
        b"</rdf:Description>"
    )

    output = xmp_for_output(source, size=(1600, 1200), color=None, modified=MODIFIED)

    assert output == (
        b"<rdf:Description tiff:ImageWidth=\"1600\" tiff:ImageLength='1200'"
        b' exif:DateTimeOriginal="2020-01-02T03:04:05" xmp:ModifyDate="' + STAMP + b'">'
        b"<xmp:MetadataDate>" + STAMP + b"</xmp:MetadataDate>"
        b"<exif:PixelXDimension>1600</exif:PixelXDimension>"
        b"<exif:PixelYDimension>1200</exif:PixelYDimension>"
        b"</rdf:Description>"
    )


def test_xmp_without_dates_gains_them_on_its_first_description() -> None:
    source = b'<rdf:RDF><rdf:Description rdf:about=""/><rdf:Description/></rdf:RDF>'

    output = xmp_for_output(source, size=(1, 1), color=None, modified=MODIFIED)

    assert output == (
        b'<rdf:RDF><rdf:Description rdf:about=""'
        b' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        b' xmp:ModifyDate="' + STAMP + b'" xmp:MetadataDate="' + STAMP + b'"'
        b"/><rdf:Description/></rdf:RDF>"
    )


def test_xmp_colour_space_follows_a_conversion() -> None:
    dates = b' xmp:ModifyDate="' + STAMP + b'" xmp:MetadataDate="' + STAMP + b'"'
    source = (
        b'<rdf:Description exif:ColorSpace="65535" photoshop:ICCProfile="Adobe RGB (1998)"'
        + dates
        + b"/>"
    )

    output = xmp_for_output(
        source, size=(1, 1), color=OutputColor("srgb", "sRGB & co"), modified=MODIFIED
    )

    assert output == (
        b'<rdf:Description exif:ColorSpace="1" photoshop:ICCProfile="sRGB &amp; co"' + dates + b"/>"
    )
    assert xmp_for_output(source, size=(1, 1), color=None, modified=MODIFIED) == source
