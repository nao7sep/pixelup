import struct
import xml.etree.ElementTree as ElementTree
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


def test_exif_without_dates_gains_them_and_keeps_every_other_value() -> None:
    exif = Image.Exif()
    exif[0x0110] = "Camera"
    exif[0x0131] = "Editor"
    source = exif.tobytes()

    output = read_exif(exif_for_output(source, size=(8, 4), color=None, modified=MODIFIED))

    assert (output[0x0110], output[0x0131]) == ("Camera", "Editor")
    assert output[0x0132] == "2026:10:05 21:30:15"
    assert output.get_ifd(EXIF_IFD) == {0x9010: "+09:00"}


def test_exif_modification_fraction_follows_the_output_and_capture_facts_stay() -> None:
    # Big-endian, hand-built with a maker note and an IFD1 thumbnail, whose offsets point
    # into the block, and all three fractional-second tags.
    maker_note = b"MAKER-NOTE-BYTES"
    thumbnail = b"\xff\xd8thumbnail\xff\xd9"
    header = b"MM\x00*" + struct.pack(">I", 8)
    ifd0 = struct.pack(">H", 1) + struct.pack(">HHII", EXIF_IFD, 4, 1, 26) + struct.pack(">I", 96)
    exif_ifd = (
        struct.pack(">H", 4)
        + struct.pack(">HHII", 0x927C, 7, len(maker_note), 80)
        + struct.pack(">HHI", 0x9290, 2, 4)
        + b"123\x00"
        + struct.pack(">HHI", 0x9291, 2, 4)
        + b"456\x00"
        + struct.pack(">HHI", 0x9292, 2, 4)
        + b"789\x00"
        + bytes(4)
    )
    ifd1 = (
        struct.pack(">H", 2)
        + struct.pack(">HHII", 0x0201, 4, 1, 126)
        + struct.pack(">HHII", 0x0202, 4, 1, len(thumbnail))
        + bytes(4)
    )
    source = header + ifd0 + exif_ifd + maker_note + ifd1 + thumbnail
    modified = MODIFIED.replace(microsecond=250000)

    output = exif_for_output(source, size=(8, 4), color=None, modified=modified)

    exif = read_exif(output)
    exif_ifd_values = exif.get_ifd(EXIF_IFD)
    assert exif_ifd_values[0x9290] == "250"
    assert (exif_ifd_values[0x9291], exif_ifd_values[0x9292]) == ("456", "789")
    assert exif_ifd_values[0x927C] == maker_note
    ifd1_values = exif.get_ifd(-1)  # IFD1
    start, length = ifd1_values[0x0201], ifd1_values[0x0202]
    assert output[6 + start : 6 + start + length] == thumbnail


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


RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
TIFF = "http://ns.adobe.com/tiff/1.0/"
EXIF = "http://ns.adobe.com/exif/1.0/"
XMP = "http://ns.adobe.com/xap/1.0/"
PS = "http://ns.adobe.com/photoshop/1.0/"


def packet(body: str) -> bytes:
    return (
        f'<rdf:RDF xmlns:rdf="{RDF}" xmlns:t="{TIFF}" xmlns:e="{EXIF}" '
        f'xmlns:m="{XMP}" xmlns:p="{PS}">{body}</rdf:RDF>'
    ).encode()


def test_xmp_namespaces_update_attributes_and_elements_preserving_capture() -> None:
    source = packet(
        '<rdf:Description t:ImageWidth="400" t:ImageLength="300" '
        'e:DateTimeOriginal="2020-01-02T03:04:05" m:ModifyDate="old">'
        "<e:PixelXDimension>400</e:PixelXDimension>"
        "<e:PixelYDimension>300</e:PixelYDimension>"
        "<m:MetadataDate>old</m:MetadataDate></rdf:Description>"
    )
    root = ElementTree.fromstring(
        xmp_for_output(source, size=(1600, 1200), color=None, modified=MODIFIED)
    )
    description = root[0]
    assert description.get(f"{{{TIFF}}}ImageWidth") == "1600"
    assert description.get(f"{{{TIFF}}}ImageLength") == "1200"
    assert description.find(f"{{{EXIF}}}PixelXDimension").text == "1600"
    assert description.find(f"{{{EXIF}}}PixelYDimension").text == "1200"
    assert description.get(f"{{{EXIF}}}DateTimeOriginal") == "2020-01-02T03:04:05"
    assert description.get(f"{{{XMP}}}ModifyDate") == STAMP.decode()
    assert description.find(f"{{{XMP}}}MetadataDate").text == STAMP.decode()


def test_xmp_prefix_rebinding_is_not_mistaken_for_the_known_namespace() -> None:
    source = packet(
        '<rdf:Description xmlns:tiff="urn:other" xmlns:xmp="urn:other" '
        'tiff:ImageWidth="other" xmp:Value="keep" p:Headline="5 &gt; 3"/>'
    )
    root = ElementTree.fromstring(
        xmp_for_output(source, size=(8, 4), color=None, modified=MODIFIED)
    )
    description = root[0]
    assert description.get("{urn:other}ImageWidth") == "other"
    assert description.get("{urn:other}Value") == "keep"
    assert description.get(f"{{{PS}}}Headline") == "5 > 3"
    assert description.get(f"{{{XMP}}}ModifyDate") == STAMP.decode()
    assert description.get(f"{{{XMP}}}MetadataDate") == STAMP.decode()


def test_xmp_dates_are_added_to_first_description_only() -> None:
    root = ElementTree.fromstring(
        xmp_for_output(
            packet("<rdf:Description/><rdf:Description/>"),
            size=(1, 1),
            color=None,
            modified=MODIFIED,
        )
    )
    assert root[0].get(f"{{{XMP}}}ModifyDate") == STAMP.decode()
    assert root[0].get(f"{{{XMP}}}MetadataDate") == STAMP.decode()
    assert not root[1].attrib


def test_xmp_color_conversion_and_capture_comments_survive() -> None:
    source = packet(
        "<!--capture--><?keep fact?><rdf:Description "
        'e:ColorSpace="65535" p:ICCProfile="Adobe RGB (1998)"/>'
    )
    output = xmp_for_output(
        source, size=(1, 1), color=OutputColor("srgb", "sRGB & co"), modified=MODIFIED
    )
    root = ElementTree.fromstring(output)
    assert root[0].get(f"{{{EXIF}}}ColorSpace") == "1"
    assert root[0].get(f"{{{PS}}}ICCProfile") == "sRGB & co"
    assert b"<!--capture-->" in output and b"<?keep fact?>" in output


def test_unreadable_xmp_is_carried_without_prefix_guessing() -> None:
    source = b'<rdf:Description tiff:ImageWidth="400"/>'
    assert xmp_for_output(source, size=(8, 4), color=None, modified=MODIFIED) == source


def test_exif_shared_date_payload_keeps_capture_date() -> None:
    # Modification and capture entries deliberately point at the same source payload.
    header = b"II*\x00" + struct.pack("<I", 8)
    ifd0 = (
        struct.pack("<H", 2)
        + struct.pack("<HHII", 0x0132, 2, 20, 68)
        + struct.pack("<HHII", EXIF_IFD, 4, 1, 38)
        + bytes(4)
    )
    exif_ifd = (
        struct.pack("<H", 2)
        + struct.pack("<HHII", 0x9003, 2, 20, 68)
        + struct.pack("<HHII", 0xA002, 4, 1, 400)
        + bytes(4)
    )
    source = header + ifd0 + exif_ifd + b"2020:01:02 03:04:05\x00"
    output = read_exif(exif_for_output(source, size=(8, 4), color=None, modified=MODIFIED))
    assert output[0x0132] == "2026:10:05 21:30:15"
    assert output.get_ifd(EXIF_IFD)[0x9003] == "2020:01:02 03:04:05"
    assert output.get_ifd(EXIF_IFD)[0xA002] == 8


def test_exif_bad_modification_pointer_does_not_discard_dimensions() -> None:
    header = b"II*\x00" + struct.pack("<I", 8)
    ifd0 = (
        struct.pack("<H", 2)
        + struct.pack("<HHII", 0x0132, 2, 20, 9999)
        + struct.pack("<HHII", 0x0100, 4, 1, 400)
        + bytes(4)
    )
    output = read_exif(exif_for_output(header + ifd0, size=(8, 4), color=None, modified=MODIFIED))
    assert output[0x0100] == 8
    assert output[0x0132] == "2026:10:05 21:30:15"


def test_exif_known_scalars_are_corrected_without_rewriting_unknown_entries() -> None:
    header = b"II*\x00" + struct.pack("<I", 8)
    ifd0 = (
        struct.pack("<H", 2)
        + struct.pack("<HHII", 0x0100, 2, 2, 0x0039)
        + struct.pack("<HHII", EXIF_IFD, 4, 1, 38)
        + bytes(4)
    )
    exif_ifd = (
        struct.pack("<H", 3)
        + struct.pack("<HHII", 0xA002, 3, 2, 0x00040003)
        + struct.pack("<HHII", 0xA001, 2, 2, 0x0039)
        + struct.pack("<HHII", 0xC001, 2, 2, 0x0078)
        + bytes(4)
    )
    output_bytes = exif_for_output(
        header + ifd0 + exif_ifd,
        size=(8, 4),
        color=OutputColor("srgb", "sRGB"),
        modified=MODIFIED,
    )
    output = read_exif(output_bytes)
    # The emitted ColorSpace field is EXIF's SHORT count=1, not just a readable number.
    color_entry = 6 + 38 + 2 + 12
    assert struct.unpack_from("<HI", output_bytes, color_entry + 2) == (3, 1)
    assert output[0x0100] == 8
    assert output.get_ifd(EXIF_IFD)[0xA002] == 8
    assert output.get_ifd(EXIF_IFD)[0xA001] == 1
    assert output.get_ifd(EXIF_IFD)[0xC001] == "x"


@pytest.mark.parametrize("payload_offset", [68, 9999])
def test_exif_bad_or_shared_output_fraction_keeps_capture_fraction(payload_offset: int) -> None:
    header = b"II*\x00" + struct.pack("<I", 8)
    ifd0 = struct.pack("<H", 1) + struct.pack("<HHII", EXIF_IFD, 4, 1, 26) + bytes(4)
    exif_ifd = (
        struct.pack("<H", 3)
        + struct.pack("<HHII", 0x9290, 2, 7, payload_offset)
        + struct.pack("<HHII", 0x9291, 2, 7, 68)
        + struct.pack("<HHII", 0xA002, 4, 1, 400)
        + bytes(4)
    )
    output = read_exif(
        exif_for_output(
            header + ifd0 + exif_ifd + b"123456\x00",
            size=(8, 4),
            color=None,
            modified=MODIFIED.replace(microsecond=250000),
        )
    )
    assert output.get_ifd(EXIF_IFD)[0x9290] == "250000"
    assert output.get_ifd(EXIF_IFD)[0x9291] == "123456"
    assert output.get_ifd(EXIF_IFD)[0xA002] == 8


def test_xmp_inherited_rebound_prefix_and_default_rdf_namespace() -> None:
    source = (
        f'<RDF xmlns="{RDF}" xmlns:xmp="urn:other"><Description xmp:Value="capture"/></RDF>'
    ).encode()
    output = ElementTree.fromstring(
        xmp_for_output(source, size=(8, 4), color=None, modified=MODIFIED)
    )
    assert output[0].get("{urn:other}Value") == "capture"
    assert output[0].get(f"{{{XMP}}}ModifyDate") == STAMP.decode()
