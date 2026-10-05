"""Source EXIF and XMP made to describe a derived output (content-lifecycle-conventions).

EXIF is patched in place rather than re-serialized, so maker notes, whose offsets point
into the block, and the IFD1 thumbnail survive. Only tags the source already has change.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from xml.sax.saxutils import escape

EXIF_PREFIX = b"Exif\x00\x00"

# TIFF field types and the tags PixelUp reads or rewrites.
_SHORT = 3
_LONG = 4
_ASCII = 2
_IFD = 13
_TAG_IMAGE_WIDTH = 0x0100
_TAG_IMAGE_LENGTH = 0x0101
_TAG_EXIF_IFD = 0x8769
_TAG_COLOR_SPACE = 0xA001
_TAG_PIXEL_X_DIMENSION = 0xA002
_TAG_PIXEL_Y_DIMENSION = 0xA003
_TAG_INTEROP_IFD = 0xA005
_TAG_INTEROP_INDEX = 0x0001

# EXIF records sRGB as ColorSpace 1 and anything else as Uncalibrated; DCF marks Adobe
# RGB as Uncalibrated with InteropIndex "R03". Some cameras write the non-standard 2.
_COLOR_SPACE_SRGB = 1
_COLOR_SPACE_ADOBE_RGB_NONSTANDARD = 2
_COLOR_SPACE_UNCALIBRATED = 0xFFFF
_INTEROP_INDEX = {"srgb": b"R98\x00", "adobergb": b"R03\x00"}


@dataclass(frozen=True, slots=True)
class OutputColor:
    """The colour space a converted output is in: a PixelUp profile name and its description."""

    name: str
    description: str


def exif_declares_adobe_rgb(exif: bytes) -> bool:
    try:
        tiff = _Tiff(exif)
        exif_ifd = tiff.sub_ifd(tiff.entries(tiff.first_ifd), _TAG_EXIF_IFD)
        if exif_ifd is None:
            return False
        entries = tiff.entries(exif_ifd)
        color_space = tiff.integer(entries.get(_TAG_COLOR_SPACE))
        if color_space == _COLOR_SPACE_ADOBE_RGB_NONSTANDARD:
            return True
        if color_space != _COLOR_SPACE_UNCALIBRATED:
            return False
        interop_ifd = tiff.sub_ifd(entries, _TAG_INTEROP_IFD)
        if interop_ifd is None:
            return False
        index = tiff.ascii(tiff.entries(interop_ifd).get(_TAG_INTEROP_INDEX))
        return index is not None and index.startswith(b"R03")
    except ValueError:
        return False


def exif_for_output(exif: bytes, *, size: tuple[int, int], color: OutputColor | None) -> bytes:
    """The source EXIF describing an output of ``size``, converted to ``color`` when given.

    Raises ValueError when the block is not a readable TIFF structure.
    """
    tiff = _Tiff(exif)
    width, height = size
    ifd0 = tiff.entries(tiff.first_ifd)
    tiff.set_integer(ifd0.get(_TAG_IMAGE_WIDTH), width)
    tiff.set_integer(ifd0.get(_TAG_IMAGE_LENGTH), height)
    exif_ifd = tiff.sub_ifd(ifd0, _TAG_EXIF_IFD)
    if exif_ifd is not None:
        entries = tiff.entries(exif_ifd)
        tiff.set_integer(entries.get(_TAG_PIXEL_X_DIMENSION), width)
        tiff.set_integer(entries.get(_TAG_PIXEL_Y_DIMENSION), height)
        if color is not None:
            tiff.set_integer(
                entries.get(_TAG_COLOR_SPACE),
                _COLOR_SPACE_SRGB if color.name == "srgb" else _COLOR_SPACE_UNCALIBRATED,
            )
            interop_ifd = tiff.sub_ifd(entries, _TAG_INTEROP_IFD)
            if interop_ifd is not None:
                tiff.set_interop_index(interop_ifd, _INTEROP_INDEX.get(color.name))
    return EXIF_PREFIX + bytes(tiff.data)


def xmp_for_output(xmp: bytes, *, size: tuple[int, int], color: OutputColor | None) -> bytes:
    """The source XMP describing an output of ``size``, converted to ``color`` when given."""
    width, height = size
    values: dict[bytes, str] = {
        b"tiff:ImageWidth": str(width),
        b"tiff:ImageLength": str(height),
        b"exif:PixelXDimension": str(width),
        b"exif:PixelYDimension": str(height),
    }
    if color is not None:
        values[b"exif:ColorSpace"] = str(
            _COLOR_SPACE_SRGB if color.name == "srgb" else _COLOR_SPACE_UNCALIBRATED
        )
        values[b"photoshop:ICCProfile"] = color.description
    for name, value in values.items():
        xmp = _set_xmp_property(xmp, name, value)
    return xmp


def _set_xmp_property(xmp: bytes, name: bytes, value: str) -> bytes:
    quoted = re.escape(name)
    attribute = re.compile(rb"(\s" + quoted + rb"\s*=\s*)([\"'])(.*?)\2", re.DOTALL)
    element = re.compile(rb"(<" + quoted + rb">)([^<]*)(</" + quoted + rb">)")
    xmp = attribute.sub(
        lambda match: (
            match[1]
            + match[2]
            + escape(value, {'"': "&quot;", "'": "&apos;"}).encode("utf-8")
            + match[2]
        ),
        xmp,
    )
    return element.sub(
        lambda match: match[1] + escape(value).encode("utf-8") + match[3],
        xmp,
    )


class _Tiff:
    """A bounds-checked view of an EXIF TIFF structure that patches values in place."""

    def __init__(self, exif: bytes) -> None:
        body = exif[len(EXIF_PREFIX) :] if exif.startswith(EXIF_PREFIX) else exif
        self.data = bytearray(body)
        order = bytes(self.data[:2])
        if order == b"II":
            self._order = "<"
        elif order == b"MM":
            self._order = ">"
        else:
            raise ValueError("not a TIFF byte order")
        if self._unpack("H", 2) != 42:
            raise ValueError("not a TIFF header")
        self.first_ifd = self._unpack("I", 4)

    def entries(self, ifd: int) -> dict[int, int]:
        """Each tag in the IFD at ``ifd``, mapped to its entry's offset."""
        count = self._unpack("H", ifd)
        self._check(ifd + 2, 12 * count + 4)
        first = ifd + 2
        return {self._unpack("H", entry): entry for entry in range(first, first + 12 * count, 12)}

    def sub_ifd(self, entries: dict[int, int], tag: int) -> int | None:
        entry = entries.get(tag)
        if entry is None or self._type(entry) not in {_LONG, _IFD} or self._count(entry) != 1:
            return None
        return self._unpack("I", entry + 8)

    def integer(self, entry: int | None) -> int | None:
        if entry is None or self._count(entry) != 1:
            return None
        kind = self._type(entry)
        if kind == _SHORT:
            return self._unpack("H", entry + 8)
        if kind == _LONG:
            return self._unpack("I", entry + 8)
        return None

    def ascii(self, entry: int | None) -> bytes | None:
        if entry is None or self._type(entry) != _ASCII:
            return None
        count = self._count(entry)
        offset = entry + 8 if count <= 4 else self._unpack("I", entry + 8)
        self._check(offset, count)
        return bytes(self.data[offset : offset + count])

    def set_integer(self, entry: int | None, value: int) -> None:
        if self.integer(entry) is None:
            return
        assert entry is not None
        if self._type(entry) == _SHORT and value <= 0xFFFF:
            self.data[entry + 8 : entry + 12] = struct.pack(self._order + "HH", value, 0)
            return
        # A dimension past the SHORT range still fits the 4-byte value field as a LONG.
        self.data[entry + 2 : entry + 4] = struct.pack(self._order + "H", _LONG)
        self.data[entry + 8 : entry + 12] = struct.pack(self._order + "I", value)

    def set_interop_index(self, ifd: int, index: bytes | None) -> None:
        """Rewrite InteropIndex, or remove it when no DCF index names the colour space."""
        entry = self.entries(ifd).get(_TAG_INTEROP_INDEX)
        if entry is None or self._type(entry) != _ASCII or self._count(entry) != 4:
            return
        if index is not None:
            self.data[entry + 8 : entry + 12] = index
            return
        count = self._unpack("H", ifd)
        end = ifd + 2 + 12 * count
        # Shift the later entries and the next-IFD pointer up over the removed entry.
        self.data[entry : end - 8] = self.data[entry + 12 : end + 4]
        self.data[end - 8 : end + 4] = bytes(12)
        self.data[ifd : ifd + 2] = struct.pack(self._order + "H", count - 1)

    def _type(self, entry: int) -> int:
        return self._unpack("H", entry + 2)

    def _count(self, entry: int) -> int:
        return self._unpack("I", entry + 4)

    def _unpack(self, kind: str, offset: int) -> int:
        size = struct.calcsize(kind)
        self._check(offset, size)
        return struct.unpack_from(self._order + kind, self.data, offset)[0]

    def _check(self, offset: int, size: int) -> None:
        if offset < 0 or offset + size > len(self.data):
            raise ValueError("EXIF offset out of range")
