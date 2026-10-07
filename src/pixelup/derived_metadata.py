"""Source EXIF and XMP made to describe a derived output (content-lifecycle-conventions).

EXIF is patched rather than re-serialized, so maker notes, whose offsets point into the
block, and the IFD1 thumbnail survive. Scalars are written in their entries; changed
strings and their IFDs are appended at the end of the block. Descriptive tags change only
where the source has them; the modification dates are always written.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime
from xml.dom import Node, minidom
from xml.parsers.expat import ExpatError

EXIF_PREFIX = b"Exif\x00\x00"

# TIFF field types and the tags PixelUp reads or rewrites.
_SHORT = 3
_LONG = 4
_ASCII = 2
_IFD = 13
_TAG_IMAGE_WIDTH = 0x0100
_TAG_IMAGE_LENGTH = 0x0101
_TAG_DATE_TIME = 0x0132
_TAG_EXIF_IFD = 0x8769
_TAG_OFFSET_TIME = 0x9010
_TAG_SUBSEC_TIME = 0x9290
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
_RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XMP_NAMESPACE = "http://ns.adobe.com/xap/1.0/"
_TIFF_NAMESPACE = "http://ns.adobe.com/tiff/1.0/"
_EXIF_NAMESPACE = "http://ns.adobe.com/exif/1.0/"
_PHOTOSHOP_NAMESPACE = "http://ns.adobe.com/photoshop/1.0/"
_XMLNS_NAMESPACE = "http://www.w3.org/2000/xmlns/"


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


def exif_for_output(
    exif: bytes,
    *,
    size: tuple[int, int],
    color: OutputColor | None,
    modified: datetime,
) -> bytes:
    """The source EXIF describing an output of ``size``, converted to ``color`` when given.

    DateTime is written as ``modified``'s local wall-clock time with OffsetTime beside it,
    and an existing SubsecTime as its fraction, in the source's number of digits; the
    capture dates, their offsets and fractions are kept. Raises ValueError when the block
    is not a readable TIFF structure.
    """
    tiff = _Tiff(exif)
    width, height = size
    ifd0 = tiff.entries(tiff.first_ifd)
    tiff.set_integer(ifd0.get(_TAG_IMAGE_WIDTH), width)
    tiff.set_integer(ifd0.get(_TAG_IMAGE_LENGTH), height)
    exif_ifd = tiff.sub_ifd(ifd0, _TAG_EXIF_IFD)
    subsec = None
    if exif_ifd is not None:
        entries = tiff.entries(exif_ifd)
        try:
            subsec = tiff.ascii(entries.get(_TAG_SUBSEC_TIME))
        except ValueError:
            # An invalid modification fraction does not invalidate other capture data.
            subsec = b"000000"
        tiff.set_integer(entries.get(_TAG_PIXEL_X_DIMENSION), width)
        tiff.set_integer(entries.get(_TAG_PIXEL_Y_DIMENSION), height)
        if color is not None:
            tiff.set_integer(
                entries.get(_TAG_COLOR_SPACE),
                _COLOR_SPACE_SRGB if color.name == "srgb" else _COLOR_SPACE_UNCALIBRATED,
                short=True,
            )
            interop_ifd = tiff.sub_ifd(entries, _TAG_INTEROP_IFD)
            if interop_ifd is not None:
                tiff.set_interop_index(interop_ifd, _INTEROP_INDEX.get(color.name))
    tiff.set_ascii(_IFD0_POINTER, _TAG_DATE_TIME, modified.strftime("%Y:%m:%d %H:%M:%S"))
    tiff.set_ascii(tiff.exif_ifd_pointer(), _TAG_OFFSET_TIME, _offset(modified))
    if subsec is not None:
        digits = len(subsec.rstrip(b"\x00")) or 1
        fraction = f"{modified.microsecond:06d}".ljust(digits, "0")[:digits]
        tiff.set_ascii(tiff.exif_ifd_pointer(), _TAG_SUBSEC_TIME, fraction)
    return EXIF_PREFIX + bytes(tiff.data)


def xmp_for_output(
    xmp: bytes,
    *,
    size: tuple[int, int],
    color: OutputColor | None,
    modified: datetime,
) -> bytes:
    """The source XMP describing an output of ``size``, converted to ``color`` when given.

    xmp:ModifyDate and xmp:MetadataDate are written as ``modified`` with its offset.
    """
    try:
        document = minidom.parseString(xmp)
    except ExpatError:
        return xmp
    try:
        descriptions = document.getElementsByTagNameNS(_RDF_NAMESPACE, "Description")
        if not descriptions:
            return xmp
        width, height = size
        values = {
            (_TIFF_NAMESPACE, "ImageWidth"): str(width),
            (_TIFF_NAMESPACE, "ImageLength"): str(height),
            (_EXIF_NAMESPACE, "PixelXDimension"): str(width),
            (_EXIF_NAMESPACE, "PixelYDimension"): str(height),
        }
        if color is not None:
            values[(_EXIF_NAMESPACE, "ColorSpace")] = str(
                _COLOR_SPACE_SRGB if color.name == "srgb" else _COLOR_SPACE_UNCALIBRATED
            )
            values[(_PHOTOSHOP_NAMESPACE, "ICCProfile")] = color.description
        stamp = modified.strftime("%Y-%m-%dT%H:%M:%S") + _offset(modified)
        values[(_XMP_NAMESPACE, "ModifyDate")] = stamp
        values[(_XMP_NAMESPACE, "MetadataDate")] = stamp
        for (namespace, name), value in values.items():
            found = False
            for description in descriptions:
                attribute = description.getAttributeNodeNS(namespace, name)
                if attribute is not None:
                    attribute.value = value
                    found = True
                for element in description.childNodes:
                    if (
                        element.nodeType == Node.ELEMENT_NODE
                        and element.namespaceURI == namespace
                        and element.localName == name
                    ):
                        for child in list(element.childNodes):
                            element.removeChild(child)
                        element.appendChild(document.createTextNode(value))
                        found = True
            if not found and namespace == _XMP_NAMESPACE:
                # The declaration is local, so an ancestor's different xmp binding is untouched.
                description = descriptions[0]
                bindings: dict[str, str] = {}
                ancestor = description
                while ancestor.nodeType == Node.ELEMENT_NODE:
                    for attribute in ancestor.attributes.values():
                        if attribute.prefix == "xmlns":
                            bindings.setdefault(attribute.localName, attribute.value)
                    ancestor = ancestor.parentNode
                prefix = "xmp"
                suffix = 0
                while prefix in bindings and bindings[prefix] != namespace:
                    suffix += 1
                    prefix = f"xmp{suffix}"
                description.setAttributeNS(_XMLNS_NAMESPACE, f"xmlns:{prefix}", namespace)
                description.setAttributeNS(namespace, f"{prefix}:{name}", value)
        return document.toxml(encoding="utf-8")

    finally:
        document.unlink()


def _offset(moment: datetime) -> str:
    if moment.utcoffset() is None:
        raise ValueError("a modification time needs its UTC offset")
    return moment.isoformat()[-6:]


_IFD0_POINTER = 4  # Where the TIFF header stores IFD0's offset.


class _Tiff:
    """A bounds-checked view of an EXIF TIFF structure that patches it without moving data."""

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
        self.first_ifd = self._unpack("I", _IFD0_POINTER)

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

    def set_integer(self, entry: int | None, value: int, *, short: bool = False) -> None:
        if entry is None:
            return
        if short or (self._type(entry) == _SHORT and value <= 0xFFFF):
            self.data[entry + 2 : entry + 8] = struct.pack(self._order + "HI", _SHORT, 1)
            self.data[entry + 8 : entry + 12] = struct.pack(self._order + "HH", value, 0)
            return
        # A dimension past the SHORT range still fits the 4-byte value field as a LONG.
        self.data[entry + 2 : entry + 8] = struct.pack(self._order + "HI", _LONG, 1)
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

    def exif_ifd_pointer(self) -> int:
        """Where IFD0 stores the Exif IFD's offset, adding an empty Exif IFD when there is none."""
        if self.sub_ifd(self.entries(self.first_ifd), _TAG_EXIF_IFD) is None:
            empty = self._append(struct.pack(self._order + "HI", 0, 0))
            self._set_entry(
                _IFD0_POINTER, _TAG_EXIF_IFD, _LONG, 1, struct.pack(self._order + "I", empty)
            )
        return self.entries(self.first_ifd)[_TAG_EXIF_IFD] + 8

    def set_ascii(self, pointer: int, tag: int, text: str) -> None:
        """Set ``tag`` in the IFD whose offset is stored at ``pointer``."""
        value = text.encode("ascii") + b"\x00"
        self._set_entry(pointer, tag, _ASCII, len(value), value)

    def _set_entry(self, pointer: int, tag: int, kind: int, count: int, value: bytes) -> None:
        ifd = self._unpack("I", pointer)
        entries = self.entries(ifd)
        # Give the changed value its own payload and IFD; a capture entry may share the
        # original payload. Existing source values and their offsets stay where they are.
        field = value.ljust(4, b"\x00") if len(value) <= 4 else self._pack_offset(value)
        raw = {existing: bytes(self.data[at : at + 12]) for existing, at in entries.items()}
        raw[tag] = struct.pack(self._order + "HHI", tag, kind, count) + field
        next_ifd = self._unpack("I", ifd + 2 + 12 * len(entries))
        rewritten = self._append(
            struct.pack(self._order + "H", len(raw))
            + b"".join(raw[key] for key in sorted(raw))
            + struct.pack(self._order + "I", next_ifd)
        )
        self.data[pointer : pointer + 4] = struct.pack(self._order + "I", rewritten)
        if pointer == _IFD0_POINTER:
            self.first_ifd = rewritten

    def _pack_offset(self, value: bytes) -> bytes:
        return struct.pack(self._order + "I", self._append(value))

    def _append(self, payload: bytes) -> int:
        if len(self.data) % 2:
            self.data.append(0)  # TIFF offsets are word-aligned.
        offset = len(self.data)
        self.data += payload
        return offset

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
