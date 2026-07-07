from __future__ import annotations

from io import BytesIO
from struct import unpack

import pytest
from PIL import ImageCms

from pixelup.icc_profiles import _inverse_3x3, _matrix_multiply, profile_bytes

GENERATED_NAMES = ("p3", "adobergb")
ALL_NAMES = ("srgb", *GENERATED_NAMES)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_profile_bytes_parse_and_describe(name: str) -> None:
    profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes(name)))
    assert ImageCms.getProfileDescription(profile)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_profile_header_is_valid_icc(name: str) -> None:
    data = profile_bytes(name)
    assert data[36:40] == b"acsp"
    (declared_size,) = unpack(">I", data[0:4])
    assert declared_size == len(data)


@pytest.mark.parametrize("name", GENERATED_NAMES)
def test_generated_profiles_carry_pixelup_signature(name: str) -> None:
    data = profile_bytes(name)
    assert data[4:8] == b"pxup"
    assert data[80:84] == b"pxup"


def test_unknown_profile_name_raises() -> None:
    with pytest.raises(ValueError):
        profile_bytes("rec2020")


def test_profile_bytes_are_cached() -> None:
    assert profile_bytes("p3") is profile_bytes("p3")


def test_inverse_3x3_round_trips_to_identity() -> None:
    matrix = (
        (2.0, 0.0, 1.0),
        (1.0, 3.0, 2.0),
        (1.0, 0.0, 3.0),
    )
    product = _matrix_multiply(matrix, _inverse_3x3(matrix))
    for row in range(3):
        for column in range(3):
            assert product[row][column] == pytest.approx(1.0 if row == column else 0.0, abs=1e-9)
