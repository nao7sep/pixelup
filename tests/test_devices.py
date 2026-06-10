from __future__ import annotations

from pixelup.devices import DEFAULT_DEVICE, DEVICE_CHOICES, DEVICE_VALUES


def test_device_values_derive_from_choices() -> None:
    assert DEVICE_VALUES == tuple(value for _label, value in DEVICE_CHOICES)


def test_default_device_is_a_known_value() -> None:
    assert DEFAULT_DEVICE in DEVICE_VALUES


def test_device_values_are_lowercase_and_unique() -> None:
    assert all(value == value.lower() for value in DEVICE_VALUES)
    assert len(set(DEVICE_VALUES)) == len(DEVICE_VALUES)
