from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from pixelup.devices import (
    DEFAULT_DEVICE,
    DEVICE_CHOICES,
    DEVICE_VALUES,
    resolve_device,
    to_torch_device,
)
from pixelup.errors import PixelupError


def test_device_values_derive_from_choices() -> None:
    assert DEVICE_VALUES == tuple(value for _label, value in DEVICE_CHOICES)


def test_default_device_is_a_known_value() -> None:
    assert DEFAULT_DEVICE in DEVICE_VALUES


def test_device_values_are_lowercase_and_unique() -> None:
    assert all(value == value.lower() for value in DEVICE_VALUES)
    assert len(set(DEVICE_VALUES)) == len(DEVICE_VALUES)


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a fake ``torch`` whose backend availability the tests drive.

    Defaults to "nothing available" so each test enables only the backend it
    exercises. ``device.type``/index are recorded as plain attributes so the
    to_torch_device tests can assert what was requested without a real torch.
    """

    class _Device:
        def __init__(self, spec: str) -> None:
            self.spec = spec

    fake = ModuleType("torch")
    fake.device = _Device  # type: ignore[attr-defined]
    fake.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_cpu_resolves_without_importing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # "cpu" must not touch torch at all — make any import explode to prove it.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("cpu", None) == "cpu"


def test_resolve_device_rejects_unknown_backend() -> None:
    with pytest.raises(PixelupError) as excinfo:
        resolve_device("gpu", None)
    assert excinfo.value.code == "invalid_argument"


def test_auto_prefers_mps_then_cuda_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_torch(monkeypatch)

    assert resolve_device("auto", None) == "cpu"
    assert resolve_device("auto", 0) == "cpu"

    fake.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    assert resolve_device("auto", None) == "cuda"
    assert resolve_device("auto", 0) == "cuda"

    fake.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    assert resolve_device("auto", None) == "mps"
    assert resolve_device("auto", 0) == "mps"


def test_forced_device_validation_uses_torch_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_torch(monkeypatch)

    with pytest.raises(PixelupError) as excinfo:
        resolve_device("mps", None)
    assert excinfo.value.code == "invalid_argument"

    with pytest.raises(PixelupError) as excinfo:
        resolve_device("cuda", None)
    assert excinfo.value.code == "invalid_argument"

    fake.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    assert resolve_device("cuda", 0) == "cuda"

    with pytest.raises(PixelupError) as excinfo:
        resolve_device("cuda", 2)
    assert excinfo.value.code == "invalid_argument"


def test_to_torch_device_maps_concrete_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)

    assert to_torch_device("cpu").spec == "cpu"
    assert to_torch_device("mps").spec == "mps"
    assert to_torch_device("cuda").spec == "cuda"
    # A specific CUDA index is encoded into the device spec.
    assert to_torch_device("cuda", 1).spec == "cuda:1"


def test_to_torch_device_does_not_re_resolve_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    # to_torch_device expects an already-resolved backend; "auto" is not one, so
    # it must reject rather than silently re-running availability detection.
    _install_fake_torch(monkeypatch)
    with pytest.raises(PixelupError) as excinfo:
        to_torch_device("auto")
    assert excinfo.value.code == "invalid_argument"
