from types import ModuleType

import pytest

from pixelup.errors import PixelupError
from pixelup.inference import (
    ModelArchitectureSpec,
    _assert_safe_torch_load,
    _build_network,
    model_architecture_spec,
)
from pixelup.upscale import count_tiles


def _fake_torch(version: str) -> ModuleType:
    module = ModuleType("torch")
    module.__version__ = version  # type: ignore[attr-defined]
    return module


def test_assert_safe_torch_load_rejects_pre_2_6() -> None:
    # torch < 2.6 loaded weights with full unpickling by default; refuse it.
    with pytest.raises(PixelupError) as excinfo:
        _assert_safe_torch_load(_fake_torch("2.5.1"))
    assert excinfo.value.code == "internal_error"


def test_assert_safe_torch_load_accepts_2_6_and_newer() -> None:
    # No raise == the safe-load floor is satisfied.
    _assert_safe_torch_load(_fake_torch("2.6.0"))
    _assert_safe_torch_load(_fake_torch("2.11.0+cpu"))


def test_assert_safe_torch_load_tolerates_unparseable_version() -> None:
    # An odd build string must not brick a legitimate install; the SHA-256 pin
    # stays the primary protection.
    _assert_safe_torch_load(_fake_torch("weird-build"))


def test_model_architecture_specs_match_builtin_models() -> None:
    x4 = model_architecture_spec("RealESRGAN_x4plus")
    assert x4.kind == "rrdb"
    assert x4.netscale == 4
    assert x4.params["num_block"] == 23

    anime = model_architecture_spec("RealESRGAN_x4plus_anime_6B")
    assert anime.kind == "rrdb"
    assert anime.params["num_block"] == 6

    general = model_architecture_spec("realesr-general-x4v3")
    assert general.kind == "srvgg"
    assert general.params["num_conv"] == 32


def test_unknown_model_uses_requested_scale_rrdb_default() -> None:
    spec = model_architecture_spec("custom-model", requested_scale=2)

    assert spec.kind == "rrdb"
    assert spec.netscale == 2
    assert spec.params["scale"] == 2


def test_count_tiles_matches_input_grid() -> None:
    assert count_tiles((400, 300), 0) == 1
    assert count_tiles((400, 300), 256) == 4
    assert count_tiles((1, 1), 512) == 1


def test_build_network_uses_srvgg_architecture() -> None:
    from pixelup.realesrgan_models import SRVGGNetCompact

    result = _build_network(model_architecture_spec("realesr-general-x4v3"))

    assert isinstance(result, SRVGGNetCompact)
    assert result.num_conv == 32


def test_tile_reporting_upsampler_emits_per_tile_callback(capsys) -> None:
    from pixelup.inference import _tile_reporting_upsampler_class

    class _FakeImg:
        shape = (1, 3, 512, 1024)

    class _FakeBase:
        def __init__(self) -> None:
            self.img = _FakeImg()
            self.tile_size = 512
            self.model = lambda x: x
            self.calls = 0

        def tile_process(self) -> None:
            tiles_x = (self.img.shape[3] + self.tile_size - 1) // self.tile_size
            tiles_y = (self.img.shape[2] + self.tile_size - 1) // self.tile_size
            for _ in range(tiles_x * tiles_y):
                self.model("tile")
                self.calls += 1
                print("upstream tile progress")

    cls = _tile_reporting_upsampler_class(_FakeBase)
    upsampler = cls()
    events: list[tuple[int, int]] = []
    upsampler._pixelup_on_tile = lambda i, n: events.append((i, n))

    upsampler.tile_process()

    assert events == [(1, 2), (2, 2)]
    assert upsampler.calls == 2
    assert capsys.readouterr().out == ""


def test_tile_reporting_upsampler_falls_back_when_shape_unexpected() -> None:
    from pixelup.inference import _tile_reporting_upsampler_class

    class _FakeBase:
        def __init__(self) -> None:
            self.img = object()  # no .shape attribute
            self.tile_size = 512
            self.model = lambda x: x
            self.ran = False

        def tile_process(self) -> None:
            self.ran = True

    cls = _tile_reporting_upsampler_class(_FakeBase)
    upsampler = cls()
    upsampler._pixelup_on_tile = lambda i, n: pytest.fail("should not be called")

    upsampler.tile_process()

    assert upsampler.ran is True


def test_tile_reporting_upsampler_swallows_callback_exception() -> None:
    from pixelup.inference import _tile_reporting_upsampler_class

    class _FakeImg:
        shape = (1, 3, 512, 512)

    class _FakeBase:
        def __init__(self) -> None:
            self.img = _FakeImg()
            self.tile_size = 512
            self.model = lambda x: x

        def tile_process(self) -> None:
            self.model("tile")

    cls = _tile_reporting_upsampler_class(_FakeBase)
    upsampler = cls()

    def bad_callback(i: int, n: int) -> None:
        raise RuntimeError("boom")

    upsampler._pixelup_on_tile = bad_callback

    upsampler.tile_process()


def test_build_network_rejects_unknown_architecture() -> None:
    spec = ModelArchitectureSpec(kind="custom", netscale=4, params={})

    with pytest.raises(PixelupError) as excinfo:
        _build_network(spec)

    assert excinfo.value.code == "internal_error"
    assert excinfo.value.details == {"kind": "custom"}
