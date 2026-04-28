import sys
from types import ModuleType

import pytest

from pixelup.errors import PixelupError
from pixelup.inference import ModelArchitectureSpec, _build_network, model_architecture_spec
from pixelup.upscale import count_tiles


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


def test_build_network_uses_srvgg_architecture(monkeypatch) -> None:
    arch_module = ModuleType("realesrgan.archs.srvgg_arch")

    class DummySRVGGNetCompact:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    arch_module.SRVGGNetCompact = DummySRVGGNetCompact
    monkeypatch.setitem(sys.modules, "realesrgan", ModuleType("realesrgan"))
    monkeypatch.setitem(sys.modules, "realesrgan.archs", ModuleType("realesrgan.archs"))
    monkeypatch.setitem(sys.modules, "realesrgan.archs.srvgg_arch", arch_module)

    result = _build_network(model_architecture_spec("realesr-general-x4v3"))

    assert isinstance(result, DummySRVGGNetCompact)
    assert result.kwargs["num_conv"] == 32


def test_build_network_rejects_unknown_architecture() -> None:
    spec = ModelArchitectureSpec(kind="custom", netscale=4, params={})

    with pytest.raises(PixelupError) as excinfo:
        _build_network(spec)

    assert excinfo.value.code == "internal_error"
    assert excinfo.value.details == {"kind": "custom"}
