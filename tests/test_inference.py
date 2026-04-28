import sys
from pathlib import Path
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


def test_face_enhance_uses_models_dir_for_helper_models_and_suppresses_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pixelup.inference import InferenceConfig, _run_face_enhance

    helper_roots: list[str] = []
    devices: list[object] = []

    class FakeFaceRestoreHelper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            helper_roots.append(str(kwargs["model_rootpath"]))
            print("helper stdout leak")

    class FakeGFPGANer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            import gfpgan.utils as gfpgan_utils

            devices.append(kwargs["device"])
            gfpgan_utils.FaceRestoreHelper(model_rootpath="gfpgan/weights")
            print("init stdout leak")

        def enhance(self, *args: object, **kwargs: object) -> tuple[None, None, str]:
            print("enhance stdout leak")
            return None, None, "enhanced"

    gfpgan_module = ModuleType("gfpgan")
    gfpgan_utils_module = ModuleType("gfpgan.utils")
    facexlib_module = ModuleType("facexlib")
    facexlib_utils_module = ModuleType("facexlib.utils")
    face_helper_module = ModuleType("facexlib.utils.face_restoration_helper")
    gfpgan_module.GFPGANer = FakeGFPGANer
    gfpgan_module.utils = gfpgan_utils_module
    gfpgan_utils_module.FaceRestoreHelper = FakeFaceRestoreHelper
    face_helper_module.FaceRestoreHelper = FakeFaceRestoreHelper

    monkeypatch.setitem(sys.modules, "gfpgan", gfpgan_module)
    monkeypatch.setitem(sys.modules, "gfpgan.utils", gfpgan_utils_module)
    monkeypatch.setitem(sys.modules, "facexlib", facexlib_module)
    monkeypatch.setitem(sys.modules, "facexlib.utils", facexlib_utils_module)
    monkeypatch.setitem(
        sys.modules,
        "facexlib.utils.face_restoration_helper",
        face_helper_module,
    )

    config = InferenceConfig(
        input_path=tmp_path / "input.png",
        models_dir=tmp_path / "models",
        model="realesr-general-x4v3",
        scale=4,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
        face_enhance=True,
        denoise_strength=1.0,
        alpha_mode="realesrgan",
        gpu_id=None,
        device="cpu",
    )

    result = _run_face_enhance(config, object(), object(), torch_device="pixelup-device")

    assert result == "enhanced"
    assert helper_roots == [str(tmp_path / "models")]
    assert devices == ["pixelup-device"]
    assert gfpgan_utils_module.FaceRestoreHelper is FakeFaceRestoreHelper
    assert capsys.readouterr().out == ""
