import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

from pixelup.errors import PixelupError
from pixelup.inference import (
    InferenceConfig,
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


def _install_fake_gfpgan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gfpganer_cls: type,
    helper_cls: type,
) -> ModuleType:
    """Install fake gfpgan/facexlib modules and return the gfpgan.utils module.

    Production reads gfpgan.utils.FaceRestoreHelper while building the enhancer,
    so the returned module is the global the substitution targets.
    """
    gfpgan_module = ModuleType("gfpgan")
    gfpgan_utils_module = ModuleType("gfpgan.utils")
    facexlib_module = ModuleType("facexlib")
    facexlib_utils_module = ModuleType("facexlib.utils")
    face_helper_module = ModuleType("facexlib.utils.face_restoration_helper")
    gfpgan_module.GFPGANer = gfpganer_cls
    gfpgan_module.utils = gfpgan_utils_module
    gfpgan_utils_module.FaceRestoreHelper = helper_cls
    face_helper_module.FaceRestoreHelper = helper_cls

    monkeypatch.setitem(sys.modules, "gfpgan", gfpgan_module)
    monkeypatch.setitem(sys.modules, "gfpgan.utils", gfpgan_utils_module)
    monkeypatch.setitem(sys.modules, "facexlib", facexlib_module)
    monkeypatch.setitem(sys.modules, "facexlib.utils", facexlib_utils_module)
    monkeypatch.setitem(
        sys.modules,
        "facexlib.utils.face_restoration_helper",
        face_helper_module,
    )
    return gfpgan_utils_module


def _face_enhance_config(models_dir: Path) -> InferenceConfig:
    return InferenceConfig(
        input_path=models_dir / "input.png",
        models_dir=models_dir,
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


def test_face_enhance_uses_models_dir_and_suppresses_only_construction_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pixelup.inference import _run_face_enhance

    helper_roots: list[str] = []
    devices: list[object] = []

    class FakeFaceRestoreHelper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            helper_roots.append(str(kwargs["model_rootpath"]))
            print("helper construction output")

    class FakeGFPGANer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            import gfpgan.utils as gfpgan_utils

            devices.append(kwargs["device"])
            gfpgan_utils.FaceRestoreHelper(model_rootpath="gfpgan/weights")
            print("gfpganer construction output")

        def enhance(self, *args: object, **kwargs: object) -> tuple[None, None, str]:
            print("enhance progress output")
            return None, None, "enhanced"

    gfpgan_utils_module = _install_fake_gfpgan(
        monkeypatch,
        gfpganer_cls=FakeGFPGANer,
        helper_cls=FakeFaceRestoreHelper,
    )
    config = _face_enhance_config(tmp_path / "models")

    result = _run_face_enhance(config, object(), object(), torch_device="pixelup-device")

    assert result == "enhanced"
    assert helper_roots == [str(tmp_path / "models")]
    assert devices == ["pixelup-device"]
    assert gfpgan_utils_module.FaceRestoreHelper is FakeFaceRestoreHelper
    captured = capsys.readouterr().out
    # Construction runs inside the serialized lock window, so its output is
    # suppressed there. enhance() runs on the worker thread without redirecting
    # the shared streams, so its output is surfaced like the plain upscale path.
    assert "helper construction output" not in captured
    assert "gfpganer construction output" not in captured
    assert "enhance progress output" in captured


def test_face_enhance_propagates_enhance_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup.inference import _GFPGAN_HELPER_LOCK, _run_face_enhance

    class FakeFaceRestoreHelper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class FakeGFPGANer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            import gfpgan.utils as gfpgan_utils

            gfpgan_utils.FaceRestoreHelper(model_rootpath="gfpgan/weights")

        def enhance(self, *args: object, **kwargs: object) -> tuple[None, None, str]:
            raise RuntimeError("enhance failed")

    gfpgan_utils_module = _install_fake_gfpgan(
        monkeypatch,
        gfpganer_cls=FakeGFPGANer,
        helper_cls=FakeFaceRestoreHelper,
    )
    config = _face_enhance_config(tmp_path / "models")

    # An enhance() failure must surface rather than being swallowed by a stream
    # redirect; the construction lock and global must already be unwound by then.
    with pytest.raises(RuntimeError, match="enhance failed"):
        _run_face_enhance(config, object(), object(), torch_device="cpu")

    assert gfpgan_utils_module.FaceRestoreHelper is FakeFaceRestoreHelper
    assert not _GFPGAN_HELPER_LOCK.locked()


def test_face_enhance_holds_helper_lock_during_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup.inference import _GFPGAN_HELPER_LOCK, _run_face_enhance

    locked_during_construction: list[bool] = []

    class FakeFaceRestoreHelper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class FakeGFPGANer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            import gfpgan.utils as gfpgan_utils

            locked_during_construction.append(_GFPGAN_HELPER_LOCK.locked())
            gfpgan_utils.FaceRestoreHelper(model_rootpath="gfpgan/weights")

        def enhance(self, *args: object, **kwargs: object) -> tuple[None, None, str]:
            return None, None, "enhanced"

    gfpgan_utils_module = _install_fake_gfpgan(
        monkeypatch,
        gfpganer_cls=FakeGFPGANer,
        helper_cls=FakeFaceRestoreHelper,
    )
    config = _face_enhance_config(tmp_path / "models")

    result = _run_face_enhance(config, object(), object(), torch_device="cpu")

    assert result == "enhanced"
    assert locked_during_construction == [True]
    assert not _GFPGAN_HELPER_LOCK.locked()
    assert gfpgan_utils_module.FaceRestoreHelper is FakeFaceRestoreHelper


def test_face_enhance_restores_helper_global_when_construction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup.inference import _GFPGAN_HELPER_LOCK, _run_face_enhance

    class FakeFaceRestoreHelper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class FailingGFPGANer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("construction failed")

    gfpgan_utils_module = _install_fake_gfpgan(
        monkeypatch,
        gfpganer_cls=FailingGFPGANer,
        helper_cls=FakeFaceRestoreHelper,
    )
    config = _face_enhance_config(tmp_path / "models")

    with pytest.raises(RuntimeError, match="construction failed"):
        _run_face_enhance(config, object(), object(), torch_device="cpu")

    assert gfpgan_utils_module.FaceRestoreHelper is FakeFaceRestoreHelper
    assert not _GFPGAN_HELPER_LOCK.locked()


def test_face_enhance_construction_is_serialized_across_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pixelup.inference import _GFPGAN_HELPER_LOCK, _run_face_enhance

    thread_count = 6
    recorded: list[str] = []
    recorded_lock = threading.Lock()
    start = threading.Barrier(thread_count)

    class FakeFaceRestoreHelper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            with recorded_lock:
                recorded.append(str(kwargs["model_rootpath"]))

    class FakeGFPGANer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            import gfpgan.utils as gfpgan_utils

            # Widen the window between the global substitution and the read it
            # guards. Without the lock, overlapping threads would observe each
            # other's substitution and record the wrong models directory.
            time.sleep(0.01)
            gfpgan_utils.FaceRestoreHelper(model_rootpath="gfpgan/weights")

        def enhance(self, *args: object, **kwargs: object) -> tuple[None, None, str]:
            return None, None, "enhanced"

    _install_fake_gfpgan(
        monkeypatch,
        gfpganer_cls=FakeGFPGANer,
        helper_cls=FakeFaceRestoreHelper,
    )

    models_dirs = [tmp_path / f"models-{index}" for index in range(thread_count)]
    expected = sorted(str(path) for path in models_dirs)
    errors: list[BaseException] = []

    def worker(models_dir: Path) -> None:
        try:
            # Bounded wait: if the lock ever regressed into a deadlock, fail the
            # test fast instead of hanging the whole suite on the barrier.
            start.wait(timeout=30)
            _run_face_enhance(
                _face_enhance_config(models_dir),
                object(),
                object(),
                torch_device="cpu",
            )
        except BaseException as exc:  # surface thread failures to the test thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(path,)) for path in models_dirs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads), "worker thread hung"
    assert errors == []
    assert sorted(recorded) == expected
    assert not _GFPGAN_HELPER_LOCK.locked()
