import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from pixelup.errors import PixelupError
from pixelup.inference import InferenceConfig, _build_network
from pixelup.model_registry import GENERAL_MODEL, model_info
from pixelup.upscale import count_tiles


def test_count_tiles_matches_input_grid() -> None:
    assert count_tiles((400, 300), 256) == 4
    assert count_tiles((1, 1), 512) == 1


def test_build_network_uses_srvgg_architecture() -> None:
    from pixelup.realesrgan_models import SRVGGNetCompact

    result = _build_network(model_info(GENERAL_MODEL).architecture)

    assert isinstance(result, SRVGGNetCompact)


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError(
            "MPS backend out of memory (MPS allocated: 1024.00 MiB, other allocations: "
            "480.00 KiB, max allowed: 909.31 MiB). Tried to allocate 256.00 MiB on shared pool."
        ),
        RuntimeError("Invalid buffer size: 52.00 GB"),
        MemoryError(),
    ],
)
def test_running_out_of_memory_is_reported_with_what_to_change(monkeypatch, failure) -> None:
    from pixelup import inference
    from pixelup.errors import ErrorCode
    from pixelup.i18n.message import Message

    def exhausted(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(inference, "_run_inference", exhausted)
    with pytest.raises(PixelupError) as caught:
        inference.run_inference(SimpleNamespace(device="cpu"), None)  # type: ignore[arg-type]
    assert caught.value.code == ErrorCode.OUT_OF_MEMORY
    assert caught.value.hint == Message("error.hintOutOfMemory")


def test_another_runtime_failure_stays_an_inference_failure(monkeypatch) -> None:
    from pixelup import inference
    from pixelup.errors import ErrorCode

    def broken(*_args, **_kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(inference, "_run_inference", broken)
    with pytest.raises(PixelupError) as caught:
        inference.run_inference(SimpleNamespace(device="cpu"), None)  # type: ignore[arg-type]
    assert caught.value.code == ErrorCode.INTERNAL_ERROR


@pytest.mark.parametrize("stage", ["construct", "cancel", "enhance", "success"])
def test_cache_cleanup_follows_tensor_lifetime(monkeypatch, tmp_path: Path, stage: str) -> None:
    import torch

    from pixelup import inference
    from pixelup.errors import ErrorCode

    config = InferenceConfig(
        models_dir=tmp_path, model=GENERAL_MODEL, scale=4, tile=2, tile_pad=0,
        pre_pad=0, fp32=True, denoise_strength=1, alpha_mode="bicubic",
        gpu_id=None, device="cpu",
    )
    references = []
    cleanup_devices = []

    class Upsampler:
        def __init__(self):
            self.tensor = torch.zeros(1)
            references.append(weakref.ref(self))
            references.append(weakref.ref(self.tensor))
            if stage == "construct":
                raise RuntimeError("constructor diagnostic")

        def enhance(self, *_args, **_kwargs):
            tile = torch.zeros(1)
            references.append(weakref.ref(tile))
            if stage == "enhance":
                raise RuntimeError("enhancement diagnostic")
            return "output", "RGB"

    checks = 0

    def should_cancel():
        nonlocal checks
        checks += 1
        return stage == "cancel" and checks == 2

    def release(device):
        assert references
        assert all(reference() is None for reference in references)
        cleanup_devices.append(device)

    monkeypatch.setattr(inference, "_create_upsampler", lambda *_a, **_kw: Upsampler())
    monkeypatch.setattr(inference, "_release_device_cache", release)
    if stage == "success":
        assert inference.run_inference(config, None, should_cancel=should_cancel) == "output"
    else:
        with pytest.raises(PixelupError) as caught:
            inference.run_inference(config, None, should_cancel=should_cancel)
        if stage == "cancel":
            assert caught.value.code == ErrorCode.JOB_CANCELLED
        else:
            cause = caught.value.__cause__
            assert isinstance(cause, RuntimeError)
            assert "diagnostic" in str(cause)
            assert cause.__traceback__ is not None
            assert caught.value.details["reason"] == str(cause)
    assert cleanup_devices == ["cpu"]

