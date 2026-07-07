from __future__ import annotations

from typing import Any

from pixelup.errors import ErrorCode, PixelupError

# Single source of truth for the compute backends PixelUp understands. Ordered
# (label, value) pairs: labels are for UI display, values are what the inference
# engine validates and what config persistence stores. Keep this the only place
# the device set is enumerated.
DEVICE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Auto", "auto"),
    ("MPS", "mps"),
    ("CUDA", "cuda"),
    ("CPU", "cpu"),
)

DEVICE_VALUES: tuple[str, ...] = tuple(value for _label, value in DEVICE_CHOICES)

DEFAULT_DEVICE = "auto"

_INVALID_DEVICE_MESSAGE = "Device must be one of Auto, MPS, CUDA, or CPU."


def resolve_device(device: str, gpu_id: int | None = None) -> str:
    """Resolve a requested backend to the concrete one inference will run on.

    This is the single place that turns ``"auto"`` into ``"mps"`` / ``"cuda"`` /
    ``"cpu"`` and validates that an explicitly requested backend is actually
    available. ``"auto"`` prefers MPS, then CUDA, then CPU. The returned value is
    always one concrete backend (never ``"auto"``); pass it to
    :func:`to_torch_device` to obtain the matching ``torch.device``.
    """
    if device not in DEVICE_VALUES:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, _INVALID_DEVICE_MESSAGE)
    if device == "cpu":
        return "cpu"

    import torch

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_available = torch.cuda.is_available()
    if device == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"
    if device == "mps":
        if not mps_available:
            raise PixelupError(ErrorCode.INVALID_ARGUMENT, "MPS is not available.")
        return "mps"
    # device == "cuda"
    if not cuda_available:
        raise PixelupError(ErrorCode.INVALID_ARGUMENT, "CUDA is not available.")
    if gpu_id is not None and gpu_id >= torch.cuda.device_count():
        raise PixelupError(
            ErrorCode.INVALID_ARGUMENT,
            "CUDA GPU index is not available.",
            details={"gpu_id": gpu_id, "device_count": torch.cuda.device_count()},
        )
    return "cuda"


def to_torch_device(concrete_device: str, gpu_id: int | None = None) -> Any:
    """Build the ``torch.device`` for an already-resolved concrete backend.

    Expects the output of :func:`resolve_device` — it does not re-resolve
    ``"auto"`` or re-check availability, so the backend decision lives in exactly
    one place. ``gpu_id`` selects a specific CUDA device when given.
    """
    import torch

    if concrete_device == "cpu":
        return torch.device("cpu")
    if concrete_device == "mps":
        return torch.device("mps")
    if concrete_device == "cuda":
        return torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    raise PixelupError(ErrorCode.INVALID_ARGUMENT, _INVALID_DEVICE_MESSAGE)
