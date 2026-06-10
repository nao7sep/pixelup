from __future__ import annotations

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
