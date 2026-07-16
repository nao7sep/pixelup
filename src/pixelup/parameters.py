from __future__ import annotations

# The valid domain of every image-processing parameter, gathered here and nowhere
# else. Three layers read this module and must agree: the Parameters panel builds
# its controls from it, the config loader clamps/coerces incoming values against
# it, and upscale.validate_options rejects out-of-domain values against it — so a
# value can never be offered by one and refused by another.
#
# This is a LEAF: it imports nothing from pixelup, exactly like devices.py. That is
# the point. These constants used to live in jobs.py, which imports upscale.py —
# so upscale.py could not import them back without a cycle and re-enumerated five
# of these domains as literals instead. The panel could then offer a value the
# validator rejected at runtime. A leaf has no such problem; keep it dependency-free.

MIN_QUALITY = 0
MAX_QUALITY = 100
MIN_DENOISE_STRENGTH = 0.0
MAX_DENOISE_STRENGTH = 1.0
DENOISE_STRENGTH_STEP = 0.1

# Ordered (label, value) pairs for the enumerated parameters, in the shape
# DEVICE_CHOICES already established: labels are for UI display, values are what a
# job carries and what config persistence stores. Keep these the only place any of
# these sets is enumerated.
SCALE_CHOICES: tuple[tuple[str, int], ...] = (
    ("2x", 2),
    ("4x", 4),
)
SCALE_VALUES: tuple[int, ...] = tuple(value for _label, value in SCALE_CHOICES)
ALPHA_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Real-ESRGAN", "realesrgan"),
    ("Bicubic", "bicubic"),
)
ALPHA_MODE_VALUES: tuple[str, ...] = tuple(value for _label, value in ALPHA_MODE_CHOICES)
TARGET_PROFILE_CHOICES: tuple[tuple[str, str | None], ...] = (
    ("Default", None),
    ("sRGB", "srgb"),
    ("Display P3", "p3"),
    ("Adobe RGB", "adobergb"),
)
TARGET_PROFILE_VALUES: tuple[str | None, ...] = tuple(
    value for _label, value in TARGET_PROFILE_CHOICES
)

# Tile size is a choice, not a range. Peak memory scales with the tile's AREA, so
# only doublings are meaningful — each rung below is 4x the one before it, and the
# in-between values a stepped spin box used to produce (768, 2816, ...) were
# arithmetic artefacts nobody would choose on purpose.
#
# Ordered by ascending peak memory, which is why "Whole image" comes LAST rather
# than leading as a 0. 0 is not the smallest tile: it disables tiling, making the
# tile the whole image — the largest possible, and the one that can exhaust GPU/MPS
# memory and hard-crash on big inputs. A spin box put it directly below 128 and
# called it "less"; here it reads as the end of the scale it actually is.
#
# The top is 2048 because tiling self-cancels above it: at 4096 a typical photo is a
# single tile, which already *is* "Whole image". The bottom is 128 because tile_pad
# overlap (~8% per edge there) and tile count both worsen sharply below it.
TILE_CHOICES: tuple[tuple[str, int], ...] = (
    ("128", 128),
    ("256", 256),
    ("512", 512),
    ("1024", 1024),
    ("2048", 2048),
    ("Whole image", 0),
)
TILE_VALUES: tuple[int, ...] = tuple(value for _label, value in TILE_CHOICES)

# 256 keeps peak memory low enough for modest GPUs and smaller-memory machines while
# giving output effectively identical to larger tiles. 128 sits one rung below for
# hardware where even 256 will not fit — reachable in one click, which the old
# 256-step spin box could not do (its only move below the default was straight to
# the whole-image pass).
DEFAULT_TILE = 256

# 4x is the scale PixelUp has always opened on, and the one every bundled model is
# trained for (the x2 model is the lone exception, and plan_warnings covers the
# mismatch), so it stays the built-in. 2x remains selectable in the Parameters panel.
DEFAULT_SCALE = 4
