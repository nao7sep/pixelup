"""PixelUp package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pixelup")
except PackageNotFoundError:  # running from a source tree without an installed dist
    __version__ = "unknown"

