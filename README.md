# PixelUp

PixelUp is a small PySide6 desktop app for upscaling local images with Real-ESRGAN. Drag images in, pick models and parameters, and a global queue processes them — each output is written beside its source with a sidecar JSON recording the settings, so a result can be reproduced. It runs on your own machine (macOS and Windows) with optional GPU/MPS/CUDA acceleration, and installs its pinned model files through Managed models when you choose them. 0.x.

## Download

Prebuilt builds for **macOS (Apple Silicon)** and **Windows (x64)** are on the [Releases](https://github.com/nao7sep/pixelup/releases/latest) page — a `.dmg` / `setup.exe` installer or a portable `.zip`. The builds are **self-contained** (no Python or uv install needed) and **unsigned**, so the OS warns the first time you open one:

- **macOS** — right-click the app and choose **Open** (or run `xattr -dr com.apple.quarantine /Applications/PixelUp.app`).
- **Windows** — on the SmartScreen prompt, click **More info → Run anyway**.

Managed models shows which pinned model files are ready and installs or repairs only what you choose; every download is verified against its pinned SHA-256 before it replaces the cache. A queue action that needs missing files discloses the exact download first and creates no jobs until installation succeeds. Allow roughly 3–67 MB of network and disk use for an upscaler model, with a small additional denoise companion for the general model. Face enhancement requires about 543 MB more. You can also place the `.pth` files in the models directory yourself.

## Features

- Drag-and-drop image list with a global queue and configurable concurrency
- Multiple Real-ESRGAN models; queue one image or all, against one model or all
- Optional face enhancement (GFPGAN), denoise, alpha handling, tiling, and output format/quality
- Per-image job summaries, a selected-image preview, and retry/cancel of jobs

## Requirements

- **macOS (Apple Silicon)** or **Windows (x64)** to run a prebuilt download — self-contained, nothing to install.
- **Python 3.12 with [uv](https://docs.astral.sh/uv/)** only if you run or build from source. Python 3.13 and newer are not supported because a required Real-ESRGAN dependency does not build on them.
- Model weights (Real-ESRGAN, plus GFPGAN and its facexlib detection/parsing weights for face enhancement) — installed explicitly from their official GitHub releases and verified against a pinned SHA-256.
- Optional: a GPU/MPS/CUDA backend for faster inference (CPU is the fallback).

## Run from source

Double-click the launcher for your platform (`scripts/run-dev.command` on macOS, `scripts/run-dev.ps1` on Windows), or:

```console
uv sync --extra dev
uv run pixelup
```

You can also pass image paths directly: `uv run pixelup image.png another.jpg`.

## License

MIT © 2026 Yoshinao Inoguchi

## Contact

Yoshinao Inoguchi — yoshinao@inoguchi.com — <https://inoguchi.com>
