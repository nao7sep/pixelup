# PixelUp

PixelUp is a small PySide6 desktop app for upscaling local images with Real-ESRGAN. Drag images in, pick models and parameters, and a global queue processes them — each output is written beside its source with a sidecar JSON recording the settings, so a result can be reproduced. It runs on your own machine (macOS and Windows) with optional GPU/MPS/CUDA acceleration, and fetches the models it needs on first use. Pre-release (0.x), currently run from source.

## Features

- Drag-and-drop image list with a global queue and configurable concurrency
- Multiple Real-ESRGAN models; queue one image or all, against one model or all
- Optional face enhancement (GFPGAN), denoise, alpha handling, tiling, and output format/quality
- Per-image job summaries, a selected-image preview, and retry/cancel of jobs

## Requirements

- macOS or Windows
- Python with [uv](https://docs.astral.sh/uv/)
- Model weights (Real-ESRGAN, plus GFPGAN for face enhancement) — fetched on demand from their official GitHub releases and verified against a pinned SHA-256
- Optional: a GPU/MPS/CUDA backend for faster inference (CPU is the fallback)

## Getting started

Double-click the launcher for your platform (`scripts/run-dev.command` on macOS, `scripts/run-dev.ps1` on Windows), or run from source:

```console
uv sync --extra dev
uv run pixelup
```

You can also pass image paths directly: `uv run pixelup image.png another.jpg`.

## License

MIT © 2026 Yoshinao Inoguchi

## Contact

Yoshinao Inoguchi — nao7sep@gmail.com
