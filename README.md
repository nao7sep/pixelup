# PixelUp

PixelUp is a simple PySide6 desktop app for upscaling local images with Real-ESRGAN models. Open or drag images in, pick models and parameters, and let a global queue process the work — outputs are written next to each source with a sidecar JSON for replication. It's a small, focused tool for batch upscaling on your own machine (macOS and Windows), with optional GPU/MPS/CUDA acceleration and automatic model download.

## Features

- Drag-and-drop image list with a global processing queue and configurable concurrency
- Multiple bundled Real-ESRGAN models; queue a selected image or all of them, with selected or all models
- Optional face enhancement (GFPGAN), denoise, alpha handling, tiling, and output format/quality
- Outputs written next to the source, with a sidecar JSON (model, scale, options, input fingerprint) for replication
- Automatic model download, plus retry and cancellation of jobs
- Per-image job summaries and a selected-image preview

## Requirements

- macOS or Windows
- Python with [uv](https://docs.astral.sh/uv/)
- Real-ESRGAN model weights (and GFPGAN for face enhancement) — fetched on demand from their official GitHub releases and verified against a pinned SHA-256 before use
- Optional: a GPU/MPS/CUDA backend for faster inference (CPU works as a fallback)

## Getting started

Double-click the launcher for your platform (`scripts/run-dev.command` on macOS, `scripts/run-dev.ps1` on Windows), or run from source:

```console
uv sync --extra dev
uv run pixelup
```

You can also pass image paths directly: `uv run pixelup image.png another.jpg`.

## Notes on parameters

- **Face enhancement** can improve recognizable faces but may alter facial details — leave it off when identity or texture must stay untouched.
- **Device** — `Auto` picks the best available backend; use CPU only when GPU/MPS/CUDA fail or are unavailable.
- **Tile size** controls memory use; `0` processes the whole image, and smaller tiles (`512`, `256`) help when memory is limited.

## License

MIT © 2026 Yoshinao Inoguchi

## Contact

Yoshinao Inoguchi — nao7sep@gmail.com
