# PixelUp

Upscale local images in batches with Real-ESRGAN, and run several models over the same images to compare. Drag images in, pick models and parameters, and a global queue processes them — each output is written beside its source with a sidecar JSON recording the settings, so a result can be reproduced. It runs on your own machine (macOS and Windows) with optional GPU/MPS/CUDA acceleration, and installs its pinned model files through Managed models when you choose them. 0.x.

## Download

Prebuilt builds for **macOS (Apple Silicon)** and **Windows (x64)** are on the [Releases](https://github.com/nao7sep/pixelup/releases/latest) page — a `.dmg` / `setup.exe` installer or a portable `.zip`. The builds are **self-contained** (no Python or uv install needed) and **unsigned**, so the OS warns the first time you open one:

- **macOS** — right-click the app and choose **Open** (or run `xattr -dr com.apple.quarantine /Applications/PixelUp.app`).
- **Windows** — on the SmartScreen prompt, click **More info → Run anyway**.

Before your first model download, choose its storage folder in **Managed models**, or leave the field blank to use the suggested default. You can change it later in the same dialog. A change leaves existing files in place; it neither moves nor deletes them. Jobs already in the queue keep using their original folder, and folder changes wait for model installations to finish.

Managed models shows which pinned model files are ready and installs or repairs only what you choose; every download is verified against its pinned SHA-256 before it replaces the cache. A queue action that needs missing files discloses the exact download first and creates no jobs until installation succeeds. You can close the download dialog and queue more images while installation continues; each request keeps the parameters you chose for it. Allow roughly 3–67 MB of network and disk use for an upscaler model, with a small additional denoise companion for the general model. You can also place the `.pth` files in the models directory yourself.

## Features

- Drag-and-drop image list with a global queue and configurable concurrency
- Multiple Real-ESRGAN models; queue one image or all, against one model or all
- Optional denoise, alpha handling, tiling, and output format/quality
- Per-image job summaries, a selected-image preview, and retry/cancel of jobs
- Remembers the main window position and size between launches
- Speaks English, Japanese, Chinese, Korean, Spanish, Portuguese, French, German, Italian, and Russian, following the computer's language unless you choose one in Settings

## Settings and results

Parameter edits save automatically. Settings changes save when you press OK; cancelling asks before discarding edits. If a saved setting cannot be used, PixelUp names it at startup and uses its default while preserving the saved value until you change that setting. Close PixelUp before editing its settings file by hand.

One PixelUp window can use a data folder at a time. Opening another copy with the same data folder asks you to close the existing window first.

Preparing a queue runs in the background, so a slow drive does not block the window. Cancel queue also cancels requests still being prepared. If images cannot be queued, the message points to Records for details before you retry.

Outputs never replace existing files. Each completed image has a JSON sidecar describing its processing settings and the source bytes used for that result. Cancelling a job removes its unfinished image and sidecar; a completed result stays. A forced quit can leave an image without its sidecar. Use Records to inspect warnings and errors from the session.

Photo orientation is applied to the pixels, including when metadata is removed. Transparent inputs keep their alpha in PNG and WebP; JPEG uses a white background. Unsigned 16-bit grayscale inputs are mapped across the model's 8-bit working range.

## Requirements

- **macOS (Apple Silicon)** or **Windows (x64)** to run a prebuilt download — self-contained, nothing to install.
- **Python 3.14 with [uv](https://docs.astral.sh/uv/)** only if you run or build from source. Python 3.15 is not supported yet because PyTorch publishes no builds for it.
- Real-ESRGAN model weights — installed explicitly from their official GitHub releases and verified against a pinned SHA-256.
- Optional: a GPU/MPS/CUDA backend for faster inference (CPU is the fallback).

## Run from source

Double-click the launcher for your platform (`scripts/run-dev.command` on macOS, `scripts/run-dev.ps1` on Windows), or:

```console
uv sync --extra dev
uv run pixelup
```

You can also pass image paths directly: `uv run pixelup image.png another.jpg`.

## Tests

`uv run ruff check .` and `uv run pytest` are what to run after a change; the ordinary suite does not download models or run real inference. `uv run pytest -m "heavy or not heavy"` is the full gate: it also runs the heavy tests, which install every managed Real-ESRGAN model through PixelUp's own model manager, about 230 MB on the first run and kept in `.pytest_cache` afterwards, and upscale a photo from the shared test-fixture corpus with each model. The corpus lives in the company repository, which must be checked out beside this one.

## License

[GNU GPL v3 or later](LICENSE) © 2026 Yoshinao Inoguchi

## Contact

Yoshinao Inoguchi — yoshinao@inoguchi.com — <https://inoguchi.com>
