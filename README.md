# PixelUp

PixelUp is a simple PySide6 desktop app for upscaling local image files with
Real-ESRGAN models.

The app is intentionally small: open or drag image files into the window, enqueue
one or more model runs per tab, and let the global queue process them. Each tab is
one unique input path. Opening the same path focuses its existing tab.

## Features

- Tab per input image
- Wrapped tab chips with truncated labels
- Drag-and-drop image opening
- Per-tab queue
- "Enqueue all models" button
- Collapsed per-tab advanced options with restore-defaults support
- Original image preview with size label
- Help dialogs for advanced settings
- Configurable global job concurrency, defaulting to 1
- Output files written next to the source image
- Sidecar JSON metadata for every successful output
- Automatic model download when enabled
- Retry failed jobs
- Proper Settings and About dialogs
- Session log file per app launch
- Optional close-tab-on-success behavior, enabled by default

## Installation

```console
uv sync --extra dev
```

Run the app:

```console
uv run pixelup
```

You can also pass image paths directly:

```console
uv run pixelup image.png another-image.jpg
```

On macOS or Windows, the `scripts/` directory also contains double-clickable
helpers:

```text
scripts/run.command
scripts/update-packages.command
scripts/run.ps1
scripts/update-packages.ps1
```

## Output Files

By default, PixelUp writes outputs next to the input image:

```text
source-model-scale.png
```

For example:

```text
a-realesr-general-x4v3-4x.png
a-realesr-general-x4v3-4x.pixelup.json
```

Model names are lowercased and underscores become hyphens. If the output filename
already exists, PixelUp appends `-2`, `-3`, and so on before the extension.

The sidecar JSON is meant for replication. It stores the model, scale, safe
options, input fingerprint, dimensions, and output filename. It does not store
absolute paths, parent directories, usernames, model directories, or temp paths.

## Models

"Enqueue all models" uses these upscale models:

```text
realesr-general-x4v3
RealESRGAN_x4plus
RealESRNet_x4plus
RealESRGAN_x2plus
RealESRGAN_x4plus_anime_6B
realesr-animevideov3
```

`GFPGANv1.4` is kept as a face-enhancement helper model and is not part of "Try
all models" in this simple GUI.

## Config

Config lives at:

```text
~/.pixelup/config.json
```

Runtime model and temp files live under:

```text
~/.pixelup/models/
~/.pixelup/temp/
~/.pixelup/logs/
```

The settings dialog writes:

```json
{
  "auto_download": true,
  "close_tab_on_success": true,
  "device": "auto",
  "max_concurrent_jobs": 1,
  "output_format": "png",
  "quality": 95,
  "tile": 0
}
```

`PIXELUP_MODELS_DIR` and `PIXELUP_TEMP_DIR` can still override the runtime
directories.

Each app launch creates a session log file using a UTC filename:

```text
~/.pixelup/logs/yyyymmdd-hhmmss-utc.log
```

Log entries also use UTC, but in ISO-style timestamps, and include settings
changes, queue decisions, job progress, warnings, outputs, and sidecar paths.

## Development

Run the normal checks:

```console
uv run --extra dev ruff check .
uv run --extra dev pytest -q
uv lock --check
```

The regular test suite does not download model weights. To run the opt-in real
inference smoke test:

```console
PIXELUP_RUN_REAL_INFERENCE=1 \
PIXELUP_REAL_INFERENCE_MODELS_DIR=/path/to/models \
uv run --extra dev pytest -q tests/test_real_inference_smoke.py
```

That directory must contain `realesr-general-x4v3.pth`.
