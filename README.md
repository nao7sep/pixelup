# PixelUp

PixelUp is a simple PySide6 desktop app for upscaling local image files with
Real-ESRGAN models.

The app is intentionally small: open or drag image files into the window, select
an image from the list, select models and parameters, and let the global queue
process the work. Opening the same path focuses its existing image row.

## Features

- Image list for open input files
- Drag-and-drop image opening
- Per-image job summaries for done, failed, cancelled, and queued work
- Shared model selection
- Always-visible parameters with restore-defaults support
- Selected image preview
- Global queue for all images
- Queue selected image with selected or all models
- Queue all images with selected or all models
- Configurable global job concurrency, defaulting to 1
- Output files written next to the source image
- Sidecar JSON metadata for every successful output
- Automatic model download when enabled
- Retry failed jobs
- Cancel pending and running jobs
- Confirmation prompt before quitting with open images
- Settings and About dialogs
- Session log file per app launch

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
scripts/run.ps1
```

## Output Files

By default, PixelUp writes outputs next to the input image:

```text
source-model-scale.png
```

For example:

```text
a-realesr-general-x4v3-4x.png
a-realesr-general-x4v3-4x.json
```

Model names are lowercased and underscores become hyphens. If the output image
or its sidecar JSON already exists at the chosen stem, PixelUp appends `-2`,
`-3`, and so on before the extension so the image and sidecar stay paired.

The sidecar JSON is meant for replication. It stores the model, scale, safe
options, input fingerprint, dimensions, and output filename. It does not store
absolute paths, parent directories, usernames, model directories, or temp paths.

## Workflow

The image list shows each input image, its size, and a compact job summary. Job
summary items are omitted when their count is zero, and appear as comma-separated
values such as:

```text
2 done, 1 failed, 1 cancelled, 3 queued
```

The preview area displays the selected original image and scales it to fit the
available space while preserving its aspect ratio.

## Parameters

Scale chooses the final output size, not always the model's native scale. Most
bundled models are native `4x`; when you choose `2x`, Real-ESRGAN still uses the
model and then rescales to the requested final size. `RealESRGAN_x2plus` is the
only bundled model trained for native `2x` output.

Face enhancement runs GFPGAN after upscaling. It can improve recognizable faces,
but it may change facial details, so leave it off for images where identity or
texture must stay untouched.

Denoise controls noise removal for `realesr-general-x4v3` only. `0` keeps more
noise, `1` removes more noise, and `0.5` is the upstream Real-ESRGAN default.
Other models ignore this control.

Alpha mode controls how transparent pixels are upscaled when the input image has
an alpha channel. `Real-ESRGAN` uses the model for transparency. `Bicubic` uses a
standard image-scaling method and can be a safer fallback for edges or masks.

Output format chooses the file type for generated images. PNG is lossless. JPG
and WebP use the Quality value.

Quality applies only to JPG and WebP. Higher values preserve more detail and
make larger files. PNG ignores this setting.

Tile size controls memory use during inference. `0` processes the whole image
and is the best default when it fits in memory. If memory is limited, try larger
tiles first; `512` and `256` are good fallback candidates.

Device chooses the compute backend. `Auto` lets Real-ESRGAN choose the best
available backend. Use CPU only when GPU/MPS/CUDA backends fail or are not
available.

Strip metadata removes embedded metadata from outputs. If it is off, PixelUp
keeps or writes an ICC profile when possible.

Target profile converts output color to a known profile. `Default` preserves the
normal PixelUp behavior. Use `sRGB`, `Display P3`, or `Adobe RGB` when you need a
specific output color space.

## Models

"Queue all images with all models" and "Queue selected image with all models" use
these upscale models:

```text
realesr-general-x4v3
RealESRGAN_x4plus
RealESRNet_x4plus
RealESRGAN_x2plus
RealESRGAN_x4plus_anime_6B
realesr-animevideov3
```

`GFPGANv1.4` is kept as a face-enhancement helper model and is not part of "all
models" queue actions in this simple GUI.

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
  "device": "auto",
  "max_concurrent_jobs": 1,
  "output_format": "png",
  "quality": 95,
  "tile": 0
}
```

Quality applies to JPG and WebP outputs. PNG ignores quality. Tile size `0`
processes the whole image. If memory is limited, try larger tiles first; `512`
and `256` are good fallback candidates. Device `auto` lets Real-ESRGAN choose
the best available backend.

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
