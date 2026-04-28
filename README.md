# PixelUp

PixelUp is a command-line tool for upscaling one image file to one output file
with Real-ESRGAN. It is designed for shell scripts, batch files, desktop apps,
and other callers that want a predictable subprocess contract.

PixelUp intentionally does not process folders, write manifests, or manage
worker pools. Run one process per image and let the caller handle batching.

## Features

- Single-image upscale command: `pixelup INPUT OUTPUT [OPTIONS]`
- Machine-readable `single` and `stream` report modes
- Human report mode with warnings, progress messages, and optional verbose timing
- Output filename placeholders such as `{stem}`, `{model_short}`, `{width}`, and `{datetime}`
- PNG, JPEG, and WebP output
- HEIC/HEIF input support through `pillow-heif`
- EXIF/XMP preservation unless `--strip-metadata` is set
- ICC conversion to sRGB, Display P3, or Adobe RGB
- Atomic temp-file writes followed by `os.replace`
- SIGINT/SIGTERM cleanup for in-flight temp output files
- Model download, check, verify, remove, and directory commands
- Cross-process model download locks under `<models-dir>/.locks/`
- Pinned Real-ESRGAN/GFPGAN inference stack bundled with the CLI

## Installation

```console
pip install -e .
```

With uv:

```console
uv sync
```

For development:

```console
uv sync --extra dev
```

The Real-ESRGAN inference stack (`torch`, `torchvision`, `realesrgan`,
`basicsr-fixed`, `gfpgan`, `opencv-python`, `numpy`) is pinned exactly and
installed as part of the base package, so a single install yields a fully
working upscaler.

## Quick Start

Download a small general model:

```console
pixelup models download realesr-general-x4v3 --models-dir ./models
```

Validate an upscale plan without running inference:

```console
pixelup input.png output.png \
  --model realesr-general-x4v3 \
  --dry-run \
  --models-dir ./models
```

Run an upscale:

```console
pixelup input.png output.png \
  --model realesr-general-x4v3 \
  --models-dir ./models \
  --report human
```

Use stream mode from an application:

```console
pixelup input.png output.png --report stream
```

## Subprocess Contract

PixelUp is safe to call from scripts and applications as a one-image,
one-output subprocess. Use `--report single` when the caller only needs the
final result, or `--report stream` when the caller needs live progress.
In machine-readable modes, expected successes and failures are JSON on stdout;
use the process exit code for coarse control flow and the JSON `code` field for
specific failure handling. `stream` mode ends with a final `result` event whose
payload matches the `single` success or failure shape.

Recommended caller flow:

```console
pixelup models check realesr-general-x4v3 \
  --download-missing \
  --models-dir ./models \
  --report single

pixelup input.png output.png \
  --model realesr-general-x4v3 \
  --dry-run \
  --models-dir ./models \
  --report single

pixelup input.png output.png \
  --model realesr-general-x4v3 \
  --models-dir ./models \
  --report stream
```

Dry-run validates the same preconditions as a real upscale at the moment it is
called: input readability, output path validity, output overwrite rules, model
presence, option values, and forced device availability. Dry-run never performs
inference and never downloads models, even when `--auto-download` is supplied.
If a required model is missing, dry-run fails with `model_not_found`; callers
that want setup to happen automatically should run `models check
--download-missing` before dry-run or before the real upscale.

In `single` report mode, a successful dry-run writes one JSON object to stdout:

```json
{
  "ok": true,
  "input": "/abs/path/input.png",
  "output": "/abs/path/output.png",
  "model": "realesr-general-x4v3",
  "scale": 4,
  "input_size": [800, 600],
  "output_size": [3200, 2400],
  "format": "png",
  "device": "mps",
  "dry_run": true,
  "models_dir": "/abs/path/models",
  "temp_dir": "/abs/path/temp",
  "message": "Dry run plan is valid.",
  "models_present": {
    "realesr-general-x4v3": true
  }
}
```

On a real successful upscale, `single` report mode writes:

```json
{
  "ok": true,
  "input": "/abs/path/input.png",
  "output": "/abs/path/output.png",
  "model": "realesr-general-x4v3",
  "scale": 4,
  "input_size": [800, 600],
  "output_size": [3200, 2400],
  "format": "png",
  "ms": 4200
}
```

## Core Usage

```console
pixelup INPUT OUTPUT [OPTIONS]
pixelup models list [OPTIONS]
pixelup models check [MODEL...] [--download-missing] [OPTIONS]
pixelup models download MODEL... [OPTIONS]
pixelup models remove MODEL... [--all] [OPTIONS]
pixelup models verify [OPTIONS]
pixelup models dir [OPTIONS]
pixelup --version [--report auto|human|single|stream]
```

Only `-h` is available as a short option, as an alias for `--help`.

## Important Options

Processing:

```console
--model NAME
--scale 2|4
--tile INT
--tile-pad INT
--pre-pad INT
--fp32
--face-enhance
--denoise-strength FLOAT
--alpha-mode realesrgan|bicubic
--device auto|mps|cuda|cpu
--gpu-id INT
```

Output:

```console
--format png|jpg|webp
--quality INT
--background COLOR
--strip-metadata
--target-profile srgb|p3|adobergb
--overwrite
```

Runtime:

```console
--auto-download
--models-dir PATH
--temp-dir PATH
--download-timeout SECONDS
--lock-timeout SECONDS
--dry-run
--report auto|human|single|stream
--quiet
--verbose
```

`--quiet` and `--verbose` are mutually exclusive.

## Output Paths

`OUTPUT` can be a file path, a directory, or a placeholder template.

If `OUTPUT` is an existing directory, PixelUp writes this default pattern:

```text
{stem}__{model_short}_{scale}x__{width}px.{ext}
```

Supported placeholders:

```text
{stem}
{ext}
{model}
{model_short}
{scale}
{width}
{height}
{denoise}
{face}
{date}
{time}
{datetime}
```

Empty optional placeholders collapse adjacent `_` or `-` separators. For
example, `{stem}__{denoise}__{model_short}.{ext}` becomes
`input__general.png` when denoise is not used.

Directory outputs and `{ext}` templates default to PNG when `--format` is not
provided.

## Report Modes

`auto` selects `human` when stdout is a terminal and `single` otherwise.

`human` writes interactive output to stderr. It shows warnings for output
extension/format mismatches and model/native scale mismatches. With `--verbose`,
it also prints start context and elapsed stage timing.

`single` writes one JSON object to stdout and is silent during processing.

`stream` writes JSON lines to stdout. Events include:

```text
start
progress
waiting
download
result
```

The final stream line is always a `result` object. Instant commands such as
`models dir` and `--version` intentionally produce the same single JSON object
in `single` and `stream` modes.

When `--tile` is greater than zero, PixelUp emits a best-effort `progress`
event with `phase: "upscale"` and `tile`/`tiles` fields after each tile
completes. This relies on the internal tile loop of the pinned `realesrgan`
release; if a future version diverges, PixelUp silently falls back to a single
per-phase `progress` event without affecting output correctness. With
`--tile 0` (no tiling) only the per-phase `progress` event is emitted.

## Models

Known public model names:

```text
RealESRGAN_x4plus
RealESRNet_x4plus
RealESRGAN_x2plus
RealESRGAN_x4plus_anime_6B
realesr-animevideov3
realesr-general-x4v3
GFPGANv1.4
```

The denoise companion model `realesr-general-wdn-x4v3` is recognized but hidden
from the default list. It is used when `--model realesr-general-x4v3` and
`--denoise-strength` is not `1.0`.

Model commands:

```console
pixelup models list --models-dir ./models
pixelup models check RealESRGAN_x4plus --models-dir ./models
pixelup models check realesr-general-x4v3 --download-missing --models-dir ./models
pixelup models download realesr-general-x4v3 --models-dir ./models
pixelup models verify --models-dir ./models
pixelup models remove realesr-general-x4v3 --models-dir ./models
pixelup models remove --all --models-dir ./models
pixelup models dir --report single
```

`models check --download-missing` may be run without model names; in that case
it downloads every missing public known model.

`models verify` validates present recognized model files by expected size and,
when available, checksum. It fails fast on the first corrupt model and reports
that mismatch as `model_corrupt`.

`models remove --all` removes all recognized model files, including the hidden
denoise companion model. Unknown files in the models directory are left alone.

## Runtime Directories

Default persistent state lives under `~/.pixelup/`:

```text
~/.pixelup/
  models/
  temp/
```

Resolution order:

```text
models: --models-dir, PIXELUP_MODELS_DIR, ~/.pixelup/models
temp:   --temp-dir,   PIXELUP_TEMP_DIR,   ~/.pixelup/temp
```

PixelUp creates these runtime directories when needed. It does not clean stale
model files or old temp directories for you.

## Inference Stack

PixelUp installs the Real-ESRGAN inference stack as part of the base package.
The pinned versions are:

```text
numpy==2.4.4
torch==2.11.0
torchvision==0.26.0
opencv-python==4.13.0.92
realesrgan==0.3.0
basicsr-fixed==1.4.2
gfpgan==1.3.8
```

`basicsr-fixed` is the compatibility fix for the removed
`torchvision.transforms.functional_tensor` module that upstream `basicsr` 1.4.2
imports. PixelUp depends on the fork directly and does not patch
`torchvision` at runtime.

## Color And Metadata

Default behavior preserves the source ICC profile, EXIF, and XMP metadata.

`--strip-metadata` drops EXIF and XMP. If the source image has an ICC profile,
PixelUp converts the image to sRGB before dropping that profile.

`--target-profile` converts and embeds one of:

```text
srgb
p3
adobergb
```

PixelUp generates Display P3 and Adobe RGB profiles from color-space constants,
so it does not depend on platform ColorSync profile files.

## Error Handling

Machine-readable failures use:

```json
{
  "ok": false,
  "code": "model_not_found",
  "message": "Human-readable explanation.",
  "hint": "Suggested next action.",
  "details": {}
}
```

Public error codes:

```text
input_not_found
input_unreadable
input_invalid_format
output_exists
output_unwritable
output_dir_missing
model_not_found
model_download_failed
model_corrupt
auto_download_disabled
denoise_strength_unsupported
face_enhance_unavailable
out_of_memory
invalid_argument
internal_error
```

Exit codes:

```text
0 success
1 internal error or uncaught exception
2 invalid argument
3 input error
4 output error
5 model error
6 out of memory
7 cancelled
```

## Development

Run the normal checks:

```console
uv run --extra dev ruff check .
uv run --extra dev pytest -q
uv lock --check
```

The regular test suite does not download ML packages or model weights. To run
the opt-in real inference smoke test:

```console
PIXELUP_RUN_REAL_INFERENCE=1 \
PIXELUP_REAL_INFERENCE_MODELS_DIR=/path/to/models \
uv run --extra dev pytest -q tests/test_real_inference_smoke.py
```

That directory must contain `realesr-general-x4v3.pth`.
