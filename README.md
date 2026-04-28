# PixelUp

PixelUp is a Python CLI for upscaling one image file to one output file with
Real-ESRGAN.

This repository is currently at **phase 1** of the blueprint implementation:

- package metadata and `pixelup` console entry point
- Typer-based CLI surface for the root command and `models` namespace
- public error-code and exit-code mapping
- `human`, `single`, and `stream` result reporting primitives
- model registry metadata and model presence checks
- output placeholder resolution, including optional-placeholder separator collapse
- input/output/model validation for `--dry-run`

Still pending for later phases:

- model download URLs, file locks, checksums, and concurrent download progress
- Real-ESRGAN and GFPGAN inference
- ICC conversion, metadata handling, and final image encoding
- atomic temp-file writes and signal cleanup

## Usage

```console
pixelup INPUT OUTPUT --dry-run --models-dir ./models
pixelup models list --models-dir ./models
pixelup models dir --report single
pixelup --version --report single
```

The non-dry-run upscale path intentionally returns `internal_error` until the
inference phase is implemented.
