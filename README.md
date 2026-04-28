# PixelUp

PixelUp is a Python CLI for upscaling one image file to one output file with
Real-ESRGAN.

This repository is currently at **phase 2** of the blueprint implementation:

- package metadata and `pixelup` console entry point
- Typer-based CLI surface for the root command and `models` namespace
- public error-code and exit-code mapping
- `human`, `single`, and `stream` result reporting primitives
- model registry metadata and model presence checks
- output placeholder resolution, including optional-placeholder separator collapse
- input/output/model validation for `--dry-run`
- official Real-ESRGAN/GFPGAN model download URLs and expected sizes
- per-model download locks under `<models-dir>/.locks/`
- model download, check, remove, and verify command behavior
- stream-mode `waiting` and `download` events for model setup

Still pending for later phases:

- Real-ESRGAN and GFPGAN inference
- ICC conversion, metadata handling, and final image encoding
- atomic temp-file writes and signal cleanup

## Usage

```console
pixelup INPUT OUTPUT --dry-run --models-dir ./models
pixelup models list --models-dir ./models
pixelup models download realesr-general-x4v3 --models-dir ./models
pixelup models check RealESRGAN_x4plus --download-missing --models-dir ./models
pixelup models dir --report single
pixelup --version --report single
```

The non-dry-run upscale path intentionally returns `internal_error` until the
inference phase is implemented, after any requested `--auto-download` setup has
completed.
