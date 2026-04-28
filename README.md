# PixelUp

PixelUp is a Python CLI for upscaling one image file to one output file with
Real-ESRGAN.

This repository is currently at **phase 3** of the blueprint implementation:

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
- lazy Real-ESRGAN/GFPGAN inference bridge for environments with the ML stack installed
- stream-mode `start` and phase `progress` events for upscale runs
- output encoding for png, jpg, and webp
- temp-file output writes followed by `os.replace`

Still pending for later phases:

- pinned packaging for the heavy inference stack (`torch`, `torchvision`, `realesrgan`,
  `basicsr-fixed`, `gfpgan`, and OpenCV)
- full ICC conversion and metadata handling for Display-P3 and Adobe RGB outputs
- signal cleanup for in-flight temp files

## Usage

```console
pixelup INPUT OUTPUT --dry-run --models-dir ./models
pixelup models list --models-dir ./models
pixelup models download realesr-general-x4v3 --models-dir ./models
pixelup models check RealESRGAN_x4plus --download-missing --models-dir ./models
pixelup models dir --report single
pixelup --version --report single
```

The non-dry-run upscale path now calls the inference bridge. If the optional ML
packages are not installed, it returns `internal_error` with the missing
dependency in `details`.
