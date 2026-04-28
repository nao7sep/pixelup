# PixelUp

PixelUp is a Python CLI for upscaling one image file to one output file with
Real-ESRGAN.

This repository is currently at **phase 9** of the blueprint implementation:

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
- signal-aware cleanup for in-flight output temp files
- ICC conversion for sRGB plus generated Display-P3/Adobe RGB profiles
- EXIF/XMP preservation unless `--strip-metadata` is set
- pinned optional inference extra for the heavy ML stack
- pinned inference-stack preflight diagnostics before model execution
- opt-in real small-image inference smoke coverage
- human-mode warnings for forced format/extension and model scale mismatches
- model verify/remove coverage for unlisted companion model weights

## Installation

For CLI validation, dry runs, model management, and image encoding:

```console
pip install -e .
```

For actual Real-ESRGAN/GFPGAN inference:

```console
pip install -e '.[inference]'
```

With uv:

```console
uv sync --extra inference
```

The `inference` extra pins `torch`, `torchvision`, `realesrgan`, `basicsr-fixed`,
`gfpgan`, `opencv-python`, and `numpy` to exact versions. `basicsr-fixed` is kept
as the primary BasicSR fork for the `torchvision.transforms.functional_tensor`
compatibility issue; PixelUp also installs a runtime fallback alias before
importing Real-ESRGAN/GFPGAN to handle environments where upstream dependency
metadata pulls `basicsr`.

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
packages are not installed or do not match the pinned versions, it returns
`internal_error` with dependency details.

## Real inference smoke test

The regular test suite does not download ML packages or model weights. To verify
the real inference path after installing the optional stack and downloading the
general v3 model:

```console
PIXELUP_RUN_REAL_INFERENCE=1 \
PIXELUP_REAL_INFERENCE_MODELS_DIR=/path/to/models \
uv run --extra dev --extra inference pytest -q tests/test_real_inference_smoke.py
```
