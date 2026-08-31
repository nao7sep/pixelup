from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pixelup.model_registry import ModelInfo, known_model
from pixelup.models import model_is_ready

GENERAL_MODEL = "realesr-general-x4v3"
GENERAL_DENOISE_MODEL = "realesr-general-wdn-x4v3"
DENOISE_NEUTRAL = 1.0

UPSCALE_MODELS = (
    GENERAL_MODEL,
    "RealESRGAN_x4plus",
    "RealESRNet_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3",
)


@dataclass(frozen=True, slots=True)
class ManagedModelBundle:
    key: str
    label: str
    purpose: str
    artifact_names: tuple[str, ...]


MANAGED_MODEL_BUNDLES = (
    *(ManagedModelBundle(name, name, "Upscaler", (name,)) for name in UPSCALE_MODELS),
    ManagedModelBundle(
        "general-denoise",
        "General x4v3 denoise",
        "Denoise support",
        (GENERAL_DENOISE_MODEL,),
    ),
    ManagedModelBundle(
        "face-enhancement",
        "Face enhancement",
        "Face restoration support",
        (
            "GFPGANv1.4",
            "facexlib-detection-retinaface-resnet50",
            "facexlib-parsing-parsenet",
        ),
    ),
)


def model_supports_denoise(model: str) -> bool:
    return model == GENERAL_MODEL


def effective_denoise_strength(model: str, denoise_strength: float) -> float:
    return denoise_strength if model_supports_denoise(model) else DENOISE_NEUTRAL


def required_artifact_names(
    models: Iterable[str],
    *,
    face_enhance: bool,
    denoise_strength: float,
) -> tuple[str, ...]:
    required: list[str] = []
    for model in models:
        required.append(model)
        if model_supports_denoise(model) and denoise_strength != DENOISE_NEUTRAL:
            required.append(GENERAL_DENOISE_MODEL)
    if face_enhance:
        required.extend(
            (
                "GFPGANv1.4",
                "facexlib-detection-retinaface-resnet50",
                "facexlib-parsing-parsenet",
            )
        )
    return tuple(dict.fromkeys(required))


def missing_artifact_names(models_dir: Path, artifact_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(name for name in artifact_names if not model_is_ready(models_dir, name))


def bundle_ready_count(models_dir: Path, bundle: ManagedModelBundle) -> int:
    return sum(model_is_ready(models_dir, name) for name in bundle.artifact_names)


def ready_artifact_count(models_dir: Path) -> tuple[int, int]:
    artifact_names = tuple(
        dict.fromkeys(name for bundle in MANAGED_MODEL_BUNDLES for name in bundle.artifact_names)
    )
    return (
        sum(model_is_ready(models_dir, name) for name in artifact_names),
        len(artifact_names),
    )


def artifact_info(name: str) -> ModelInfo:
    info = known_model(name)
    if info is None or info.url is None or info.expected_size is None:
        raise ValueError(f"Managed model artifact is incomplete: {name}")
    return info


def bundle_size_bytes(bundle: ManagedModelBundle) -> int:
    return sum(artifact_info(name).expected_size or 0 for name in bundle.artifact_names)


def artifact_size_bytes(artifact_names: Iterable[str]) -> int:
    return sum(artifact_info(name).expected_size or 0 for name in artifact_names)
