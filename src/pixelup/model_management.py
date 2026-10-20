from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pixelup.i18n.message import Message
from pixelup.model_registry import (
    GENERAL_DENOISE_MODEL,
    GENERAL_MODEL,
    UPSCALE_MODELS,
    model_info,
)

DENOISE_NEUTRAL = 1.0


@dataclass(frozen=True, slots=True)
class ManagedModelBundle:
    key: str
    # A model's own name is a plain string, the same in every language; a name
    # PixelUp gives is interface text.
    label: Message | str
    purpose: Message
    artifact_names: tuple[str, ...]


MANAGED_MODEL_BUNDLES = (
    *(
        ManagedModelBundle(name, name, Message("managedModels.purposeUpscaler"), (name,))
        for name in UPSCALE_MODELS
    ),
    ManagedModelBundle(
        "general-denoise",
        Message("managedModels.generalDenoise"),
        Message("managedModels.purposeDenoise"),
        (GENERAL_DENOISE_MODEL,),
    ),
)

MANAGED_ARTIFACT_NAMES = tuple(
    dict.fromkeys(name for bundle in MANAGED_MODEL_BUNDLES for name in bundle.artifact_names)
)


def model_supports_denoise(model: str) -> bool:
    return model == GENERAL_MODEL


def effective_denoise_strength(model: str, denoise_strength: float) -> float:
    return denoise_strength if model_supports_denoise(model) else DENOISE_NEUTRAL


def required_artifact_names(
    models: Iterable[str],
    *,
    denoise_strength: float,
) -> tuple[str, ...]:
    required: list[str] = []
    for model in models:
        required.append(model)
        if model_supports_denoise(model) and denoise_strength != DENOISE_NEUTRAL:
            required.append(GENERAL_DENOISE_MODEL)
    return tuple(dict.fromkeys(required))


def bundle_size_bytes(bundle: ManagedModelBundle) -> int:
    return artifact_size_bytes(bundle.artifact_names)


def artifact_size_bytes(artifact_names: Iterable[str]) -> int:
    return sum(model_info(name).expected_size for name in artifact_names)
