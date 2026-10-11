from __future__ import annotations

from pixelup.model_management import (
    GENERAL_DENOISE_MODEL,
    MANAGED_MODEL_BUNDLES,
    UPSCALE_MODELS,
    artifact_info,
    bundle_size_bytes,
    required_artifact_names,
)


def test_managed_bundles_cover_every_runtime_artifact_once() -> None:
    artifact_names = [name for bundle in MANAGED_MODEL_BUNDLES for name in bundle.artifact_names]

    assert len(artifact_names) == len(set(artifact_names))
    assert set(UPSCALE_MODELS).issubset(artifact_names)
    assert GENERAL_DENOISE_MODEL in artifact_names
    for name in artifact_names:
        info = artifact_info(name)
        assert info.url is not None
        assert info.expected_size is not None
        assert info.checksum_sha256 is not None


def test_required_artifacts_are_ordered_and_deduplicated() -> None:
    required = required_artifact_names(
        ("realesr-general-x4v3", "RealESRGAN_x4plus", "realesr-general-x4v3"),
        denoise_strength=0.5,
    )

    assert required == (
        "realesr-general-x4v3",
        GENERAL_DENOISE_MODEL,
        "RealESRGAN_x4plus",
    )


def test_neutral_denoise_omits_the_general_companion() -> None:
    required = required_artifact_names(
        ("realesr-general-x4v3", "RealESRGAN_x4plus"),
        denoise_strength=1.0,
    )

    assert required == ("realesr-general-x4v3", "RealESRGAN_x4plus")


def test_bundle_size_is_the_sum_of_its_pinned_artifacts() -> None:
    first = MANAGED_MODEL_BUNDLES[0]

    assert bundle_size_bytes(first) == artifact_info(first.artifact_names[0]).expected_size
