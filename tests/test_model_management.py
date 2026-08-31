from __future__ import annotations

from pathlib import Path

from pixelup.model_management import (
    GENERAL_DENOISE_MODEL,
    MANAGED_MODEL_BUNDLES,
    UPSCALE_MODELS,
    artifact_info,
    bundle_ready_count,
    bundle_size_bytes,
    missing_artifact_names,
    ready_artifact_count,
    required_artifact_names,
)
from pixelup.models import model_file


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
        face_enhance=True,
        denoise_strength=0.5,
    )

    assert required == (
        "realesr-general-x4v3",
        GENERAL_DENOISE_MODEL,
        "RealESRGAN_x4plus",
        "GFPGANv1.4",
        "facexlib-detection-retinaface-resnet50",
        "facexlib-parsing-parsenet",
    )


def test_neutral_denoise_omits_the_general_companion() -> None:
    required = required_artifact_names(
        ("realesr-general-x4v3", "RealESRGAN_x4plus"),
        face_enhance=False,
        denoise_strength=1.0,
    )

    assert required == ("realesr-general-x4v3", "RealESRGAN_x4plus")


def test_readiness_is_derived_from_artifact_files(tmp_path: Path) -> None:
    first = MANAGED_MODEL_BUNDLES[0]
    first_path = model_file(tmp_path, first.artifact_names[0])
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_bytes(b"present")

    assert bundle_ready_count(tmp_path, first) == 1
    assert missing_artifact_names(tmp_path, first.artifact_names) == ()
    assert ready_artifact_count(tmp_path) == (1, 10)

    first_path.write_bytes(b"")
    assert bundle_ready_count(tmp_path, first) == 0
    assert missing_artifact_names(tmp_path, first.artifact_names) == first.artifact_names


def test_bundle_size_is_the_sum_of_its_pinned_artifacts() -> None:
    face = next(bundle for bundle in MANAGED_MODEL_BUNDLES if bundle.key == "face-enhancement")

    assert bundle_size_bytes(face) == 543_461_828
