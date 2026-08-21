from __future__ import annotations

from dataclasses import dataclass

REAL_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
GFPGAN_RELEASES = "https://github.com/TencentARC/GFPGAN/releases/download"
FACEXLIB_RELEASES = "https://github.com/xinntao/facexlib/releases/download"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    filename: str
    url: str | None
    expected_size: int | None = None
    checksum_sha256: str | None = None
    listed: bool = True


# Each model is pinned to an immutable upstream release artifact (the URL's tag)
# and to a SHA-256 of that artifact's bytes. The hashes were computed from the
# official releases (xinntao/Real-ESRGAN, TencentARC/GFPGAN, xinntao/facexlib) and
# confirmed byte-identical to what those pinned URLs serve; that is the trust
# anchor, since these old releases publish no upstream checksum of their own. A
# download is verified against its hash before it is cached (see verify_model_file
# in models.py), so a corrupted or substituted same-size file never reaches the
# cache; a file already on disk is then trusted and not re-hashed on use.
# These projects froze in 2022; each entry is already the latest of its model.
ALL_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        67040989,
        checksum_sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
    ModelInfo(
        "RealESRNet_x4plus",
        "RealESRNet_x4plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.1.1/RealESRNet_x4plus.pth",
        67040989,
        checksum_sha256="a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df",
    ),
    ModelInfo(
        "RealESRGAN_x2plus",
        "RealESRGAN_x2plus.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        67061725,
        checksum_sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    ),
    ModelInfo(
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus_anime_6B.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        17938799,
        checksum_sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    ),
    ModelInfo(
        "realesr-animevideov3",
        "realesr-animevideov3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        2504012,
        checksum_sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
    ),
    ModelInfo(
        "realesr-general-x4v3",
        "realesr-general-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        4885111,
        checksum_sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    ),
    ModelInfo(
        "realesr-general-wdn-x4v3",
        "realesr-general-wdn-x4v3.pth",
        f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        4885111,
        checksum_sha256="1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        listed=False,
    ),
    ModelInfo(
        "GFPGANv1.4",
        "GFPGANv1.4.pth",
        f"{GFPGAN_RELEASES}/v1.3.4/GFPGANv1.4.pth",
        348632874,
        checksum_sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
    ),
    ModelInfo(
        "facexlib-detection-retinaface-resnet50",
        "detection_Resnet50_Final.pth",
        f"{FACEXLIB_RELEASES}/v0.1.0/detection_Resnet50_Final.pth",
        109497761,
        checksum_sha256="6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
        listed=False,
    ),
    ModelInfo(
        "facexlib-parsing-parsenet",
        "parsing_parsenet.pth",
        f"{FACEXLIB_RELEASES}/v0.2.2/parsing_parsenet.pth",
        85331193,
        checksum_sha256="3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2",
        listed=False,
    ),
)

KNOWN_MODELS: tuple[ModelInfo, ...] = tuple(model for model in ALL_MODELS if model.listed)
_MODEL_BY_NAME = {model.name: model for model in ALL_MODELS}


def known_model(name: str) -> ModelInfo | None:
    return _MODEL_BY_NAME.get(name)
