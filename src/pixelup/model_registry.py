from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message

REAL_ESRGAN_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"

GENERAL_MODEL = "realesr-general-x4v3"
GENERAL_DENOISE_MODEL = "realesr-general-wdn-x4v3"


@dataclass(frozen=True, slots=True)
class ModelArchitecture:
    """The network a model's weights load into, and the scale it upscales by."""

    kind: Literal["rrdb", "srvgg"]
    netscale: int
    params: Mapping[str, int | str]


def _rrdb(*, scale: int, num_block: int) -> ModelArchitecture:
    return ModelArchitecture(
        kind="rrdb",
        netscale=scale,
        params={
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": num_block,
            "num_grow_ch": 32,
            "scale": scale,
        },
    )


def _srvgg(*, scale: int, num_conv: int) -> ModelArchitecture:
    return ModelArchitecture(
        kind="srvgg",
        netscale=scale,
        params={
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_conv": num_conv,
            "upscale": scale,
            "act_type": "prelu",
        },
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelInfo:
    name: str
    filename: str
    url: str
    expected_size: int
    checksum_sha256: str
    architecture: ModelArchitecture
    # An upscaler the user picks; the general model's denoise companion is not one.
    upscaler: bool


# Listed in the order the main window and Managed models show the upscalers.
# Each model is pinned to an immutable upstream release artifact (the URL's tag)
# and to a SHA-256 of that artifact's bytes. The hashes were computed from the
# official xinntao/Real-ESRGAN releases and confirmed byte-identical to what those
# pinned URLs serve; that is the trust anchor, since these old releases publish no
# upstream checksum of their own. A download is verified against its hash before
# it is cached, so a corrupted or substituted same-size file never reaches the
# cache; a file already on disk is then trusted and not re-hashed on use.
# These projects froze in 2022; each entry is already the latest of its model.
ALL_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        name=GENERAL_MODEL,
        filename="realesr-general-x4v3.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        expected_size=4885111,
        checksum_sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        architecture=_srvgg(scale=4, num_conv=32),
        upscaler=True,
    ),
    ModelInfo(
        name="RealESRGAN_x4plus",
        filename="RealESRGAN_x4plus.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        expected_size=67040989,
        checksum_sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        architecture=_rrdb(scale=4, num_block=23),
        upscaler=True,
    ),
    ModelInfo(
        name="RealESRNet_x4plus",
        filename="RealESRNet_x4plus.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.1.1/RealESRNet_x4plus.pth",
        expected_size=67040989,
        checksum_sha256="a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df",
        architecture=_rrdb(scale=4, num_block=23),
        upscaler=True,
    ),
    ModelInfo(
        name="RealESRGAN_x2plus",
        filename="RealESRGAN_x2plus.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        expected_size=67061725,
        checksum_sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        architecture=_rrdb(scale=2, num_block=23),
        upscaler=True,
    ),
    ModelInfo(
        name="RealESRGAN_x4plus_anime_6B",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        expected_size=17938799,
        checksum_sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        architecture=_rrdb(scale=4, num_block=6),
        upscaler=True,
    ),
    ModelInfo(
        name="realesr-animevideov3",
        filename="realesr-animevideov3.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        expected_size=2504012,
        checksum_sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        architecture=_srvgg(scale=4, num_conv=16),
        upscaler=True,
    ),
    ModelInfo(
        name=GENERAL_DENOISE_MODEL,
        filename="realesr-general-wdn-x4v3.pth",
        url=f"{REAL_ESRGAN_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        expected_size=4885111,
        checksum_sha256="1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        # Interpolated into the general model's weights, so it shares its network.
        architecture=_srvgg(scale=4, num_conv=32),
        upscaler=False,
    ),
)

UPSCALE_MODELS: tuple[str, ...] = tuple(model.name for model in ALL_MODELS if model.upscaler)
_MODEL_BY_NAME = {model.name: model for model in ALL_MODELS}


def model_info(name: str) -> ModelInfo:
    try:
        return _MODEL_BY_NAME[name]
    except KeyError:
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            Message.of("error.modelUnknown", model=name),
            details={"model": name},
        ) from None
