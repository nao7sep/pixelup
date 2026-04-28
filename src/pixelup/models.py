from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    alias: str | None
    filename: str
    expected_size: int | None = None
    checksum_sha256: str | None = None


KNOWN_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("RealESRGAN_x4plus", "x4plus", "RealESRGAN_x4plus.pth"),
    ModelInfo("RealESRNet_x4plus", "x4plusnet", "RealESRNet_x4plus.pth"),
    ModelInfo("RealESRGAN_x2plus", "x2plus", "RealESRGAN_x2plus.pth"),
    ModelInfo("RealESRGAN_x4plus_anime_6B", "anime", "RealESRGAN_x4plus_anime_6B.pth"),
    ModelInfo("realesr-animevideov3", "animevideo", "realesr-animevideov3.pth"),
    ModelInfo("realesr-general-x4v3", "general", "realesr-general-x4v3.pth"),
    ModelInfo("GFPGANv1.4", "face", "GFPGANv1.4.pth"),
)

_MODEL_BY_NAME = {model.name: model for model in KNOWN_MODELS}
_ALIAS_BY_NAME = {model.name: model.alias for model in KNOWN_MODELS if model.alias}


def model_short_name(name: str) -> str:
    return _ALIAS_BY_NAME.get(name, name)


def known_model(name: str) -> ModelInfo | None:
    return _MODEL_BY_NAME.get(name)


def model_file(models_dir: Path, name: str) -> Path:
    info = known_model(name)
    filename = info.filename if info else f"{name}.pth"
    return models_dir / filename


def model_present(models_dir: Path, name: str) -> bool:
    path = model_file(models_dir, name)
    return path.is_file() and path.stat().st_size > 0


def model_record(models_dir: Path, info: ModelInfo) -> dict[str, object]:
    path = models_dir / info.filename
    present = path.is_file()
    return {
        "name": info.name,
        "alias": info.alias,
        "present": present,
        "size_bytes": path.stat().st_size if present else None,
    }


def list_model_records(models_dir: Path, names: list[str] | None = None) -> list[dict[str, object]]:
    if not names:
        return [model_record(models_dir, info) for info in KNOWN_MODELS]
    records: list[dict[str, object]] = []
    for name in names:
        info = known_model(name)
        if info is None:
            path = model_file(models_dir, name)
            present = path.is_file()
            records.append(
                {
                    "name": name,
                    "alias": None,
                    "present": present,
                    "size_bytes": path.stat().st_size if present else None,
                }
            )
        else:
            records.append(model_record(models_dir, info))
    return records


def require_model_present(models_dir: Path, name: str) -> Path:
    path = model_file(models_dir, name)
    if not path.is_file():
        raise PixelupError(
            ErrorCode.MODEL_NOT_FOUND,
            f"Model '{name}' is not present in the models directory.",
            hint=(
                "Run 'pixelup models download MODEL' in a later phase, "
                "or place the .pth file there."
            ),
            details={"model": name, "models_dir": str(models_dir), "path": str(path)},
        )
    if path.stat().st_size <= 0:
        raise PixelupError(
            ErrorCode.MODEL_CORRUPT,
            f"Model '{name}' is empty.",
            details={"model": name, "path": str(path)},
        )
    return path


def verify_present_models(models_dir: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for info in KNOWN_MODELS:
        path = models_dir / info.filename
        if not path.is_file():
            continue
        size = path.stat().st_size
        ok = size > 0 and (info.expected_size is None or size == info.expected_size)
        results.append(
            {
                "name": info.name,
                "path": str(path),
                "ok": ok,
                "size_bytes": size,
                "expected_size_bytes": info.expected_size,
                "checksum_sha256": info.checksum_sha256,
            }
        )
    corrupt = [item for item in results if not item["ok"]]
    if corrupt:
        raise PixelupError(
            ErrorCode.MODEL_CORRUPT,
            "One or more model files failed verification.",
            details={"models": corrupt},
        )
    return results
