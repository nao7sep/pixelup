from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

from pixelup.errors import ErrorCode, PixelupError

APP_NAME = "pixelup"
MODELS_ENV = "PIXELUP_MODELS_DIR"
TEMP_ENV = "PIXELUP_TEMP_DIR"


@dataclass(frozen=True, slots=True)
class RuntimeDirs:
    models_dir: Path
    temp_dir: Path


def resolve_runtime_dirs(
    *,
    models_dir: Path | None = None,
    temp_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> RuntimeDirs:
    source_env = env if env is not None else os.environ
    return RuntimeDirs(
        models_dir=resolve_models_dir(models_dir, source_env),
        temp_dir=resolve_temp_dir(temp_dir, source_env),
    )


def resolve_models_dir(override: Path | None, env: dict[str, str] | None = None) -> Path:
    return _resolve_dir(override, MODELS_ENV, "models", env)


def resolve_temp_dir(override: Path | None, env: dict[str, str] | None = None) -> Path:
    return _resolve_dir(override, TEMP_ENV, "temp", env)


def ensure_models_dir(path: Path) -> Path:
    return _ensure_dir(path, ErrorCode.MODEL_NOT_FOUND, "Could not create the models directory.")


def ensure_temp_dir(path: Path) -> Path:
    return _ensure_dir(path, ErrorCode.OUTPUT_UNWRITABLE, "Could not create the temp directory.")


def _resolve_dir(
    override: Path | None,
    env_name: str,
    leaf: str,
    env: dict[str, str] | None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    source_env = env if env is not None else os.environ
    if env_value := source_env.get(env_name):
        return Path(env_value).expanduser().resolve()
    return _default_state_dir().joinpath(leaf).resolve()


def _default_state_dir() -> Path:
    primary = Path.home() / f".{APP_NAME}"
    if primary.parent.exists() and os.access(primary.parent, os.W_OK):
        return primary
    return Path(user_data_dir(APP_NAME, appauthor=False))


def _ensure_dir(path: Path, code: ErrorCode, message: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(code, message, details={"path": str(path), "reason": str(exc)}) from exc
    if not path.is_dir():
        raise PixelupError(code, message, details={"path": str(path)})
    return path

