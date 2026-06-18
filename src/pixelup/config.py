from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError

APP_NAME = "pixelup"
HOME_ENV = "PIXELUP_HOME"
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


def resolve_state_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return _default_state_dir().resolve()


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
    return _default_state_dir(source_env).joinpath(leaf).resolve()


def _default_state_dir(env: dict[str, str] | None = None) -> Path:
    """Resolve the storage root: ``PIXELUP_HOME`` when set, else ``~/.pixelup``.

    Resolution is lazy on purpose — every call reads the environment afresh — so
    ``PIXELUP_HOME`` set before launch is honored and tests can vary it without a
    private setter. The override is expanded (leading ``~``, ``$VAR``/``%VAR%``)
    and made absolute against the *home* directory, never the working directory,
    so the override can never reintroduce a cwd dependence. There is exactly one
    root: an unusable home is a reported startup error, never a silent second
    root under the OS-native per-app directory.
    """
    source_env = env if env is not None else os.environ
    override = source_env.get(HOME_ENV, "")
    if override.strip():
        expanded = os.path.expandvars(os.path.expanduser(override.strip()))
        root = Path(expanded)
        if not root.is_absolute():
            root = Path.home() / root
        return _ensure_state_root(root.resolve())
    return _ensure_state_root((Path.home() / f".{APP_NAME}").resolve())


def _ensure_state_root(root: Path) -> Path:
    """Create the storage root and its standard subdirectories on first use.

    Raises a clear startup error and stops if the root cannot be created or is
    not a usable directory, rather than relocating data somewhere the user did
    not ask for.
    """
    try:
        for path in (root, root / "logs", root / "models", root / "temp"):
            path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "Could not create the PixelUp storage directory.",
            hint=f"Set {HOME_ENV} to a writable location, or make the path usable.",
            details={"path": str(root), "reason": str(exc)},
        ) from exc
    if not root.is_dir():
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            "The PixelUp storage path is not a usable directory.",
            hint=f"Set {HOME_ENV} to a writable directory.",
            details={"path": str(root)},
        )
    return root


def _ensure_dir(path: Path, code: ErrorCode, message: str) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(code, message, details={"path": str(path), "reason": str(exc)}) from exc
    if not path.is_dir():
        raise PixelupError(code, message, details={"path": str(path)})
    return path
