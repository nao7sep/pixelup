from __future__ import annotations

import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pixelup.errors import ErrorCode, PixelupError
from pixelup.i18n.message import Message

APP_NAME = "pixelup"
HOME_ENV = "PIXELUP_DATA_DIR"
MODELS_ENV = "PIXELUP_MODELS_DIR"

# An environment reference in a path override: ${VAR}, $VAR or %VAR%.
_ENV_REF = re.compile(r"\$\{(\w+)\}|\$(\w+)|%(\w+)%")


@dataclass(frozen=True, slots=True)
class RuntimeDirs:
    models_dir: Path


def resolve_runtime_dirs(
    *,
    models_dir: Path | None = None,
    env: dict[str, str] | None = None,
) -> RuntimeDirs:
    source_env = env if env is not None else os.environ
    return RuntimeDirs(
        models_dir=resolve_models_dir(models_dir, source_env),
    )


def resolve_models_dir(override: Path | None, env: dict[str, str] | None = None) -> Path:
    return _resolve_dir(override, MODELS_ENV, "models", env)


def saved_models_dir(value: str, default: Path) -> Path:
    """Resolve a saved literal folder without touching a possibly unavailable disk.

    Empty follows the runtime default; relative authored paths are home-relative.
    Picker paths keep literal environment-looking characters. Symlinks need not be
    resolved to use the folder and resolving them here could stall the GUI thread.
    """
    if not value:
        return default
    path = Path(value)
    if path.parts and path.parts[0] == "~":
        path = Path.home().joinpath(*path.parts[1:])
    return Path(os.path.normpath(path if path.is_absolute() else Path.home() / path))


def resolve_state_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return _default_state_dir().resolve()


def window_settings_path() -> Path:
    """Return the native Qt settings file used for disposable window geometry."""
    return resolve_state_dir() / "window.ini"


def _resolve_dir(
    override: Path | None,
    env_name: str,
    leaf: str,
    env: dict[str, str] | None,
) -> Path:
    if override is not None:
        return _anchored_to_home(override.expanduser())
    source_env = env if env is not None else os.environ
    if env_value := source_env.get(env_name):
        return _anchored_to_home(_expand_override(env_value, env_name, source_env))
    return _default_state_dir(source_env).joinpath(leaf).resolve()


def _anchored_to_home(path: Path) -> Path:
    """``path`` made absolute against the home directory, never the working directory."""
    return (path if path.is_absolute() else Path.home() / path).resolve()


def _default_state_dir(env: dict[str, str] | None = None) -> Path:
    """Resolve the storage root: ``PIXELUP_DATA_DIR`` when set, else ``~/.pixelup``.

    Resolution is lazy on purpose — every call reads the environment afresh — so
    ``PIXELUP_DATA_DIR`` set before launch is honored and tests can vary it without a
    private setter. The override is expanded (leading ``~``, ``$VAR``/``%VAR%``)
    and made absolute against the *home* directory, never the working directory,
    so the override can never reintroduce a cwd dependence. There is exactly one
    root: an unusable home is a reported startup error, never a silent second
    root under the OS-native per-app directory.
    """
    source_env = env if env is not None else os.environ
    override = source_env.get(HOME_ENV, "")
    if override.strip():
        expanded = _expand_override(override.strip(), HOME_ENV, source_env)
        return _ensure_state_root(_anchored_to_home(expanded))
    return _ensure_state_root((Path.home() / f".{APP_NAME}").resolve())


def _expand_override(raw: str, variable: str, env: Mapping[str, str]) -> Path:
    """Expand ``~`` and environment references in the raw value of ``variable``.

    ``$FOO``, ``${FOO}`` and ``%FOO%`` are all expanded on every platform; the host's
    ``os.path.expandvars`` leaves ``%FOO%`` literal on macOS. A reference to a
    variable that is unset or empty does not resolve to the path the user meant, so
    per the storage-path-conventions it is a startup error, never a fallback.
    """
    unresolved = False

    def value(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = next(group for group in match.groups() if group is not None)
        found = env.get(name, "")
        unresolved = unresolved or not found
        return found

    expanded = _ENV_REF.sub(value, os.path.expanduser(raw))
    if not expanded or unresolved:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message.of("error.homeUnusable", variable=variable),
            hint=Message.of("error.hintHomeVariables", variable=variable),
            details={"value": raw, "expanded": expanded},
        )
    return Path(expanded)


def _ensure_state_root(root: Path) -> Path:
    """Create the storage root and its models folder on first use.

    ``logs/`` is not made here: it holds only the fallback lines written when the
    records store cannot take an entry, and that writer creates it with the first
    one, so an ordinary launch leaves no empty folder behind.

    Raises a clear startup error and stops if the root cannot be created or is
    not a usable directory, rather than relocating data somewhere the user did
    not ask for.
    """
    try:
        # The root is created with the owner-only (0700) mode directly, rather than under the
        # default umask and relying solely on `_tighten_root_permissions` afterward — that
        # sequence would leave a brief window where a freshly created root is world-readable.
        # The mode has no group/other bits, so a stricter umask can only narrow it further,
        # never widen it. Subdirectories keep their existing default-mode creation;
        # `_tighten_root_permissions` below only ever touches the root itself.
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _tighten_root_permissions(root)
        (root / "models").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.storageCreateFailed"),
            hint=Message.of("error.hintHomeWritableLocation", variable=HOME_ENV),
            details={"path": str(root), "reason": str(exc)},
        ) from exc
    if not root.is_dir():
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.storageNotDirectory"),
            hint=Message.of("error.hintHomeWritableDirectory", variable=HOME_ENV),
            details={"path": str(root)},
        )
    return root


def _tighten_root_permissions(root: Path) -> None:
    """Enforce an owner-only (0700) storage root on POSIX.

    Per the storage-path-conventions, the root is created owner-only and
    tightened at each launch when an existing root is broader, because
    derived data and logs must never be readable by accounts that cannot
    read their sources. Windows uses its own permission model and is
    unaffected. A failure to establish 0700 prevents startup; only the root
    itself is touched, never its contents.
    """
    if sys.platform == "win32":
        return
    try:
        mode = stat.S_IMODE(root.stat().st_mode)
        if mode != 0o700:
            root.chmod(0o700)
    except OSError as exc:
        raise PixelupError(
            ErrorCode.OUTPUT_UNWRITABLE,
            Message("error.storageCreateFailed"),
            hint=Message.of("error.hintHomeWritableLocation", variable=HOME_ENV),
            details={"path": str(root), "reason": str(exc)},
        ) from exc


def write_managed_text(path: Path, text: str) -> None:
    """The single managed-text atomic-write choke point for PixelUp.

    Every durable text file the app owns, including the settings files, is written
    through here. A managed-text write that bypasses this helper is a silent backup
    gap; there is deliberately no second atomic-write path for managed text in the app.

    Writes ``text`` (UTF-8) to a temp file the standard library creates exclusively
    beside ``path`` (``<stem>-<random>.tmp``, the storage-path conventions'
    derived-filename grammar), then atomically renames it over ``path``, so a crash
    mid-write cannot corrupt the target. Flushes and syncs the staged bytes before
    the rename. Raises on failure, preserving the previous target.

    **The data-backup record fires strictly AFTER the rename lands
    (data-backup-conventions).** Recording before the rename would risk a "backup of
    a save that never happened": if the rename then failed, the history would hold a
    version that never reached disk. So: rename lands, *then* record the exact bytes
    just written — the same ``data`` buffer already in hand, never a re-read of the
    file. The record is best-effort and silent; it never raises back into this
    write and never affects the save's success (see :mod:`pixelup.backup_store`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}-", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            if sys.platform == "darwin":
                _keep_replaced_file_mode(path, temp_path)
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        # Cleanup must not obscure the failure that prevented publication.
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise
    # After the rename: the file is exactly where it belongs, so record the bytes we
    # just wrote. Imported lazily to avoid a config <-> backup_store import cycle
    # (backup_store resolves its store path through resolve_state_dir here). record()
    # catches, logs once, and swallows every failure, so a backup problem can never
    # break the save that already succeeded above.
    from pixelup.backup_store import record

    record(path, data)


def _keep_replaced_file_mode(path: Path, temp_path: Path) -> None:
    """Keep ordinary permission bits on a macOS replacement through the runtime."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return
    os.chmod(temp_path, mode)
