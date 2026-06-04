from __future__ import annotations

from pathlib import Path

import pytest

from pixelup import config as config_module
from pixelup.config import (
    ensure_models_dir,
    ensure_temp_dir,
    resolve_models_dir,
    resolve_runtime_dirs,
    resolve_state_dir,
    resolve_temp_dir,
)
from pixelup.errors import ErrorCode, PixelupError


def test_override_takes_precedence_over_env(tmp_path: Path) -> None:
    override = tmp_path / "explicit"
    resolved = resolve_models_dir(override, {"PIXELUP_MODELS_DIR": str(tmp_path / "from-env")})
    assert resolved == override.resolve()


def test_env_used_when_no_override(tmp_path: Path) -> None:
    env_dir = tmp_path / "from-env"
    assert resolve_temp_dir(None, {"PIXELUP_TEMP_DIR": str(env_dir)}) == env_dir.resolve()


def test_default_leaf_used_when_neither_override_nor_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_default_state_dir", lambda: tmp_path / "state")
    assert resolve_models_dir(None, {}) == (tmp_path / "state" / "models").resolve()
    assert resolve_temp_dir(None, {}) == (tmp_path / "state" / "temp").resolve()


def test_resolve_runtime_dirs_composes_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_default_state_dir", lambda: tmp_path / "state")
    dirs = resolve_runtime_dirs(env={})
    assert dirs.models_dir == (tmp_path / "state" / "models").resolve()
    assert dirs.temp_dir == (tmp_path / "state" / "temp").resolve()


def test_override_expands_user() -> None:
    resolved = resolve_models_dir(Path("~/pixelup-models"))
    assert resolved == (Path.home() / "pixelup-models").resolve()


def test_resolve_state_dir_expands_and_resolves_override() -> None:
    assert resolve_state_dir(Path("~/pixelup-state")) == (Path.home() / "pixelup-state").resolve()


def test_ensure_models_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "models"
    assert ensure_models_dir(target) == target
    assert target.is_dir()


def test_ensure_models_dir_wraps_oserror(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PixelupError) as exc_info:
        ensure_models_dir(blocker / "models")
    assert exc_info.value.code == ErrorCode.MODEL_NOT_FOUND


def test_ensure_temp_dir_wraps_oserror(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PixelupError) as exc_info:
        ensure_temp_dir(blocker / "temp")
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE
