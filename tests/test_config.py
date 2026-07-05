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
    monkeypatch.setattr(config_module, "_default_state_dir", lambda env=None: tmp_path / "state")
    assert resolve_models_dir(None, {}) == (tmp_path / "state" / "models").resolve()
    assert resolve_temp_dir(None, {}) == (tmp_path / "state" / "temp").resolve()


def test_resolve_runtime_dirs_composes_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_default_state_dir", lambda env=None: tmp_path / "state")
    dirs = resolve_runtime_dirs(env={})
    assert dirs.models_dir == (tmp_path / "state" / "models").resolve()
    assert dirs.temp_dir == (tmp_path / "state" / "temp").resolve()


def test_override_expands_user() -> None:
    resolved = resolve_models_dir(Path("~/pixelup-models"))
    assert resolved == (Path.home() / "pixelup-models").resolve()


def test_resolve_state_dir_expands_and_resolves_override() -> None:
    assert resolve_state_dir(Path("~/pixelup-state")) == (Path.home() / "pixelup-state").resolve()


def test_pixelup_home_relocates_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "relocated"
    monkeypatch.setenv("PIXELUP_HOME", str(root))
    assert resolve_state_dir() == root.resolve()
    # The root and its standard subdirectories are created on first use.
    assert (root / "logs").is_dir()
    assert (root / "models").is_dir()
    assert (root / "temp").is_dir()


def test_default_root_is_dot_pixelup_when_home_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PIXELUP_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resolve_state_dir() == (tmp_path / ".pixelup").resolve()


def test_pixelup_home_resolution_is_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # app_config is imported at the top of this module; the root must still be
    # resolved on first use, so a PIXELUP_HOME set *after* import takes effect
    # (it was never frozen into a module-level constant).
    from pixelup.app_config import config_path

    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path / "first"))
    assert config_path() == (tmp_path / "first" / "config.json").resolve()

    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path / "second"))
    assert config_path() == (tmp_path / "second" / "config.json").resolve()


def test_pixelup_home_relative_value_anchors_to_home_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("PIXELUP_HOME", "pixelup-data")
    # A relative override resolves against home, never the working directory.
    assert resolve_state_dir() == (fake_home / "pixelup-data").resolve()


def test_unusable_pixelup_home_is_a_reported_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("PIXELUP_HOME", str(blocker / "root"))
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


def test_pixelup_home_with_unset_env_reference_is_a_reported_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unset $VAR is left literal by os.path.expandvars rather than raising;
    # that must not silently become a directory literally named "$PIXELUP_NOPE".
    monkeypatch.delenv("PIXELUP_NOPE", raising=False)
    monkeypatch.setenv("PIXELUP_HOME", "$PIXELUP_NOPE/data")
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


def test_pixelup_home_with_empty_env_reference_is_a_reported_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A variable that is set but empty must not silently collapse the
    # storage root onto bare $HOME.
    monkeypatch.setenv("PIXELUP_EMPTY", "")
    monkeypatch.setenv("PIXELUP_HOME", "$PIXELUP_EMPTY")
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


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
