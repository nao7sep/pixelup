from __future__ import annotations

import re
import stat
import sys
from pathlib import Path

import pytest

from pixelup import config as config_module
from pixelup.config import (
    RuntimeDirs,
    quarantine_corrupt_file,
    resolve_models_dir,
    resolve_runtime_dirs,
    resolve_state_dir,
    window_settings_path,
)
from pixelup.errors import ErrorCode, PixelupError


def test_override_takes_precedence_over_env(tmp_path: Path) -> None:
    override = tmp_path / "explicit"
    resolved = resolve_models_dir(override, {"PIXELUP_MODELS_DIR": str(tmp_path / "from-env")})
    assert resolved == override.resolve()


def test_env_used_when_no_override(tmp_path: Path) -> None:
    env_dir = tmp_path / "from-env"
    assert resolve_models_dir(None, {"PIXELUP_MODELS_DIR": str(env_dir)}) == env_dir.resolve()


def test_default_leaf_used_when_neither_override_nor_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_default_state_dir", lambda env=None: tmp_path / "state")
    assert resolve_models_dir(None, {}) == (tmp_path / "state" / "models").resolve()


def test_resolve_runtime_dirs_resolves_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "_default_state_dir", lambda env=None: tmp_path / "state")
    dirs = resolve_runtime_dirs(env={})
    assert dirs.models_dir == (tmp_path / "state" / "models").resolve()


def test_override_expands_user() -> None:
    resolved = resolve_models_dir(Path("~/pixelup-models"))
    assert resolved == (Path.home() / "pixelup-models").resolve()


def test_relative_runtime_overrides_anchor_to_home_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # Path.home() and a leading ~ both read these.
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    env = {"PIXELUP_MODELS_DIR": "models"}
    expected = RuntimeDirs(models_dir=(fake_home / "models").resolve())
    for working_directory in (tmp_path / "one", tmp_path / "two"):
        working_directory.mkdir()
        monkeypatch.chdir(working_directory)
        assert resolve_runtime_dirs(env=env) == expected
        assert resolve_runtime_dirs(
            models_dir=Path("models"), env={}
        ) == expected


def test_runtime_override_with_unset_env_reference_is_a_reported_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PIXELUP_NOPE", raising=False)
    with pytest.raises(PixelupError) as exc_info:
        resolve_models_dir(None, {"PIXELUP_MODELS_DIR": "$PIXELUP_NOPE/models"})
    assert exc_info.value.message.value("variable") == "PIXELUP_MODELS_DIR"


def test_resolve_state_dir_expands_and_resolves_override() -> None:
    assert resolve_state_dir(Path("~/pixelup-state")) == (Path.home() / "pixelup-state").resolve()


def test_pixelup_home_relocates_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "relocated"
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    assert resolve_state_dir() == root.resolve()
    # The root and its models folder are created on first use; logs/ waits for its
    # first fallback entry.
    assert (root / "models").is_dir()
    assert not (root / "logs").exists()
    assert not (root / "temp").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only permission model")
def test_fresh_root_is_created_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "relocated"
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    resolve_state_dir()
    mode = stat.S_IMODE(root.stat().st_mode)
    assert mode == 0o700


def test_windows_leaves_the_root_to_its_own_permission_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "relocated"
    root.mkdir()
    monkeypatch.setattr(config_module.sys, "platform", "win32")
    monkeypatch.setattr(Path, "chmod", lambda *_args: pytest.fail("chmod on Windows"))
    config_module._tighten_root_permissions(root)


def test_only_macos_copies_the_replaced_files_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows' chmod sets only the read-only flag, which would make the replacement
    # itself refuse to land; ordinary permission bits are a POSIX contract.
    monkeypatch.setattr("pixelup.backup_store.record", lambda *_args: None)
    copied: list[str] = []
    monkeypatch.setattr(
        config_module, "_keep_replaced_file_mode", lambda path, _temp: copied.append(path.name)
    )
    target = tmp_path / "config.json"
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(config_module.sys, "platform", platform)
        config_module.write_managed_text(target, platform)
        assert target.read_text(encoding="utf-8") == platform
    assert copied == ["config.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only permission model")
def test_existing_broader_root_is_tightened_on_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "relocated"
    root.mkdir()
    root.chmod(0o755)
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    resolve_state_dir()
    mode = stat.S_IMODE(root.stat().st_mode)
    assert mode == 0o700


def test_pixelup_home_relocates_window_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "relocated"
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    assert window_settings_path() == (root / "window.ini").resolve()


def test_default_root_is_dot_pixelup_when_home_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PIXELUP_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resolve_state_dir() == (tmp_path / ".pixelup").resolve()


def test_pixelup_home_resolution_is_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # app_config is imported at the top of this module; the root must still be
    # resolved on first use, so a PIXELUP_DATA_DIR set *after* import takes effect
    # (it was never frozen into a module-level constant).
    from pixelup.app_config import config_path

    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "first"))
    assert config_path() == (tmp_path / "first" / "config.json").resolve()

    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "second"))
    assert config_path() == (tmp_path / "second" / "config.json").resolve()


def test_pixelup_home_relative_value_anchors_to_home_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("PIXELUP_DATA_DIR", "pixelup-data")
    # A relative override resolves against home, never the working directory.
    assert resolve_state_dir() == (fake_home / "pixelup-data").resolve()


def test_unusable_pixelup_home_is_a_reported_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(blocker / "root"))
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


def test_pixelup_home_with_unset_env_reference_is_a_reported_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unset $VAR is left literal by os.path.expandvars rather than raising;
    # that must not silently become a directory literally named "$PIXELUP_NOPE".
    monkeypatch.delenv("PIXELUP_NOPE", raising=False)
    monkeypatch.setenv("PIXELUP_DATA_DIR", "$PIXELUP_NOPE/data")
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


def test_pixelup_home_with_empty_env_reference_is_a_reported_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A variable that is set but empty must not silently collapse the
    # storage root onto bare $HOME.
    monkeypatch.setenv("PIXELUP_EMPTY", "")
    monkeypatch.setenv("PIXELUP_DATA_DIR", "$PIXELUP_EMPTY")
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


@pytest.mark.parametrize("reference", ["$PIXELUP_ROOT", "${PIXELUP_ROOT}", "%PIXELUP_ROOT%"])
def test_pixelup_home_expands_every_reference_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    monkeypatch.setenv("PIXELUP_ROOT", str(tmp_path / "profile"))
    monkeypatch.setenv("PIXELUP_DATA_DIR", f"{reference}/pixelup")

    assert resolve_state_dir() == (tmp_path / "profile" / "pixelup").resolve()


@pytest.mark.parametrize("reference", ["$PIXELUP_ROOT", "${PIXELUP_ROOT}", "%PIXELUP_ROOT%"])
@pytest.mark.parametrize("value", ["", None])
def test_pixelup_home_with_an_empty_or_absent_reference_is_a_reported_error(
    monkeypatch: pytest.MonkeyPatch, reference: str, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("PIXELUP_ROOT", raising=False)
    else:
        monkeypatch.setenv("PIXELUP_ROOT", value)
    monkeypatch.setenv("PIXELUP_DATA_DIR", f"{reference}/pixelup")

    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE


_QUARANTINE_NAME = re.compile(r"^config-\d{8}-\d{6}-utc\.invalid$")


def test_quarantine_corrupt_file_moves_bytes_to_invalid_sibling(tmp_path: Path) -> None:
    corrupt = tmp_path / "config.json"
    corrupt.write_text("{ not valid json", encoding="utf-8")

    quarantined = quarantine_corrupt_file(corrupt)

    # The original is gone from its place; its bytes live on under the .invalid name.
    assert not corrupt.exists()
    assert quarantined.parent == tmp_path
    assert quarantined.read_text(encoding="utf-8") == "{ not valid json"
    # The name follows <stem>-<utc>.invalid: the role extension replaces .json (so a
    # *.json scan can never pick the debris up), and the discriminator is the compact
    # second UTC stamp.
    assert _QUARANTINE_NAME.match(quarantined.name)
    assert quarantined.suffix == ".invalid"


def test_quarantine_never_replaces_an_earlier_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two corrupt versions quarantined in one second clash on the name: the
    # second fails like any read failure and leaves
    # both the earlier quarantine and its own original in place.
    monkeypatch.setattr(config_module, "utc_now_stamp", lambda: "20260705-010203-utc")

    first = tmp_path / "config.json"
    first.write_text("first corrupt", encoding="utf-8")
    quarantined_first = quarantine_corrupt_file(first)

    second = tmp_path / "config.json"
    second.write_text("second corrupt", encoding="utf-8")
    with pytest.raises(FileExistsError):
        quarantine_corrupt_file(second)

    assert quarantined_first.read_text(encoding="utf-8") == "first corrupt"
    assert second.read_text(encoding="utf-8") == "second corrupt"


def test_a_failed_quarantine_move_releases_its_claimed_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupt = tmp_path / "config.json"
    corrupt.write_text("corrupt", encoding="utf-8")

    def fail(_source: object, _target: object) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(config_module.os, "replace", fail)
    with pytest.raises(OSError, match="rename failed"):
        quarantine_corrupt_file(corrupt)

    assert [item.name for item in tmp_path.iterdir()] == ["config.json"]


def test_saved_model_folder_is_literal_home_relative_and_never_resolves_disk(
    tmp_path, monkeypatch
):
    from pixelup.config import saved_models_dir

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "resolve", lambda *a, **kw: pytest.fail("filesystem resolution"))
    default = tmp_path / "default"
    assert saved_models_dir("", default) == default
    assert saved_models_dir("~/models", default) == tmp_path / "models"
    assert saved_models_dir("relative/models", default) == tmp_path / "relative/models"
    assert saved_models_dir("~literal/models", default) == tmp_path / "~literal/models"
    literal = tmp_path / "$literal%PATH%"
    assert saved_models_dir(str(literal), default) == literal


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only permission model")
@pytest.mark.parametrize("initial_mode", [0o500, 0o750, 0o770, 0o1700])
def test_root_establishes_exact_owner_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_mode: int
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    root.chmod(initial_mode)
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    resolve_state_dir()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only permission model")
def test_root_permission_failure_stops_startup_before_creating_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    failure = PermissionError("permission denied")

    def fail_chmod(_path: Path, _mode: int) -> None:
        raise failure

    monkeypatch.setattr(Path, "chmod", fail_chmod)
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(root))
    with pytest.raises(PixelupError) as exc_info:
        resolve_state_dir()
    assert exc_info.value.code == ErrorCode.OUTPUT_UNWRITABLE
    assert exc_info.value.__cause__ is failure
    assert not (root / "logs").exists()


def test_managed_write_syncs_flushed_bytes_before_publication_and_records_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.json"
    target.write_text("old", encoding="utf-8")
    data = "new \u65e5\u672c\u8a9e\n".encode()
    events: list[str] = []
    fsync = config_module.os.fsync
    replace = config_module.os.replace

    def observe_sync(descriptor: int) -> None:
        events.append("file sync")
        # fstat observes bytes flushed out of Python's buffer.
        assert config_module.os.fstat(descriptor).st_size == len(data)
        assert target.read_text(encoding="utf-8") == "old"
        fsync(descriptor)

    def observe_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        assert source.read_bytes() == data
        replace(source, destination)

    def observe_backup(path: Path, saved: bytes) -> None:
        events.append("backup")
        assert path == target
        assert saved == target.read_bytes() == data

    monkeypatch.setattr(config_module.os, "fsync", observe_sync)
    monkeypatch.setattr(config_module.os, "replace", observe_replace)
    monkeypatch.setattr("pixelup.backup_store.record", observe_backup)
    config_module.write_managed_text(target, data.decode())
    assert events == ["file sync", "replace", "backup"]
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failed_step", ["file sync", "replace"])
def test_managed_write_failure_keeps_previous_bytes_and_does_not_record_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_step: str
) -> None:
    target = tmp_path / "config.json"
    target.write_text("old", encoding="utf-8")
    failure = OSError("primary write failed")

    def fail(*_args: object) -> None:
        raise failure

    monkeypatch.setattr(
        config_module.os, "fsync" if failed_step == "file sync" else "replace", fail
    )
    monkeypatch.setattr(
        "pixelup.backup_store.record", lambda *_args: pytest.fail("backup before publication")
    )
    with pytest.raises(OSError) as exc_info:
        config_module.write_managed_text(target, "new")
    assert exc_info.value is failure
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_staging_cleanup_does_not_replace_the_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.json"
    target.write_text("old", encoding="utf-8")
    failure = OSError("primary write failed")

    def fail_replace(*_args: object) -> None:
        raise failure

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(config_module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as exc_info:
        config_module.write_managed_text(target, "new")
    assert exc_info.value is failure
    assert target.read_text(encoding="utf-8") == "old"

