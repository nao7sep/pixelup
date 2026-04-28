from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from pixelup import cli as cli_module
from pixelup.cli import _version_command, main, models_check, models_remove
from pixelup.reporting import ReportMode


def test_main_registers_image_plugins_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli_module, "register_image_plugins", lambda: calls.append("plugins"))
    monkeypatch.setattr(cli_module, "_version_command", lambda args: calls.append("version"))

    main(["--version"])

    assert calls == ["plugins", "version"]


def test_models_remove_all_removes_unlisted_companion_model(
    tmp_path: Path,
    capsys,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    files = {
        "realesr-general-x4v3": models_dir / "realesr-general-x4v3.pth",
        "realesr-general-wdn-x4v3": models_dir / "realesr-general-wdn-x4v3.pth",
        "GFPGANv1.4": models_dir / "GFPGANv1.4.pth",
        "facexlib-detection-retinaface-resnet50": models_dir / "detection_Resnet50_Final.pth",
        "facexlib-parsing-parsenet": models_dir / "parsing_parsenet.pth",
    }
    unknown_file = models_dir / "custom.pth"
    for path in files.values():
        path.write_bytes(b"weights")
    unknown_file.write_bytes(b"weights")

    models_remove(
        models=None,
        all_models=True,
        report=ReportMode.SINGLE,
        models_dir=models_dir,
    )

    output = json.loads(capsys.readouterr().out)
    assert output["removed"] == list(files)
    assert all(not path.exists() for path in files.values())
    assert unknown_file.is_file()


def test_models_check_download_missing_without_names_targets_all_public_models(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_download_models(
        models_dir: Path,
        names: list[str],
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        calls["download_names"] = names
        return []

    def fake_list_model_records(
        models_dir: Path,
        names: list[str] | None = None,
    ) -> list[dict[str, object]]:
        calls["record_names"] = names
        return []

    monkeypatch.setattr(cli_module, "download_models", fake_download_models)
    monkeypatch.setattr(cli_module, "list_model_records", fake_list_model_records)

    models_check(
        models=None,
        download_missing=True,
        report=ReportMode.SINGLE,
        models_dir=tmp_path / "models",
    )

    output = json.loads(capsys.readouterr().out)
    expected = cli_module.all_model_names()
    assert calls["download_names"] == expected
    assert calls["record_names"] == expected
    assert output == {"models_dir": str((tmp_path / "models").resolve()), "models": []}


def test_version_stream_report_is_single_result_without_event(capsys) -> None:
    _version_command(["--report", "stream"])

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert "event" not in output


def test_version_invalid_report_returns_machine_readable_invalid_argument(capsys) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        _version_command(["--report", "bogus"])

    output = json.loads(capsys.readouterr().out)
    assert excinfo.value.exit_code == 2
    assert output["ok"] is False
    assert output["code"] == "invalid_argument"
    assert output["details"] == {"report": "bogus"}


def test_main_version_invalid_report_exits_without_traceback(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version", "--report", "bogus"])

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert excinfo.value.code == 2
    assert output["code"] == "invalid_argument"
    assert captured.err == ""


def test_show_completion_accepts_explicit_shell(capsys) -> None:
    main(["--show-completion", "zsh"])

    captured = capsys.readouterr()
    assert "#compdef pixelup" in captured.out
    assert "_PIXELUP_COMPLETE=complete_zsh" in captured.out
    assert captured.err == ""


def test_show_completion_uses_shell_env(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SHELL", "/bin/bash")

    main(["--show-completion"])

    captured = capsys.readouterr()
    assert "complete -o default -F _pixelup_completion pixelup" in captured.out
    assert captured.err == ""


def test_install_completion_accepts_explicit_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    main(["--install-completion", "zsh"])

    captured = capsys.readouterr()
    assert "zsh completion installed" in captured.out
    assert (tmp_path / ".zfunc" / "_pixelup").is_file()
    assert (tmp_path / ".zshrc").is_file()
