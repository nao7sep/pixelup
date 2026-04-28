from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from pixelup.cli import _version_command, main, models_remove
from pixelup.reporting import ReportMode


def test_models_remove_all_removes_unlisted_companion_model(
    tmp_path: Path,
    capsys,
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    public_model = models_dir / "realesr-general-x4v3.pth"
    companion_model = models_dir / "realesr-general-wdn-x4v3.pth"
    public_model.write_bytes(b"weights")
    companion_model.write_bytes(b"weights")

    models_remove(
        models=None,
        all_models=True,
        report=ReportMode.SINGLE,
        models_dir=models_dir,
    )

    output = json.loads(capsys.readouterr().out)
    assert output["removed"] == ["realesr-general-x4v3", "realesr-general-wdn-x4v3"]
    assert not public_model.exists()
    assert not companion_model.exists()


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
