from __future__ import annotations

import json
from pathlib import Path

from pixelup.cli import models_remove
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
