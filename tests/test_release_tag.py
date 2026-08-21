from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release_tag.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_release_tag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_project_version_accepts_its_exact_v_prefixed_tag() -> None:
    checker = _load_script()
    tag = checker.expected_release_tag()

    checker.validate_release_tag(tag)
    assert checker.main([tag]) == 0


def test_mismatched_release_tag_fails(capsys: pytest.CaptureFixture[str]) -> None:
    checker = _load_script()

    with pytest.raises(ValueError, match="does not match project version"):
        checker.validate_release_tag("v999.0.0")
    assert checker.main(["v999.0.0"]) == 1
    assert "does not match project version" in capsys.readouterr().err


def test_release_workflow_runs_the_exact_tag_gate_before_builds() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert 'uv run python scripts/check_release_tag.py "${{ github.ref_name }}"' in workflow
    assert workflow.index("scripts/check_release_tag.py") < workflow.index("  build:")
