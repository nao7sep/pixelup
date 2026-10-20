"""The launchers' shared runtime ownership: which processes a launch replaces.

The decisions run here with a supplied process list; nothing real is signalled.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _bash(script: str, repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "launcher-runtime.sh"}"\n{script}', "bash", str(repo)],
        capture_output=True, text=True, timeout=10, check=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_only_the_latest_claim_owns_and_releases_the_runtime(repo: Path) -> None:
    result = _bash(
        """
repo="$1"
claim_launcher_runtime first "$repo"
claim_launcher_runtime second "$repo"
is_launcher_runtime_owner first "$repo" && echo "first owns"
is_launcher_runtime_owner second "$repo" && echo "second owns"
release_launcher_runtime first "$repo"
[[ -f "$repo/.git/launcher-runtime-owner" ]] && echo "kept after stale release"
release_launcher_runtime second "$repo"
[[ -f "$repo/.git/launcher-runtime-owner" ]] || echo "released"
""",
        repo,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["second owns", "kept after stale release", "released"]


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_a_launch_replaces_only_this_repositorys_pixelup(repo: Path) -> None:
    built = f"{repo}/dist/PixelUp.app/Contents/MacOS/PixelUp"
    other = "/elsewhere/pixelup"
    listing = repo / "ps.txt"
    listing.write_text(
        "\n".join(
            [
                f"101 /bin/bash {repo}/scripts/run-dev.command",
                f"102 {repo}/.venv/bin/python {repo}/.venv/bin/pixelup",
                f"103 uv run --project {repo} pixelup",
                f"104 {built} --flag",
                f"201 /bin/bash {other}/scripts/run-dev.command",
                f"202 {other}/.venv/bin/python {other}/.venv/bin/pixelup",
                f"203 {repo}/.venv/bin/python -m pytest",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = _bash(
        f"""
repo="$1"
ps() {{ cat "{listing}"; echo "$$ {repo}/scripts/run-dev.command"; }}
echo "owned: $(owned_runtime_pids "$repo" pixelup "{built}" | tr '\\n' ' ')"
echo "ready: $(ready_runtime_pids "$repo" pixelup "{built}" | tr '\\n' ' ')"
""",
        repo,
    )
    assert result.returncode == 0, result.stderr
    # A launcher or uv wrapper is replaced, but readiness waits for the app itself.
    assert result.stdout.splitlines() == ["owned: 101 102 103 104 ", "ready: 102 104 "]


@pytest.mark.skipif(sys.platform != "win32", reason="the PowerShell launcher is for Windows")
def test_windows_only_the_latest_claim_owns_and_releases_the_runtime(repo: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    script = f"""
$ErrorActionPreference = 'Stop'
Import-Module '{SCRIPTS / "launcher-runtime.psm1"}' -Force
$repo = '{repo}'
Claim-LauncherRuntime -Token first -RepoDir $repo
Claim-LauncherRuntime -Token second -RepoDir $repo
if (Test-LauncherRuntimeOwner -Token first -RepoDir $repo) {{ 'first owns' }}
if (Test-LauncherRuntimeOwner -Token second -RepoDir $repo) {{ 'second owns' }}
Release-LauncherRuntime -Token first -RepoDir $repo
if (Test-Path -LiteralPath "$repo\\.git\\launcher-runtime-owner") {{ 'kept after stale release' }}
Release-LauncherRuntime -Token second -RepoDir $repo
if (-not (Test-Path -LiteralPath "$repo\\.git\\launcher-runtime-owner")) {{ 'released' }}
"""
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["second owns", "kept after stale release", "released"]
