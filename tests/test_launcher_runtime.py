"""The launchers' shared runtime ownership: which processes a launch replaces.

Only disposable children started by these tests may be signalled.
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
        capture_output=True, text=True, timeout=20, check=False,
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


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_replacement_stops_only_the_direct_recorded_child(repo: Path) -> None:
    result = _bash(
        r'''
set -eu
repo="$1"
sleep 30 & unrelated=$!
trap 'kill "$unrelated" 2>/dev/null || true; wait "$unrelated" 2>/dev/null || true' EXIT
claim_launcher_runtime first "$repo"
start_owned_runtime "$repo" first sleep 30
owned=$LAUNCHED_RUNTIME_PID
identity=$LAUNCHED_RUNTIME_IDENTITY
wait_for_captured_runtime StandIn "$owned" "$identity" 1
stop_owned_runtime StandIn "$repo" first
wait "$owned" 2>/dev/null || true
kill -0 "$unrelated"
! is_captured_runtime_alive "$owned" "$identity"
[[ ! -f "$repo/.git/launcher-runtime-process" ]]
echo "only recorded child stopped"
''', repo,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "only recorded child stopped"


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
@pytest.mark.parametrize("receipt", ["missing", "stale", "malformed"])
def test_unrecorded_or_stale_identity_never_stops_a_child(repo: Path, receipt: str) -> None:
    result = _bash(
        r'''
set -eu
repo="$1"
sleep 30 & child=$!
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
claim_launcher_runtime first "$repo"
''' + {
            "missing": "",
            "stale": (
                'launcher_runtime_transaction "$repo" register first "$child" '
                '"stale creation identity"'
            ),
            "malformed": 'printf "invalid\n" > "$repo/.git/launcher-runtime-process"',
        }[receipt] + r'''
stop_owned_runtime StandIn "$repo" first
kill -0 "$child"
echo "child kept"
''', repo,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "child kept"


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_superseded_starter_stops_its_child_and_keeps_newer_receipt(repo: Path) -> None:
    result = _bash(
        r'''
set -eu
repo="$1"
claim_launcher_runtime first "$repo"
# Interleave a newer launcher between the old launch and its registration.
original_transaction=$(declare -f launcher_runtime_transaction)
eval "${original_transaction/launcher_runtime_transaction/original_transaction}"
launcher_runtime_transaction() {
  if [[ "$2" == register && "$3" == first ]]; then
    original_transaction "$repo" claim second
    sleep 30 & newer=$!
    newer_identity=$(runtime_creation_identity "$newer")
    original_transaction "$repo" register second "$newer" "$newer_identity"
  fi
  original_transaction "$@"
}
cleanup() {
  if [[ -n "${newer:-}" ]]; then
    kill "$newer" 2>/dev/null || true
    wait "$newer" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if start_owned_runtime "$repo" first sleep 30; then exit 1; else [[ "$?" == 2 ]]; fi
! is_captured_runtime_alive "$LAUNCHED_RUNTIME_PID" "$LAUNCHED_RUNTIME_IDENTITY"
kill -0 "$newer"
stop_owned_runtime StandIn "$repo" first first
release_launcher_runtime first "$repo"
is_launcher_runtime_owner second "$repo"
[[ $(head -n 1 "$repo/.git/launcher-runtime-process") == second ]]
echo "newer runtime kept"
''', repo,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "newer runtime kept"


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_built_receipt_remains_available_to_the_next_launcher(repo: Path) -> None:
    result = _bash(
        r'''
set -eu
repo="$1"
claim_launcher_runtime built "$repo"
start_owned_runtime "$repo" built sleep 30
child=$LAUNCHED_RUNTIME_PID
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
claim_launcher_runtime replacement "$repo"
stop_owned_runtime StandIn "$repo" replacement
wait "$child" 2>/dev/null || true
! is_captured_runtime_alive "$child" "$LAUNCHED_RUNTIME_IDENTITY"
echo "prior built runtime replaced"
''', repo,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "prior built runtime replaced"


@pytest.mark.skipif(sys.platform != "win32", reason="the PowerShell launcher is for Windows")
def test_windows_receipts_limit_replacement_to_owned_live_identity(repo: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    python = sys.executable.replace("'", "''")
    script = f'''
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module '{SCRIPTS / "launcher-runtime.psm1"}' -Force
$repo = '{repo}'
$python = '{python}'
$unrelated = Start-Process -FilePath $python `
    -ArgumentList '-c "import time; time.sleep(30)"' -PassThru
$owned = $null
try {{
    Claim-LauncherRuntime -Token first -RepoDir $repo
    $owned = Start-OwnedRuntime -RepoDir $repo -Token first -FilePath $python `
        -ArgumentList @('-c', 'import time; time.sleep(30)')
    Wait-OwnedRuntime -Label StandIn -Process $owned
    Stop-OwnedRuntime -Label StandIn -RepoDir $repo -Token first
    if (-not $owned.WaitForExit(5000)) {{ throw 'owned child kept running' }}
    $unrelated.Refresh()
    if ($unrelated.HasExited) {{ throw 'unrelated child stopped' }}

    # A syntactically valid stale creation identity grants no stopping authority.
    $receiptFile = Join-Path (Join-Path $repo '.git') 'launcher-runtime-process'
    @{{ Token = 'first'; ProcessId = $unrelated.Id; StartTicks = 1 }} |
        ConvertTo-Json | Set-Content -LiteralPath $receiptFile
    Stop-OwnedRuntime -Label StandIn -RepoDir $repo -Token first
    $unrelated.Refresh()
    if ($unrelated.HasExited) {{ throw 'stale receipt stopped child' }}

    Claim-LauncherRuntime -Token second -RepoDir $repo
    $owned = Start-OwnedRuntime -RepoDir $repo -Token second -FilePath $python `
        -ArgumentList @('-c', 'import time; time.sleep(30)')
    $obsolete = Start-OwnedRuntime -RepoDir $repo -Token first -FilePath $python `
        -ArgumentList @('-c', 'raise SystemExit(99)')
    if ($null -ne $obsolete) {{ throw 'superseded launcher started a runtime' }}
    Stop-OwnedRuntime -Label StandIn -RepoDir $repo -Token first -ReceiptToken first
    Release-LauncherRuntime -Token first -RepoDir $repo
    $owned.Refresh()
    if ($owned.HasExited) {{ throw 'stale cleanup stopped newer child' }}
    $receipt = Get-Content -LiteralPath $receiptFile -Raw | ConvertFrom-Json
    if ($receipt.Token -cne 'second') {{ throw 'newer receipt overwritten' }}
    'recorded ownership respected'
}} finally {{
    foreach ($child in @($owned, $unrelated)) {{
        if ($null -ne $child) {{
            if (-not $child.HasExited) {{ $child.Kill(); $child.WaitForExit() }}
            $child.Dispose()
        }}
    }}
}}
'''
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "recorded ownership respected"


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_recorded_child_that_ignores_term_is_killed_and_settled(repo: Path) -> None:
    standin = repo / "ignore_term.py"
    standin.write_text(
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text('ready')\n"
        "time.sleep(30)\n"
    )
    # Pass the interpreter as an argument, rather than embedding shell syntax.
    script = r'''
set -eu
repo="$1"
claim_launcher_runtime first "$repo"
start_owned_runtime "$repo" first "$2" "$repo/ignore_term.py" "$repo/ready"
child=$LAUNCHED_RUNTIME_PID
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
for attempt in {1..100}; do
  [[ ! -f "$repo/ready" ]] || break
  sleep 0.01
done
[[ -f "$repo/ready" ]]
stop_owned_runtime StandIn "$repo" first
! is_captured_runtime_alive "$child" "$LAUNCHED_RUNTIME_IDENTITY"
wait "$child" 2>/dev/null || true
echo "forced stop settled"
'''
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "launcher-runtime.sh"}"\n{script}',
         "bash", str(repo), sys.executable],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "forced stop settled"


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_readiness_rejects_a_different_creation_identity(repo: Path) -> None:
    result = _bash(
        r'''
set -eu
sleep 30 & child=$!
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
if wait_for_captured_runtime StandIn "$child" "stale identity" 1; then exit 1; fi
kill -0 "$child"
echo "readiness stayed with captured identity"
''', repo,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "readiness stayed with captured identity"


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_snapshot_failure_stops_the_launcher_before_start(repo: Path) -> None:
    result = _bash(
        r'''
set -eu
repo="$1"
launcher_runtime_transaction() { return 73; }
start_owned_runtime() { printf "launched" > "$repo/unexpected-start"; }
stop_owned_runtime StandIn "$repo" current
start_owned_runtime "$repo" current sleep 30
''', repo,
    )
    assert result.returncode == 73
    assert not (repo / "unexpected-start").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the bash launcher runtime is for macOS")
def test_start_owner_check_keeps_transaction_failure_distinct_from_supersession(repo: Path) -> None:
    result = _bash(
        r'''
set -eu
repo="$1"
launcher_runtime_transaction() { return 73; }
start_owned_runtime "$repo" current touch "$repo/unexpected-start"
''', repo,
    )
    assert result.returncode == 73
    assert not (repo / "unexpected-start").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the PowerShell launcher is for Windows")
def test_windows_receipt_io_failure_prevents_the_next_launch_step(repo: Path) -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    script = f'''
$ErrorActionPreference = 'Stop'
Import-Module '{SCRIPTS / "launcher-runtime.psm1"}' -Force
$repo = '{repo}'
Claim-LauncherRuntime -Token first -RepoDir $repo
$receiptFile = Join-Path (Join-Path $repo '.git') 'launcher-runtime-process'
$null = [System.IO.Directory]::CreateDirectory($receiptFile)
$marker = Join-Path $repo 'unexpected-start'
$failed = $false
try {{
    Stop-OwnedRuntime -Label StandIn -RepoDir $repo -Token first
    [System.IO.File]::WriteAllText($marker, 'next launch step reached')
}} catch [System.UnauthorizedAccessException] {{ $failed = $true }}
if (-not $failed -or [System.IO.File]::Exists($marker)) {{
    throw 'Receipt storage failure was treated as missing state.'
}}
'''
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
