Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$scriptExitCode = 0

# rebuild: freeze a fresh PixelUp.exe (onedir) with PyInstaller, then launch it —
# the same frozen output the release pipeline builds (scripts/package.ps1 wraps it
# into the setup.exe/.zip; rebuild stops at the launchable exe, no installer).
# Slow; run after changing source. run-built is the no-build fast path afterward.

function Set-Utf8Console {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $global:OutputEncoding = $utf8NoBom
    if (Get-Command chcp.com -ErrorAction SilentlyContinue) {
        & chcp.com 65001 > $null
        $null = $LASTEXITCODE
    }
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [int[]]$AllowedExitCodes = @(0)
    )

    & $FilePath @ArgumentList
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    if ($AllowedExitCodes -notcontains $exitCode) {
        throw "Command failed with exit code ${exitCode}: $FilePath $($ArgumentList -join ' ')"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Split-Path -Parent $scriptDir
$distDir = Join-Path $repoDir "dist"
$workDir = Join-Path $repoDir "build-pyinstaller"
$exePath = Join-Path $repoDir "dist/PixelUp/PixelUp.exe"

try {
    Set-Utf8Console
    Require-Command uv

    Set-Location $repoDir

    Write-Step "Removing stale build output"
    # Clear output first so a build that fails to emit a file cannot be masked by a
    # leftover artifact from a previous run.
    if (Test-Path $distDir) { Remove-Item -Recurse -Force $distDir }
    if (Test-Path $workDir) { Remove-Item -Recurse -Force $workDir }

    Write-Step "Freezing PixelUp.exe (uv installs the build deps, then PyInstaller runs)"
    Invoke-Native -FilePath "uv" -ArgumentList @(
        "run", "--extra", "build", "pyinstaller", "pixelup.spec",
        "--workpath", "build-pyinstaller", "--distpath", "dist", "--noconfirm"
    )

    # Fail fast if freezing dropped a lazily-imported dependency (see gui._selftest).
    # The exe is windowed, so wait on it to read the exit code; PIXELUP_SELFTEST=1
    # imports the full stack and exits before any window is created.
    Write-Step "Self-testing the frozen bundle"
    $env:PIXELUP_SELFTEST = "1"
    $proc = Start-Process -FilePath $exePath -Wait -PassThru
    Remove-Item Env:PIXELUP_SELFTEST
    if ($proc.ExitCode -ne 0) { throw "Frozen self-test failed (exit $($proc.ExitCode))" }

    Write-Step "Launching PixelUp"
    # GUI app: launch non-blocking via Start-Process (the Windows counterpart to
    # macOS `open`), so the console does not wait on the app's lifetime.
    Start-Process -FilePath $exePath
}
catch {
    Write-Host ""
    Write-Host "pixelup rebuild failed: $($_.Exception.Message)" -ForegroundColor Red
    $scriptExitCode = 1
}
finally {
    Read-Host "Press Enter to close" | Out-Null
}

exit $scriptExitCode
