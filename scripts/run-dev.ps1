Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$scriptExitCode = 0

# run-dev: run PixelUp (a PySide6 GUI app) from source via uv. This is the fast
# dev-loop launcher; rebuild and run-built cover the frozen PyInstaller build.

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
$appId = $null

try {
    Set-Utf8Console
    Import-Module (Join-Path $scriptDir "launcher-runtime.psm1") -Force
    Require-Command uv

    Set-Location $repoDir

    Write-Step "Replacing any existing PixelUp runtime"
    Stop-RecordedRuntime

    Write-Step "Installing dependencies required for launch"
    Invoke-Native -FilePath "uv" -ArgumentList @("sync", "--extra", "dev")

    Write-Step "Starting PixelUp"
    # Forward any script arguments to the app, matching run-dev.command's `"$@"`.
    $devProcess = Start-RecordedRuntime -FilePath (Join-Path $repoDir ".venv/Scripts/python.exe") -ArgumentList (@("-m", "pixelup") + $args) -Wrapper
    $appId = Get-RecordedRuntimeId
    Write-Step "PixelUp is ready"
    $devProcess.WaitForExit()
    if ($devProcess.ExitCode -notin @(0, 130, -1073741510)) {
        throw "PixelUp development runtime failed with exit code $($devProcess.ExitCode)."
    }
}
catch {
    Write-Host ""
    Write-Host "pixelup run-dev failed: $($_.Exception.Message)" -ForegroundColor Red
    $scriptExitCode = 1
}
finally {
    # Closing this launcher closes the PixelUp it started, unless a later launcher has replaced it.
    if ($appId) {
        try { Stop-RecordedRuntime -OnlyId $appId } catch { Write-Host $_.Exception.Message }
    }
    Read-Host "Press Enter to close" | Out-Null
}

exit $scriptExitCode
