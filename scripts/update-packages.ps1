Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$scriptExitCode = 0

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

try {
    Set-Utf8Console
    Require-Command uv

    Set-Location $repoDir

    Write-Step "Updating locked packages within declared constraints"
    Invoke-Native -FilePath "uv" -ArgumentList @("lock", "--upgrade")

    Write-Step "Installing updated dependencies"
    Invoke-Native -FilePath "uv" -ArgumentList @("sync", "--extra", "dev")

    Write-Step "Running lint"
    Invoke-Native -FilePath "uv" -ArgumentList @("run", "--extra", "dev", "ruff", "check", ".")

    Write-Step "Running tests"
    Invoke-Native -FilePath "uv" -ArgumentList @("run", "--extra", "dev", "pytest", "-q")

    Write-Step "Checking lockfile"
    Invoke-Native -FilePath "uv" -ArgumentList @("lock", "--check")
}
catch {
    Write-Host ""
    Write-Host "pixelup update-packages failed: $($_.Exception.Message)" -ForegroundColor Red
    $scriptExitCode = 1
}
finally {
    Read-Host "Press Enter to close" | Out-Null
}

exit $scriptExitCode
