# pixelup Windows packaging: freeze PixelUp.exe (onedir) with PyInstaller, then emit
# dist\pixelup-<version>-setup.exe (Inno Setup installer) + dist\pixelup-<version>-win.zip
# (portable). Run by CI on windows-latest; iscc is pre-installed there. Per the
# app-release-conventions the packaging lives here so the release workflow just calls it.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Version = ([regex]::Match((Get-Content pyproject.toml -Raw), '(?m)^version = "(.*?)"')).Groups[1].Value

Remove-Item -Recurse -Force dist, build-pyinstaller -ErrorAction SilentlyContinue

# Freeze -> dist\PixelUp\PixelUp.exe (+ runtime). uv run --extra build auto-syncs deps.
uv run --extra build pyinstaller pixelup.spec --workpath build-pyinstaller --distpath dist --noconfirm

# Fail fast if freezing dropped a lazily-imported dependency (see gui._selftest).
# The exe is a windowed (GUI-subsystem) binary, so wait on it explicitly to get the
# exit code — PIXELUP_SELFTEST=1 makes it import the full stack and exit before any
# window is created.
$env:PIXELUP_SELFTEST = "1"
$proc = Start-Process -FilePath "dist\PixelUp\PixelUp.exe" -Wait -PassThru
Remove-Item Env:PIXELUP_SELFTEST
if ($proc.ExitCode -ne 0) { throw "Frozen self-test failed (exit $($proc.ExitCode))" }

# Portable: zip the onedir as-is.
Compress-Archive -Path dist\PixelUp\* -DestinationPath "dist\pixelup-$Version-win.zip" -Force

# Installer: Inno Setup. iscc is on PATH on windows-latest; fall back to its standard path.
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc "/DMyAppVersion=$Version" scripts\pixelup.iss

Get-ChildItem dist
