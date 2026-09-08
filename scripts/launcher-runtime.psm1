function Get-LauncherOwnerFile {
    param([Parameter(Mandatory = $true)][string]$RepoDir)
    return Join-Path (Join-Path $RepoDir '.git') 'launcher-runtime-owner'
}

function Claim-LauncherRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$Token,
        [Parameter(Mandatory = $true)][string]$RepoDir
    )
    $ownerFile = Get-LauncherOwnerFile -RepoDir $RepoDir
    $temporaryFile = "$ownerFile.$PID.$([guid]::NewGuid().ToString('N'))"
    [System.IO.File]::WriteAllText($temporaryFile, "$Token`n", [System.Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporaryFile -Destination $ownerFile -Force
}

function Test-LauncherRuntimeOwner {
    param(
        [Parameter(Mandatory = $true)][string]$Token,
        [Parameter(Mandatory = $true)][string]$RepoDir
    )
    $ownerFile = Get-LauncherOwnerFile -RepoDir $RepoDir
    return (Test-Path -LiteralPath $ownerFile) -and
        ((Get-Content -LiteralPath $ownerFile -Raw).Trim() -ceq $Token)
}

function Release-LauncherRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$Token,
        [Parameter(Mandatory = $true)][string]$RepoDir
    )
    if (Test-LauncherRuntimeOwner -Token $Token -RepoDir $RepoDir) {
        Remove-Item -LiteralPath (Get-LauncherOwnerFile -RepoDir $RepoDir) -Force -ErrorAction SilentlyContinue
    }
}

function Get-OwnedRuntimeProcess {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('dotnet', 'python')][string]$Kind,
        [Parameter(Mandatory = $true)][string]$RepoDir,
        [string]$ProjectFile,
        [Parameter(Mandatory = $true)][string]$ExecutableName,
        [Parameter(Mandatory = $true)][string]$BuiltExecutable
    )

    $repoPrefix = [System.IO.Path]::GetFullPath($RepoDir).TrimEnd('\') + '\'
    $builtPath = [System.IO.Path]::GetFullPath($BuiltExecutable)
    Get-CimInstance Win32_Process | Where-Object {
        $path = if ($_.ExecutablePath) { [System.IO.Path]::GetFullPath($_.ExecutablePath) } else { '' }
        $command = if ($_.CommandLine) { $_.CommandLine } else { '' }
        if ($path -eq $builtPath) { return $true }
        foreach ($launcherName in @('run-dev.ps1', 'run-built.ps1', 'rebuild.ps1')) {
            $launcherPath = Join-Path (Join-Path $RepoDir 'scripts') $launcherName
            if ($command.IndexOf($launcherPath, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                return $true
            }
        }

        if ($Kind -eq 'dotnet') {
            return (($path.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
                    [System.IO.Path]::GetFileName($path) -ieq "$ExecutableName.exe") -or
                    ($ProjectFile -and $command.IndexOf($ProjectFile, [System.StringComparison]::OrdinalIgnoreCase) -ge 0))
        }

        $venvPrefix = Join-Path $repoPrefix '.venv'
        return (($path.StartsWith($venvPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
                $command.IndexOf($ExecutableName, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -or
                ($command.IndexOf($RepoDir, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $command.IndexOf('uv run', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $command.IndexOf($ExecutableName, [System.StringComparison]::OrdinalIgnoreCase) -ge 0))
    }
}

function Stop-OwnedRuntime {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('dotnet', 'python')][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$RepoDir,
        [string]$ProjectFile,
        [Parameter(Mandatory = $true)][string]$ExecutableName,
        [Parameter(Mandatory = $true)][string]$BuiltExecutable
    )

    $owned = @(Get-OwnedRuntimeProcess -Kind $Kind -RepoDir $RepoDir -ProjectFile $ProjectFile -ExecutableName $ExecutableName -BuiltExecutable $BuiltExecutable |
        Where-Object { $_.ProcessId -ne $PID })
    if ($owned.Count -eq 0) { return }
    $originalProcessIds = @($owned.ProcessId)

    Write-Host "Stopping the existing $Label runtime (pid $($owned.ProcessId -join ', '))."
    foreach ($item in $owned) {
        try {
            $process = Get-Process -Id $item.ProcessId -ErrorAction Stop
            $null = $process.CloseMainWindow()
        } catch {}
    }

    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    do {
        Start-Sleep -Milliseconds 100
        $remainingProcessIds = @($originalProcessIds | Where-Object {
            Get-Process -Id $_ -ErrorAction SilentlyContinue
        })
    } while ($remainingProcessIds.Count -gt 0 -and [DateTime]::UtcNow -lt $deadline)

    $owned = @(Get-OwnedRuntimeProcess -Kind $Kind -RepoDir $RepoDir -ProjectFile $ProjectFile -ExecutableName $ExecutableName -BuiltExecutable $BuiltExecutable |
        Where-Object { $originalProcessIds -contains $_.ProcessId })
    foreach ($item in $owned) {
        Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Wait-OwnedRuntime {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('dotnet', 'python')][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$RepoDir,
        [string]$ProjectFile,
        [Parameter(Mandatory = $true)][string]$ExecutableName,
        [Parameter(Mandatory = $true)][string]$BuiltExecutable,
        [int]$TimeoutSeconds = 30
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $repoPrefix = [System.IO.Path]::GetFullPath($RepoDir).TrimEnd('\') + '\'
        $builtPath = [System.IO.Path]::GetFullPath($BuiltExecutable)
        $owned = @(Get-OwnedRuntimeProcess -Kind $Kind -RepoDir $RepoDir -ProjectFile $ProjectFile -ExecutableName $ExecutableName -BuiltExecutable $BuiltExecutable |
            Where-Object {
                if ($_.ProcessId -eq $PID) { return $false }
                $path = if ($_.ExecutablePath) { [System.IO.Path]::GetFullPath($_.ExecutablePath) } else { '' }
                if ($path -eq $builtPath) { return $true }
                if ($Kind -eq 'dotnet') {
                    return $path.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
                        [System.IO.Path]::GetFileName($path) -ieq "$ExecutableName.exe"
                }
                $command = if ($_.CommandLine) { $_.CommandLine } else { '' }
                return $path.StartsWith((Join-Path $repoPrefix '.venv'), [System.StringComparison]::OrdinalIgnoreCase) -and
                    $command.IndexOf($ExecutableName, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                    $command.IndexOf('uv run', [System.StringComparison]::OrdinalIgnoreCase) -lt 0
            })
        if ($owned.Count -gt 0) { return }
        Start-Sleep -Milliseconds 100
    } while ([DateTime]::UtcNow -lt $deadline)

    throw "$Label did not start within ${TimeoutSeconds}s."
}

Export-ModuleMember -Function Claim-LauncherRuntime, Test-LauncherRuntimeOwner, Release-LauncherRuntime, Stop-OwnedRuntime, Wait-OwnedRuntime
