function Invoke-LauncherTransaction {
    param([string]$RepoDir, [scriptblock]$Action)
    $normalizedPath = [System.IO.Path]::GetFullPath($RepoDir).ToUpperInvariant()
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $hasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($normalizedPath))
        $name = 'Local\PixelUpLauncher-' + ([BitConverter]::ToString($digest)).Replace('-', '')
    } finally { $hasher.Dispose() }
    $mutex = [System.Threading.Mutex]::new($false, $name)
    $acquired = $false
    try {
        try { $acquired = $mutex.WaitOne() }
        catch [System.Threading.AbandonedMutexException] { $acquired = $true }
        & $Action
    } finally {
        if ($acquired) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

function Write-AtomicRuntimeText {
    param([string]$Path, [string]$Text)
    $temporaryFile = "$Path.$PID.$([guid]::NewGuid().ToString('N'))"
    [System.IO.File]::WriteAllText($temporaryFile, $Text, [System.Text.UTF8Encoding]::new($false))
    if ([System.IO.File]::Exists($Path)) {
        [System.IO.File]::Replace($temporaryFile, $Path, $null)
    } else {
        [System.IO.File]::Move($temporaryFile, $Path)
    }
}

function Get-LauncherOwnerFile {
    param([string]$RepoDir)
    Join-Path (Join-Path $RepoDir '.git') 'launcher-runtime-owner'
}

function Test-OwnerToken {
    param([string]$Token, [string]$RepoDir)
    $ownerFile = Get-LauncherOwnerFile $RepoDir
    try {
        return [System.IO.File]::ReadAllText($ownerFile).Trim() -ceq $Token
    } catch [System.IO.FileNotFoundException] { return $false }
    catch [System.IO.DirectoryNotFoundException] { return $false }
}

function Claim-LauncherRuntime {
    param([string]$Token, [string]$RepoDir)
    Invoke-LauncherTransaction $RepoDir {
        Write-AtomicRuntimeText (Get-LauncherOwnerFile $RepoDir) "$Token`n"
    }
}

function Test-LauncherRuntimeOwner {
    param([string]$Token, [string]$RepoDir)
    Invoke-LauncherTransaction $RepoDir { Test-OwnerToken $Token $RepoDir }
}

function Release-LauncherRuntime {
    param([string]$Token, [string]$RepoDir)
    Invoke-LauncherTransaction $RepoDir {
        if (Test-OwnerToken $Token $RepoDir) {
            [System.IO.File]::Delete((Get-LauncherOwnerFile $RepoDir))
        }
    }
}

function Get-RuntimeReceipt {
    param([string]$RepoDir)
    $receiptFile = Join-Path (Join-Path $RepoDir '.git') 'launcher-runtime-process'
    try {
        $text = [System.IO.File]::ReadAllText($receiptFile)
    } catch [System.IO.FileNotFoundException] { return $null }
    catch [System.IO.DirectoryNotFoundException] { return $null }
    try {
        $receipt = $text | ConvertFrom-Json
        if ($receipt.ProcessId -gt 0 -and $receipt.StartTicks -gt 0 -and $receipt.Token) {
            return $receipt
        }
    } catch {}
    return $null
}

function Get-CapturedRuntime {
    param([int]$RuntimeId, [long]$StartTicks)
    $process = $null
    try {
        $process = [System.Diagnostics.Process]::GetProcessById($RuntimeId)
        # Acquire the handle before checking identity; stopping then targets this
        # exact process even if Windows later recycles its numeric PID.
        $null = $process.Handle
        if ($process.StartTime.ToUniversalTime().Ticks -eq $StartTicks -and -not $process.HasExited) {
            return $process
        }
    } catch {}
    if ($null -ne $process) { $process.Dispose() }
    return $null
}

function Stop-CapturedRuntime {
    param([int]$RuntimeId, [long]$StartTicks)
    $process = Get-CapturedRuntime $RuntimeId $StartTicks
    if ($null -eq $process) { return }
    try {
        $null = $process.CloseMainWindow()
        if (-not $process.WaitForExit(5000)) {
            $process.Kill()
            if (-not $process.WaitForExit(5000)) { throw 'The recorded runtime did not stop.' }
        }
    } catch [System.InvalidOperationException] {
        if (-not $process.HasExited) { throw }
    } finally { $process.Dispose() }
}

function Stop-OwnedRuntime {
    param([string]$Label, [string]$RepoDir, [string]$Token, [string]$ReceiptToken = '')
    $receipt = Invoke-LauncherTransaction $RepoDir {
        if (Test-OwnerToken $Token $RepoDir) {
            $saved = Get-RuntimeReceipt $RepoDir
            if ($null -ne $saved -and (-not $ReceiptToken -or $saved.Token -ceq $ReceiptToken)) {
                return $saved
            }
        }
    }
    if ($null -eq $receipt) { return }
    Write-Host "Stopping the existing $Label runtime (pid $($receipt.ProcessId))."
    Stop-CapturedRuntime $receipt.ProcessId $receipt.StartTicks
    Invoke-LauncherTransaction $RepoDir {
        $current = Get-RuntimeReceipt $RepoDir
        if ((Test-OwnerToken $Token $RepoDir) -and $null -ne $current -and
            $current.Token -ceq $receipt.Token -and $current.ProcessId -eq $receipt.ProcessId -and
            $current.StartTicks -eq $receipt.StartTicks) {
            [System.IO.File]::Delete((Join-Path (Join-Path $RepoDir '.git') 'launcher-runtime-process'))
        }
    }
}

function ConvertTo-RuntimeArgument {
    param([AllowEmptyString()][string]$Value)
    # Start-Process joins ArgumentList into a native Windows command line.
    # Quote each argument, doubling backslashes only before quotes and the end.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Start-OwnedRuntime {
    param([string]$RepoDir, [string]$Token, [string]$FilePath, [string[]]$ArgumentList = @())
    # Spawn and publish together under the short metadata lock. Waiting and
    # stopping always happen outside it.
    return Invoke-LauncherTransaction $RepoDir {
        if (-not (Test-OwnerToken $Token $RepoDir)) { return $null }
        $parameters = @{ FilePath = $FilePath; PassThru = $true; NoNewWindow = $true }
        if ($ArgumentList.Count -gt 0) {
            $parameters.ArgumentList = ($ArgumentList | ForEach-Object { ConvertTo-RuntimeArgument $_ }) -join ' '
        }
        $process = Start-Process @parameters
        try {
            $null = $process.Handle
            $receipt = @{
                Token = $Token
                ProcessId = $process.Id
                StartTicks = $process.StartTime.ToUniversalTime().Ticks
            }
            $receiptFile = Join-Path (Join-Path $RepoDir '.git') 'launcher-runtime-process'
            Write-AtomicRuntimeText $receiptFile ($receipt | ConvertTo-Json -Compress)
            return $process
        } catch {
            # This is our direct launch result; it cannot refer to an independent process.
            if (-not $process.HasExited) { $process.Kill() }
            $process.Dispose()
            throw
        }
    }
}

function Wait-OwnedRuntime {
    param([string]$Label, [System.Diagnostics.Process]$Process, [int]$TimeoutSeconds = 30)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $Process.Refresh()
        if (-not $Process.HasExited) { return }
        break
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "$Label did not start within ${TimeoutSeconds}s."
}

Export-ModuleMember -Function Claim-LauncherRuntime, Test-LauncherRuntimeOwner, Release-LauncherRuntime, Stop-OwnedRuntime, Start-OwnedRuntime, Wait-OwnedRuntime
