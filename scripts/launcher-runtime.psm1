# Shared by the Windows launchers (run-dev, run-built, rebuild): one PixelUp
# runtime per repository. The launcher that starts PixelUp records its process ID
# and executable under .git; the next launcher asks that process to quit only
# while it still runs that executable, and stops with a message rather than
# force-closing it when it stays open. PixelUp started any other way is left to
# its single-writer admission, which refuses a second copy on the same data folder.

$script:RecordFile = Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) '.git') 'launcher-runtime-process'

function Get-ExecutableOf {
    param([int]$ProcessId)
    (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue).Path
}

# The recorded process ID, returned only while that process still runs the recorded executable.
function Get-RecordedRuntimeId {
    if (-not (Test-Path -LiteralPath $script:RecordFile)) { return $null }
    $lines = @(Get-Content -LiteralPath $script:RecordFile)
    $id = 0
    if ($lines.Count -lt 2 -or -not [int]::TryParse($lines[0], [ref]$id) -or $id -le 0 -or -not $lines[1]) { return $null }
    $current = Get-ExecutableOf $id
    if ($current -and $current -ieq $lines[1]) { return $id }
    return $null
}

# Ask the recorded PixelUp to quit as a user would, so it can still ask about
# running work. Given a process ID, only while that is the recorded one.
function Stop-RecordedRuntime {
    param([int]$OnlyId = 0)
    $id = Get-RecordedRuntimeId
    if ($null -eq $id -or ($OnlyId -and $id -ne $OnlyId)) { return }
    Write-Host "Asking the running PixelUp (pid $id) to quit."
    $process = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($process) { $null = $process.CloseMainWindow() }
    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ((Get-RecordedRuntimeId) -ne $id) { return }
        Start-Sleep -Milliseconds 100
    }
    throw "PixelUp is still open, perhaps finishing or asking about running work. Close it, then run this launcher again."
}

function ConvertTo-RuntimeArgument {
    param([AllowEmptyString()][string]$Value)
    # Start-Process joins ArgumentList into a native Windows command line.
    # Quote each argument, doubling backslashes only before quotes and the end.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

# Start the executable and record PixelUp's process. With -Wrapper, the started
# process is a launcher that runs PixelUp as its child (a virtual environment's
# python.exe starts the base interpreter that way), and the child is recorded.
# Returns the started process.
function Start-RecordedRuntime {
    param([string]$FilePath, [string[]]$ArgumentList = @(), [switch]$Wrapper)
    $parameters = @{ FilePath = $FilePath; PassThru = $true; NoNewWindow = $true }
    if ($ArgumentList.Count -gt 0) {
        $parameters.ArgumentList = ($ArgumentList | ForEach-Object { ConvertTo-RuntimeArgument $_ }) -join ' '
    }
    $process = Start-Process @parameters
    $app = $process
    if ($Wrapper) {
        $deadline = [DateTime]::UtcNow.AddSeconds(30)
        do {
            $child = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($process.Id)" | Select-Object -First 1
            if ($child) { $app = Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue; break }
            Start-Sleep -Milliseconds 100
        } while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline)
    }
    $executable = if ($app) { $app.Path } else { $null }
    if (-not $executable) { throw "PixelUp did not start." }
    Set-Content -LiteralPath $script:RecordFile -Value @($app.Id, $executable) -Encoding utf8
    return $process
}

Export-ModuleMember -Function Stop-RecordedRuntime, Start-RecordedRuntime, Get-RecordedRuntimeId
