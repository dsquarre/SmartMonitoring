# Pauses all FL training while a game is running and resumes it when the game closes.
# Windows cannot cap one process's GPU share, so pausing is the only way to give the game the whole GPU.
# Paused runs keep all their progress (they are frozen, not killed) and continue where they stopped.
# Usage: powershell -ExecutionPolicy Bypass -File gpu_governor.ps1            (keep this window open)
param(
    [string]$GamePattern = '^(Cities2|Cities)$',   # process names to watch (Cities: Skylines II = Cities2.exe)
    [int]$PollSeconds = 10
)

Add-Type -Namespace Native -Name Proc -MemberDefinition @"
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr processHandle);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr processHandle);
"@

$Host.UI.RawUI.WindowTitle = "GPU governor (pauses training while gaming)"
$paused = @{}   # pid -> description

function Get-TrainingProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'simulate_fl\.py|nonfl_baselines\.py' }
}

function Set-Paused([int]$procId, [bool]$pause) {
    try {
        $p = [System.Diagnostics.Process]::GetProcessById($procId)
        if ($pause) { [void][Native.Proc]::NtSuspendProcess($p.Handle) } else { [void][Native.Proc]::NtResumeProcess($p.Handle) }
        return $true
    } catch { return $false }
}

function Resume-All {
    foreach ($procId in @($paused.Keys)) {
        if (Set-Paused $procId $false) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] resumed  $($paused[$procId])" -ForegroundColor Green }
        $paused.Remove($procId)
    }
}

Write-Host "Watching for game processes matching '$GamePattern' every $PollSeconds s. Close this window to stop (training is resumed first)." -ForegroundColor Cyan
try {
    while ($true) {
        $game = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match $GamePattern } | Select-Object -First 1
        if ($game) {
            foreach ($t in Get-TrainingProcesses) {
                if (-not $paused.ContainsKey([int]$t.ProcessId)) {
                    $desc = ($t.CommandLine -replace '.*python(\.exe)?"?\s+(-u\s+)?', '')
                    if (Set-Paused ([int]$t.ProcessId) $true) {
                        $paused[[int]$t.ProcessId] = $desc
                        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $($game.ProcessName) running -> paused  $desc" -ForegroundColor Yellow
                    }
                }
            }
        } elseif ($paused.Count -gt 0) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] game closed -> resuming training at full speed" -ForegroundColor Cyan
            Resume-All
        }
        # forget processes that have exited
        foreach ($procId in @($paused.Keys)) { if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { $paused.Remove($procId) } }
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    Resume-All
}
