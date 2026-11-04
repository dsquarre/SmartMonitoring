# Windows counterpart of stop.sh: stops the clients, FL server and Celery worker started by start_windows.ps1.
# Redis is left running, as with stop.sh; pass -StopRedis to stop its container (smartmon-redis) too.
param([switch]$StopRedis)
$Root = $PSScriptRoot
$RunDir = Join-Path $Root "run"

function Stop-Tree($procId, $label) {
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        cmd /c "taskkill /F /T /PID $procId >nul 2>&1"
        if ($label) { Write-Host "${label}: stopped (pid $procId)" }
    }
}
function Stop-PidFile($file, $label) {
    $path = Join-Path $RunDir $file
    if (Test-Path $path) {
        foreach ($p in Get-Content $path) { if ($p.Trim()) { Stop-Tree $p.Trim() $label } }
        Remove-Item $path -Force
    }
}

Write-Host "Stopping clients..."
Stop-PidFile "clients.pid" $null
Stop-PidFile "server.pid" "Server"
Stop-PidFile "celery.pid" "Celery"
if ($StopRedis) { cmd /c "docker stop smartmon-redis >nul 2>&1"; Write-Host "Redis: stopped (container smartmon-redis)" }
Write-Host "All stopped."
