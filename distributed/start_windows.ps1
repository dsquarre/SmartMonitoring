# Windows counterpart of start.sh for this PC: runs the same stack (Redis, Celery worker, FL server, clients,
# dashboard) with the conda env py310 instead of a .venv. start.sh/stop.sh are unchanged for Linux.
#
#   start_windows.bat                                  # settings from .env
#   start_windows.bat -Clients 5 -Rounds 5 -K 3        # override FL_N / FL_ROUNDS / FL_K for this run
#   stop_windows.bat                                   # stop everything (Redis keeps running, as with stop.sh)
#
# Redis runs in a Docker container (smartmon-redis); Docker Desktop is started if needed.
# Logs: run\server.log, run\celery.log, client\run_logs\client_N.log
param(
    [int]$Clients = 0,
    [int]$Rounds = 0,
    [int]$K = 0,
    [string]$Python = "$env:USERPROFILE\miniconda3\envs\py310\python.exe"
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$RunDir = Join-Path $Root "run"
New-Item -ItemType Directory -Force $RunDir | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Root "client\run_logs") | Out-Null

if (-not (Test-Path $Python)) { throw "Python not found at $Python (pass -Python <path to python.exe>)" }
# What `conda activate` adds to PATH: without the env's Library\bin, TensorFlow cannot load the CUDA/cuDNN DLLs and
# silently runs on the CPU.
$EnvDir = Split-Path $Python -Parent
$env:PATH = (@($EnvDir, "$EnvDir\Library\mingw-w64\bin", "$EnvDir\Library\usr\bin", "$EnvDir\Library\bin",
               "$EnvDir\Scripts", "$EnvDir\bin") -join ";") + ";" + $env:PATH

# --- 0. Load .env into this process's environment (children inherit it) -------------------------------------
$EnvFile = Join-Path $Root ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "No .env found - creating one from .env.example (edit it to change settings)."
    Copy-Item (Join-Path $Root ".env.example") $EnvFile
}
foreach ($line in Get-Content $EnvFile) {
    $t = $line.Trim()
    if ($t -eq "" -or $t.StartsWith("#") -or -not $t.Contains("=")) { continue }
    $name, $value = $t.Split("=", 2)
    $value = $value.Trim()
    if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") { $value = $Matches[1] } else { $value = ($value -replace '\s+#.*$', '') }
    Set-Item -Path "env:$($name.Trim())" -Value $value
}
if ($Clients -gt 0) { $env:FL_N = "$Clients" }
if ($Rounds -gt 0) { $env:FL_ROUNDS = "$Rounds" }
if ($K -gt 0) { $env:FL_K = "$K" }
function Get-Setting($name, $default) { $v = [Environment]::GetEnvironmentVariable($name); if ($v) { $v } else { $default } }
$FL_N = [int](Get-Setting "FL_N" "10")
$SERVER_HOST = Get-Setting "SERVER_HOST" "http://127.0.0.1:8000"
$SERVER_IP = Get-Setting "SERVER_IP" "127.0.0.1:8000"
$PASSWORD = Get-Setting "PASSWORD" "P7h1!quiBO0no96"
$BIND_HOST = Get-Setting "HOST" "0.0.0.0"
$PORT = Get-Setting "PORT" "8000"
$REDIS_URL = Get-Setting "REDIS_URL" "redis://localhost:6379/0"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"   # several TensorFlow processes share the one GPU
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:PYTHONUNBUFFERED = "1"
Write-Host "Settings: FL_N=$FL_N FL_K=$env:FL_K FL_ROUNDS=$env:FL_ROUNDS aggregator=$env:FL_AGGREGATOR selector=$env:FL_SELECTOR"

function Test-Running($pidFile) {
    if (-not (Test-Path $pidFile)) { return $false }
    $p = Get-Content $pidFile -TotalCount 1
    return [bool](Get-Process -Id $p -ErrorAction SilentlyContinue)
}
function Start-Logged($workDir, $arguments, $log) {
    # cmd /s /c keeps one log for stdout+stderr (like start.sh); stop_windows kills the cmd process tree.
    $cmdLine = "/s /c `"`"$Python`" $arguments > `"$log`" 2>&1`""
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdLine -WorkingDirectory $workDir -WindowStyle Hidden -PassThru
    return $p.Id
}
function Test-Port($hostName, $port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.Connect($hostName, $port); return $true } catch { return $false } finally { $c.Close() }
}

# --- 1. Redis (Docker container) -----------------------------------------------------------------------------
$redisUri = [Uri]$REDIS_URL
$redisHost = $redisUri.Host; $redisPort = $redisUri.Port
if (Test-Port $redisHost $redisPort) {
    Write-Host "Redis:   already running"
} else {
    cmd /c "docker info >nul 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host -NoNewline "Docker:  starting Docker Desktop"
        Start-Process "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep -Seconds 2; Write-Host -NoNewline "."
            cmd /c "docker info >nul 2>&1"; if ($LASTEXITCODE -eq 0) { break }
        }
        Write-Host ""
        cmd /c "docker info >nul 2>&1"; if ($LASTEXITCODE -ne 0) { throw "Docker did not start; start Docker Desktop and run again." }
    }
    Write-Host "Redis:   starting container smartmon-redis..."
    $exists = (cmd /c "docker ps -a --filter name=^smartmon-redis$ --format {{.Names}}")
    if ($exists -eq "smartmon-redis") { cmd /c "docker start smartmon-redis >nul" }
    else { cmd /c "docker run -d --name smartmon-redis -p ${redisPort}:6379 redis:7-alpine >nul" }
    for ($i = 0; $i -lt 20; $i++) { if (Test-Port $redisHost $redisPort) { break }; Start-Sleep -Milliseconds 500 }
    if (-not (Test-Port $redisHost $redisPort)) { throw "Redis failed to start." }
}
# Flush stale state only when no server is running (same reason as start.sh: a stale fl:coordinator_lock).
if (-not (Test-Running (Join-Path $RunDir "server.pid"))) {
    Write-Host "Redis:   flushing stale state for a clean start..."
    & $Python -c "import redis, sys; redis.from_url(sys.argv[1]).flushall()" $REDIS_URL
}

# --- 2. Dashboard build (first run only) ---------------------------------------------------------------------
if (-not (Test-Path (Join-Path $Root "server\static\index.html"))) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "Dashboard: building (first run)..."
        Push-Location (Join-Path $Root "dashboard"); npm install --silent; npm run build; Pop-Location
    } else { Write-Host "Dashboard: npm not found, skipping build - the API still works but '/' won't serve the UI." }
} else { Write-Host "Dashboard: already built" }

# --- 3. Celery worker (solo pool: the default prefork pool does not run on Windows) -------------------------
$celeryPid = Join-Path $RunDir "celery.pid"
if (Test-Running $celeryPid) { Write-Host "Celery:  already running (pid $(Get-Content $celeryPid))" }
else {
    Write-Host "Celery:  starting..."
    Start-Logged (Join-Path $Root "server") "-m celery -A celery_app.celery_app worker --loglevel=info --pool=solo" (Join-Path $RunDir "celery.log") | Set-Content $celeryPid
    Start-Sleep -Seconds 3
}

# --- 4. FL server ----------------------------------------------------------------------------------------------
$serverPid = Join-Path $RunDir "server.pid"
if (Test-Running $serverPid) { Write-Host "Server:  already running (pid $(Get-Content $serverPid))" }
else {
    Write-Host "Server:  starting..."
    Start-Logged (Join-Path $Root "server") "-m uvicorn main:app --host $BIND_HOST --port $PORT" (Join-Path $RunDir "server.log") | Set-Content $serverPid
}
Write-Host -NoNewline "Server:  waiting for it to come up"
$up = $false
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing "$SERVER_HOST/api/status" -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } } catch { }
    Write-Host -NoNewline "."; Start-Sleep -Seconds 1
}
if ($up) { Write-Host " - up." } else { Write-Host " - not up yet; see run\server.log" }

# --- 5. Clients from ..\datasets\non_iid (partitions 0..FL_N-1; 90-99 are the server's validation set) -------
$DataDir = if ($env:CLIENT_DATA_DIR) { $env:CLIENT_DATA_DIR } else { Join-Path $Root "..\datasets\non_iid" }
$DataDir = (Resolve-Path $DataDir).Path
if (-not (Test-Path (Join-Path $Root "server\server_val.npz"))) {
    & $Python (Join-Path $Root "scripts\make_server_val.py") --src $DataDir --clients 90-99 --out (Join-Path $Root "server\server_val.npz")
}
$available = [Math]::Min(90, (Get-ChildItem $DataDir -Filter "client_*.npz").Count)
$n = $FL_N
if ($n -gt $available) { Write-Host "Clients: FL_N=$FL_N but only $available dataset file(s) in $DataDir - launching $available."; $n = $available }
$clientsPid = Join-Path $RunDir "clients.pid"
Set-Content $clientsPid -Value $null
Write-Host "Clients: launching $n (client_0..client_$($n - 1))..."
for ($i = 0; $i -lt $n; $i++) {
    $data = Join-Path $DataDir "client_$i.npz"
    $log = Join-Path $Root "client\run_logs\client_$i.log"
    Start-Logged (Join-Path $Root "client") "main.py -d `"$data`" -s $SERVER_IP -p `"$PASSWORD`" -c client_$i" $log | Add-Content $clientsPid
}

# --- 6. Dashboard ------------------------------------------------------------------------------------------------
Write-Host "Dashboard: opening $SERVER_HOST/ ..."
Start-Process "$SERVER_HOST/"
Write-Host ""
Write-Host "All started. Logs: run\server.log, run\celery.log, client\run_logs\client_*.log"
Write-Host "Stop everything with: stop_windows.bat"
