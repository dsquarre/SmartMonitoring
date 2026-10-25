# Runs reference baselines (v1 code) for comparison with the H1 v1 results, in a visible window.
# Jobs: centralized, local_only (nonfl_baselines.py), fedavg_random, qfedavg_random, fedprox_random, linucb_fedavg (simulate_fl.py)
# Usage: powershell -ExecutionPolicy Bypass -Command "& .\run_baselines.ps1 -Jobs centralized,fedavg_random -Seeds 42,43,44,45,46"
param(
    [string[]]$Jobs = @("centralized"),
    [int[]]$Seeds = @(42, 43, 44, 45, 46)
)

$CondaActivate = "C:\Users\hlp06\miniconda3\Scripts\activate.bat"
$CondaEnv = "py310"
Set-Location $PSScriptRoot
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"

# Same FL configuration as the H1 v1 runs
$fl = "--data-dir datasets/iid -n 100 -k 20 -r 120 -e 5 -b 100"
$jobArgs = @{
    "centralized"    = "nonfl_baselines.py --mode centralized --data-dir datasets/iid --epochs 120"
    "local_only"     = "nonfl_baselines.py --mode local --data-dir datasets/iid --epochs 120 --eval-every 10"
    "fedavg_random"  = "simulate_fl.py $fl -s random -a fedavg"
    "qfedavg_random" = "simulate_fl.py $fl -s random -a qfedavg"
    "fedprox_random" = "simulate_fl.py $fl -s random -a fedprox"
    "linucb_fedavg"  = "simulate_fl.py $fl -s linucb -a fedavg"
}

foreach ($job in $Jobs) {
    if (-not $jobArgs.ContainsKey($job)) { Write-Host "Unknown job '$job' - skipping" -ForegroundColor Red; continue }
    foreach ($seed in $Seeds) {
        $outDir = "results/baselines/$job/s$seed"
        $log = "$outDir/run.log"
        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Completed Successfully" -Quiet)) {
            Write-Host "[$(Get-Date -Format 'HH:mm')] $outDir already completed - skipping" -ForegroundColor DarkGray
            continue
        }
        New-Item -ItemType Directory -Force $outDir | Out-Null
        $Host.UI.RawUI.WindowTitle = "Baseline $job - seed $seed"
        Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] ===== $job | seed $seed =====" -ForegroundColor Cyan
        $cmd = "call `"$CondaActivate`" $CondaEnv && python -u $($jobArgs[$job]) --seed $seed -o $outDir 2>&1"
        & cmd /c $cmd | Tee-Object -FilePath $log
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] finished (exit code $LASTEXITCODE)" -ForegroundColor Green
    }
}
Write-Host "`nAll baseline jobs done." -ForegroundColor Green
Read-Host "Press Enter to close"
