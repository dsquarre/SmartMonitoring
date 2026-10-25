# Runs H1 seeds sequentially for one selector arm on the GPU, in a visible window with live output.
# Usage: powershell -ExecutionPolicy Bypass -File run_h1_seeds.ps1 -Selector hierarchical -OutRoot results/h1_custom_fixed
param(
    [Parameter(Mandatory = $true)][string]$Selector,
    [Parameter(Mandatory = $true)][string]$OutRoot,
    [int]$FirstSeed = 42,
    [int]$LastSeed = 71,
    [int]$LocalEpochs = 5,
    # Extra simulate_fl.py flags, e.g. "--gamma-meta 1.0 --gamma-sub 1.0"
    [string]$ExtraArgs = "",
    # Simulator script to run (v1: simulate_fl.py)
    [string]$Script = "simulate_fl.py"
)

# conda env py310 (TF 2.10.1 + CUDA 11.2/cuDNN 8.1 in Library\bin); activation puts the CUDA DLLs on PATH
$CondaActivate = "C:\Users\hlp06\miniconda3\Scripts\activate.bat"
$CondaEnv = "py310"
Set-Location $PSScriptRoot

# Set here (inherited by cmd/python) rather than with cmd's `set X=v &&`, which keeps a trailing space in the value
$env:TF_CPP_MIN_LOG_LEVEL = "3"
# Lets both arms share the GPU instead of the first process reserving all VRAM
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

# Below-normal priority (inherited by python): leaves the CPU responsive for browsing
(Get-Process -Id $PID).PriorityClass = "BelowNormal"
$Host.UI.RawUI.WindowTitle = "H1 $Selector seeds $FirstSeed-$LastSeed"

foreach ($seed in $FirstSeed..$LastSeed) {
    $outDir = "$OutRoot/s$seed"
    $log = "$outDir/run.log"
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Completed Successfully" -Quiet)) {
        Write-Host "[$(Get-Date -Format 'HH:mm')] Seed $seed already completed - skipping" -ForegroundColor DarkGray
        continue
    }
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $Host.UI.RawUI.WindowTitle = "H1 $Selector - seed $seed ($FirstSeed-$LastSeed)"
    Write-Host "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] ===== $Selector seed $seed =====" -ForegroundColor Cyan

    # -b 100: one memory cleanup per phase instead of per client (gc.collect dominated runtime)
    $cmd = "call `"$CondaActivate`" $CondaEnv && " +
           "python -u $Script --data-dir datasets/iid -n 100 -k 20 -r 120 -e $LocalEpochs -b 100 --seed $seed -s $Selector $ExtraArgs -o $outDir 2>&1"
    & cmd /c $cmd | Tee-Object -FilePath $log

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] Seed $seed finished (exit code $LASTEXITCODE)" -ForegroundColor Green
}

Write-Host "`nAll seeds done for $Selector." -ForegroundColor Green
Read-Host "Press Enter to close"
