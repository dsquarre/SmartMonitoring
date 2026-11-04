#!/usr/bin/env bash
# Writes experiments/jobs_tests.txt: the test runs of ecg_protocol.md section 2, with the calibrated settings.
#   bash experiments/make_test_jobs.sh <ft_lr_scale from calibration step 4>
# P1c (random_mix) needs the selector's FedYogi rate from P1a and is written separately afterwards.
set -euo pipefail
FT="$1"
BASE="--rounds 60 --seeds 42-46 --normalize record --lr 0.1 --server-lr 0.01 --ft-lr-scale $FT"
OUT=experiments/jobs_tests.txt
{
  echo "# ECG tests (ecg_protocol.md section 2). Calibrated: record normalisation, client lr 0.1, FedYogi eta 0.01,"
  echo "# server-training step ${FT} x client lr. Test seeds 42-46, 60 rounds."
  echo "# P1a + P2a: full matched V"
  for s in fedavg fedyogi select train select_train; do echo "$BASE --strategy $s --out results/ecg/main"; done
  echo "$BASE --strategy train_select --select-frac 0.2 --out results/ecg/main"
  echo "# P1b: mechanism controls (FedYogi candidate replaced)"
  for c in reject half normmatch; do echo "$BASE --strategy select --rules fedavg,$c --name c_$c --out results/ecg/main"; done
  echo "# P2b: |V| = 280, matched"
  for s in select train select_train; do echo "$BASE --strategy $s --val-size 280 --out results/ecg/v280"; done
  echo "$BASE --strategy train_select --select-frac 0.2 --val-size 280 --out results/ecg/v280"
  echo "# P2c: |V| = 280 at 50% AFib (label-skewed server set)"
  for s in select train select_train; do echo "$BASE --strategy $s --val-size 280 --val-pos-share 0.5 --out results/ecg/v280_afib50"; done
  echo "$BASE --strategy train_select --select-frac 0.2 --val-size 280 --val-pos-share 0.5 --out results/ecg/v280_afib50"
  for flip in "" "--flip"; do
    dir=results/ecg/poison_standard; [ -n "$flip" ] && dir=results/ecg/poison_flip
    P="$BASE --num-malicious 20 --attack standard $flip --out $dir"
    if [ -n "$flip" ]; then echo "# P3b: 20 malicious clients, standard attack, direction flipped"; else echo "# P3a: 20 malicious clients, standard attack"; fi
    for s in fedavg median krum; do echo "$P --strategy $s"; done
    echo "$P --strategy fedstrategist --controller-args '{\"preset\":\"paper\"}' --name fs_paper"
    echo "$P --strategy fedstrategist --controller-args '{\"preset\":\"authors\"}' --name fs_authors"
    echo "$P --strategy fedstrategist --controller-args '{\"preset\":\"paper\",\"reward_metric\":\"loss\"}' --name fs_paper_loss"
    echo "$P --strategy select --rules fedavg,median,krum --name select_robust"
    echo "$P --strategy sara"
  done
} > "$OUT"
grep -vc '^#' "$OUT" | xargs echo "jobs:"
