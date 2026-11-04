# ECG experiments: protocol (written 2026-10-31, before any calibration or test run)

Data: `datasets/non_iid`, 100 label-skewed PTB-XL clients (AFib share per client 0-100%, sizes 5-1354), 17,441
training records. Checked before this protocol: every record maps to its PTB-XL record with the right label; no
patient is in both the train and test split of a client; 3 of 3,854 test-split patients also appear in some
client's training data (left in; negligible). The pooled client test splits are split per seed into V (2,198 records,
~146 AFib) and the reported test set T (2,198, ~157 AFib). T is never used for a decision on a test seed.

Fixed for every run: Oort (K = 10 of 100), E = 5 local epochs, batch 32, plain SGD clients (TMLR supplement), ECG
1-D CNN, FedYogi tau = 1e-3, b1 0.9, b2 0.99, one server epoch for server training.

Primary metric: **AUPRC-AUC** = mean test AUPRC over all rounds (the supplement's accuracy-AUC on AUPRC). Also
reported: late AUPRC-AUC (second half), final AUPRC (mean of the last 5 rounds), F2 at 0.3, Brier, server cost.

## 1. Calibration (tuning seeds 1 and 2, 40 rounds; never reported as results)

Choice rule: the setting with the highest AUPRC-AUC averaged over the two tuning seeds, measured on T of the tuning
seeds (representative held-out data, the supplement's "fair tuning"); within 0.005 of the best, the setting listed
first wins. Steps run in order; each fixes its setting for the next.

1. Preprocessing: `none` vs `record` (per-record, per-lead z-score). Static FedAvg, client lr 0.05.
2. Client learning rate: 0.01, 0.03, 0.05, 0.1. Static FedAvg.
3. FedYogi step eta: 0.01, 0.003, 0.03. Static FedYogi. (The selector and C3 use the same eta.)
4. Server-training step (multiple of the client lr): 1.0, 0.1, 0.01. `train` (FedAvg base). Used by every
   controller that trains on V.

## 2. Tests (test seeds 42-46, 60 rounds, settings from calibration)

Comparisons are paired by seed. "Wins" counts seeds where the difference is positive.

**Paper 1 - measure after the updates arrive (ECG replication of the supplement's Point B).**
- P1a: `fedavg`, `fedyogi`, `select` (FedAvg vs FedYogi by validation loss).
  H1: `select` beats the per-seed best static rule in AUPRC-AUC on at least 4 of 5 seeds.
- P1b: mechanism controls, `select` with FedYogi replaced by `reject` (C1), `half` (C2), `normmatch` (C3).
  A control "matches" if it attains at least 80% of the selector's gain over the best static rule on most seeds.
- P1c: `random_mix` with p = the selector's mean FedYogi rate in P1a (computed from P1a before P1c runs).
  H1c: `select` beats `random_mix` in AUPRC-AUC on at least 4 of 5 seeds.
  Not run: the oracle and predictability battery (needs FedProx client branches).

**Paper 2 - select and train.** Controllers: `select`, `train`, `select_train`, `train_select` (selection split 20%).
- P2a: full matched V (2,198). P2b: |V| = 280, uniform (matched). P2c: |V| = 280 at 50% AFib (label-skewed,
  "curated" server set).
- H2: in each condition, `select_train` is at least as good as the better of `select` and `train` (AUPRC-AUC,
  paired). H2r (robustness): in P2c, `select_train` does not fall below the per-seed best static rule.

**Paper 3 - measurement vs a learned controller under poisoning.** 20 of the 100 clients are malicious (fixed per
seed); a selected malicious client uploads 5x its honest update (FedStrategist's standard attack). Krum is told the
number of malicious clients in the round.
- P3a (standard) and P3b (standard, direction flipped: real poisoning): `fedavg`, `median`, `krum`,
  `fedstrategist` (paper preset, accuracy reward), `fedstrategist` (authors preset), `fedstrategist` (paper preset,
  loss reward: the strongest variant for imbalanced ECG), and `select` over {fedavg, median, krum}.
- H3: `select` matches or beats the best FedStrategist variant in AUPRC-AUC on at least 4 of 5 seeds. Server cost
  (seconds and models scored on V per round) is reported for every method, whichever way it goes.

## 3. Calibration results (2026-10-31, tuning seeds 1-2, 40 rounds; test AUPRC-AUC of the tuning seeds)

| Step | Settings (protocol order) | AUPRC-AUC | Chosen |
|---|---|---|---|
| 1 Preprocessing (FedAvg, lr 0.05) | none / record | 0.479 / 0.590 | record |
| 2 Client lr (FedAvg) | 0.01 / 0.03 / 0.05 / 0.1 | 0.396 / 0.540 / 0.590 / 0.628 | 0.1 (edge of the grid; not extended) |
| 3 FedYogi eta (static FedYogi) | 0.01 / 0.003 / 0.03 | 0.627 / 0.541 / 0.598 | 0.01 |
| 4 Server-training step (train) | 1.0 / 0.1 / 0.01 x client lr | 0.752 / 0.755 / 0.679 | 1.0 (within 0.005 of 0.1; first listed) |

Job files: experiments/jobs_calib_step{1..4}.txt; results: results/ecg/calib/step{1..4}. The step-2 lr 0.05 run is the
step-1 `record` run (identical configuration). Test jobs: experiments/jobs_tests.txt (make_test_jobs.sh 1.0).

P1c: the selector kept FedYogi in 35.7% of P1a rounds, so `random_mix` ran with p = 0.357
(experiments/jobs_p1c.txt). Test results: results/ecg/REPORT.md (experiments/analyze_ecg.py).

## 4. Amendment (2026-10-31, after the main battery, before any SARA run): SARA added to Paper 3

SARA is Hu et al. (SEAMS 2025), the second learned controller cited by the TMLR paper. There is no public code, so it is
reconstructed from that paper's description (`sara` controller): UCB1 (c = 1, one pull of each arm first) over FedAvg,
median and Krum, updating only the applied rule, reward = accuracy - loss of the new global model on V. It runs in
P3a and P3b with the same settings as the other methods (experiments/jobs_sara.txt).
- H3s: `select` over {fedavg, median, krum} matches or beats SARA in AUPRC-AUC on at least 4 of 5 seeds.
- H3 is also reported against the best of all four learned controllers (three FedStrategist variants and SARA).
