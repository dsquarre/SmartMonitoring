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

## 5. Amendment 2 (2026-11-05, written before any run below): next experiments as hypotheses

**Data.** The label-skewed split `datasets/non_iid` stays the primary data: the papers are about heterogeneous
clients, and its clients differ in size (5-1,354 records) and AFib share (0-100%). The older patient split
`datasets/data/non_iid` (same 21,837 records, 100 clients of about 189 patients, AFib 5-10% each) is close to IID; it
is used only as the middle heterogeneity level of N4.

**Rules for every hypothesis below.**
- Metric: AUPRC-AUC unless stated; comparisons paired by seed; settings from calibration (section 3).
- Seeds: R1 uses test seeds 42-51. New experiments use fresh test seeds 52-61 (10 seeds; the TMLR paper used 5-6
  test seeds per test and 10-11 in its server-learning tests); any new tuning uses seeds 1-2.
- "Beats" = superiority: one-sided paired t-test, H0 mean difference <= 0. "At least as good" / "matches or beats" =
  non-inferiority with margin 0.01 AUPRC-AUC: H0 mean difference <= -0.01.
- Holm correction across the hypotheses of one paper within one experiment, alpha 0.05. Wins out of 10 and the TMLR
  paper's "every seed" criterion are reported as descriptors, not as the decision.
- Analysis: `experiments/analyze_ecg.py --seeds <seeds>` (confirmatory summary table).

**R1 - replication at 10 seeds (run first).** Every hypothesis of sections 2 and 4 (H1, H1c, H2 in three conditions,
H2r, H3 under both attacks, H3s under both attacks) is re-tested on seeds 42-51 under the rules above, adding seeds
47-51 with the same job lines (`experiments/jobs_r1.txt`; random_mix keeps p = 0.357). The 5-seed results stay
reported as they were. A hypothesis supported at 5 seeds but not at 10 is reported as not confirmed. P1b stays
descriptive (a control "matches" if it reaches 80% of the selector's gain on most seeds).

**Paper 1 (selection by validation loss)**
- N1, speed (no new runs; R1 data). Rounds-to-target = first round whose 3-round moving mean of test AUPRC reaches
  0.70 (61 if never). H: the selector needs fewer rounds than the per-seed best static rule (superiority).
- N2, oracle and predictability (needs FedProx client branches, not yet implemented). The TMLR paper's definitions and
  thresholds applied to ECG: H-N2a the per-round oracle beats the noise-only control on every seed; H-N2b no
  pre-round predictor reaches the 25% bar; H-N2c the oracle-derived schedule does not beat the best static rule on
  held-out seeds.
- N3, selection score. Selector variants scored by validation loss (current), AUPRC and Brier. H: neither AUPRC nor
  Brier scoring beats loss scoring (both superiority tests fail after Holm); otherwise the better score is adopted
  for N4-N12 and reported as such.
- N4, heterogeneity. Data: `datasets/iid`, the patient split, `datasets/non_iid`. H-N4a: the selector is at least as
  good as the best static rule at every level (non-inferiority). H-N4b: its gain under label skew exceeds its gain
  under IID (superiority on the paired difference of gains).

**Paper 2 (selection plus server training)**
- N5, server-set size. |V| in {50, 100, 280, 560, 1100, 2198}, matched and 50% AFib; select, train, select_train,
  train_select. H-N5a: train beats select at |V| = 2198 (matched). H-N5b: select is at least as good as train at
  |V| = 50 (matched). H-N5c: select_train is at least as good as the better of select and train at every matched
  size (Holm across sizes). The 50% AFib curves are reported descriptively.
- N6, one-hospital server set. V = 280 records from one PTB-XL site (site metadata), not used by any client.
  H: select beats train (the TMLR finding for a label-skewed V, here a site-skewed V).
- N7, separate scoring split. select_train scoring on a held-out 20% of V vs scoring on all of V, at |V| = 280
  (matched and 50% AFib). H: the held-out split beats scoring on all of V in both conditions.
- N8, baselines that use server data. H: select_train beats FLTrust and beats Zeno (full V, no attack). Secondary:
  sensitivity at 90% specificity, threshold chosen on V.

**Paper 3 (measurement vs learned controllers under poisoning)**
- N9, changing attacks (SARA's setting). Malicious clients active in rounds 1-20 and 41-60, honest in 21-40; standard
  and flipped attacks. H: select_robust is at least as good as SARA and as the best FedStrategist variant under both.
- N10, attack sweep. Malicious share {10, 20, 30, 40}% x attacks {standard, flipped, stealth, label flipping}; rules
  add trimmed mean, Multi-Krum and FLTrust. H-N10a: select_robust is at least as good as the best learned controller
  in every cell (Holm across cells). H-N10b (expected failure, declared in advance): under a backdoor attack,
  select_robust does not lower the attack success rate relative to FedAvg.
- N11, adaptive attacker that knows the selector: exploratory, no confirmatory claim.

**Across papers**
- N12, second dataset: PhysioNet 2021 (CPSC, Chapman-Shaoxing, Georgia), clients split by site, 1-D ResNet;
  calibration on tuning seeds 1-2 as in section 1. H: H1, H2 (full V), H3 and H3s replicate under the rules above.
- N13, clean-up before N1-N12: remove the 3 test patients that appear in client training data; extend the client lr
  grid to 0.2 and 0.3 on the tuning seeds. If a new lr wins by more than 0.005, the N-series uses it; R1 keeps 0.1.

Order: R1, N1, N13, N5, N7, N4, N9, N10, N3, N6, N8, N2, N12.

## 6. Amendment 3 (2026-11-05, before any run at these settings): base parameters 120 rounds x 30 seeds

- R1 (section 5) was stopped after about 11 rounds of its first two jobs, before any job finished. It was also running
  on the CPU by mistake (TensorFlow started without the conda env's CUDA libraries). No R1 result was looked at.
- New base for every test: **120 rounds, 30 test seeds (42-71), E = 5 local epochs**, with the fixed settings of
  section 2 and the calibrated values of section 3. Results go to `results/ecg120/` (`experiments/jobs_base.txt`,
  written by `experiments/make_test_jobs.sh`).
- The 60-round, 5-seed battery (sections 2-4, `results/ecg/`) becomes the pilot. It is reported as such and never
  pooled with the base runs.
- R1 is replaced by R1-120: every hypothesis of sections 2 and 4 on seeds 42-71 at 120 rounds, decided by the rules
  of section 5 (one-sided paired t, Holm within each paper, non-inferiority margin 0.01). The win descriptor at 30
  seeds is 24 of 30 (80%).
- P1c's p is recomputed from the new P1a (the selector's FedYogi rate over seeds 42-71) before random_mix runs.
- N-series: fresh test seeds 72-101 (30 seeds); tuning seeds 1-2. N1's "never reached" value becomes 121.
- Calibration (section 3) was done at 40 rounds and is kept. Limitation: at 120 rounds another client lr might win;
  N13's lr check runs at 120 rounds.
- Compute: 33 jobs x 30 seeds x 120 rounds = 118,800 rounds, about 2 days on the laptop GPU at 2 jobs in parallel.

## 7. Amendment 4 (2026-11-05, before any run at the base parameters): hypotheses run one at a time

- The base run of section 6 was stopped in its first rounds; no result was looked at. Experiments are no longer run as
  one batch: each hypothesis is run by hand with `simulate_fl.py` and decided with `compare_experiments.py`, using the
  commands in `EXPERIMENTS.md`. Results go to `results/h<N>_<method>/s<seed>/`.
- `simulate_fl.py` and `compare_experiments.py` replace `run.py`, `smartmon/`, `experiments/run_jobs.py`,
  `analyze_ecg.py` and `calibrate.py`. The simulation code is the same (same data split, model, training, seeds and
  plugins); the pilot (sections 2-4, `results/ecg/`) and its job files are in git commit 39d8a2e.
- Base parameters for every hypothesis (section 6): 120 rounds, test seeds 42-71, E = 5, calibrated settings.
- Decision rule: one-sided paired t-test per comparison, Holm across the comparisons of the hypothesis's
  `compare_experiments.py` call (this includes descriptive rows, so it is more conservative than Holm over the
  deciding rows alone), alpha 0.05; non-inferiority margin 0.01 AUPRC-AUC. Wins out of 30, Wilcoxon and the TMLR
  paper's "every seed" criterion are reported as descriptors.
- H2 is now decided by superiority of the selector over each control (it was "a control matches if it reaches 80% of
  the selector's gain on most seeds"); a supported H2 means no control explains the gain.

| Guide | Protocol ID | Claim |
|---|---|---|
| H1 | H1 | selector beats the per-seed best static rule |
| H2 | P1b | selector beats each mechanism control |
| H3 | H1c | selector beats random mixing at its own FedYogi rate |
| H4 | N1 | selector reaches AUPRC 0.70 in fewer rounds |
| H5 | N4 | selector is at least as good as the best static rule on IID and patient-split data |
| H6 | N3 | neither AUPRC nor Brier scoring beats loss scoring |
| H7 | H2 (full V) | select_train is at least as good as the better of select and train |
| H8 | H2 (V = 280) | the same with 280 matched records |
| H9 | H2 (V = 280, 50% AFib), H2r | the same with a skewed server set; and not below the best static rule |
| H10 | N5 | server-set size sweep (50 to 2,198) |
| H11 | N7 | a held-out scoring split beats scoring on all of V |
| H12 | H3, H3s (standard) | selector over FedAvg/median/Krum is at least as good as FedStrategist and SARA |
| H13 | H3, H3s (flipped) | the same under the flipped attack |
| H14 | N9 | the same with attacks active in rounds 1-40 and 81-120 |
| H15 | N10 | attack strength sweep (implemented attacks only) |
| H16-H19 | N6, N8, N2, N12 | need code or data first |
- Clients per round: K = 20 of 100 (was 10), set on 2026-11-05 before any base run; Oort still selects them.

## 8. Amendment 5 (2026-11-12, before any run below): one paper, and two falsification tests first

### 8.1 Scope

- **One paper.** There is one paper, for IEEE: select-then-train (`select_train`). It cites the TMLR paper "Predict, Measure, or Train?" (our co-author's) for selection by validation measurement and for server training. Paper 3's poisoning setting becomes this paper's Byzantine setting.
- **Paper 1 hypotheses are not this paper's claims.** They are the TMLR paper's results.
  - The ECG reruns already done (H1-H5, plus H3b: a post-hoc follow-up written after H3 was not supported, so exploratory) stay as background.
  - No further Paper 1 runs. H6 is dropped.
- **Dropped or on hold:**
  - H10 and H11 are dropped.
  - H8 and H9 (small and skewed server sets) are on hold, because the TMLR paper already covers server-set composition.
  - H7 and H12-H15 are replaced by A1 and A2 below.
- **Why A1 and A2 come first.** The method rests on two untested assumptions, A1 and A2. Each gets one quick falsification test before any other experiment. Every outcome's conclusion is written below, before any run.

### 8.2 Common rules for A1 and A2

- **Data and training:**
  - `datasets/non_iid`, 120 rounds, E = 5, K = 20 of 100 with Oort.
  - Calibrated settings (section 3); V = 2,198 records, matched.
- **Seeds:** fresh test seeds 72-81 (10 seeds). No run on these seeds exists or has been looked at.
- **Metric:** AUPRC-AUC on T. Comparisons are paired by seed.
- **One primary test per assumption:** the one-sided paired t-test of the deciding row. Because there is exactly one primary test, there is no Holm correction on it. The other rows of the `compare_experiments.py` call are descriptive. This departs from section 7, which applied Holm across a whole call.
- **Outcomes**, where d = mean improvement on the deciding row:
  - **PASS:** d > 0 and p < 0.05.
  - **FAIL (falsified):** d <= 0.
  - **UNDECIDED:** d > 0 and p >= 0.05. Extend once to seeds 72-101 (30 seeds) with the same commands, and decide with the same rule, except that UNDECIDED at 30 seeds counts as FAIL. There is no further extension.
- **Order:**
  - A1 runs first, then A2, one process at a time.
  - A2 is analysed whatever A1's outcome is.
  - Nothing else is run until both are decided.

### 8.3 A1: select-then-train beats each of its parts (no attack)

- **Claim:** `select_train` beats both `select` and `train`.
- **Runs** (default rules: fedavg, fedyogi; `train` uses the FedAvg base):
  - `a1_select_train`, `a1_select`, `a1_train`;
  - `a1_fedavg`, a clean reference that A2's validity check also uses.
- **Deciding row:** `select_train` against the per-seed best of `select` and `train`.
- **Why this is a severe test.** In calibration, training on V gave about +0.12 AUPRC-AUC over FedAvg (section 3, at 40 rounds), while selection gave about +0.02 (H1). If the selection gain does not survive on top of training, the combination is just training.
- **Conclusions, declared now:**
  - **PASS:** A1 survives, and `select_train` stays the method.
  - **FAIL with d(vs train) <= 0:** selection adds nothing once the server trains on V. The method reduces to server training (Mai et al. 2023; the TMLR paper's "Train"), and the paper cannot claim the combination. Any contribution would have to rest on A2, reframed as server training under attack.
  - **FAIL with d(vs select) <= 0:** training adds nothing. This contradicts calibration, so check the code for a bug before drawing any conclusion.
  - **FAIL with both d(vs train) > 0 and d(vs select) > 0:** each part is beaten on average, but the better part of each seed is not. A1 is false as stated: the combination does not reliably beat the better of its parts.

### 8.4 A2: under Byzantine clients, select-then-train beats existing defenses

- **Claim:** with malicious clients, `select_train` over {fedavg, median, krum} beats existing defenses.
- **Attack:**
  - 20 of 100 clients are malicious, fixed per seed.
  - Each uploads its honest update scaled ×5 with its direction reversed (`--attack standard --flip`), in every round.
- **Krum's f:** fixed at 4 (the expected number of attackers among 20 selected) for every method that uses Krum, our candidate included. It is not the per-round true count used in sections 2 and 7.
- **Ours:** `a2_select_train`, which is `select_train --rules fedavg,median,krum`.
- **Existing methods:**
  - `a2_fedavg`, `a2_median`, `a2_krum`;
  - `a2_fltrust` (Cao et al. 2021, root set = V);
  - `a2_train` (FedAvg plus server training);
  - `a2_fs_paper` (FedStrategist as published: LinUCB, accuracy reward);
  - `a2_sara`.
- **SARA:** runs only after the `sara` controller has been checked against the original paper (Hu et al., SEAMS 2025), not the secondary description it was built from (section 4).
  - If the paper cannot be obtained, SARA is left out of the best-of, and the paper must not claim to beat SARA.
  - A2 is not decided before one of these two outcomes holds, with one exception. Adding a method to the best-of can only lower d, so a FAIL without SARA is final. A PASS or UNDECIDED without SARA waits.
- **Deciding row:** `select_train` against the per-seed best of all existing methods.
- **Validity check:** the attack must hurt. Attacked FedAvg (`a2_fedavg`) must score at least 0.05 AUPRC-AUC below clean FedAvg (`a1_fedavg`, paired by seed).
  - If it does not, A2 is uninformative: neither PASS nor FAIL.
  - In that case the existing-method runs are repeated once at ×10 before any decision.
- **Conclusions, declared now:**
  - **PASS:** A2 survives the quick test. A confirmatory Byzantine battery is pre-registered next: stealth attack, attacks switching on and off, attacker shares of 10-40%, 30 seeds.
  - **FAIL, best existing method is `fltrust` or `train`:** server data alone defends as well, and A2 is false as stated. The paper may then claim at most "as good as". That needs its own pre-registered non-inferiority test, or the Byzantine setting is dropped.
  - **FAIL, best existing method is median, Krum, FedStrategist or SARA:** A2 is false. Report which method wins.

### 8.5 Commands

The commands are in `EXPERIMENTS.md`, section 3.
