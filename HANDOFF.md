# Handoff: SmartMonitoring ECG experiments (state on 2026-11-10)

Read this first if you are picking up this work, human or Claude. It covers what changed since the project was split
into `distributed/` and `simple/`, what has been run, what the results say, and what to do next. Branch:
`ecg-plugins-oort-sara`.

## 1. What the project is

Federated learning for AFib detection on 12-lead ECG (PTB-XL), supporting three papers:

| Paper | Question | Hypotheses |
|---|---|---|
| 1 | Replicate the TMLR "Predict, Measure, or Train?" supplement on ECG: does choosing the server rule each round by measuring candidates on a server validation set V help? | H1-H6 |
| 2 | Selection plus training the global model on V (`select_train`, our method) | H7-H11 |
| 3 | Measurement vs learned controllers (FedStrategist, SARA) under poisoning | H12-H15 |

Read next, in this order:
1. `simple/README.md`: the simulator.
2. `simple/EXPERIMENTS.md`: CLI reference and the exact command for every hypothesis.
3. `simple/experiments/ecg_protocol.md`: pre-declared hypotheses, decision rules and amendments. Section 7 maps H1-H19 to protocol IDs.
4. `distributed/README.md`: the deployed system and its plugins.
5. `docs/poisoning_simulation.html`: the attacks and the defenses.

## 2. Code layout after the restructure

Commits:
- `39d8a2e` split the repo into `distributed/` and `simple/` and made every method a plugin.
- `d4833bd` reduced `simple/` to one simulator and one comparison script.

```
distributed/                 deployed system: FastAPI server, Redis, Celery, S3, WebSocket clients, dashboard
  server/plugins/            THE methods, shared by both halves
    selectors.py, oort.py    client selection: oort (TMLR supplement's port, FedScale defaults), random
    rules.py                 fedavg, fedyogi, median, krum; mechanism controls reject, half, normmatch
    controllers.py           static, select, select_train, train, train_select, random_mix, fedstrategist, sara
    bandits.py               LinUCB, CountUCB, update diagnostics (FedStrategist)
    attacks.py               poisoning: standard (x5), stealth, flip; copied byte-identical to client/attacks.py
  server/aggregator.py       adapter that runs a controller on a round's model files
  tests/                     plugin unit tests (test_plugins.py and others)
simple/                      single-process simulator for experiments, using the same plugins
  simulate_fl.py             data, model, training, round loop, plots, CLI (one strategy x many seeds)
  compare_experiments.py     paired comparisons across seeds: t-test, Wilcoxon, Holm; tables and plots
  tests/test_simulate_fl.py  end-to-end tests on synthetic clients (CPU, about a minute)
```

To add a method, register a class with `@RULES.register("name")` or `@CONTROLLERS.register("name")` and add a test.
It is then available as `-a name` in the simulator and as `FL_AGGREGATOR=name` in the deployed system.

Do not change `distributed/start.sh` or `distributed/stop.sh`.

### Changes since `d4833bd` (in the same commit as this file)

- `simple/compare_experiments.py` now writes:
  - every metric: accuracy, loss, F2, Brier, energy, latency and rounds-to-target;
  - `per_seed.csv`, `comparison_summary.csv` and `hypothesis_test_results.csv` (every pair of groups × every metric);
  - curves against round, cost bars, Pareto plots (2D and 3D), the rule-selection frequency chart and per-seed box plots.
- `simple/simulate_fl.py`: **bug fix.** `random_mix` used the same coin-flip schedule for every seed. It now draws one schedule per seed (`controller_args.setdefault("seed", seed)`). Test: `test_random_mix_draws_a_schedule_per_seed`.
- `simple/EXPERIMENTS.md`: new outputs documented, and Hypothesis 3b added (see section 4).
- `docs/poisoning_simulation.html`: the poisoning setup and server defenses, for the PR.

Tests: `cd simple && python -m unittest discover -s tests` (11 pass) and `cd distributed && python -m unittest discover -s tests`.

## 3. Running experiments

- **Environment:** `cd simple`, then `conda activate py310`. Without the env TensorFlow silently runs on the CPU; the banner must say `Device : GPU`.
- **Data (not in git):** `datasets/` is gitignored and must be copied separately.
  - `datasets/non_iid`: primary, label-skewed, 100 clients, 2 to 1,120 training records each, AFib share 0-100%.
  - `datasets/iid`
  - `datasets/data/non_iid`: patient split, close to IID.
- **Base parameters (the defaults):** 120 rounds, seeds 42-71, E = 5, K = 20 of 100 with Oort, client lr 0.1, FedYogi η 0.01, per-record z-score. V and the test set T are the pooled client test splits (4,396 records), halved per seed. T is never used for a decision.
- **Resume:** each seed writes `checkpoint.pkl` every round and `summary.json` when it finishes. Re-running the same command skips finished seeds and resumes unfinished ones from their last round.
- **Speed:** about 2-4 s per round on `non_iid` (Oort favours small clients there) and about 7-9 s on iid and patient. One method × 30 seeds takes about 3-4 h on `non_iid` and about 8-9 h on the others.
- **Memory (Windows):** each `simulate_fl.py` process commits about 8-9.5 GB of virtual memory (TF/CUDA start-up). Two in parallel plus another heavy program crashed a run. **Run one process at a time.**
- **Results (not in git):** `simple/results/` is gitignored. `compare_experiments.py` needs each run's `metrics.csv` and `summary.json`, so copy the `results/h*` folders (CSV/JSON only) to compare against earlier runs.

## 4. Results so far (30 seeds, 120 rounds; metric AUPRC-AUC = mean test AUPRC over all rounds)

Decision rule: one-sided paired t-test with Holm correction across the rows of one `compare_experiments.py` call, at α = 0.05. "At least as good" means non-inferiority with a margin of 0.01. Every hypothesis below was run on the label-skewed data unless stated.

| H | Claim | Result | Verdict |
|---|---|---|---|
| H1 | Selector (FedAvg vs FedYogi by V loss) beats the per-seed best static rule | Selector 0.7515, FedAvg 0.7334, FedYogi 0.7318. Gain over best of the two: +0.011 ±0.005, 24/30 seeds, Holm p 1e-4. FedYogi kept in 32.4% of rounds | **supported** |
| H2 | Selector beats each mechanism control | Reject 0.621, Half 0.702, NormMatch 0.729. The selector beats each: +0.131 (30/30 seeds), +0.050 (30/30), +0.023 (27/30) | **supported** |
| H3 | Selector beats random mixing at its own FedYogi rate (p = 0.324) | Mix 0.7472. Gain +0.0044 ±0.0070, 16/30 seeds, p 0.106 | **not supported** |
| H3b | (post hoc) Selector beats random mixing at p = 0.10 and 0.50 | Mix0.10 0.7397 (gain +0.0118, Holm p 0.008). Mix0.50 0.7500 (gain +0.0015, p 0.32) | **not supported** |
| H4 | Selector reaches AUPRC 0.70 in fewer rounds | 20.4 vs 24.3 (FedAvg) and 24.7 (FedYogi). Gain over best of the two: +1.5 ±2.7 rounds, p 0.13 | **not supported** |
| H5 | Selector at least as good as the best static rule on IID and patient data | IID: −0.0004 ±0.0042, as good (p 3e-5). Patient: Selector 0.8062 vs FedYogi 0.8152, −0.0094 ±0.0052, p 0.72 | **not supported** (fails on the patient split) |

**Interpretation so far:**
- **Mixing helps, not per-round measurement.** Alternating FedAvg and FedYogi beats either rule alone by about +0.015-0.02. The gain is flat for any FedYogi share of at least about 1/3, and 10% loses part of it.
- **The selector finds the plateau but adds nothing on top.** It lands on that plateau without tuning, but does no better than random mixing at a good ratio. Its value is that it needs no tuning.
- **Loss misleads the selector on the patient split.** FedYogi is the best rule there by AUPRC but has a higher validation loss (0.107 vs 0.085). Scoring by loss therefore keeps FedAvg in 70% of rounds. This motivates H6 (scoring by AUPRC or Brier).

Flawed runs, kept for the record: `results/h3_random_mix_fixedschedule` and `results/hypothesis3_fixedschedule`, which used one shared schedule for every seed (the bug above). Use `h3_random_mix`.

## 5. Next steps

1. **Record H3b in the protocol.** Add H3b to `simple/experiments/ecg_protocol.md` as Amendment 5: a post-hoc follow-up added after H3 failed. It is already in `EXPERIMENTS.md`.
2. **H6, selection score.** Run `-a select --select-score auprc` and `--select-score brier` on `non_iid`. It is also worth running on the patient split, where loss misled the selector (an extra run not in the protocol; declare it before running).
3. **H7-H11 (Paper 2):** commands are in `EXPERIMENTS.md`.
4. **H12-H15 (Paper 3):** commands are in `EXPERIMENTS.md`. Before running:
   - The commands use `-a select`, so they include no server training. Consider adding `-a select_train --rules fedavg,median,krum` so the full method is tested, and declare it in the protocol first.
   - In the simulator, Krum is given the true number of attackers each round, which no real server knows. Report that, or fix f with `--rule-args '{"krum": {"num_malicious": 4}}'`.
   - SARA has no public code; the `sara` controller is a reconstruction from the TMLR paper's description.
5. **Commit:** commit the work when the user asks.

**Not implemented:**
- backdoor and label-flipping attacks;
- the trimmed mean, Multi-Krum, FLTrust and Zeno rules;
- FedProx client branches, needed for H18;
- PTB-XL site metadata, needed for H16;
- the PhysioNet 2021 data, needed for H19.

The MAD norm filter from `main` was removed in the restructure, so FedAvg has no defense on this branch.

## 6. How the project owner works

- One hypothesis at a time, run by hand. Never change the base parameters without saying so.
- Run long jobs in a separate window and resume them from checkpoints. Keep console output to progress lines.
- When polling a long run, check about every 30 minutes, not every minute.
- For each result, report the verdict, the means, and the deciding row (gain, CI, wins, Holm p). Say plainly when a hypothesis is not supported.
- Do not commit or push unless asked.
