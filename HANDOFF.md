# Handoff: SmartMonitoring ECG experiments (state on 2026-11-12)

Read this first if you are picking up this work, human or Claude. It covers what changed since the project was split
into `distributed/` and `simple/`, what has been run, what the results say, and what to do next. Branch:
`ecg-plugins-oort-sara`.

## 1. What the project is

Federated learning for AFib detection on 12-lead ECG (PTB-XL). **There is one paper (IEEE): select-then-train
(`select_train`).** Each round the server keeps the candidate rule with the lowest loss on its validation set V, then
trains it on V.
- It cites the TMLR paper "Predict, Measure, or Train?" (our co-author's) for selection and for server training.
- The poisoning (Byzantine) setting is part of this paper.
- The earlier "Paper 1" hypotheses (H1-H5, H3b) re-ran the TMLR paper's results on ECG. They are done and are background, not this paper's claims.

**Current step (protocol section 8, pre-registered 2026-11-12):** quick falsification tests of the two assumptions the
method rests on.
- **A1:** select-then-train beats both selection alone and training alone, with no attack.
- **A2:** under Byzantine clients, it beats existing defenses (FedAvg, median, Krum, FLTrust, server training, FedStrategist, SARA).

Nothing else is run until both are decided.

Read next, in this order:
1. `simple/README.md`: the simulator.
2. `simple/EXPERIMENTS.md`: CLI reference, and the commands for the current tests (section 3).
3. `simple/experiments/ecg_protocol.md`: pre-registered claims, decision rules and amendments. Section 8 is the current plan.
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
    rules.py                 fedavg, fedyogi, median, krum, fltrust; mechanism controls reject, half, normmatch
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

### Changes since `d4833bd`

- `simple/compare_experiments.py` now writes:
  - every metric: accuracy, loss, F2, Brier, energy, latency and rounds-to-target;
  - `per_seed.csv`, `comparison_summary.csv` and `hypothesis_test_results.csv` (every pair of groups × every metric);
  - curves against round, cost bars, Pareto plots (2D and 3D), the rule-selection frequency chart and per-seed box plots.
- `simple/simulate_fl.py`: **bug fix.** `random_mix` used the same coin-flip schedule for every seed. It now draws one schedule per seed (`controller_args.setdefault("seed", seed)`). Test: `test_random_mix_draws_a_schedule_per_seed`.
- `simple/EXPERIMENTS.md`: new outputs documented, and Hypothesis 3b added (see section 4).
- `docs/poisoning_simulation.html`: the poisoning setup and server defenses, for the PR.
- 2026-11-12:
  - The plan was narrowed to one paper.
  - A1 and A2 were pre-registered (protocol section 8).
  - The `fltrust` rule was added, with tests.
  - Static controllers now report a rule's own training passes on V.
  - The `sara` controller is marked unverified until it is checked against the SARA paper.

Tests: `cd simple && python -m unittest discover -s tests` (11 pass; the plugin tests in `distributed/tests/test_plugins.py` pass too, while 4 `test_client_model` tests fail on Windows temp-file cleanup, before and after these changes) and `cd distributed && python -m unittest discover -s tests`.

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

## 4. Background results: ECG reruns of the TMLR paper (30 seeds, 120 rounds; metric AUPRC-AUC = mean test AUPRC over all rounds)

These reproduce the TMLR paper's selection results on ECG. They are not claims of the IEEE paper.

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

1. **Run A1, then A2.** The commands are in `simple/EXPERIMENTS.md` section 3. The claims, the decision rule and the conclusion for every outcome are fixed in `simple/experiments/ecg_protocol.md` section 8.
   - **Seeds and settings:** fresh seeds 72-81, 120 rounds, one process at a time.
   - **Run time:** A1 is 4 runs, about 5 h; A2 is 7 runs, about 10-12 h.
   - **Decisions:** decide each assumption exactly as section 8 says. Do not change a claim or a rule after seeing results; an extension to seeds 72-101 is allowed only for an UNDECIDED outcome.
2. **SARA must be checked against the original paper before `a2_sara` runs.** The `sara` controller was built from secondary descriptions (the TMLR paper and FedStrategist), not from SARA itself.
   - The paper is Hu et al., "Adapting Aggregation Rule for Robust Federated Learning under Dynamic Attacks", SEAMS 2025 (IEEE Xplore 11025919). It is paywalled and has no preprint or code.
   - Get the PDF through institutional access and save it as `docs/papers/hu2025_sara.pdf` (gitignored, so keep it local).
   - Then compare its arms ("modules within the aggregation rules"), reward, UCB form and attack schedule with the `sara` controller, and fix the controller.
   - If the paper can't be obtained, A2 is decided without SARA, and the paper must not claim to beat SARA.
3. **After both decisions:** pre-register the follow-ups for the outcomes in protocol section 8 before running anything.
   - **If A2 passes:** stealth attacks, attacks switching on and off, 10-40% attackers, 30 seeds.

The retired hypotheses and the reasons are listed in `EXPERIMENTS.md` section 3.

**Not implemented:**
- backdoor and label-flipping attacks;
- the trimmed mean, Multi-Krum and Zeno rules.

FLTrust was added on 2026-11-12 (`-a fltrust`, Cao et al. 2021, with V as its root set). The MAD norm filter from
`main` was removed in the restructure, so FedAvg has no defense on this branch.

## 6. How the project owner works

- One hypothesis at a time, run by hand. Never change the base parameters without saying so.
- Run long jobs in a separate window and resume them from checkpoints. Keep console output to progress lines.
- When polling a long run, check about every 30 minutes, not every minute.
- For each result, report the verdict, the means, and the deciding row (gain, CI, wins, Holm p). Say plainly when a hypothesis is not supported.
- Do not commit or push unless asked.
