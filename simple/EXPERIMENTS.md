# SmartMonitoring Experimentation Guide (simple)

Run each test by hand, one at a time: simulate the methods it needs with `simulate_fl.py`, then test them with
`compare_experiments.py`. Claims, decision rules and conclusions are pre-registered in `experiments/ecg_protocol.md`
before any run; the current plan is its section 8.

**Before anything:** open a terminal in `simple/` and activate the conda env, or TensorFlow runs on the CPU (the
banner prints `Device : GPU` when it is right):

```bash
cd simple
conda activate py310
```

---

## 1. CLI reference (`simulate_fl.py`)

The base parameters are the defaults: 120 rounds, 5 local epochs, 30 seeds (42-71), K = 20 of 100 clients,
label-skewed data `../datasets/non_iid`, Oort client selection, and the calibrated client lr 0.1, FedYogi step 0.01,
per-record normalisation. Change any of them with the flags below. Every test uses the label-skewed
split (the default `--data-dir`); only the background H5 runs also used the IID split and the patient split.

### FL controls
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--data-dir` | `../datasets/non_iid` | Client partitions folder (`../datasets/iid`, `../datasets/data/non_iid` = patient split) |
| `-n`, `--num-clients` | `0` (all) | Total clients N |
| `-k`, `--select-k` | `20` | Clients selected per round K |
| `-r`, `--rounds` | `120` | Rounds |
| `-e`, `--local-epochs` | `5` | Local epochs per round |
| `--batch-size` | `32` | Local batch size |
| `--lr` | `0.1` | Client SGD learning rate |
| `--seeds` | `42-71` | Seed range; each seed goes to `-o/s<seed>` |
| `--seed` | | One seed, written to `-o` itself |
| `-o`, `--output-dir` | `results/run` | Output folder |
| `--overwrite` | off | Run again even if finished (finished runs are skipped; interrupted ones resume) |

### Server strategy and client selection
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `-a`, `--aggregator` | `select_train` | A rule (`fedavg`, `fedyogi`, `median`, `krum`, `fltrust`) or a controller: `select` (keep the candidate with the lower validation loss), `train` (FedAvg, then train on V), `select_train` (ours), `train_select`, `random_mix`, `fedstrategist`, `sara` |
| `--rules` | controller's own | Candidate rules of a controller, e.g. `fedavg,median,krum` |
| `-s`, `--selector` | `oort` | Client selector: `oort`, `random` |
| `--mix-p` | | `random_mix`: probability of FedYogi each round |
| `--fs-preset` | `paper` | `fedstrategist`: `paper` (LinUCB) or `authors` (their code) |
| `--fs-reward` | `accuracy` | `fedstrategist`: `accuracy` or `loss` |
| `--server-lr` | `0.01` | FedYogi step |
| `--ft-lr-scale` | `1.0` | Server-training step = client lr x this |
| `--controller-args`, `--rule-args` | `{}` | Any other plugin argument, as JSON |

### Server set V and poisoning
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--val-size` | all (2,198) | Records in V |
| `--val-pos-share` | matched | With `--val-size`: AFib share of V (e.g. `0.5`) |
| `--select-frac` | `0` | Share of V used only for scoring candidates (rest only for training) |
| `--select-score` | `loss` | What the server scores candidates by: `loss`, `brier`, `auprc` |
| `--num-malicious` | `0` | Malicious clients (fixed per seed) |
| `--attack` | `none` | `standard` (5x the honest update), `stealth` |
| `--flip` | off | Reverse the malicious update direction |
| `--attack-scale` | `5.0` | Attack scale |
| `--attack-rounds` | all | Rounds when attackers are active, e.g. `1-40,81-120` |

**Run time:** one method = 30 seeds x 120 rounds with 20 clients per round, roughly 8 h on the laptop GPU (estimate: about twice the measured 10-client time). To halve it, run the same command
twice in two terminals with `--seeds 42-56` and `--seeds 57-71` (same `-o`).

**Quick check before a long run:** `python simulate_fl.py -a select --seed 42 -r 5 -o results/test`.

---

## 2. `compare_experiments.py`

```bash
python compare_experiments.py --dirs <first group> <other groups...> --labels <names...> [--best-of A,B] [--test ni] [--metric ...] -o <folder>
```

The first group is the method under test. It is compared, paired by seed, with every other group and with the
per-seed best of each `--best-of` list. Each row gives the mean improvement with a 95% CI, wins out of 30, a one-sided
paired t-test, Wilcoxon, and the Holm-adjusted p across the rows of that call. A hypothesis is **supported** when its
deciding row has Holm p < 0.05. `--test ni` tests "at least as good" (non-inferiority, margin 0.01 AUPRC-AUC).
`--metric` defaults to `auprc_auc` (mean test AUPRC over all rounds); also `auprc_late`, `auprc_final`,
`rounds_to_target` (`--target 0.70`), `f2_final`, `accuracy_final`, `loss_final`, `brier_final`, `energy_kj`,
`latency_h`. `summary.md` also lists the first group against each comparator on every metric (descriptive, two-sided,
uncorrected).

Output files:
- **Tables:** `summary.md`, `per_seed.csv`, `comparison_summary.csv` (mean and std per group), `hypothesis_test_results.csv` (every pair of groups × every metric) and `comparison_strategy_selection_distribution.csv`.
- **Curves vs round** (mean with 95% band): AUPRC, loss, F2, accuracy, cumulative energy and cumulative latency.
- **Costs:** a bar chart of energy, latency and server time.
- **Pareto plots:** AUPRC-AUC and final F2 against energy, against latency, and against energy × latency, plus 3D plots.
- **Rule selection:** server rule kept, as a frequency chart and as shares.
- **Per seed:** box plots.

---

## 3. The paper: select-then-train, and the two assumptions it rests on

There is one paper, for IEEE: select-then-train (`select_train`). Each round the server builds every rule's
candidate, keeps the one with the lowest loss on its validation set V, then trains it on V. The paper cites the TMLR
paper "Predict, Measure, or Train?" for selection and for server training, and takes Paper 3's poisoning setting as
its Byzantine setting.

The method rests on two untested assumptions. Each gets one quick falsification test before anything else is run.
The claims, the decision rule and the conclusion for every outcome were pre-registered in
`experiments/ecg_protocol.md` section 8, before any of these runs. Both tests use:
- fresh seeds 72-81 (10 seeds), 120 rounds, `non_iid`;
- one primary test: the deciding row, a one-sided paired t-test with no Holm correction;
- outcomes PASS (gain > 0, p < 0.05), FAIL (gain <= 0) or UNDECIDED. UNDECIDED extends once to seeds 72-101.

### Assumption 1: select-then-train beats each of its parts (no attack)
```bash
python simulate_fl.py -a fedavg --seeds 72-81 -o results/a1_fedavg
python simulate_fl.py -a select --seeds 72-81 -o results/a1_select
python simulate_fl.py -a train --seeds 72-81 -o results/a1_train
python simulate_fl.py -a select_train --seeds 72-81 -o results/a1_select_train
python compare_experiments.py --dirs results/a1_select_train results/a1_select results/a1_train results/a1_fedavg --labels SelectTrain Selector Training FedAvg --best-of Selector,Training -o results/assumption1
```
Deciding row: `best of Selector,Training`. FedAvg is the clean reference for A2's validity check.

### Assumption 2: under Byzantine clients, select-then-train beats existing defenses
In this test, 20 of 100 clients upload their update scaled ×5 and reversed, every round. Krum's f is fixed at 4 for
every method that uses Krum.
```bash
P="--seeds 72-81 --num-malicious 20 --attack standard --flip --rule-args {\"krum\":{\"num_malicious\":4}}"
python simulate_fl.py -a select_train --rules fedavg,median,krum $P -o results/a2_select_train
python simulate_fl.py -a fedavg $P -o results/a2_fedavg
python simulate_fl.py -a median $P -o results/a2_median
python simulate_fl.py -a krum $P -o results/a2_krum
python simulate_fl.py -a fltrust $P -o results/a2_fltrust
python simulate_fl.py -a train $P -o results/a2_train
python simulate_fl.py -a fedstrategist --fs-preset paper $P -o results/a2_fs_paper
python simulate_fl.py -a sara $P -o results/a2_sara          # only after the sara controller is checked against the SARA paper
python compare_experiments.py --dirs results/a2_select_train results/a2_fedavg results/a2_median results/a2_krum results/a2_fltrust results/a2_train results/a2_fs_paper results/a2_sara --labels SelectTrain FedAvg Median Krum FLTrust Training FS-paper SARA --best-of FedAvg,Median,Krum,FLTrust,Training,FS-paper,SARA -o results/assumption2
```
Deciding row: `best of FedAvg,...,SARA`. If the SARA paper cannot be obtained, drop SARA from the call and from the
claim. Until SARA is checked, run the same call without SARA into `results/assumption2_without_sara`. A FAIL there is
final, since adding SARA can only raise the best-of.

Validity check: attacked FedAvg must score at least 0.05 AUPRC-AUC below `a1_fedavg`. If it does not, the attack is
too weak and A2 is undecided.

### After the two tests
What follows depends on the outcomes declared in the protocol (section 8). Nothing is run until both are decided.
- **Both pass:** pre-register the confirmatory runs: A1 at 30 seeds, and the Byzantine battery (stealth, attacks
  switching on and off as in SARA's setting, 10-40% attackers).
- **A1 fails:** the combination is not the contribution. See the protocol for what the paper can still claim.

### Background: ECG reruns of the TMLR paper's results (done; not this paper's claims)
H1-H5 and H3b reran the TMLR paper's Paper 1 hypotheses on ECG (seeds 42-71). Results and their reading are in
`../HANDOFF.md` section 4, and the comparison folders in `results/hypothesis*`. The commands, for reproducibility:
```bash
python simulate_fl.py -a fedavg -o results/h1_fedavg            # likewise fedyogi, select
python simulate_fl.py -a select --rules fedavg,reject -o results/h2_reject      # likewise half, normmatch
python simulate_fl.py -a random_mix --mix-p 0.324 -o results/h3_random_mix      # H3b: --mix-p 0.10 and 0.50
python simulate_fl.py -a fedavg --data-dir ../datasets/iid -o results/h5_iid_fedavg   # H5: fedavg, fedyogi, select on iid and data/non_iid
```

### Retired
| Was | Why |
| :--- | :--- |
| H6 (selection score) | A Paper 1 question, answered by the TMLR paper |
| H7 | Replaced by Assumption 1 |
| H8, H9 (small and skewed server sets) | On hold: the TMLR paper covers server-set composition |
| H10 (server-set size sweep), H11 (scoring split) | Dropped |
| H12-H15 (poisoning) | Replaced by Assumption 2 and, if it passes, its confirmatory battery |
| H16-H19 | Not planned |


---

## 4. Outputs

Each run folder (`results/<name>/s<seed>/`) holds `metrics.csv` and `round_history.json` (one row per round: test
AUPRC, ROC-AUC, F2, F1, recall, precision, Brier, loss, confusion counts, validation loss, the rule kept, server time
and cost, latency and energy), `config.json`, `summary.json` (when finished), and plots: `auprc_vs_round.png`,
`loss_vs_round.png`, `accuracy_vs_round.png`, `confusion_matrix_latest.png`, `strategy_selection_freq.png`.
