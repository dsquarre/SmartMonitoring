# SmartMonitoring Experimentation Guide (simple)

Run each hypothesis by hand, one at a time: simulate the methods it needs with `simulate_fl.py`, then test them with
`compare_experiments.py`. The hypotheses and decision rules are pre-declared in `experiments/ecg_protocol.md`
(section 7 maps these numbers to the protocol).

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
per-record normalisation. Change any of them with the flags below. Every hypothesis uses the label-skewed
split (the default `--data-dir`); only H5 also runs the IID split and the patient split, as milder heterogeneity levels.

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
| `-a`, `--aggregator` | `select_train` | A rule (`fedavg`, `fedyogi`, `median`, `krum`) or a controller: `select` (keep the candidate with the lower validation loss), `train` (FedAvg, then train on V), `select_train` (ours), `train_select`, `random_mix`, `fedstrategist`, `sara` |
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

## 3. Hypotheses

### Paper 1: selecting the aggregator by measuring on validation data

#### Hypothesis 1: the selector beats both static rules
```bash
python simulate_fl.py -a fedavg -o results/h1_fedavg
python simulate_fl.py -a fedyogi -o results/h1_fedyogi
python simulate_fl.py -a select -o results/h1_select
python compare_experiments.py --dirs results/h1_select results/h1_fedavg results/h1_fedyogi --labels Selector FedAvg FedYogi --best-of FedAvg,FedYogi -o results/hypothesis1
```
Deciding row: `best of FedAvg,FedYogi`. The "rules kept" column gives the selector's FedYogi share, needed for H3.

#### Hypothesis 2: no simpler mechanism explains the gain
The FedYogi candidate is replaced by rejecting the round, half a FedAvg step, or FedAvg's direction at FedYogi's step size.
```bash
python simulate_fl.py -a select --rules fedavg,reject -o results/h2_reject
python simulate_fl.py -a select --rules fedavg,half -o results/h2_half
python simulate_fl.py -a select --rules fedavg,normmatch -o results/h2_normmatch
python compare_experiments.py --dirs results/h1_select results/h2_reject results/h2_half results/h2_normmatch --labels Selector Reject Half NormMatch -o results/hypothesis2
```
Supported when the selector beats all three controls.

#### Hypothesis 3: measuring beats mixing the two rules at random
`P` = the selector's FedYogi share from H1 (e.g. `0.36`).
```bash
python simulate_fl.py -a random_mix --mix-p P -o results/h3_random_mix
python compare_experiments.py --dirs results/h1_select results/h3_random_mix --labels Selector RandomMix -o results/hypothesis3
```

#### Hypothesis 3b (follow-up): the mixing ratio matters, and the selector finds a good one without tuning
Added after H3 was not supported (the selector did not beat random mixing at its own ratio, p = 0.324). Random
mixing at other ratios should do worse than at the selector's ratio and worse than the selector.
```bash
python simulate_fl.py -a random_mix --mix-p 0.10 -o results/h3b_mix_p010
python simulate_fl.py -a random_mix --mix-p 0.50 -o results/h3b_mix_p050
python compare_experiments.py --dirs results/h1_select results/h3_random_mix results/h3b_mix_p010 results/h3b_mix_p050 --labels Selector Mix0.32 Mix0.10 Mix0.50 -o results/hypothesis3b
python compare_experiments.py --dirs results/h3_random_mix results/h3b_mix_p010 results/h3b_mix_p050 --labels Mix0.32 Mix0.10 Mix0.50 -o results/hypothesis3b_ratio
```
Supported when the selector beats both Mix0.10 and Mix0.50 (`hypothesis3b`). `hypothesis3b_ratio` shows whether 0.32
itself beats the other ratios.

#### Hypothesis 4: the selector reaches AUPRC 0.70 in fewer rounds (no new runs)
```bash
python compare_experiments.py --dirs results/h1_select results/h1_fedavg results/h1_fedyogi --labels Selector FedAvg FedYogi --best-of FedAvg,FedYogi --metric rounds_to_target --target 0.70 -o results/hypothesis4
```

#### Hypothesis 5: the selector helps at every level of client heterogeneity
The label-skewed level is H1. Run the same three methods on the IID split and on the patient split:
```bash
python simulate_fl.py -a fedavg --data-dir ../datasets/iid -o results/h5_iid_fedavg
python simulate_fl.py -a fedyogi --data-dir ../datasets/iid -o results/h5_iid_fedyogi
python simulate_fl.py -a select --data-dir ../datasets/iid -o results/h5_iid_select
python simulate_fl.py -a fedavg --data-dir ../datasets/data/non_iid -o results/h5_patient_fedavg
python simulate_fl.py -a fedyogi --data-dir ../datasets/data/non_iid -o results/h5_patient_fedyogi
python simulate_fl.py -a select --data-dir ../datasets/data/non_iid -o results/h5_patient_select
python compare_experiments.py --dirs results/h5_iid_select results/h5_iid_fedavg results/h5_iid_fedyogi --labels Selector FedAvg FedYogi --best-of FedAvg,FedYogi --test ni -o results/hypothesis5_iid
python compare_experiments.py --dirs results/h5_patient_select results/h5_patient_fedavg results/h5_patient_fedyogi --labels Selector FedAvg FedYogi --best-of FedAvg,FedYogi --test ni -o results/hypothesis5_patient
```
Supported when the selector is at least as good as the best static rule at every level (H1 covers label skew).

#### Hypothesis 6: validation loss is an adequate selection score
```bash
python simulate_fl.py -a select --select-score auprc -o results/h6_select_auprc
python simulate_fl.py -a select --select-score brier -o results/h6_select_brier
python compare_experiments.py --dirs results/h6_select_auprc results/h1_select --labels ScoreAUPRC ScoreLoss -o results/hypothesis6_auprc
python compare_experiments.py --dirs results/h6_select_brier results/h1_select --labels ScoreBrier ScoreLoss -o results/hypothesis6_brier
```
Supported (loss stays the score) when neither alternative beats loss.

### Paper 2: selection plus training on the server's data

#### Hypothesis 7: with the full server set, select-then-train is at least as good as either step alone
```bash
python simulate_fl.py -a train -o results/h7_train
python simulate_fl.py -a select_train -o results/h7_select_train
python simulate_fl.py -a train_select --select-frac 0.2 -o results/h7_train_select
python compare_experiments.py --dirs results/h7_select_train results/h1_select results/h7_train results/h7_train_select results/h1_fedavg results/h1_fedyogi --labels SelectTrain Selector Training TrainSelect FedAvg FedYogi --best-of Selector,Training --best-of FedAvg,FedYogi --test ni -o results/hypothesis7
```
Deciding row: `best of Selector,Training`.

#### Hypothesis 8: the same with a small server set (280 records, matched)
```bash
python simulate_fl.py -a select --val-size 280 -o results/h8_select
python simulate_fl.py -a train --val-size 280 -o results/h8_train
python simulate_fl.py -a select_train --val-size 280 -o results/h8_select_train
python simulate_fl.py -a train_select --select-frac 0.2 --val-size 280 -o results/h8_train_select
python compare_experiments.py --dirs results/h8_select_train results/h8_select results/h8_train results/h8_train_select --labels SelectTrain Selector Training TrainSelect --best-of Selector,Training --test ni -o results/hypothesis8
```

#### Hypothesis 9: the same with a skewed small server set (280 records, 50% AFib)
```bash
python simulate_fl.py -a select --val-size 280 --val-pos-share 0.5 -o results/h9_select
python simulate_fl.py -a train --val-size 280 --val-pos-share 0.5 -o results/h9_train
python simulate_fl.py -a select_train --val-size 280 --val-pos-share 0.5 -o results/h9_select_train
python simulate_fl.py -a train_select --select-frac 0.2 --val-size 280 --val-pos-share 0.5 -o results/h9_train_select
python compare_experiments.py --dirs results/h9_select_train results/h9_select results/h9_train results/h9_train_select results/h1_fedavg results/h1_fedyogi --labels SelectTrain Selector Training TrainSelect FedAvg FedYogi --best-of Selector,Training --best-of FedAvg,FedYogi --test ni -o results/hypothesis9
```
Two deciding rows: `best of Selector,Training` (H9) and `best of FedAvg,FedYogi` (H9r: robustness to a skewed server set).

#### Hypothesis 10: where training on V overtakes selection (server-set size sweep)
Sizes 280 (H8) and 2,198 (H1, H7) are done; run 50, 100, 560 and 1100 the same way, e.g. for 50:
```bash
python simulate_fl.py -a select --val-size 50 -o results/h10_v50_select
python simulate_fl.py -a train --val-size 50 -o results/h10_v50_train
python simulate_fl.py -a select_train --val-size 50 -o results/h10_v50_select_train
python compare_experiments.py --dirs results/h10_v50_select results/h10_v50_train --labels Selector Training --test ni -o results/hypothesis10_v50_select
python compare_experiments.py --dirs results/h10_v50_select_train results/h10_v50_select results/h10_v50_train --labels SelectTrain Selector Training --best-of Selector,Training --test ni -o results/hypothesis10_v50
```
Predictions: training beats selection at 2,198 (`--dirs results/h7_train results/h1_select`, superiority); selection is
at least as good as training at 50; select-then-train is at least as good as the better step at every size.

#### Hypothesis 11: a held-out scoring split fixes select-then-train on a small server set
```bash
python simulate_fl.py -a select_train --val-size 280 --select-frac 0.2 -o results/h11_select_train_split
python simulate_fl.py -a select_train --val-size 280 --val-pos-share 0.5 --select-frac 0.2 -o results/h11_select_train_split_afib50
python compare_experiments.py --dirs results/h11_select_train_split results/h8_select_train --labels Split NoSplit -o results/hypothesis11_matched
python compare_experiments.py --dirs results/h11_select_train_split_afib50 results/h9_select_train --labels Split NoSplit -o results/hypothesis11_afib50
```

### Paper 3: measurement vs learned controllers under poisoning (20 of 100 clients malicious)

#### Hypothesis 12: standard attack (malicious clients upload 5x their honest update)
```bash
python simulate_fl.py -a fedavg --num-malicious 20 --attack standard -o results/h12_fedavg
python simulate_fl.py -a median --num-malicious 20 --attack standard -o results/h12_median
python simulate_fl.py -a krum --num-malicious 20 --attack standard -o results/h12_krum
python simulate_fl.py -a fedstrategist --fs-preset paper --num-malicious 20 --attack standard -o results/h12_fs_paper
python simulate_fl.py -a fedstrategist --fs-preset authors --num-malicious 20 --attack standard -o results/h12_fs_authors
python simulate_fl.py -a fedstrategist --fs-preset paper --fs-reward loss --num-malicious 20 --attack standard -o results/h12_fs_loss
python simulate_fl.py -a sara --num-malicious 20 --attack standard -o results/h12_sara
python simulate_fl.py -a select --rules fedavg,median,krum --num-malicious 20 --attack standard -o results/h12_select_robust
python compare_experiments.py --dirs results/h12_select_robust results/h12_fs_paper results/h12_fs_authors results/h12_fs_loss results/h12_sara results/h12_fedavg results/h12_median results/h12_krum --labels Selector FS-paper FS-authors FS-loss SARA FedAvg Median Krum --best-of FS-paper,FS-authors,FS-loss --best-of FS-paper,FS-authors,FS-loss,SARA --best-of FedAvg,Median,Krum --test ni -o results/hypothesis12
```
Deciding rows: `best of FS-paper,FS-authors,FS-loss` (vs FedStrategist) and `SARA`. The table also shows server cost.

#### Hypothesis 13: flipped attack (the malicious update also points the wrong way)
The same eight commands as H12 with `--flip` added and `h13_` folders, then the same comparison into `results/hypothesis13`.

#### Hypothesis 14: attacks that switch on and off (SARA's setting)
Attackers active in rounds 1-40 and 81-120:
```bash
python simulate_fl.py -a select --rules fedavg,median,krum --num-malicious 20 --attack standard --attack-rounds 1-40,81-120 -o results/h14_select_robust
python simulate_fl.py -a sara --num-malicious 20 --attack standard --attack-rounds 1-40,81-120 -o results/h14_sara
python simulate_fl.py -a fedstrategist --fs-preset paper --num-malicious 20 --attack standard --attack-rounds 1-40,81-120 -o results/h14_fs_paper
python simulate_fl.py -a fedstrategist --fs-preset authors --num-malicious 20 --attack standard --attack-rounds 1-40,81-120 -o results/h14_fs_authors
python simulate_fl.py -a fedstrategist --fs-preset paper --fs-reward loss --num-malicious 20 --attack standard --attack-rounds 1-40,81-120 -o results/h14_fs_loss
python compare_experiments.py --dirs results/h14_select_robust results/h14_sara results/h14_fs_paper results/h14_fs_authors results/h14_fs_loss --labels Selector SARA FS-paper FS-authors FS-loss --best-of FS-paper,FS-authors,FS-loss --test ni -o results/hypothesis14
```

#### Hypothesis 15: attack strength sweep
Repeat H12 with `--num-malicious 10`, `30` and `40`, and with `--attack stealth`, into folders such as
`results/h15_m30_standard_<method>`, and compare as in H12. Label flipping, trimmed mean, Multi-Krum and FLTrust are
not implemented yet.

### Not runnable yet (need code or data first)
| # | Hypothesis | Needs |
| :--- | :--- | :--- |
| 16 | A server set from one hospital: selection beats training | PTB-XL site metadata in the data loader |
| 17 | Select-then-train beats FLTrust and Zeno | FLTrust and Zeno rules |
| 18 | Oracle and predictability tests of the TMLR paper on ECG | FedProx client branches |
| 19 | H1, H7, H12 replicate on PhysioNet 2021 (clients by site) | the dataset and a 1-D ResNet |

**Checks before H16-H19:** client lr 0.2 and 0.3 on tuning seeds 1-2 (`--seeds 1-2 --lr 0.2 -o results/check_lr0.2`),
and removing the 3 test patients that also appear in client training data.

---

## 4. Outputs

Each run folder (`results/<name>/s<seed>/`) holds `metrics.csv` and `round_history.json` (one row per round: test
AUPRC, ROC-AUC, F2, F1, recall, precision, Brier, loss, confusion counts, validation loss, the rule kept, server time
and cost, latency and energy), `config.json`, `summary.json` (when finished), and plots: `auprc_vs_round.png`,
`loss_vs_round.png`, `accuracy_vs_round.png`, `confusion_matrix_latest.png`, `strategy_selection_freq.png`.
