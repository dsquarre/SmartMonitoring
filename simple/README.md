# SmartMonitoring: simple version

A single-process simulation for testing quickly on one machine and for running the papers' experiments. It needs no
Redis, Celery or S3; the server and all the clients run in one Python process.

The methods are not implemented here. They are the distributed system's plugins (`../distributed/server/plugins`),
which `smartmon` loads as the package `plugins`, so a result from the simulator is a result of the code the deployed
system runs. See [the distributed README](../distributed/README.md#plugins) for the plugin list and for how to add one.

```
smartmon/
  __init__.py     loads ../distributed/server/plugins as `plugins`
  ecg_model.py    ECG AFib 1-D CNN, local training (plain SGD, as the TMLR supplement), metrics
  data.py         client files -> client train splits; pooled client test splits -> server V + test T
  profiles.py     client hardware tiers -> per-round latency / energy (Oort durations, cost reporting)
  simulation.py   the federated loop: selector -> local training (+ attacks) -> controller; CSV, checkpoints
run.py            CLI: strategies x seeds
compare.py        summary table (accuracy, energy, server cost, rule mix) + paired comparisons with a target
tests/            python -m unittest discover -s tests -t .
```

## Strategies

| `--strategy` | What the server does | Paper |
|---|---|---|
| `fedavg`, `fedyogi`, `median`, `krum` | The same rule every round | baselines |
| `select` | Builds every rule's candidate, keeps the lowest loss on V (the supplement's greedy-val) | 1, 2 |
| `train` | FedAvg (or `{"base": "fedyogi"}`), then training on V (the supplement's server-ft) | 1, 2 |
| `select_train` | Selects as above, then trains the kept model on V (ours) | 2 |
| `train_select` | Every rule with and without training on V competes on V; use `--select-frac` | 2 |
| `fedstrategist` | A contextual bandit picks FedAvg, median or Krum each round (Haque et al. 2025) | 3 |
| `sara` | UCB1 picks FedAvg, median or Krum, rewarded with accuracy − loss on V (Hu et al. 2025, reconstructed) | 3 |

The supplement found two things. Selection alone beats every static aggregator. Server training beats selection when
V is representative, and loses to it when V is strongly skewed and the step is tuned on V. It left the
**combination untested**, and `select_train` and `train_select` are the two ways to combine them.

For paper 3, `select` with `--rules-for-select fedavg,median,krum` measures every robust rule directly; FedStrategist
learns to pick among the same rules. Every run logs the server's cost of the decision per round (`server_s`,
`server_evals`, `server_finetunes`), and `compare.py` reports it.

**Data.**
- Each `client_<i>.npz` keeps its train split.
- The clients' test splits are pooled and split once per seed into V (`--val-fraction`, default 50%) and the reported
  test set T. T is never used for any decision.
- `--val-size N` subsamples V to test small server sets.
- `--select-frac F` keeps a share of V for scoring candidates only and trains on the rest.

**Poisoning.** `--num-malicious M` fixes M malicious clients per seed; they attack whenever Oort selects them
(`--attack standard|stealth`, `--flip`). Krum is told how many of the round's clients are malicious, as in
FedStrategist.

**Metrics** (logged every round on T): AUPRC (primary), ROC-AUC, F1, F2, precision, recall and accuracy at threshold
0.3, Brier score, BCE, and client latency and energy from the device tiers.

## Usage

Use the `py310` environment (TF 2.10 on the GPU).

```bash
cd simple
python run.py --strategy select_train --seeds 42 --rounds 100                         # one run
python run.py --strategy fedavg,fedyogi,select,train,select_train --seeds 42-46 --out results/main
python run.py --strategy train_select --select-frac 0.2 --seeds 42-46 --out results/main
python run.py --strategy fedavg,median,krum,fedstrategist,select --rules-for-select fedavg,median,krum \
              --num-malicious 20 --attack standard --seeds 42-46 --out results/poison
python run.py --strategy fedstrategist --controller-args '{"preset": "authors"}' --name fs-authors --out results/poison
python compare.py results/main                                                        # writes comparison.md
python -m unittest discover -s tests -t .                                            # ~35 s, CPU
```

Default settings:
- `../datasets/non_iid`, 100 clients, K = 10, E = 5, batch 32.
- Client SGD learning rate 0.1 and per-record, per-lead z-scoring (`--normalize record`), both chosen by the ECG
  calibration on tuning seeds 1-2 (`experiments/ecg_protocol.md` section 3).
- FedYogi η = 0.01, τ = 1e-3; server training 1 epoch at the client learning rate.
- 100 rounds.

`--controller-args` goes to every strategy in the list, so use it with a single strategy and `--name`. Runs that
have finished are skipped. An interrupted run resumes from `<name>_s<seed>.ckpt.pkl` the next time you run the same
command.
