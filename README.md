# SmartMonitoring

Federated learning for atrial-fibrillation detection from 12-lead ECGs (PTB-XL), with:
- **Oort** client selection;
- a server that **selects, then trains**. Each round it builds the FedAvg and FedYogi aggregates from the same client
  updates, keeps the one with the lower loss on its own validation data, then trains that model on the same data.

Every method is a plugin (client selector, server rule, server controller, client attack) in
`distributed/server/plugins`. One system runs them all, so methods can be compared by changing a name. The
repository has two hosts for the same plugins:

| Folder | What it is | Use it to |
|---|---|---|
| [`simple/`](simple) | A single-process simulation: the ECG model, data split, device profiles, a runner and a comparison script | Run the papers' experiments (many seeds, GPU) |
| [`distributed/`](distributed) | The original deployable system (FastAPI coordinator, Redis, Celery worker, S3, WebSocket clients, React dashboard, AWS CloudFormation, Docker), with the RL code replaced by the plugins | Run real clients on separate machines or containers |

Plugins: Oort and random selection; FedAvg, FedYogi, coordinate-wise median and Krum; the controllers `select` (the
TMLR supplement's greedy-val), `train` (server-ft), `select_train` (ours), `train_select` and `fedstrategist`; and
the FedStrategist attacks. See [distributed/README.md](distributed/README.md#plugins).

## Quick start

```bash
# simple: one seed of the method on the non-IID partitions (hypotheses and commands: simple/EXPERIMENTS.md)
cd simple && python simulate_fl.py -a select_train --seed 42 -o results/test

# distributed: whole stack in Docker (Redis, LocalStack as S3, server, Celery, 4 clients); see distributed/README.md
cd distributed && python scripts/make_server_val.py && docker compose up --build
```

## Data

`datasets/` is not tracked by git. It holds `iid/` and `non_iid/` folders, each with 100 `client_<i>.npz`
partitions of PTB-XL (`X_train`, `y_train`, `X_test`, `y_test`; X has shape (n, 1000, 12) and y is AFib or not).

## Background

The method builds on the TMLR supplement (`tmlr_supplement.zip`, not tracked):
- Its Oort port is reused unchanged.
- Its "greedy-val" selector (FedAvg vs FedYogi chosen by validation loss) and its "server-ft" baseline (training on
  the server's validation set) are the two parts that `select_train` combines.

The earlier hierarchical-RL selector and its experiments were removed. Archives:
- `archive_rl_experiments_2026-10-30.zip`: the last working tree, including uncommitted work.
- The `experiments-v2-v3` and `paper-final-v1` branches.
