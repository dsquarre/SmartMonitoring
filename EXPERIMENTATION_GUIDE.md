# SmartMonitoring Experimentation Guide

A concise reference for running Federated Learning experiments, customizing multi-objective rewards, persisting trained selectors, and generating 2D/3D Pareto trade-off plots.

---

## 1. CLI Parameters Reference (`simulate_fl.py`)

### General FL Controls
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--data-dir` | `data/iid` | Path to client dataset folder |
| `-n`, `--num-clients` | `10` | Total number of clients ($N$) |
| `-k`, `--select-k` | `max(5, 0.2N)` | Selected clients per round ($K$). Omit for dynamic calculation. |
| `-r`, `--rounds` | `5` | Total federated rounds ($R$) |
| `-e`, `--local-epochs` | `1` | Local epochs per round |
| `-b`, `--batch-clients` | `1` | Sequential client batch size (for low CPU memory consumption) |
| `--seed` | `42` | Random seed for reproducibility across seeds |

### Selector & Aggregator Selection
| Parameter | Options / Default | Description |
| :--- | :--- | :--- |
| `-a`, `--aggregator` | `fedavg` (default), `qfedavg`, `fedfv`, `fedadam`, `fedprox`, `krum`, `scaffold` | Global aggregation algorithm |
| `-s`, `--selector` | `random` (default), `linucb`, `wls-ts`, `dqn`, `hierarchical`, `oort` | Client selection algorithm |
| `-g`, `--gamma` | `0.95` (default, range `(0, 1.0]`) | Exponential discount factor. Set `-g 1.0` for undiscounted RL. |

### Reward Weight Customization & Accuracy-Only Mode
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--w-loss` | `1.0` | Sub-controller client loss reward weight |
| `--w-acc` | `10.0` | Meta-controller accuracy improvement weight |
| `--w-lat` | `1.0` | Latency penalty weight ($w_L$) |
| `--w-eng` | `1.0` | Energy penalty weight ($w_E$) |
| `--accuracy-only` | `False` | Disables latency and energy penalties (`w_lat=0.0`, `w_eng=0.0`) |

### Selector Persistence & Policy Freezing
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--save-selector` | `None` | Saves learned selector state ($A, b$, weights) to a `.npz` file |
| `--load-selector` | `None` | Loads pre-trained selector parameters from a `.npz` file |
| `--freeze-selector` | `False` | Freezes policy updates (`update_policy` becomes no-op for weights) |

---

## 2. Experiment Workflows

### Workflow 1: Multi-Objective vs. Accuracy-Only Optimization
Compare multi-objective optimization ($w_L=1.0, w_E=1.0$) against accuracy-only optimization ($w_L=0, w_E=0$).

```bash
# Run 1: Multi-Objective Optimization
conda run -n web python simulate_fl.py \
  --data-dir data/iid -n 10 -r 10 --seed 42 \
  -s hierarchical \
  --w-lat 1.0 --w-eng 1.0 \
  -o results/exp_multiobj_s42

# Run 2: Accuracy-Only Optimization
conda run -n web python simulate_fl.py \
  --data-dir data/iid -n 10 -r 10 --seed 42 \
  -s hierarchical \
  --accuracy-only \
  -o results/exp_acconly_s42

# Plot Pareto & Comparative Overlays
conda run -n web python compare_experiments.py \
  --dirs results/exp_multiobj_s42 results/exp_acconly_s42 \
  --labels "Multi-Objective Pareto" "Accuracy-Only Baseline" \
  -o results/pareto_comparison
```

---

### Workflow 2: Aggregator Comparison with a Frozen Trained Selector
Train a client selector once, save its parameters, and evaluate different fixed aggregators holding the selector policy constant.

```bash
# Step 1: Train and Save Client Selector
conda run -n web python simulate_fl.py \
  --data-dir data/iid -n 10 -r 10 --seed 42 \
  -s linucb \
  --save-selector results/trained_selector.npz \
  -o results/train_selector

# Step 2: Evaluate Aggregators with Frozen Selector
# A. Dynamic Meta-Aggregator + Frozen Selector
conda run -n web python simulate_fl.py \
  --data-dir data/iid -n 10 -r 5 --seed 42 \
  -s hierarchical \
  --load-selector results/trained_selector.npz --freeze-selector \
  -o results/eval_meta_aggregator

# B. Fixed FedAvg + Frozen Selector
conda run -n web python simulate_fl.py \
  --data-dir data/iid -n 10 -r 5 --seed 42 \
  -s linucb -a fedavg \
  --load-selector results/trained_selector.npz --freeze-selector \
  -o results/eval_fedavg

# C. Fixed FedProx + Frozen Selector
conda run -n web python simulate_fl.py \
  --data-dir data/iid -n 10 -r 5 --seed 42 \
  -s linucb -a fedprox \
  --load-selector results/trained_selector.npz --freeze-selector \
  -o results/eval_fedprox

# Step 3: Generate Multi-Strategy Comparison
conda run -n web python compare_experiments.py \
  --dirs results/eval_meta_aggregator results/eval_fedavg results/eval_fedprox \
  --labels "Meta-Aggregator" "FedAvg" "FedProx" \
  -o results/aggregator_ablation_comparison
```

---

### Workflow 3: Multi-Seed Statistical Hypothesis Testing
Run multiple random seeds and perform paired $t$-test / Wilcoxon signed-rank analysis.

```bash
# Multi-seed runs for Strategy A
for seed in 42 43 44; do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 5 --seed $seed -s hierarchical -o results/stratA_s${seed}
done

# Multi-seed runs for Strategy B
for seed in 42 43 44; do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 5 --seed $seed -s linucb -a fedavg -o results/stratB_s${seed}
done

# Perform Paired Hypothesis Testing & Plot Shaded CIs
conda run -n web python compare_experiments.py \
  --dirs "results/stratA_s*" "results/stratB_s*" \
  --labels "Hierarchical RL" "LinUCB + FedAvg" \
  -o results/multi_seed_hypothesis_analysis
```

---

## 3. Generated Analysis Outputs (`compare_experiments.py`)

Running `compare_experiments.py` populates the output folder with:
- **`comparison_loss_overlay.png`**: Mean loss curves with shaded $\pm 1 \text{ stderr}$ confidence bands.
- **`comparison_f2_accuracy_overlay.png`**: Mean $F_2$-Score accuracy curves.
- **`comparison_pareto_accuracy_vs_energy.png`**: 2D Accuracy vs Energy Pareto chart (mean $\pm 95\%$ CI).
- **`comparison_pareto_accuracy_vs_latency.png`**: 2D Accuracy vs Latency Pareto chart.
- **`comparison_pareto_accuracy_vs_joint_cost.png`**: 2D Accuracy vs Energy-Delay Product (EDP) joint cost.
- **`comparison_pareto_3d_accuracy_vs_energy_latency.png`**: 3D Tradeoff plot (Latency $\times$ Energy $\times$ Accuracy).
- **`hypothesis_test_results.csv`**: Paired $t$-test statistic, $p$-value, and Wilcoxon signed-rank test results.
- **`comparison_summary.csv`**: Aggregated final round metrics per strategy group.
