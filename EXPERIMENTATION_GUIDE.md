# SmartMonitoring Experimentation Guide

A concise reference for running Federated Learning experiments, customizing multi-objective rewards, persisting trained selectors, and generating 2D/3D Pareto trade-off plots with 30-seed paired hypothesis testing.

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
| `--gamma-meta` | `None` (defaults to `--gamma`) | Level 1 Meta-Aggregator discount factor $\gamma_{\text{meta}}$ |
| `--gamma-sub` | `None` (defaults to `--gamma`) | Level 2 Sub-controller discount factor $\gamma_{\text{sub}}$ |

### Reward Weight Customization & Accuracy-Only Mode
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--w-loss` | `1.0` | Sub-controller client loss reward weight |
| `--w-acc` | `10.0` | Meta-controller accuracy improvement weight |
| `--w-lat` | `1.0` | Latency penalty weight ($w_L$) |
| `--w-eng` | `1.0` | Energy penalty weight ($w_E$) |
| `--accuracy-only` | `False` | Disables latency and energy penalties (`w_lat=0.0`, `w_eng=0.0`) |

### Hardware Profile Tiers & Profile Calculations
Clients are deterministically assigned one of 4 standard hardware tiers at startup using the `--seed` parameter (`random.Random(seed)`):

| Hardware Tier | CPU Frequency | Power Draw ($P_{\text{draw}}$) | Transmit Power ($P_{\text{tx}}$) | Network Rate ($r_{\text{trans}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (High-Performance)** | 2.5 GHz | 8.0 Watts | 0.5 Watts | 25 Mbps |
| **Tier 2 (Mid-Range)** | 1.8 GHz | 4.5 Watts | 0.3 Watts | 15 Mbps |
| **Tier 3 (Constrained)** | 1.2 GHz | 2.5 Watts | 0.15 Watts | 8 Mbps |
| **Tier 4 (Low-Power IoT)** | 0.8 GHz | 1.2 Watts | 0.1 Watts | 3 Mbps |

**Latency & Energy Formulas**:
* $\text{Training Latency } t_{\text{comp}} = t_{\text{wall}} \times \left( \frac{2.0 \text{ GHz}}{f_{\text{client}}} \right)$
* $\text{Training Energy } E_{\text{comp}} = t_{\text{comp}} \times P_{\text{draw}}$
* $\text{Transmission Latency } t_{\text{trans}} = \frac{\text{Model Size (bits)}}{r_{\text{trans}}}$
* $\text{Transmission Energy } E_{\text{trans}} = t_{\text{trans}} \times P_{\text{tx}}$

---

## 2. 30-Seed Hypothesis Testing Workflows

The following bash commands run 30 distinct seeds (`42` to `71`) and execute paired $t$-tests and Wilcoxon signed-rank tests for:
1. **Max/Final $F_2$ Accuracy in Round $T$**
2. **Total Energy across all rounds**
3. **Total/Average Latency**
4. **Round to reach 90% Target Accuracy**

---

### Hypothesis 1: Proposed Custom State ($G_t, X_t$) vs. Simple Oort-Style State

```bash
# 1. Run 30 Seeds for Proposed Custom State (Hierarchical RL)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical -o results/h1_custom/s${seed}
done

# 2. Run 30 Seeds for Simple Oort-Style State (Oort RL)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s oort -o results/h1_oort/s${seed}
done

# 3. Paired Hypothesis Test & Pareto Comparison
conda run -n web python compare_experiments.py \
  --dirs "results/h1_custom/s*" "results/h1_oort/s*" \
  --labels "Proposed Custom State" "Oort-Style State" \
  -o results/hypothesis1_results
```

---

### Hypothesis 2: Decoupled Discounted vs. Undiscounted LinUCB/TS

```bash
# 1. Both Discounted (gamma_meta=0.95, gamma_sub=0.95)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical --gamma-meta 0.95 --gamma-sub 0.95 -o results/h2_discount_both/s${seed}
done

# 2. Level 1 Meta-Controller Discounted Only (gamma_meta=0.95, gamma_sub=1.0)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical --gamma-meta 0.95 --gamma-sub 1.0 -o results/h2_meta_only/s${seed}
done

# 3. Level 2 Sub-Controller Discounted Only (gamma_meta=1.0, gamma_sub=0.95)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical --gamma-meta 1.0 --gamma-sub 0.95 -o results/h2_sub_only/s${seed}
done

# 4. Both Undiscounted (gamma_meta=1.0, gamma_sub=1.0)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical --gamma-meta 1.0 --gamma-sub 1.0 -o results/h2_undiscounted_both/s${seed}
done

# 5. Paired Hypothesis Test & 4-Overlay Comparison
conda run -n web python compare_experiments.py \
  --dirs "results/h2_discount_both/s*" "results/h2_meta_only/s*" "results/h2_sub_only/s*" "results/h2_undiscounted_both/s*" \
  --labels "Both Discounted (0.95)" "Meta Discounted Only" "Sub Discounted Only" "Both Undiscounted (1.0)" \
  -o results/hypothesis2_results
```

---

### Hypothesis 3: Aggregator Selection is Valuable and Converges (8-Way Comparison)

```bash
# Step A: Pre-train and Save Selector over 30 Seeds
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s linucb --save-selector results/h3_saved_selectors/selector_s${seed}.npz -o results/h3_train/s${seed}
done

# Step B: Run Frozen Selector with Meta-Aggregator & 7 Fixed Aggregators across 30 Seeds
for seed in $(seq 42 71); do
  # 1. Dynamic Meta-Aggregator + Frozen Selector
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 5 --seed $seed -s hierarchical --load-selector results/h3_saved_selectors/selector_s${seed}.npz --freeze-selector -o results/h3_meta/s${seed}

  # 2-8. Fixed Aggregators + Frozen Selector
  for agg in fedavg fedprox scaffold krum fedadam fedfv qfedavg; do
    conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 5 --seed $seed -s linucb -a $agg --load-selector results/h3_saved_selectors/selector_s${seed}.npz --freeze-selector -o results/h3_${agg}/s${seed}
  done
done

# Step C: 8-Way Overlay & Pareto Hypothesis Comparison
conda run -n web python compare_experiments.py \
  --dirs "results/h3_meta/s*" "results/h3_fedavg/s*" "results/h3_fedprox/s*" "results/h3_scaffold/s*" "results/h3_krum/s*" "results/h3_fedadam/s*" "results/h3_fedfv/s*" "results/h3_qfedavg/s*" \
  --labels "Meta-Aggregator" "FedAvg" "FedProx" "SCAFFOLD" "Krum" "FedAdam" "FedFV" "qFedAvg" \
  -o results/hypothesis3_results
```

---

### Hypothesis 4: Multi-Objective Pareto vs. Accuracy-Only Objective

```bash
# 1. Run 30 Seeds for Multi-Objective Pareto (w_lat=1.0, w_eng=1.0)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical --w-lat 1.0 --w-eng 1.0 -o results/h4_multiobj/s${seed}
done

# 2. Run 30 Seeds for Accuracy-Only Objective (w_lat=0.0, w_eng=0.0)
for seed in $(seq 42 71); do
  conda run -n web python simulate_fl.py --data-dir data/iid -n 10 -r 10 --seed $seed -s hierarchical --accuracy-only -o results/h4_acconly/s${seed}
done

# 3. Paired Hypothesis Test & Pareto Comparison
conda run -n web python compare_experiments.py \
  --dirs "results/h4_multiobj/s*" "results/h4_acconly/s*" \
  --labels "Multi-Objective Pareto" "Accuracy-Only Baseline" \
  -o results/hypothesis4_results
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
- **`hypothesis_test_results.csv`**: Paired $t$-test statistic, $p$-value, and Wilcoxon signed-rank test results for:
  1. Final Loss ($L_T$)
  2. Final $F_2$ Score (Accuracy)
  3. Total Energy
  4. Total Latency
  5. Round to 90% Target Accuracy
- **`comparison_summary.csv`**: Aggregated final round metrics per strategy group.
